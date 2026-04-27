// Adreno OpenCL backend implementing IBackend.
//
// All five kernels (rotate, mse_encode, mse_score, qjl_score, value_dequant)
// run on the GPU. Kernel sources are compiled in as const byte arrays via the
// embed_resource CMake helper. libOpenCL.so is dlopen'd at init time.

#include "turboquant/backend.hpp"

#include "opencl_loader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" {
extern const unsigned char tq_opencl_mse_score[];     extern const size_t tq_opencl_mse_score_size;
extern const unsigned char tq_opencl_qjl_score[];     extern const size_t tq_opencl_qjl_score_size;
extern const unsigned char tq_opencl_mse_encode[];    extern const size_t tq_opencl_mse_encode_size;
extern const unsigned char tq_opencl_value_dequant[]; extern const size_t tq_opencl_value_dequant_size;
extern const unsigned char tq_opencl_rotate[];        extern const size_t tq_opencl_rotate_size;
}

namespace turboquant {
namespace {

#define CL_OK(call)                                                            \
    do {                                                                        \
        cl_int err__ = (call);                                                  \
        if (err__ != CL_SUCCESS) {                                              \
            std::fprintf(stderr, "[opencl] %s -> %d at %s:%d\n",                \
                         #call, (int)err__, __FILE__, __LINE__);                \
            return;                                                             \
        }                                                                       \
    } while (0)

class OpenCLBackend final : public IBackend {
public:
    const char* name() const override { return "opencl"; }

    bool init() override {
        if (!load_opencl(&api_)) return false;

        cl_uint nplat = 0;
        if (api_.GetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) {
            std::fprintf(stderr, "[opencl] no platforms\n");
            return false;
        }
        std::vector<cl_platform_id> plats(nplat);
        api_.GetPlatformIDs(nplat, plats.data(), nullptr);

        cl_device_id dev = nullptr;
        for (cl_platform_id p : plats) {
            cl_uint nd = 0;
            if (api_.GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd > 0) {
                api_.GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
                platform_ = p;
                break;
            }
        }
        if (!dev) {
            std::fprintf(stderr, "[opencl] no GPU device\n");
            return false;
        }
        device_ = dev;

        cl_int err = 0;
        ctx_ = api_.CreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (!ctx_ || err != CL_SUCCESS) return false;
        queue_ = api_.CreateCommandQueue(ctx_, device_, 0, &err);
        if (!queue_ || err != CL_SUCCESS) return false;

        // Print device info to stdout for diagnostics.
        char buf[256] = {0};
        api_.GetDeviceInfo(device_, CL_DEVICE_NAME,   sizeof(buf), buf, nullptr);
        std::fprintf(stderr, "[opencl] device: %s\n", buf);
        std::memset(buf, 0, sizeof(buf));
        api_.GetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
        std::fprintf(stderr, "[opencl] vendor: %s\n", buf);

        // Compile each kernel program.
        program_score_  = build_program(reinterpret_cast<const char*>(tq_opencl_mse_score),     tq_opencl_mse_score_size);
        program_qjl_    = build_program(reinterpret_cast<const char*>(tq_opencl_qjl_score),     tq_opencl_qjl_score_size);
        program_encode_ = build_program(reinterpret_cast<const char*>(tq_opencl_mse_encode),    tq_opencl_mse_encode_size);
        program_vdq_    = build_program(reinterpret_cast<const char*>(tq_opencl_value_dequant), tq_opencl_value_dequant_size);
        program_rot_    = build_program(reinterpret_cast<const char*>(tq_opencl_rotate),        tq_opencl_rotate_size);
        if (!program_score_ || !program_qjl_ || !program_encode_ || !program_vdq_ || !program_rot_) return false;

        kernel_score_uchar_ = api_.CreateKernel(program_score_,  "mse_score_uchar",   &err);
        kernel_score_u32_b3_= api_.CreateKernel(program_score_,  "mse_score_u32_b3",  &err);
        kernel_qjl_         = api_.CreateKernel(program_qjl_,    "qjl_score",         &err);
        kernel_encode_uchar_= api_.CreateKernel(program_encode_, "mse_encode_uchar",  &err);
        kernel_encode_u32_b3_=api_.CreateKernel(program_encode_, "mse_encode_u32_b3", &err);
        kernel_vdq_         = api_.CreateKernel(program_vdq_,    "value_dequant",     &err);
        kernel_rot_         = api_.CreateKernel(program_rot_,    "tq_rotate",         &err);
        return kernel_score_uchar_ && kernel_score_u32_b3_ && kernel_qjl_
            && kernel_encode_uchar_ && kernel_encode_u32_b3_
            && kernel_vdq_ && kernel_rot_;
    }

    ~OpenCLBackend() override {
        // P2 Stage B: drop any pooled K-cache buffers we still hold. The
        // session layer normally calls release_keys() before destruction,
        // but a hard process exit can skip that — be defensive so we
        // don't leak GPU-side allocations across hot-restart of the host.
        {
            std::lock_guard<std::mutex> lk(key_pool_mu_);
            for (auto& kv : key_pool_) {
                if (kv.second.buf) api_.ReleaseMemObject(kv.second.buf);
            }
            key_pool_.clear();
            buf_to_shape_.clear();
        }

        for (cl_kernel k : {kernel_score_uchar_, kernel_score_u32_b3_, kernel_qjl_,
                            kernel_encode_uchar_, kernel_encode_u32_b3_,
                            kernel_vdq_, kernel_rot_}) {
            if (k) api_.ReleaseKernel(k);
        }
        for (cl_program p : {program_score_, program_qjl_, program_encode_,
                             program_vdq_, program_rot_}) {
            if (p) api_.ReleaseProgram(p);
        }
        if (queue_) api_.ReleaseCommandQueue(queue_);
        if (ctx_)   api_.ReleaseContext(ctx_);
        unload_opencl(&api_);
    }

    void rotate(const float* in, const float* Pi,
                int n, int D, float* out) override {
        cl_int err;
        cl_mem mIn  = make_buf(CL_MEM_READ_ONLY,  n * D * sizeof(float), in,  &err);
        cl_mem mPi  = make_buf(CL_MEM_READ_ONLY,  D * D * sizeof(float), Pi,  &err);
        cl_mem mOut = make_buf(CL_MEM_WRITE_ONLY, n * D * sizeof(float), nullptr, &err);
        cl_uint a = 0;
        api_.SetKernelArg(kernel_rot_, a++, sizeof(mIn),  &mIn);
        api_.SetKernelArg(kernel_rot_, a++, sizeof(mPi),  &mPi);
        api_.SetKernelArg(kernel_rot_, a++, sizeof(mOut), &mOut);
        api_.SetKernelArg(kernel_rot_, a++, sizeof(int),  &n);
        api_.SetKernelArg(kernel_rot_, a++, sizeof(int),  &D);
        size_t global[2] = { (size_t)D, (size_t)n };
        CL_OK(api_.EnqueueNDRangeKernel(queue_, kernel_rot_, 2, nullptr, global, nullptr, 0, nullptr, nullptr));
        CL_OK(api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, n * D * sizeof(float), out, 0, nullptr, nullptr));
        release_if_uncached(mIn);
        release_if_uncached(mPi);
        release_if_uncached(mOut);
    }

    void mse_encode(const float* rotated, const float* boundaries,
                    int N, int D, int bits, void* packed_out) override {
        const int n_boundaries = (1 << bits) - 1;
        cl_int err;
        cl_mem mRot = make_buf(CL_MEM_READ_ONLY,  N * D * sizeof(float), rotated,    &err);
        cl_mem mBnd = make_buf(CL_MEM_READ_ONLY,  n_boundaries * sizeof(float), boundaries, &err);

        if (bits == 3) {
            int n_words = (D + 9) / 10;
            cl_mem mOut = make_buf(CL_MEM_WRITE_ONLY, N * n_words * sizeof(uint32_t), nullptr, &err);
            cl_uint a = 0;
            api_.SetKernelArg(kernel_encode_u32_b3_, a++, sizeof(mRot), &mRot);
            api_.SetKernelArg(kernel_encode_u32_b3_, a++, sizeof(mBnd), &mBnd);
            api_.SetKernelArg(kernel_encode_u32_b3_, a++, sizeof(mOut), &mOut);
            api_.SetKernelArg(kernel_encode_u32_b3_, a++, sizeof(int), &N);
            api_.SetKernelArg(kernel_encode_u32_b3_, a++, sizeof(int), &D);
            api_.SetKernelArg(kernel_encode_u32_b3_, a++, sizeof(int), &n_words);
            size_t global[2] = { (size_t)n_words, (size_t)N };
            api_.EnqueueNDRangeKernel(queue_, kernel_encode_u32_b3_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
            api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, N * n_words * sizeof(uint32_t), packed_out, 0, nullptr, nullptr);
            release_if_uncached(mOut);
        } else {
            int vals_per_byte, eff_bits, mask;
            switch (bits) {
                case 1: vals_per_byte = 8; eff_bits = 1; mask = 0x1; break;
                case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3; break;
                case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF; break;
                case 8: default:
                        vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
            }
            int packed_d = (D + vals_per_byte - 1) / vals_per_byte;
            cl_mem mOut = make_buf(CL_MEM_WRITE_ONLY, N * packed_d, nullptr, &err);
            cl_uint a = 0;
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(mRot), &mRot);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(mBnd), &mBnd);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(mOut), &mOut);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(int), &N);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(int), &D);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(int), &packed_d);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(int), &n_boundaries);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(int), &eff_bits);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(int), &vals_per_byte);
            api_.SetKernelArg(kernel_encode_uchar_, a++, sizeof(int), &mask);
            size_t global[2] = { (size_t)packed_d, (size_t)N };
            api_.EnqueueNDRangeKernel(queue_, kernel_encode_uchar_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
            api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, N * packed_d, packed_out, 0, nullptr, nullptr);
            release_if_uncached(mOut);
        }
        release_if_uncached(mRot);
        release_if_uncached(mBnd);
    }

    void mse_score(const float* q_rot, const void* mse_packed,
                   const float* norms, const float* centroids,
                   int BH, int N, int D, int bits, float* out) override {
        cl_int err;
        cl_mem mQ   = make_buf(CL_MEM_READ_ONLY, BH * D * sizeof(float),       q_rot,     &err);
        cl_mem mNm  = make_buf(CL_MEM_READ_ONLY, BH * N * sizeof(float),       norms,     &err);
        cl_mem mCen = make_buf(CL_MEM_READ_ONLY, (1 << bits) * sizeof(float),  centroids, &err);
        cl_mem mOut = make_buf(CL_MEM_WRITE_ONLY, BH * N * sizeof(float),      nullptr,   &err);

        if (bits == 3) {
            int n_words = (D + 9) / 10;
            size_t bytes = (size_t)BH * N * n_words * sizeof(uint32_t);
            cl_mem mPk = make_buf(CL_MEM_READ_ONLY, bytes, mse_packed, &err);
            cl_uint a = 0;
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mQ),   &mQ);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mPk),  &mPk);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mNm),  &mNm);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mCen), &mCen);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mOut), &mOut);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &BH);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &N);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &D);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &n_words);
            size_t global[2] = { (size_t)N, (size_t)BH };
            api_.EnqueueNDRangeKernel(queue_, kernel_score_u32_b3_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
            api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, BH * N * sizeof(float), out, 0, nullptr, nullptr);
            release_if_uncached(mPk);
        } else {
            int vals_per_byte, eff_bits, mask;
            switch (bits) {
                case 1: vals_per_byte = 8; eff_bits = 1; mask = 0x1; break;
                case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3; break;
                case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF; break;
                case 8: default:
                        vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
            }
            int packed_d = (D + vals_per_byte - 1) / vals_per_byte;
            size_t bytes = (size_t)BH * N * packed_d;
            cl_mem mPk = make_buf(CL_MEM_READ_ONLY, bytes, mse_packed, &err);
            cl_uint a = 0;
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mQ),   &mQ);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mPk),  &mPk);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mNm),  &mNm);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mCen), &mCen);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mOut), &mOut);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &BH);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &N);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &D);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &packed_d);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &eff_bits);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &vals_per_byte);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &mask);
            size_t global[2] = { (size_t)N, (size_t)BH };
            api_.EnqueueNDRangeKernel(queue_, kernel_score_uchar_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
            api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, BH * N * sizeof(float), out, 0, nullptr, nullptr);
            release_if_uncached(mPk);
        }
        release_if_uncached(mQ);
        release_if_uncached(mNm);
        release_if_uncached(mCen);
        release_if_uncached(mOut);
    }

    // ---------------- P2 Stage B: pooled K-cache plumbing ----------------
    //
    // The default (unpooled) mse_score() above re-allocates a cl_mem for
    // the BH×N×packed_d mse-packed K state on every call via
    // CL_MEM_COPY_HOST_PTR. On Adreno that allocation alone runs 17–23 ms
    // per attention call — see backend_factory.cpp's prioritization
    // comment. The pool keeps one cl_mem per (BH, D, key_bits) shape
    // sized for a max-expected sequence; mse_score_pooled() then uploads
    // the new bytes via clEnqueueWriteBuffer (no allocation) and runs
    // the existing kernels against that buffer.
    //
    // The handle returned to the caller is the cl_mem cast to void* so
    // the SessionEntry can stash it without depending on cl.h. We also
    // keep a reverse map from cl_mem -> shape key so release_keys() can
    // look up the entry without a second hash.

    void* prepare_keys(int BH, int D, int key_bits) override {
        const uint64_t key = pool_key(BH, D, key_bits);
        std::lock_guard<std::mutex> lk(key_pool_mu_);
        auto it = key_pool_.find(key);
        if (it != key_pool_.end()) {
            // Already allocated for this shape — reuse.
            return reinterpret_cast<void*>(it->second.buf);
        }

        // Size the buffer for the worst-case sequence we expect to see in
        // a single decode. 16384 tokens at D=128 / key_bits=3 is
        // 16384 * 16 = 256 KiB per BH; for BH=32 that's 8 MiB — well
        // within Adreno's GPU memory budget. Bigger contexts can extend
        // this cap by re-running the prepare with a larger N (the pool
        // would need to grow, but we keep it simple for now).
        constexpr int kMaxTokens = 16384;
        const size_t row_bytes = packed_len_mse_for(D, key_bits);
        const size_t bytes = static_cast<size_t>(BH) *
                             static_cast<size_t>(kMaxTokens) * row_bytes;

        cl_int err = 0;
        cl_mem buf = api_.CreateBuffer(ctx_, CL_MEM_READ_ONLY, bytes,
                                       nullptr, &err);
        if (!buf || err != CL_SUCCESS) {
            std::fprintf(stderr,
                "[opencl] prepare_keys: CreateBuffer(%zu bytes) -> %d\n",
                bytes, (int)err);
            return nullptr;
        }

        KeyCachePool entry{ buf, bytes };
        key_pool_.emplace(key, entry);
        buf_to_shape_.emplace(buf, key);
        return reinterpret_cast<void*>(buf);
    }

    void release_keys(void* handle) override {
        if (!handle) return;
        cl_mem buf = reinterpret_cast<cl_mem>(handle);
        std::lock_guard<std::mutex> lk(key_pool_mu_);
        auto it_shape = buf_to_shape_.find(buf);
        if (it_shape == buf_to_shape_.end()) {
            // Foreign handle — refuse to free. Better than a double-free.
            std::fprintf(stderr,
                "[opencl] release_keys: unknown handle %p (ignoring)\n", handle);
            return;
        }
        const uint64_t key = it_shape->second;
        // Make sure no kernels are still referencing this buffer.
        if (queue_) api_.Finish(queue_);
        api_.ReleaseMemObject(buf);
        key_pool_.erase(key);
        buf_to_shape_.erase(it_shape);
    }

    void mse_score_pooled(const float* q_rot, const void* mse_packed,
                          const float* norms, const float* centroids,
                          int BH, int N, int D, int bits, float* out,
                          void* gpu_key_handle) override {
        if (!gpu_key_handle) {
            // No pool entry — fall back to the unpooled path. (Stateless
            // callers and any session whose prepare_keys() returned
            // nullptr land here.)
            mse_score(q_rot, mse_packed, norms, centroids,
                      BH, N, D, bits, out);
            return;
        }

        cl_mem mPk = reinterpret_cast<cl_mem>(gpu_key_handle);
        const size_t row_bytes = packed_len_mse_for(D, bits);
        const size_t bytes     = static_cast<size_t>(BH) *
                                 static_cast<size_t>(N) * row_bytes;

        // Defensive: if the pool entry is too small for the requested N,
        // fall back to the unpooled path rather than risk a write-past-
        // end (which on Adreno is an undefined-behavior crash, not a
        // clean error). 16384 tokens is a generous cap so this is rare.
        size_t pool_bytes = 0;
        {
            std::lock_guard<std::mutex> lk(key_pool_mu_);
            auto it_shape = buf_to_shape_.find(mPk);
            if (it_shape != buf_to_shape_.end()) {
                auto it_pool = key_pool_.find(it_shape->second);
                if (it_pool != key_pool_.end()) pool_bytes = it_pool->second.bytes;
            }
        }
        if (pool_bytes < bytes) {
            std::fprintf(stderr,
                "[opencl] mse_score_pooled: pool too small (%zu < %zu) "
                "for BH=%d N=%d D=%d bits=%d — falling back to unpooled\n",
                pool_bytes, bytes, BH, N, D, bits);
            mse_score(q_rot, mse_packed, norms, centroids,
                      BH, N, D, bits, out);
            return;
        }

        cl_int err = 0;
        cl_mem mQ   = make_buf(CL_MEM_READ_ONLY,  BH * D * sizeof(float),       q_rot,     &err);
        cl_mem mNm  = make_buf(CL_MEM_READ_ONLY,  BH * N * sizeof(float),       norms,     &err);
        cl_mem mCen = make_buf(CL_MEM_READ_ONLY,  (1 << bits) * sizeof(float),  centroids, &err);
        cl_mem mOut = make_buf(CL_MEM_WRITE_ONLY, BH * N * sizeof(float),       nullptr,   &err);

        // Upload the new K state into the pre-allocated pool buffer. This
        // is the only host->device transfer that scales with sequence
        // length — Pi/centroids/q_rot are all O(D) or O(BH·D), tiny next
        // to BH·N·packed_d.
        CL_OK(api_.EnqueueWriteBuffer(queue_, mPk, CL_FALSE, 0, bytes,
                                      mse_packed, 0, nullptr, nullptr));

        if (bits == 3) {
            int n_words = (D + 9) / 10;
            cl_uint a = 0;
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mQ),   &mQ);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mPk),  &mPk);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mNm),  &mNm);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mCen), &mCen);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(mOut), &mOut);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &BH);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &N);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &D);
            api_.SetKernelArg(kernel_score_u32_b3_, a++, sizeof(int),  &n_words);
            size_t global[2] = { (size_t)N, (size_t)BH };
            api_.EnqueueNDRangeKernel(queue_, kernel_score_u32_b3_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
            api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, BH * N * sizeof(float), out, 0, nullptr, nullptr);
        } else {
            int vals_per_byte, eff_bits, mask;
            switch (bits) {
                case 1: vals_per_byte = 8; eff_bits = 1; mask = 0x1; break;
                case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3; break;
                case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF; break;
                case 8: default:
                        vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
            }
            int packed_d = (D + vals_per_byte - 1) / vals_per_byte;
            cl_uint a = 0;
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mQ),   &mQ);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mPk),  &mPk);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mNm),  &mNm);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mCen), &mCen);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(mOut), &mOut);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &BH);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &N);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &D);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &packed_d);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &eff_bits);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &vals_per_byte);
            api_.SetKernelArg(kernel_score_uchar_, a++, sizeof(int),  &mask);
            size_t global[2] = { (size_t)N, (size_t)BH };
            api_.EnqueueNDRangeKernel(queue_, kernel_score_uchar_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
            api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, BH * N * sizeof(float), out, 0, nullptr, nullptr);
        }
        // NB: do NOT release mPk here — that's the pooled buffer and its
        // lifetime is owned by the SessionEntry.
        release_if_uncached(mQ);
        release_if_uncached(mNm);
        release_if_uncached(mCen);
        release_if_uncached(mOut);
    }

    void qjl_score(const float* q_sketch, const uint8_t* signs,
                   const float* res_norms, const float* mse_in,
                   int BH, int N, int D, float qjl_scale, float* out) override {
        const int packed_d_signs = (D + 7) / 8;
        cl_int err;
        cl_mem mQ  = make_buf(CL_MEM_READ_ONLY,  BH * D * sizeof(float), q_sketch,  &err);
        cl_mem mSg = make_buf(CL_MEM_READ_ONLY,  (size_t)BH * N * packed_d_signs, signs, &err);
        cl_mem mRn = make_buf(CL_MEM_READ_ONLY,  BH * N * sizeof(float), res_norms, &err);
        cl_mem mMs = make_buf(CL_MEM_READ_ONLY,  BH * N * sizeof(float), mse_in,    &err);
        cl_mem mOut= make_buf(CL_MEM_WRITE_ONLY, BH * N * sizeof(float), nullptr,   &err);
        cl_uint a = 0;
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(mQ),  &mQ);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(mSg), &mSg);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(mRn), &mRn);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(mMs), &mMs);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(mOut), &mOut);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(int), &BH);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(int), &N);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(int), &D);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(int), &packed_d_signs);
        api_.SetKernelArg(kernel_qjl_, a++, sizeof(float), &qjl_scale);
        size_t global[2] = { (size_t)N, (size_t)BH };
        api_.EnqueueNDRangeKernel(queue_, kernel_qjl_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, BH * N * sizeof(float), out, 0, nullptr, nullptr);
        release_if_uncached(mQ);  release_if_uncached(mSg);
        release_if_uncached(mRn); release_if_uncached(mMs); release_if_uncached(mOut);
    }

    void value_dequant(const uint8_t* packed, const float* scales, const float* zeros,
                       int N, int D, int bits, int group_size, float* out) override {
        int vals_per_byte, eff_bits, mask;
        switch (bits) {
            case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3; break;
            case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF; break;
            case 8: default:
                    vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
        }
        int n_groups = D / group_size;
        int packed_d = D / vals_per_byte;
        cl_int err;
        cl_mem mPk  = make_buf(CL_MEM_READ_ONLY, (size_t)N * packed_d, packed, &err);
        cl_mem mSc  = make_buf(CL_MEM_READ_ONLY, (size_t)N * n_groups * sizeof(float), scales, &err);
        cl_mem mZe  = make_buf(CL_MEM_READ_ONLY, (size_t)N * n_groups * sizeof(float), zeros,  &err);
        cl_mem mOut = make_buf(CL_MEM_WRITE_ONLY, (size_t)N * D * sizeof(float), nullptr, &err);
        cl_uint a = 0;
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(mPk),  &mPk);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(mSc),  &mSc);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(mZe),  &mZe);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(mOut), &mOut);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &N);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &D);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &eff_bits);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &vals_per_byte);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &mask);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &group_size);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &n_groups);
        api_.SetKernelArg(kernel_vdq_, a++, sizeof(int), &packed_d);
        size_t global[2] = { (size_t)D, (size_t)N };
        api_.EnqueueNDRangeKernel(queue_, kernel_vdq_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        api_.EnqueueReadBuffer(queue_, mOut, CL_TRUE, 0, (size_t)N * D * sizeof(float), out, 0, nullptr, nullptr);
        release_if_uncached(mPk);  release_if_uncached(mSc);
        release_if_uncached(mZe);  release_if_uncached(mOut);
    }

private:
    // Per-row mse_packed byte stride for a given (D, key_bits). Mirrors
    // the layout the existing mse_score() / quantizer.cpp produce so the
    // pooled buffer is always sized identically. Kept here rather than
    // calling into packing.hpp so the OpenCL backend stays self-contained
    // (matters when this TU is built in isolation for the Android shared
    // lib).
    static size_t packed_len_mse_for(int D, int key_bits) {
        if (key_bits == 3) {
            const int n_words = (D + 9) / 10;
            return static_cast<size_t>(n_words) * sizeof(uint32_t);
        }
        int vals_per_byte = 1;
        switch (key_bits) {
            case 1: vals_per_byte = 8; break;
            case 2: vals_per_byte = 4; break;
            case 4: vals_per_byte = 2; break;
            case 8: default: vals_per_byte = 1; break;
        }
        return static_cast<size_t>((D + vals_per_byte - 1) / vals_per_byte);
    }

    static uint64_t pool_key(int BH, int D, int key_bits) {
        return (static_cast<uint64_t>(BH) << 32) |
               (static_cast<uint64_t>(D)  << 16) |
                static_cast<uint64_t>(key_bits);
    }

    cl_mem make_buf(cl_mem_flags flags, size_t bytes, const void* host_ptr, cl_int* err) {
        if (host_ptr) {
            return api_.CreateBuffer(ctx_, flags | CL_MEM_COPY_HOST_PTR, bytes,
                                     const_cast<void*>(host_ptr), err);
        }
        return api_.CreateBuffer(ctx_, flags, bytes, nullptr, err);
    }

    void release_if_uncached(cl_mem buf) { api_.ReleaseMemObject(buf); }

    cl_program build_program(const char* src, size_t src_len) {
        cl_int err = 0;
        size_t lengths[1] = { src_len };
        const char* sources[1] = { src };
        cl_program p = api_.CreateProgramWithSource(ctx_, 1, sources, lengths, &err);
        if (!p || err != CL_SUCCESS) {
            std::fprintf(stderr, "[opencl] CreateProgramWithSource -> %d\n", err);
            return nullptr;
        }
        err = api_.BuildProgram(p, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_len = 0;
            api_.GetProgramBuildInfo(p, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
            std::string log(log_len, ' ');
            api_.GetProgramBuildInfo(p, device_, CL_PROGRAM_BUILD_LOG, log_len, log.data(), nullptr);
            std::fprintf(stderr, "[opencl] BuildProgram failed: %s\n", log.c_str());
            api_.ReleaseProgram(p);
            return nullptr;
        }
        return p;
    }

    OpenCLApi        api_{};
    cl_platform_id   platform_ = nullptr;
    cl_device_id     device_   = nullptr;
    cl_context       ctx_      = nullptr;
    cl_command_queue queue_    = nullptr;

    cl_program program_score_ = nullptr, program_qjl_ = nullptr,
               program_encode_= nullptr, program_vdq_ = nullptr,
               program_rot_   = nullptr;
    cl_kernel  kernel_score_uchar_  = nullptr,
               kernel_score_u32_b3_ = nullptr,
               kernel_qjl_          = nullptr,
               kernel_encode_uchar_ = nullptr,
               kernel_encode_u32_b3_= nullptr,
               kernel_vdq_          = nullptr,
               kernel_rot_          = nullptr;

    // P2 Stage B: the cl_mem pool.
    struct KeyCachePool {
        cl_mem buf   = nullptr;
        size_t bytes = 0;
    };
    std::mutex                              key_pool_mu_;
    std::unordered_map<uint64_t, KeyCachePool> key_pool_;     // shape -> pool entry
    std::unordered_map<cl_mem,  uint64_t>      buf_to_shape_;  // reverse lookup
};

}  // namespace

std::unique_ptr<IBackend> create_opencl_backend() {
    return std::make_unique<OpenCLBackend>();
}

}  // namespace turboquant
