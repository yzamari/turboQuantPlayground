// Compute-backend abstraction. Each backend (cpu_scalar, cpu_neon, qnn_htp,
// opencl, vulkan) implements these five kernels — one per existing Metal
// kernel in the Python reference plus the rotation matmul.
//
// Kernels operate on flat row-major buffers. Outer (batch, head) dimensions
// are flattened by the caller to BH = batch * heads.
//
// All shapes and bit-packing layouts must match types.hpp exactly.

#pragma once

#include <cstdint>
#include <memory>

namespace turboquant {

enum class BackendKind {
    CpuScalar,
    CpuNeon,
    QnnHtp,
    OpenCL,
    Vulkan,
};

const char* backend_kind_name(BackendKind);
BackendKind backend_kind_from_name(const char* name);  // returns CpuScalar if unknown

class IBackend {
public:
    virtual ~IBackend() = default;
    virtual const char* name() const = 0;

    // One-time setup: load drivers, build kernel programs, etc. Returns false
    // if the backend cannot run (driver missing, etc.) — caller falls back.
    virtual bool init() = 0;

    // out[n, D] = in[n, D] @ Pi^T   (Pi is row-major [D, D])
    virtual void rotate(const float* in,
                        const float* Pi,
                        int n, int D,
                        float* out) = 0;

    // Fused encode: rotated[N, D] (float) -> packed_out (bit-packed indices).
    // boundaries has size (2^bits - 1) — interior decision boundaries only.
    // Output buffer size is determined by bits and D — caller pre-sizes.
    virtual void mse_encode(const float* rotated,
                            const float* boundaries,
                            int N, int D, int bits,
                            void* packed_out) = 0;

    // Fused MSE attention score:
    //   out[bh, n] = norms[bh, n] * sum_j q_rot[bh, j] * centroids[idx[bh, n, j]]
    // mse_packed: bit-packed indices for [BH, N, packed_d]
    // centroids:  [2^bits]
    virtual void mse_score(const float* q_rot,
                           const void* mse_packed,
                           const float* norms,
                           const float* centroids,
                           int BH, int N, int D, int bits,
                           float* out) = 0;

    // Fused QJL correction score:
    //   out[bh, n] = mse_in[bh, n]
    //              + qjl_scale * res_norms[bh, n]
    //                          * sum_j q_sketch[bh, j] * sign[bh, n, j]
    // signs: 1 bit per coord, 8/byte LSB-first; bit=1 -> +1.0, bit=0 -> -1.0
    virtual void qjl_score(const float* q_sketch,
                           const uint8_t* signs,
                           const float* res_norms,
                           const float* mse_in,
                           int BH, int N, int D,
                           float qjl_scale,
                           float* out) = 0;

    // Fused value dequant:
    //   out[n, coord] = (float)qval[n, coord] * scales[n, coord/group_size]
    //                                         + zeros [n, coord/group_size]
    // packed: [N, packed_d] uint8 (vals_per_byte = 8/bits, supports 2/4/8 bits)
    // scales/zeros: [N, n_groups]
    virtual void value_dequant(const uint8_t* packed,
                               const float* scales,
                               const float* zeros,
                               int N, int D, int bits, int group_size,
                               float* out) = 0;

    // ---- Optional: stateful GPU buffer pool for session-cached decode ----
    //
    // Backends with a per-call upload bottleneck (chiefly OpenCL: ~17–23 ms
    // per attention call to re-upload Pi / quantized K / centroids via
    // CL_MEM_COPY_HOST_PTR) can override these to allocate a long-lived GPU
    // buffer keyed by (BH, D, key_bits) and return an opaque handle. The
    // handle is stored in attention_turboquant.cpp's SessionEntry and is
    // released via release_keys() when the session is invalidated (model
    // unload).
    //
    // CPU backends (scalar, NEON) leave these at their no-op defaults — no
    // GPU memory to manage, no per-call upload to amortize.
    //
    // Phase A2 (F3) of the HW-accel plan. Default returns nullptr so this
    // commit is plumbing-only; the OpenCL implementation that actually
    // uses the handle in mse_score() lands separately. See
    // ~/.claude/plans/async-baking-hopper.md P2 Stage B.
    virtual void* prepare_keys(int /*BH*/, int /*D*/, int /*key_bits*/) {
        return nullptr;
    }
    virtual void release_keys(void* /*handle*/) {}
};

// Factory. Returns nullptr if the requested backend wasn't compiled in or
// failed to init.
std::unique_ptr<IBackend> create_backend(BackendKind kind);

// Returns the best available backend in priority order:
//   QnnHtp > OpenCL > Vulkan > CpuNeon > CpuScalar
// Each call constructs a fresh backend — including kernel-program compilation
// for OpenCL (~500 ms first call). Use get_best_backend() instead when the
// backend should persist across many calls (e.g. one per attention layer
// per token).
std::unique_ptr<IBackend> create_best_backend();

// Process-wide singleton variant of create_best_backend(). Lazily creates
// the best available backend on first call, caches it, returns the same
// pointer on every subsequent call. The pointer is owned by the singleton —
// do NOT delete it. NOT thread-safe to call concurrently with backend
// teardown (we never tear down a singleton in practice).
//
// The backend's compilation cost (program build for OpenCL, graph finalize
// for QNN) happens inside the first call, so callers that care about
// startup latency should call ensure_backends_initialized() during
// program init.
//
// Returns nullptr only if no backend at all is available (extremely
// unlikely — CpuScalar is always present when TQ_WITH_CPU_SCALAR is on).
IBackend * get_best_backend();

// Idempotent. If the singleton hasn't been built yet, build it now.
// Useful during program init (e.g. JNI_OnLoad / first JNI call) so
// the OpenCL program-compilation cost doesn't fall on the first
// inference token.
void ensure_backends_initialized();

}  // namespace turboquant
