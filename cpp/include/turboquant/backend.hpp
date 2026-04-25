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
};

// Factory. Returns nullptr if the requested backend wasn't compiled in or
// failed to init.
std::unique_ptr<IBackend> create_backend(BackendKind kind);

// Returns the best available backend in priority order:
//   QnnHtp > OpenCL > Vulkan > CpuNeon > CpuScalar
std::unique_ptr<IBackend> create_best_backend();

}  // namespace turboquant
