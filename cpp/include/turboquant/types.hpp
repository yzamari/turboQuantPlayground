// Plain-data types shared across the algorithm core and every backend.
//
// Storage layout matches the Python reference exactly so golden corpora
// produced by tools/gen_golden.py can be byte-compared.
//
// Per the automotive design rules: no STL containers in the on-the-wire types,
// only POD-friendly structs. Owning containers (std::vector) appear in the
// algorithm classes, not here.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace turboquant {

// Output of TurboQuantMSE::quantize(x).
//
// `indices` layout:
//   - bits == 1, 2, 4, 8: tightly packed uint8 bytes, LSB = element 0.
//                          packed_len = ceil(d / vals_per_byte).
//   - bits == 3:           uint32 words, 10 values per word, value i at bit
//                          offset i*3. packed_len_u32 = ceil(d / 10).
//                          The byte buffer holds packed_len_u32 * 4 bytes,
//                          interpreted as uint32 little-endian.
//
// The shape (..., d) is flattened to (n_vec, d) in C++; the caller restores
// any outer dimensions.
struct MSEQuantized {
    std::vector<uint8_t> indices;   // packed bytes (also covers uint32 case)
    std::vector<float>   norms;     // per-vector L2 norm
    int                  n_vec = 0;
    int                  bits  = 0;
    int                  d     = 0;
    int                  packed_len = 0;  // bytes per vector
};

// Output of TurboQuantProd::quantize(x). Stage-1 indices + stage-2 sign bits.
struct ProdQuantized {
    std::vector<uint8_t> mse_indices;     // bits = (top-level bits - 1)
    std::vector<uint8_t> qjl_signs;       // 1 bit per coord, 8/byte LSB-first
    std::vector<float>   residual_norms;  // per-vector ||x - x_hat||
    std::vector<float>   norms;           // per-vector ||x||
    int                  n_vec      = 0;
    int                  d          = 0;
    int                  mse_bits   = 0;  // = (top-level bits - 1)
    int                  packed_len_mse  = 0;
    int                  packed_len_signs = 0;  // = ceil(d / 8)
};

// Output of value-cache asymmetric per-group quantization.
struct ValueQuantized {
    std::vector<uint8_t> data;     // packed groupwise quantized values
    std::vector<float>   scales;   // per (vector, group)
    std::vector<float>   zeros;    // per (vector, group)
    int                  n_vec       = 0;
    int                  d           = 0;
    int                  bits        = 2;   // 2 or 4
    int                  group_size  = 32;
    int                  n_groups    = 0;
    int                  packed_d    = 0;   // bytes per vector
};

}  // namespace turboquant
