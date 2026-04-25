// Internal header for the bit-packing primitives. Not part of the public API
// (the public API exposes only quantize/dequantize/attention_score, which call
// into these).

#pragma once

#include <cstdint>

namespace turboquant {

int packed_len_for(int d, int bits);
int packed_signs_len(int d);

// Pack/unpack n_vec vectors of d integers each (1/2/3/4/8-bit).
void pack_indices  (const int32_t* idx, int n_vec, int d, int bits, uint8_t* out);
void unpack_indices(const uint8_t* packed, int n_vec, int d, int bits, int32_t* out);

void pack_qjl_signs            (const float* projected, int n_vec, int d, uint8_t* out);
void unpack_qjl_signs_to_float (const uint8_t* packed,  int n_vec, int d, float*   out);

}  // namespace turboquant
