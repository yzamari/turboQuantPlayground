// Mirrors src/turboquant_mac/backends/metal/mse_score.py
// One thread per (bh, n) output element.
//
// Inputs:
//   q_rot[BH, D]       float
//   mse[BH, N, packed_d]   uchar (or uint for bits=3, see *_b3 kernel)
//   norms[BH, N]       float
//   centroids[2^bits]  float
// Output:
//   out[BH, N]         float

__kernel void mse_score_uchar(
    __global const float*  q_rot,
    __global const uchar*  mse,
    __global const float*  norms,
    __global const float*  centroids,
    __global       float*  out,
    const int   BH,
    const int   N,
    const int   D,
    const int   PACKED_D,
    const int   BITS,
    const int   VALS_PER_BYTE,
    const int   MASK)
{
    const int n  = get_global_id(0);
    const int bh = get_global_id(1);
    if (n >= N || bh >= BH) return;

    const int row_off = (bh * N + n) * PACKED_D;
    const int q_off   = bh * D;

    float score = 0.0f;
    for (int byte_idx = 0; byte_idx < PACKED_D; ++byte_idx) {
        uchar packed = mse[row_off + byte_idx];
        for (int sub = 0; sub < VALS_PER_BYTE; ++sub) {
            int coord = byte_idx * VALS_PER_BYTE + sub;
            if (coord >= D) break;
            int idx = (packed >> (sub * BITS)) & MASK;
            score += q_rot[q_off + coord] * centroids[idx];
        }
    }
    out[bh * N + n] = score * norms[bh * N + n];
}

// 3-bit special case: 10 vals per uint32 word.
__kernel void mse_score_u32_b3(
    __global const float* q_rot,
    __global const uint*  mse,
    __global const float* norms,
    __global const float* centroids,
    __global       float* out,
    const int  BH,
    const int  N,
    const int  D,
    const int  N_WORDS)
{
    const int n  = get_global_id(0);
    const int bh = get_global_id(1);
    if (n >= N || bh >= BH) return;

    const int row_off = (bh * N + n) * N_WORDS;
    const int q_off   = bh * D;

    float score = 0.0f;
    for (int j = 0; j < D; ++j) {
        int word = j / 10;
        int sub  = j % 10;
        int idx  = (mse[row_off + word] >> (sub * 3)) & 0x7;
        score += q_rot[q_off + j] * centroids[idx];
    }
    out[bh * N + n] = score * norms[bh * N + n];
}
