// Mirrors src/turboquant_mac/backends/metal/qjl_score.py
// Adds the QJL correction term to the incoming MSE score.
__kernel void qjl_score(
    __global const float* q_sketch,
    __global const uchar* signs,
    __global const float* res_norms,
    __global const float* mse_in,
    __global       float* out,
    const int   BH,
    const int   N,
    const int   D,
    const int   PACKED_D_SIGNS,
    const float QJL_SCALE)
{
    const int n  = get_global_id(0);
    const int bh = get_global_id(1);
    if (n >= N || bh >= BH) return;

    const int row_off = (bh * N + n) * PACKED_D_SIGNS;
    const int q_off   = bh * D;

    float dot = 0.0f;
    for (int byte_idx = 0; byte_idx < PACKED_D_SIGNS; ++byte_idx) {
        uchar packed = signs[row_off + byte_idx];
        for (int bit = 0; bit < 8; ++bit) {
            int coord = byte_idx * 8 + bit;
            if (coord >= D) break;
            float s = ((packed >> bit) & 1) ? 1.0f : -1.0f;
            dot += q_sketch[q_off + coord] * s;
        }
    }
    out[bh * N + n] = mse_in[bh * N + n] + dot * res_norms[bh * N + n] * QJL_SCALE;
}
