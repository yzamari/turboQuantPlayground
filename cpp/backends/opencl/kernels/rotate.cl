// Simple GEMV-shaped rotate: out[i, j] = sum_k in[i, k] * Pi[j, k]   (= in @ Pi^T)
// One thread per output element. For our shapes (n small, D=128) this is
// adequate; a tiled SGEMM is overkill for the rotation step.
__kernel void tq_rotate(
    __global const float* in,
    __global const float* Pi,
    __global       float* out,
    const int N,
    const int D)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    if (j >= D || i >= N) return;
    float s = 0.0f;
    for (int k = 0; k < D; ++k) {
        s += in[i * D + k] * Pi[j * D + k];
    }
    out[i * D + j] = s;
}
