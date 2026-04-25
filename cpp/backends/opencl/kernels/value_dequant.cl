// Mirrors src/turboquant_mac/backends/metal/value_dequant.py
// One thread per (batch_idx, coord) output element.
__kernel void value_dequant(
    __global const uchar* packed,
    __global const float* scales,
    __global const float* zeros,
    __global       float* out,
    const int N,
    const int D,
    const int BITS,
    const int VALS_PER_BYTE,
    const int MASK,
    const int GROUP_SIZE,
    const int N_GROUPS,
    const int PACKED_D)
{
    const int coord     = get_global_id(0);
    const int batch_idx = get_global_id(1);
    if (coord >= D || batch_idx >= N) return;

    int byte = coord / VALS_PER_BYTE;
    int sub  = coord % VALS_PER_BYTE;
    int qval = (packed[batch_idx * PACKED_D + byte] >> (sub * BITS)) & MASK;
    int g    = coord / GROUP_SIZE;
    out[batch_idx * D + coord] =
        (float)qval * scales[batch_idx * N_GROUPS + g] +
                      zeros [batch_idx * N_GROUPS + g];
}
