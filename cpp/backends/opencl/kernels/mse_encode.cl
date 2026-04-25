// Mirrors src/turboquant_mac/backends/metal/mse_encode.py
// One thread per (batch, byte/word) output unit; fuses searchsorted + bit-pack.

__kernel void mse_encode_uchar(
    __global const float* rotated,
    __global const float* boundaries,
    __global       uchar* out,
    const int N_BATCH,
    const int D,
    const int PACKED_D,
    const int N_BOUNDARIES,
    const int BITS,
    const int VALS_PER_BYTE,
    const int MASK)
{
    const int byte_idx   = get_global_id(0);
    const int batch_idx  = get_global_id(1);
    if (byte_idx >= PACKED_D || batch_idx >= N_BATCH) return;

    uchar packed = 0;
    for (int sub = 0; sub < VALS_PER_BYTE; ++sub) {
        int coord = byte_idx * VALS_PER_BYTE + sub;
        if (coord < D) {
            float val = rotated[batch_idx * D + coord];
            int idx = 0;
            for (int b = 0; b < N_BOUNDARIES; ++b) {
                idx += (val >= boundaries[b]);
            }
            packed |= (uchar)((idx & MASK) << (sub * BITS));
        }
    }
    out[batch_idx * PACKED_D + byte_idx] = packed;
}

// 3-bit u32: 10 vals/word.
__kernel void mse_encode_u32_b3(
    __global const float* rotated,
    __global const float* boundaries,
    __global       uint*  out,
    const int N_BATCH,
    const int D,
    const int N_WORDS)
{
    const int word_idx  = get_global_id(0);
    const int batch_idx = get_global_id(1);
    if (word_idx >= N_WORDS || batch_idx >= N_BATCH) return;

    uint packed = 0;
    for (int sub = 0; sub < 10; ++sub) {
        int coord = word_idx * 10 + sub;
        if (coord < D) {
            float val = rotated[batch_idx * D + coord];
            int idx = 0;
            // 3-bit -> 7 boundaries, fully unrolled
            idx += (val >= boundaries[0]);
            idx += (val >= boundaries[1]);
            idx += (val >= boundaries[2]);
            idx += (val >= boundaries[3]);
            idx += (val >= boundaries[4]);
            idx += (val >= boundaries[5]);
            idx += (val >= boundaries[6]);
            packed |= ((uint)(idx & 0x7)) << (sub * 3);
        }
    }
    out[batch_idx * N_WORDS + word_idx] = packed;
}
