"""
Metal shader source for TurboQuant MSE attention score computation.

Ports Triton kernel _turboquant_mse_score_kernel to Apple Silicon Metal.

For each (batch_head, token), iterates over packed bytes, extracts indices
via bit-shift, looks up centroids, and accumulates q_rot[j] * centroid[idx[j]].
Final score is multiplied by the original vector norm.

Key insight: query is rotated forward (q @ Pi^T) once, then scores are computed
directly from packed indices — avoids materializing D-dim dequantized keys.
"""

# Metal shader source — compiled at runtime by mx.fast.metal_kernel()
# Template parameters {BITS}, {VALS_PER_BYTE}, {BIT_MASK}, {D}, {PACKED_D}
# are substituted before compilation.

MSE_SCORE_SOURCE = """
    // Thread handles one (batch_head, token) pair
    uint bh = thread_position_in_grid.y;
    uint n = thread_position_in_grid.x;

    uint N = mse_shape[1];   // number of KV tokens
    uint BH = q_rot_shape[0];

    if (n >= N || bh >= BH) return;

    float score = 0.0f;

    uint BITS = {BITS};
    uint VALS_PER_BYTE = {VALS_PER_BYTE};
    uint BIT_MASK = {BIT_MASK};
    uint D = {D};
    uint PACKED_D = {PACKED_D};

    // Iterate over packed bytes
    for (uint byte_idx = 0; byte_idx < PACKED_D; byte_idx++) {{
        // Load packed byte for this token
        uint8_t packed = mse[bh * N * PACKED_D + n * PACKED_D + byte_idx];
        uint packed_int = (uint)packed;

        // Extract each index from the packed byte
        for (uint sub = 0; sub < VALS_PER_BYTE; sub++) {{
            uint coord = byte_idx * VALS_PER_BYTE + sub;
            if (coord < D) {{
                uint idx = (packed_int >> (sub * BITS)) & BIT_MASK;
                float centroid_val = centroids[idx];
                float q_val = q_rot[bh * D + coord];
                score += q_val * centroid_val;
            }}
        }}
    }}

    // Multiply by original vector norm
    float norm_val = norms[bh * N + n];
    out[bh * N + n] = score * norm_val;
"""


MSE_SCORE_3BIT_U32_SOURCE = """
    // Thread handles one (batch_head, token) pair
    // 3-bit uint32 packing: 10 values per uint32 word (30 of 32 bits used)
    uint bh = thread_position_in_grid.y;
    uint n = thread_position_in_grid.x;

    uint N = mse_shape[1];
    uint BH = q_rot_shape[0];

    if (n >= N || bh >= BH) return;

    float score = 0.0f;

    uint D = {D};
    uint PACKED_D = {PACKED_D};
    uint VALS_PER_WORD = 10;
    uint BIT_MASK = 0x7;  // 3-bit mask

    for (uint word_idx = 0; word_idx < PACKED_D; word_idx++) {{
        uint packed = mse[bh * N * PACKED_D + n * PACKED_D + word_idx];

        for (uint sub = 0; sub < VALS_PER_WORD; sub++) {{
            uint coord = word_idx * VALS_PER_WORD + sub;
            if (coord < D) {{
                uint idx = (packed >> (sub * 3)) & BIT_MASK;
                float centroid_val = centroids[idx];
                float q_val = q_rot[bh * D + coord];
                score += q_val * centroid_val;
            }}
        }}
    }}

    float norm_val = norms[bh * N + n];
    out[bh * N + n] = score * norm_val;
"""


def get_mse_score_source(bits: int, d: int, packed_d: int) -> str:
    """Return Metal shader source with template parameters filled in."""
    # Use specialized uint32 kernel for 3-bit
    if bits == 3:
        return MSE_SCORE_3BIT_U32_SOURCE.format(D=d, PACKED_D=packed_d)

    if bits == 1:
        eff_bits, vals_per_byte = 1, 8
    elif bits == 2:
        eff_bits, vals_per_byte = 2, 4
    elif bits <= 4:
        eff_bits, vals_per_byte = 4, 2
    else:
        eff_bits, vals_per_byte = 8, 1

    bit_mask = (1 << eff_bits) - 1

    return MSE_SCORE_SOURCE.format(
        BITS=eff_bits,
        VALS_PER_BYTE=vals_per_byte,
        BIT_MASK=bit_mask,
        D=d,
        PACKED_D=packed_d,
    )
