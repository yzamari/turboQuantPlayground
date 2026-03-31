"""
Metal shader source for fused TurboQuant MSE encoding (quantize + bit-pack).

Takes rotated float values and decision boundaries, outputs packed uint8 indices.
Combines searchsorted + bit-packing into a single GPU dispatch, avoiding
intermediate arrays and multiple kernel launches.
"""

# Template parameters: {N_BOUNDARIES}, {BITS}, {VALS_PER_BYTE}, {D}, {PACKED_D}
MSE_ENCODE_SOURCE = """
    // Thread handles one (batch, coordinate_group) pair
    uint batch_idx = thread_position_in_grid.y;
    uint byte_idx = thread_position_in_grid.x;

    uint N_BATCH = rotated_shape[0];  // total batch elements (flattened)
    uint D = {D};
    uint PACKED_D = {PACKED_D};
    uint BITS = {BITS};
    uint VALS_PER_BYTE = {VALS_PER_BYTE};
    uint N_BOUNDARIES = {N_BOUNDARIES};

    if (batch_idx >= N_BATCH || byte_idx >= PACKED_D) return;

    uint8_t packed = 0;

    for (uint sub = 0; sub < VALS_PER_BYTE; sub++) {{
        uint coord = byte_idx * VALS_PER_BYTE + sub;
        if (coord < D) {{
            float val = rotated[batch_idx * D + coord];

            // searchsorted: count how many boundaries val >= boundary
            uint idx = 0;
            for (uint b = 0; b < N_BOUNDARIES; b++) {{
                if (val >= boundaries[b]) {{
                    idx++;
                }}
            }}

            // Pack into byte at correct bit position
            packed |= (uint8_t)(idx << (sub * BITS));
        }}
    }}

    out[batch_idx * PACKED_D + byte_idx] = packed;
"""


MSE_ENCODE_3BIT_U32_SOURCE = """
    // Thread handles one (batch, word) pair
    // 3-bit uint32 packing: 10 values per uint32 word
    uint batch_idx = thread_position_in_grid.y;
    uint word_idx = thread_position_in_grid.x;

    uint N_BATCH = rotated_shape[0];
    uint D = {D};
    uint PACKED_D = {PACKED_D};
    uint N_BOUNDARIES = {N_BOUNDARIES};
    uint VALS_PER_WORD = 10;

    if (batch_idx >= N_BATCH || word_idx >= PACKED_D) return;

    uint packed = 0;

    for (uint sub = 0; sub < VALS_PER_WORD; sub++) {{
        uint coord = word_idx * VALS_PER_WORD + sub;
        if (coord < D) {{
            float val = rotated[batch_idx * D + coord];

            uint idx = 0;
            for (uint b = 0; b < N_BOUNDARIES; b++) {{
                if (val >= boundaries[b]) {{
                    idx++;
                }}
            }}

            packed |= (idx << (sub * 3));
        }}
    }}

    out[batch_idx * PACKED_D + word_idx] = packed;
"""


def get_mse_encode_source(n_boundaries: int, bits: int, d: int, packed_d: int) -> str:
    """Return Metal shader source with template parameters filled in."""
    # Use specialized uint32 kernel for 3-bit
    if bits == 3:
        return MSE_ENCODE_3BIT_U32_SOURCE.format(
            N_BOUNDARIES=n_boundaries, D=d, PACKED_D=packed_d,
        )

    if bits == 1:
        eff_bits, vals_per_byte = 1, 8
    elif bits == 2:
        eff_bits, vals_per_byte = 2, 4
    elif bits <= 4:
        eff_bits, vals_per_byte = 4, 2
    else:
        eff_bits, vals_per_byte = 8, 1

    return MSE_ENCODE_SOURCE.format(
        N_BOUNDARIES=n_boundaries,
        BITS=eff_bits,
        VALS_PER_BYTE=vals_per_byte,
        D=d,
        PACKED_D=packed_d,
    )
