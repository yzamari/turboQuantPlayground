"""
Metal shader source for TurboQuant QJL residual correction score.

Ports Triton kernel _turboquant_qjl_score_kernel to Apple Silicon Metal.

For each (batch_head, token), iterates over packed sign bytes, extracts
8 sign bits per byte, converts to {-1, +1}, and accumulates
q_sketch[j] * sign_val. Final score is scaled by residual_norm * qjl_scale.

The QJL scores are ADDED to existing MSE scores in the output buffer.
"""

# Template parameters: {D}, {PACKED_D_SIGNS}
QJL_SCORE_SOURCE = """
    uint bh = thread_position_in_grid.y;
    uint n = thread_position_in_grid.x;

    uint N = signs_shape[1];
    uint BH = q_sketch_shape[0];

    if (n >= N || bh >= BH) return;

    float dot = 0.0f;
    uint D = {D};
    uint PACKED_D_SIGNS = {PACKED_D_SIGNS};

    for (uint byte_idx = 0; byte_idx < PACKED_D_SIGNS; byte_idx++) {{
        uint8_t packed = signs[bh * N * PACKED_D_SIGNS + n * PACKED_D_SIGNS + byte_idx];
        uint packed_int = (uint)packed;

        for (uint bit = 0; bit < 8; bit++) {{
            uint coord = byte_idx * 8 + bit;
            if (coord < D) {{
                uint sign_bit = (packed_int >> bit) & 1;
                float sign_val = (sign_bit == 1) ? 1.0f : -1.0f;
                float q_val = q_sketch[bh * D + coord];
                dot += q_val * sign_val;
            }}
        }}
    }}

    float res_norm = res_norms[bh * N + n];
    float qjl_contribution = dot * res_norm * {QJL_SCALE};

    // Add to existing MSE scores
    out[bh * N + n] = mse_scores_in[bh * N + n] + qjl_contribution;
"""


def get_qjl_score_source(d: int, packed_d_signs: int, qjl_scale: float) -> str:
    """Return Metal shader source with template parameters filled in."""
    return QJL_SCORE_SOURCE.format(
        D=d,
        PACKED_D_SIGNS=packed_d_signs,
        QJL_SCALE=f"{qjl_scale:.10f}f",
    )
