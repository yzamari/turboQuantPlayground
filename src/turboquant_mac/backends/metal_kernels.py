"""
MLX Metal kernel wrappers for TurboQuant attention scoring.

Compiles and caches Metal shaders via mx.fast.metal_kernel().
Provides turboquant_mse_score_metal() and turboquant_qjl_score_metal()
that compute attention scores directly from packed quantized data on
Apple Silicon GPU.
"""

import math
import mlx.core as mx

from turboquant_mac.backends.metal.mse_score import get_mse_score_source
from turboquant_mac.backends.metal.qjl_score import get_qjl_score_source

# Kernel cache: (bits, d, packed_d) -> compiled kernel
_mse_kernel_cache: dict = {}
_qjl_kernel_cache: dict = {}


def _get_packing_params(bits: int) -> tuple[int, int]:
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2
    else:
        return 8, 1


def turboquant_mse_score_metal(
    q_rot: mx.array,       # (BH, D) rotated query: q @ Pi^T
    mse_packed: mx.array,  # (BH, N, packed_d) uint8 bit-packed indices
    norms: mx.array,       # (BH, N) original vector norms
    centroids: mx.array,   # (n_clusters,) codebook centroids
    mse_bits: int,
) -> mx.array:
    """
    Compute MSE attention scores using Metal kernel.

    Returns: (BH, N) attention logits.
    """
    BH, D = q_rot.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]

    cache_key = (mse_bits, D, packed_d)
    if cache_key not in _mse_kernel_cache:
        source = get_mse_score_source(mse_bits, D, packed_d)
        _mse_kernel_cache[cache_key] = mx.fast.metal_kernel(
            name=f"turboquant_mse_score_b{mse_bits}_d{D}",
            input_names=["q_rot", "mse", "norms", "centroids"],
            output_names=["out"],
            source=source,
        )

    kernel = _mse_kernel_cache[cache_key]

    # Ensure correct dtypes
    q_rot = q_rot.astype(mx.float32)
    norms = norms.astype(mx.float32)
    centroids = centroids.astype(mx.float32)
    mse_packed = mse_packed.astype(mx.uint8)

    out = kernel(
        inputs=[q_rot, mse_packed, norms, centroids],
        output_shapes=[(BH, N)],
        output_dtypes=[mx.float32],
        grid=(N, BH, 1),
        threadgroup=(min(N, 256), 1, 1),
    )

    return out[0]


def turboquant_qjl_score_metal(
    q_sketch: mx.array,       # (BH, D) sketched query: q @ S^T
    qjl_signs: mx.array,      # (BH, N, packed_d_signs) uint8 packed signs
    residual_norms: mx.array,  # (BH, N)
    qjl_scale: float,
    mse_scores: mx.array,      # (BH, N) existing MSE scores to add to
) -> mx.array:
    """
    Compute QJL score contribution and add to existing MSE scores.

    Returns: (BH, N) combined scores (MSE + QJL).
    """
    BH, D = q_sketch.shape
    N = qjl_signs.shape[1]
    packed_d_signs = qjl_signs.shape[2]

    cache_key = (D, packed_d_signs, round(qjl_scale, 10))
    if cache_key not in _qjl_kernel_cache:
        source = get_qjl_score_source(D, packed_d_signs, qjl_scale)
        _qjl_kernel_cache[cache_key] = mx.fast.metal_kernel(
            name=f"turboquant_qjl_score_d{D}",
            input_names=["q_sketch", "signs", "res_norms", "mse_scores_in"],
            output_names=["out"],
            source=source,
        )

    kernel = _qjl_kernel_cache[cache_key]

    q_sketch = q_sketch.astype(mx.float32)
    residual_norms = residual_norms.astype(mx.float32)
    qjl_signs = qjl_signs.astype(mx.uint8)
    mse_scores = mse_scores.astype(mx.float32)

    out = kernel(
        inputs=[q_sketch, qjl_signs, residual_norms, mse_scores],
        output_shapes=[(BH, N)],
        output_dtypes=[mx.float32],
        grid=(N, BH, 1),
        threadgroup=(min(N, 256), 1, 1),
    )

    return out[0]


def turboquant_attention_score_metal(
    query: mx.array,          # (BH, D) or (BH, 1, D)
    mse_packed: mx.array,     # (BH, N, packed_d) uint8
    qjl_signs: mx.array,     # (BH, N, packed_d_signs) uint8
    norms: mx.array,         # (BH, N)
    residual_norms: mx.array, # (BH, N)
    Pi: mx.array,            # (D, D) rotation matrix
    S: mx.array,             # (D, D) QJL matrix
    centroids: mx.array,     # (n_clusters,)
    mse_bits: int,
    qjl_scale: float,
) -> mx.array:
    """
    High-level: compute TurboQuant attention scores using Metal kernels.

    1. Precomputes q_rot = q @ Pi^T and q_sketch = q @ S^T
    2. Runs MSE Metal kernel
    3. Runs QJL Metal kernel (adds to MSE scores)

    Returns: (BH, N) raw logits.
    """
    if query.ndim == 3:
        query = query.squeeze(axis=1)

    # Precompute rotated and sketched queries (once per decode step)
    q_rot = mx.matmul(query.astype(mx.float32), mx.transpose(Pi))    # (BH, D)
    q_sketch = mx.matmul(query.astype(mx.float32), mx.transpose(S))  # (BH, D)

    # MSE scores
    scores = turboquant_mse_score_metal(q_rot, mse_packed, norms, centroids, mse_bits)

    # Add QJL scores
    scores = turboquant_qjl_score_metal(q_sketch, qjl_signs, residual_norms, qjl_scale, scores)

    return scores
