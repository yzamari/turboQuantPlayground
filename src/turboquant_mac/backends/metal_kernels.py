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
from turboquant_mac.backends.metal.mse_encode import get_mse_encode_source
from turboquant_mac.backends.metal.value_dequant import get_value_dequant_source

# Kernel cache: (bits, d, packed_d) -> compiled kernel
_mse_kernel_cache: dict = {}
_qjl_kernel_cache: dict = {}
_mse_encode_kernel_cache: dict = {}
_value_dequant_kernel_cache: dict = {}


def _get_packing_params(bits: int) -> tuple[int, int]:
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2
    else:
        return 8, 1


def turboquant_value_dequant_metal(
    packed: mx.array,    # (..., packed_d) uint8 packed values
    scales: mx.array,    # (..., n_groups) per-group scales
    zeros: mx.array,     # (..., n_groups) per-group zeros
    bits: int,
    d: int,
    group_size: int = 32,
) -> mx.array:
    """
    Fused Metal kernel: unpack bits + dequantize values in one GPU dispatch.

    Combines bit-unpacking, int-to-float conversion, and affine transform
    (val * scale + zero) into a single kernel, avoiding intermediate arrays.
    """
    orig_shape = packed.shape  # (..., packed_d)
    batch_shape = orig_shape[:-1]
    n_batch = 1
    for s in batch_shape:
        n_batch *= s

    flat_packed = packed.reshape(n_batch, -1).astype(mx.uint8)
    flat_scales = scales.reshape(n_batch, -1).astype(mx.float32)
    flat_zeros = zeros.reshape(n_batch, -1).astype(mx.float32)

    cache_key = ("value_dequant", bits, d, group_size)
    if cache_key not in _value_dequant_kernel_cache:
        source = get_value_dequant_source(bits, d, group_size)
        _value_dequant_kernel_cache[cache_key] = mx.fast.metal_kernel(
            name=f"turboquant_value_dequant_b{bits}_d{d}_g{group_size}",
            input_names=["packed", "scales", "zeros"],
            output_names=["out"],
            source=source,
        )

    kernel = _value_dequant_kernel_cache[cache_key]

    out = kernel(
        inputs=[flat_packed, flat_scales, flat_zeros],
        output_shapes=[(n_batch, d)],
        output_dtypes=[mx.float32],
        grid=(d, n_batch, 1),
        threadgroup=(min(d, 256), 1, 1),
    )

    return out[0].reshape(*batch_shape, d)


def turboquant_mse_encode_metal(
    rotated: mx.array,       # (..., D) rotated values: Pi @ x_unit
    boundaries: mx.array,    # (n_boundaries,) decision boundaries (interior)
    bits: int,
    d: int,
) -> mx.array:
    """
    Fused Metal kernel: searchsorted + bit-pack in one GPU dispatch.

    Takes rotated float values and codebook boundaries, returns packed indices.
    Uses uint32 for 3-bit (10 values per word), uint8 for other bit widths.
    """
    is_3bit_u32 = (bits == 3)

    if is_3bit_u32:
        vals_per_word = 10
        packed_d = (d + vals_per_word - 1) // vals_per_word
        out_dtype = mx.uint32
    else:
        eff_bits, vals_per_byte = _get_packing_params(bits)
        packed_d = (d + vals_per_byte - 1) // vals_per_byte
        out_dtype = mx.uint8

    n_boundaries = boundaries.shape[0]

    # Flatten batch dims
    orig_shape = rotated.shape
    flat = rotated.reshape(-1, d).astype(mx.float32)
    n_batch = flat.shape[0]

    cache_key = ("encode", bits, d, packed_d, n_boundaries)
    if cache_key not in _mse_encode_kernel_cache:
        source = get_mse_encode_source(n_boundaries, bits, d, packed_d)
        _mse_encode_kernel_cache[cache_key] = mx.fast.metal_kernel(
            name=f"turboquant_mse_encode_b{bits}_d{d}",
            input_names=["rotated", "boundaries"],
            output_names=["out"],
            source=source,
        )

    kernel = _mse_encode_kernel_cache[cache_key]
    boundaries = boundaries.astype(mx.float32)

    out = kernel(
        inputs=[flat, boundaries],
        output_shapes=[(n_batch, packed_d)],
        output_dtypes=[out_dtype],
        grid=(packed_d, n_batch, 1),
        threadgroup=(min(packed_d, 256), 1, 1),
    )

    batch_shape = orig_shape[:-1]
    return out[0].reshape(*batch_shape, packed_d)


def turboquant_mse_score_metal(
    q_rot: mx.array,       # (BH, D) rotated query: q @ Pi^T
    mse_packed: mx.array,  # (BH, N, packed_d) bit-packed indices (uint8 or uint32)
    norms: mx.array,       # (BH, N) original vector norms
    centroids: mx.array,   # (n_clusters,) codebook centroids
    mse_bits: int,
) -> mx.array:
    """
    Compute MSE attention scores using Metal kernel.

    Uses uint32 packed data for 3-bit, uint8 for other bit widths.
    Returns: (BH, N) attention logits.
    """
    BH, D = q_rot.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    is_3bit_u32 = (mse_bits == 3)

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
    mse_packed = mse_packed.astype(mx.uint32 if is_3bit_u32 else mx.uint8)

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
