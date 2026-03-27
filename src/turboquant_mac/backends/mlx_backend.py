"""
MLX backend for TurboQuant — optimized for Apple Silicon.

Provides the same API as pytorch_backend using MLX arrays.
MLX uses lazy evaluation: call eval_() to force computation when needed.
Unified memory means no device transfers.
"""

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    raise ImportError("MLX is not installed. Run: pip install mlx")

BACKEND_NAME = "mlx"


def from_numpy(arr: np.ndarray) -> mx.array:
    return mx.array(arr)


def to_numpy(t: mx.array) -> np.ndarray:
    return np.array(t)


def matmul(a: mx.array, b: mx.array) -> mx.array:
    return mx.matmul(a, b)


def transpose(t: mx.array, dim0: int, dim1: int) -> mx.array:
    axes = list(range(t.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return mx.transpose(t, axes=axes)


def searchsorted(sorted_seq: mx.array, values: mx.array) -> mx.array:
    # MLX doesn't have searchsorted; use numpy fallback
    sorted_np = np.array(sorted_seq)
    values_np = np.array(values)
    result_np = np.searchsorted(sorted_np, values_np)
    return mx.array(result_np)


def norm(t: mx.array, dim: int = -1, keepdim: bool = False) -> mx.array:
    return mx.sqrt(mx.sum(t * t, axis=dim, keepdims=keepdim))


def pad(t: mx.array, pad_width: int, value: int = 0) -> mx.array:
    """Pad the last dimension on the right."""
    ndim = t.ndim
    pad_widths = [(0, 0)] * (ndim - 1) + [(0, pad_width)]
    return mx.pad(t, pad_widths, constant_values=value)


def arange(n: int, dtype=None) -> mx.array:
    return mx.arange(n)


def zeros(shape, dtype=None) -> mx.array:
    return mx.zeros(shape, dtype=dtype or mx.float32)


def ones(shape, dtype=None) -> mx.array:
    return mx.ones(shape, dtype=dtype or mx.float32)


def cat(tensors: list, dim: int = 0) -> mx.array:
    return mx.concatenate(tensors, axis=dim)


def stack(tensors: list, dim: int = 0) -> mx.array:
    return mx.stack(tensors, axis=dim)


def softmax(t: mx.array, dim: int = -1) -> mx.array:
    return mx.softmax(t, axis=dim)


def tensor(data, dtype=None) -> mx.array:
    return mx.array(data, dtype=dtype)


def to_float(t: mx.array) -> mx.array:
    return t.astype(mx.float32)


def to_uint8(t: mx.array) -> mx.array:
    return t.astype(mx.uint8)


def to_long(t: mx.array) -> mx.array:
    return t.astype(mx.int64)


def clamp(t: mx.array, min_val=None, max_val=None) -> mx.array:
    return mx.clip(t, a_min=min_val, a_max=max_val)


def round_(t: mx.array) -> mx.array:
    return mx.round(t)


def unsqueeze(t: mx.array, dim: int) -> mx.array:
    return mx.expand_dims(t, axis=dim)


def reshape(t: mx.array, shape) -> mx.array:
    return mx.reshape(t, shape)


def sum_(t: mx.array, dim: int = -1, dtype=None) -> mx.array:
    result = mx.sum(t, axis=dim)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def min_(t: mx.array, dim: int = -1):
    values = mx.min(t, axis=dim, keepdims=True)
    return values


def max_(t: mx.array, dim: int = -1):
    values = mx.max(t, axis=dim, keepdims=True)
    return values


def bitwise_and(a: mx.array, b) -> mx.array:
    if not isinstance(b, mx.array):
        b = mx.array(b, dtype=a.dtype)
    return mx.bitwise_and(a, b)


def bitwise_or(a: mx.array, b: mx.array) -> mx.array:
    return mx.bitwise_or(a, b)


def left_shift(a: mx.array, b) -> mx.array:
    if not isinstance(b, mx.array):
        b = mx.array(b, dtype=a.dtype)
    return mx.left_shift(a, b)


def right_shift(a: mx.array, b) -> mx.array:
    if not isinstance(b, mx.array):
        b = mx.array(b, dtype=a.dtype)
    return mx.right_shift(a, b)


def greater_than(a: mx.array, b) -> mx.array:
    return a > b


def index_select(t: mx.array, indices: mx.array) -> mx.array:
    """Gather from last dimension using indices: t[indices]."""
    return t[indices]


def eval_(*tensors):
    """Force evaluation of lazy MLX arrays."""
    mx.eval(*tensors)


def empty_like(t: mx.array) -> mx.array:
    return mx.zeros_like(t)


def nelement(t: mx.array) -> int:
    return t.size
