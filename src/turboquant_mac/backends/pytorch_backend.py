"""
PyTorch CPU backend for TurboQuant.

Provides a uniform API for array operations using PyTorch tensors on CPU.
This backend is for learning/debugging — no GPU acceleration.
"""

import numpy as np
import torch
import torch.nn.functional as F

BACKEND_NAME = "pytorch"


def from_numpy(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).clone()


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def transpose(t: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return t.transpose(dim0, dim1)


def searchsorted(sorted_seq: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(sorted_seq, values.contiguous())


def norm(t: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return t.norm(dim=dim, keepdim=keepdim)


def pad(t: torch.Tensor, pad_width: int, value: int = 0) -> torch.Tensor:
    """Pad the last dimension on the right."""
    return F.pad(t.to(torch.uint8), (0, pad_width), value=value)


def arange(n: int, dtype=None) -> torch.Tensor:
    return torch.arange(n, dtype=dtype or torch.int64)


def zeros(shape, dtype=None) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype or torch.float32)


def ones(shape, dtype=None) -> torch.Tensor:
    return torch.ones(shape, dtype=dtype or torch.float32)


def cat(tensors: list, dim: int = 0) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def stack(tensors: list, dim: int = 0) -> torch.Tensor:
    return torch.stack(tensors, dim=dim)


def softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(t, dim=dim)


def tensor(data, dtype=None) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype)


def to_float(t: torch.Tensor) -> torch.Tensor:
    return t.float()


def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.uint8)


def to_long(t: torch.Tensor) -> torch.Tensor:
    return t.long()


def clamp(t: torch.Tensor, min_val=None, max_val=None) -> torch.Tensor:
    return t.clamp(min=min_val, max=max_val)


def round_(t: torch.Tensor) -> torch.Tensor:
    return t.round()


def unsqueeze(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.unsqueeze(dim)


def reshape(t: torch.Tensor, shape) -> torch.Tensor:
    return t.reshape(shape)


def sum_(t: torch.Tensor, dim: int = -1, dtype=None) -> torch.Tensor:
    if dtype is not None:
        return t.sum(dim=dim).to(dtype)
    return t.sum(dim=dim)


def min_(t: torch.Tensor, dim: int = -1):
    return t.min(dim=dim, keepdim=True).values


def max_(t: torch.Tensor, dim: int = -1):
    return t.max(dim=dim, keepdim=True).values


def bitwise_and(a: torch.Tensor, b) -> torch.Tensor:
    return a & b


def bitwise_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a | b


def left_shift(a: torch.Tensor, b) -> torch.Tensor:
    return a << b


def right_shift(a: torch.Tensor, b) -> torch.Tensor:
    return a >> b


def greater_than(a: torch.Tensor, b) -> torch.Tensor:
    return (a > b)


def index_select(t: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather from last dimension using indices: t[..., indices]."""
    return t[indices]


def eval_(*tensors):
    """No-op for PyTorch (eager evaluation)."""
    pass


def empty_like(t: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(t)


def nelement(t: torch.Tensor) -> int:
    return t.nelement()
