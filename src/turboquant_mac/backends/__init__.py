"""
Backend abstraction for TurboQuant.

Auto-detects MLX (preferred on Apple Silicon) or PyTorch CPU.
Each backend module exports the same set of functions for array operations.
"""

_active_backend = None
_active_backend_name = None


def get_backend(name: str = None):
    """Return the backend module. Auto-detects if name is None."""
    global _active_backend, _active_backend_name

    if name is None and _active_backend is not None:
        return _active_backend

    if name is None:
        # Auto-detect: prefer MLX on Apple Silicon
        try:
            from turboquant_mac.backends import mlx_backend
            _active_backend = mlx_backend
            _active_backend_name = "mlx"
            return _active_backend
        except ImportError:
            pass

        try:
            from turboquant_mac.backends import pytorch_backend
            _active_backend = pytorch_backend
            _active_backend_name = "pytorch"
            return _active_backend
        except ImportError:
            pass

        raise ImportError(
            "No backend available. Install either mlx (pip install mlx) "
            "or torch (pip install torch)."
        )

    if name == "mlx":
        from turboquant_mac.backends import mlx_backend
        _active_backend = mlx_backend
        _active_backend_name = "mlx"
    elif name == "pytorch":
        from turboquant_mac.backends import pytorch_backend
        _active_backend = pytorch_backend
        _active_backend_name = "pytorch"
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'mlx' or 'pytorch'.")

    return _active_backend


def get_backend_name() -> str:
    """Return the name of the active backend."""
    if _active_backend_name is None:
        get_backend()
    return _active_backend_name


def reset_backend():
    """Reset backend selection (for testing)."""
    global _active_backend, _active_backend_name
    _active_backend = None
    _active_backend_name = None
