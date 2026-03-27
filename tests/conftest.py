"""Shared test fixtures for TurboQuant tests."""

import pytest
import numpy as np
from turboquant_mac.backends import get_backend, reset_backend


def _available_backends():
    backends = []
    try:
        import torch
        backends.append("pytorch")
    except ImportError:
        pass
    try:
        import mlx.core
        backends.append("mlx")
    except ImportError:
        pass
    return backends


@pytest.fixture(params=_available_backends())
def backend_name(request):
    """Parametrize tests over all available backends."""
    reset_backend()
    return request.param


@pytest.fixture
def B(backend_name):
    """Return the backend module."""
    return get_backend(backend_name)


@pytest.fixture
def rng():
    """Deterministic numpy RNG for test data."""
    return np.random.RandomState(42)
