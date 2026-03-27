"""Tests for TurboQuant KV cache."""

import math
import numpy as np
import pytest
from turboquant_mac.kv_cache import (
    TurboQuantKVCache,
    quantize_values,
    dequantize_values,
)
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
    reset_backend()
    return request.param


class TestValueQuantization:
    def test_roundtrip_shape(self, backend_name):
        """Quantize -> dequantize should preserve shape."""
        B = get_backend(backend_name)
        rng = np.random.RandomState(42)
        v_np = rng.randn(2, 4, 32, 128).astype(np.float32)
        v = B.from_numpy(v_np)

        vq = quantize_values(v, bits=2, group_size=32, backend=backend_name)
        v_hat = dequantize_values(vq, group_size=32, backend=backend_name)
        v_hat_np = B.to_numpy(v_hat)

        assert v_hat_np.shape == v_np.shape

    def test_roundtrip_accuracy(self, backend_name):
        """2-bit group quantization should have reasonable accuracy."""
        B = get_backend(backend_name)
        rng = np.random.RandomState(42)
        v_np = rng.randn(2, 4, 32, 128).astype(np.float32)
        v = B.from_numpy(v_np)

        vq = quantize_values(v, bits=2, group_size=32, backend=backend_name)
        v_hat = dequantize_values(vq, group_size=32, backend=backend_name)
        v_hat_np = B.to_numpy(v_hat)

        mse = np.mean((v_np - v_hat_np) ** 2)
        # 2-bit group quantization should have reasonable MSE
        assert mse < 1.0, f"Value quantization MSE = {mse:.4f}"


class TestKVCache:
    def test_prefill_only(self, backend_name):
        """Short sequences should stay in buffer."""
        B = get_backend(backend_name)
        cache = TurboQuantKVCache(
            head_dim=64, key_bits=3, value_bits=2, buffer_size=128, backend=backend_name,
        )

        rng = np.random.RandomState(42)
        keys_np = rng.randn(1, 2, 32, 64).astype(np.float32)
        values_np = rng.randn(1, 2, 32, 64).astype(np.float32)

        cache.prefill(B.from_numpy(keys_np), B.from_numpy(values_np))
        assert cache.get_seq_length() == 32
        assert cache.key_quantized is None  # all in buffer

    def test_prefill_with_quantization(self, backend_name):
        """Long sequences should partially quantize."""
        B = get_backend(backend_name)
        cache = TurboQuantKVCache(
            head_dim=64, key_bits=3, value_bits=2, buffer_size=32, backend=backend_name,
        )

        rng = np.random.RandomState(42)
        seq_len = 128
        keys_np = rng.randn(1, 2, seq_len, 64).astype(np.float32)
        values_np = rng.randn(1, 2, seq_len, 64).astype(np.float32)

        cache.prefill(B.from_numpy(keys_np), B.from_numpy(values_np))
        assert cache.get_seq_length() == seq_len
        assert cache.key_quantized is not None
        assert cache.key_buffer.shape[-2] == 32

    def test_attention_scores_shape(self, backend_name):
        """Attention scores should have correct shape."""
        B = get_backend(backend_name)
        cache = TurboQuantKVCache(
            head_dim=64, key_bits=3, value_bits=2, buffer_size=16, backend=backend_name,
        )

        rng = np.random.RandomState(42)
        seq_len = 64
        keys_np = rng.randn(1, 2, seq_len, 64).astype(np.float32)
        values_np = rng.randn(1, 2, seq_len, 64).astype(np.float32)
        query_np = rng.randn(1, 2, 1, 64).astype(np.float32)

        cache.prefill(B.from_numpy(keys_np), B.from_numpy(values_np))
        scores = cache.attention_scores(B.from_numpy(query_np))
        scores_np = B.to_numpy(scores)

        assert scores_np.shape == (1, 2, 1, seq_len), f"Got shape {scores_np.shape}"

    def test_memory_savings(self, backend_name):
        """Compressed cache should use less memory than FP16."""
        B = get_backend(backend_name)
        cache = TurboQuantKVCache(
            head_dim=128, key_bits=3, value_bits=2, buffer_size=32, backend=backend_name,
        )

        rng = np.random.RandomState(42)
        seq_len = 256
        keys_np = rng.randn(1, 8, seq_len, 128).astype(np.float32)
        values_np = rng.randn(1, 8, seq_len, 128).astype(np.float32)

        cache.prefill(B.from_numpy(keys_np), B.from_numpy(values_np))
        mem = cache.memory_bytes()

        fp16_bytes = 2 * seq_len * 128 * 8 * 2  # keys + values, fp16
        assert mem["total"] < fp16_bytes, (
            f"Compressed: {mem['total']} bytes, FP16: {fp16_bytes} bytes"
        )
