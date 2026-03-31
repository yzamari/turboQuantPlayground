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


class TestLayerAdaptive:
    """Tests for layer-adaptive quantization (per-layer bit widths)."""

    def test_bit_schedule_applies_correct_bits(self, backend_name):
        """Different layers should use different bit widths from schedule."""
        B = get_backend(backend_name)
        schedule = {0: 4, 1: 3, "default": 2}

        cache_layer0 = TurboQuantKVCache(
            head_dim=64, buffer_size=16, layer_idx=0,
            backend=backend_name, bit_schedule=schedule,
        )
        cache_layer1 = TurboQuantKVCache(
            head_dim=64, buffer_size=16, layer_idx=1,
            backend=backend_name, bit_schedule=schedule,
        )
        cache_layer5 = TurboQuantKVCache(
            head_dim=64, buffer_size=16, layer_idx=5,
            backend=backend_name, bit_schedule=schedule,
        )

        assert cache_layer0.key_bits == 4
        assert cache_layer1.key_bits == 3
        assert cache_layer5.key_bits == 2  # default

    def test_bit_schedule_tuple_kv(self, backend_name):
        """Schedule with (key_bits, value_bits) tuples should work."""
        B = get_backend(backend_name)
        schedule = {0: (4, 4), 1: (3, 2), "default": (2, 2)}

        cache = TurboQuantKVCache(
            head_dim=64, buffer_size=16, layer_idx=0,
            backend=backend_name, bit_schedule=schedule,
        )
        assert cache.key_bits == 4
        assert cache.value_bits == 4

        cache1 = TurboQuantKVCache(
            head_dim=64, buffer_size=16, layer_idx=1,
            backend=backend_name, bit_schedule=schedule,
        )
        assert cache1.key_bits == 3
        assert cache1.value_bits == 2

    def test_no_schedule_uses_explicit_bits(self, backend_name):
        """Without bit_schedule, explicit key_bits/value_bits should work as before."""
        B = get_backend(backend_name)
        cache = TurboQuantKVCache(
            head_dim=64, key_bits=4, value_bits=4, buffer_size=16,
            backend=backend_name,
        )
        assert cache.key_bits == 4
        assert cache.value_bits == 4

    def test_adaptive_layers_quantize_differently(self, backend_name):
        """Layers with different bits should produce different compressed sizes."""
        B = get_backend(backend_name)
        schedule = {0: 4, "default": 2}

        cache_4bit = TurboQuantKVCache(
            head_dim=64, buffer_size=8, layer_idx=0,
            backend=backend_name, bit_schedule=schedule,
        )
        cache_2bit = TurboQuantKVCache(
            head_dim=64, buffer_size=8, layer_idx=5,
            backend=backend_name, bit_schedule=schedule,
        )

        rng = np.random.RandomState(42)
        keys = B.from_numpy(rng.randn(1, 2, 32, 64).astype(np.float32))
        values = B.from_numpy(rng.randn(1, 2, 32, 64).astype(np.float32))

        cache_4bit.prefill(keys, values)
        cache_2bit.prefill(keys, values)

        mem_4bit = cache_4bit.memory_bytes()["total"]
        mem_2bit = cache_2bit.memory_bytes()["total"]
        assert mem_4bit > mem_2bit, (
            f"4-bit ({mem_4bit}B) should use more memory than 2-bit ({mem_2bit}B)"
        )


class TestSparseVAttention:
    """Tests for top-K sparse value attention."""

    def _make_cache_and_prefill(self, backend_name, seq_len=64, head_dim=64, buffer_size=16):
        B = get_backend(backend_name)
        cache = TurboQuantKVCache(
            head_dim=head_dim, key_bits=3, value_bits=2,
            buffer_size=buffer_size, backend=backend_name,
        )
        rng = np.random.RandomState(42)
        keys = B.from_numpy(rng.randn(1, 2, seq_len, head_dim).astype(np.float32))
        values = B.from_numpy(rng.randn(1, 2, seq_len, head_dim).astype(np.float32))
        query = B.from_numpy(rng.randn(1, 2, 1, head_dim).astype(np.float32))
        cache.prefill(keys, values)
        return cache, query, B

    def test_top_k_output_shape(self, backend_name):
        """attend(top_k=K) should return same shape as attend(top_k=None)."""
        cache, query, B = self._make_cache_and_prefill(backend_name)
        scores = cache.attention_scores(query)
        attn_weights = B.softmax(scores, dim=-1)

        out_full = cache.attend(attn_weights)
        out_sparse = cache.attend(attn_weights, top_k=16)

        assert B.to_numpy(out_sparse).shape == B.to_numpy(out_full).shape

    def test_top_k_equals_n_matches_full(self, backend_name):
        """top_k = total_seq_len should give identical results to full attention."""
        cache, query, B = self._make_cache_and_prefill(backend_name, seq_len=64)
        scores = cache.attention_scores(query)
        attn_weights = B.softmax(scores, dim=-1)

        out_full = cache.attend(attn_weights)
        out_topk = cache.attend(attn_weights, top_k=64)

        np.testing.assert_allclose(
            B.to_numpy(out_full), B.to_numpy(out_topk), atol=1e-4,
        )

    def test_top_k_small_is_approximate(self, backend_name):
        """Small top_k should give a reasonable approximation."""
        cache, query, B = self._make_cache_and_prefill(backend_name, seq_len=64)
        scores = cache.attention_scores(query)
        attn_weights = B.softmax(scores, dim=-1)

        out_full = cache.attend(attn_weights)
        out_sparse = cache.attend(attn_weights, top_k=32)

        full_np = B.to_numpy(out_full)
        sparse_np = B.to_numpy(out_sparse)
        # Cosine similarity should be high
        cos_sim = np.sum(full_np * sparse_np) / (
            np.linalg.norm(full_np) * np.linalg.norm(sparse_np) + 1e-10
        )
        assert cos_sim > 0.8, f"Cosine similarity = {cos_sim:.4f} (expected > 0.8)"

    def test_top_k_none_is_full_attention(self, backend_name):
        """top_k=None should be the same as not passing top_k at all."""
        cache, query, B = self._make_cache_and_prefill(backend_name)
        scores = cache.attention_scores(query)
        attn_weights = B.softmax(scores, dim=-1)

        out1 = cache.attend(attn_weights)
        out2 = cache.attend(attn_weights, top_k=None)

        np.testing.assert_allclose(
            B.to_numpy(out1), B.to_numpy(out2), atol=1e-6,
        )
