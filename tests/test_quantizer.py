"""Tests for MSE and inner-product quantizers."""

import math
import numpy as np
import pytest
from turboquant_mac.quantizer import (
    TurboQuantMSE,
    TurboQuantProd,
    _pack_indices,
    _unpack_indices,
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


class TestBitPacking:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_roundtrip(self, backend_name, bits):
        """pack -> unpack should recover original indices."""
        B = get_backend(backend_name)
        d = 128
        n_clusters = 2**bits
        rng = np.random.RandomState(42)
        indices_np = rng.randint(0, n_clusters, size=(4, d)).astype(np.int64)
        indices = B.from_numpy(indices_np)

        packed = _pack_indices(indices, bits, B)
        unpacked = _unpack_indices(packed, bits, d, B)

        unpacked_np = B.to_numpy(unpacked)
        assert np.array_equal(indices_np, unpacked_np), (
            f"bits={bits}: roundtrip failed"
        )


class TestTurboQuantMSE:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_distortion_bounds(self, backend_name, bits):
        """MSE should be close to paper's theoretical values."""
        B = get_backend(backend_name)
        d = 128
        mse_q = TurboQuantMSE(dim=d, bits=bits, backend=backend_name)

        rng = np.random.RandomState(42)
        x_np = rng.randn(500, d).astype(np.float32)
        norms = np.linalg.norm(x_np, axis=-1, keepdims=True)
        x_np = x_np / norms  # unit vectors for clean MSE measurement

        x = B.from_numpy(x_np)
        x_hat = mse_q.forward(x)
        x_hat_np = B.to_numpy(x_hat)

        mse = np.mean((x_np - x_hat_np) ** 2)
        # Paper bound: MSE <= sqrt(3)*pi/2 * (1/4^b) ~ 2.72 / 4^b
        upper_bound = math.sqrt(3) * math.pi / 2 * (1.0 / 4**bits)
        assert mse < upper_bound * 1.2, (
            f"bits={bits}: MSE={mse:.4f} exceeds bound {upper_bound:.4f}"
        )

    def test_norm_preservation(self, backend_name):
        """Dequantized vectors should have similar norms to input."""
        B = get_backend(backend_name)
        d = 128
        mse_q = TurboQuantMSE(dim=d, bits=3, backend=backend_name)

        rng = np.random.RandomState(42)
        x_np = rng.randn(100, d).astype(np.float32) * 5.0  # non-unit
        x = B.from_numpy(x_np)

        x_hat = mse_q.forward(x)
        x_hat_np = B.to_numpy(x_hat)

        norms_orig = np.linalg.norm(x_np, axis=-1)
        norms_recon = np.linalg.norm(x_hat_np, axis=-1)
        rel_error = np.abs(norms_orig - norms_recon) / (norms_orig + 1e-10)
        assert np.mean(rel_error) < 0.25, f"Mean relative norm error: {np.mean(rel_error):.4f}"


class TestTurboQuantProd:
    def test_inner_product_unbiasedness(self, backend_name):
        """Inner product estimation should be approximately unbiased."""
        B = get_backend(backend_name)
        d = 128
        prod_q = TurboQuantProd(dim=d, bits=3, backend=backend_name)

        rng = np.random.RandomState(42)
        n_pairs = 500
        x_np = rng.randn(n_pairs, d).astype(np.float32)
        y_np = rng.randn(n_pairs, d).astype(np.float32)

        # True inner products
        true_ip = np.sum(x_np * y_np, axis=-1)

        # Estimated inner products
        x = B.from_numpy(x_np)
        y = B.from_numpy(y_np)

        q = prod_q.quantize(x)
        x_hat = prod_q.dequantize(q)
        x_hat_np = B.to_numpy(x_hat)
        est_ip = np.sum(x_hat_np * y_np, axis=-1)

        # Bias should be small relative to signal
        bias = np.mean(est_ip - true_ip)
        signal = np.std(true_ip)
        assert abs(bias / signal) < 0.1, (
            f"Bias/signal ratio = {abs(bias/signal):.4f} (should be < 0.1)"
        )

    def test_attention_score_shape(self, backend_name):
        """attention_score should return correct shape."""
        B = get_backend(backend_name)
        d = 64
        prod_q = TurboQuantProd(dim=d, bits=3, backend=backend_name)

        rng = np.random.RandomState(42)
        n_q, n_k = 4, 32
        queries_np = rng.randn(n_q, d).astype(np.float32)
        keys_np = rng.randn(n_k, d).astype(np.float32)

        queries = B.from_numpy(queries_np)
        keys = B.from_numpy(keys_np)

        q_keys = prod_q.quantize(keys)
        scores = prod_q.attention_score(queries, q_keys)
        scores_np = B.to_numpy(scores)

        assert scores_np.shape == (n_q, n_k), f"Expected ({n_q}, {n_k}), got {scores_np.shape}"

    def test_attention_score_correlation(self, backend_name):
        """Estimated scores should correlate well with true scores."""
        B = get_backend(backend_name)
        d = 128
        prod_q = TurboQuantProd(dim=d, bits=3, backend=backend_name)

        rng = np.random.RandomState(42)
        n_q, n_k = 8, 64
        queries_np = rng.randn(n_q, d).astype(np.float32)
        keys_np = rng.randn(n_k, d).astype(np.float32)

        true_scores = queries_np @ keys_np.T

        queries = B.from_numpy(queries_np)
        keys = B.from_numpy(keys_np)
        q_keys = prod_q.quantize(keys)
        est_scores = B.to_numpy(prod_q.attention_score(queries, q_keys))

        corr = np.corrcoef(true_scores.ravel(), est_scores.ravel())[0, 1]
        assert corr > 0.75, f"Correlation = {corr:.4f} (should be > 0.75)"
