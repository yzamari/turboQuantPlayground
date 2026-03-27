"""Tests for Metal kernels — compare against PyTorch CPU reference."""

import math
import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


from turboquant_mac.quantizer import TurboQuantProd, TurboQuantMSE, _pack_indices, _unpack_indices
from turboquant_mac.backends import get_backend, reset_backend


def _make_quantized_data(B_H, N, D, bits, seed=42):
    """Create quantized key data for testing."""
    reset_backend()
    B = get_backend("mlx")
    rng = np.random.RandomState(seed)

    keys_np = rng.randn(B_H, N, D).astype(np.float32)
    prod_q = TurboQuantProd(dim=D, bits=bits, seed=42, backend="mlx")

    keys = B.from_numpy(keys_np)
    quantized = prod_q.quantize(keys)

    return prod_q, quantized, keys_np


class TestMSEScoreKernel:
    @pytest.mark.parametrize("D,bits", [(64, 3), (128, 3), (64, 2), (128, 4)])
    def test_matches_pytorch_reference(self, D, bits):
        """Metal MSE score should match pure-PyTorch computation."""
        from turboquant_mac.backends.metal_kernels import turboquant_mse_score_metal

        BH, N = 4, 64
        rng = np.random.RandomState(42)

        # Create quantized keys via MLX backend
        prod_q, quantized, keys_np = _make_quantized_data(BH, N, D, bits)

        # Query
        query_np = rng.randn(BH, D).astype(np.float32)

        # --- MLX Metal path ---
        Pi_mx = prod_q.mse_quantizer.Pi
        q_rot_mx = mx.matmul(mx.array(query_np), mx.transpose(Pi_mx))
        centroids_mx = prod_q.mse_quantizer.centroids
        norms_mx = quantized.norms
        mse_packed_mx = quantized.mse_indices

        metal_scores = turboquant_mse_score_metal(
            q_rot_mx, mse_packed_mx, norms_mx, centroids_mx, bits - 1
        )
        mx.eval(metal_scores)
        metal_scores_np = np.array(metal_scores)

        # --- PyTorch reference path ---
        reset_backend()
        B_pt = get_backend("pytorch")
        import torch
        mse_q_pt = TurboQuantMSE(dim=D, bits=bits - 1, seed=42, backend="pytorch")
        keys_pt = torch.from_numpy(keys_np)
        q_pt = mse_q_pt.quantize(keys_pt)
        k_mse_pt = mse_q_pt.dequantize(q_pt).numpy()

        # Batched dot: query (BH, D) @ k_mse (BH, N, D)^T -> (BH, N)
        ref_scores = np.einsum("bd,bnd->bn", query_np, k_mse_pt)

        # Compare — allow tolerance for float32 accumulation differences
        # The scores should be in the same ballpark
        assert metal_scores_np.shape == (BH, N), f"Shape mismatch: {metal_scores_np.shape}"

        # Check correlation (more robust than absolute comparison)
        for bh in range(BH):
            corr = np.corrcoef(metal_scores_np[bh], ref_scores[bh])[0, 1]
            assert corr > 0.95, (
                f"BH={bh}: correlation={corr:.4f} (expected > 0.95)"
            )

    @pytest.mark.parametrize("N", [32, 128, 512])
    def test_various_sequence_lengths(self, N):
        """Metal kernel should handle various sequence lengths."""
        from turboquant_mac.backends.metal_kernels import turboquant_mse_score_metal

        BH, D, bits = 2, 64, 3
        prod_q, quantized, _ = _make_quantized_data(BH, N, D, bits)

        query_np = np.random.RandomState(99).randn(BH, D).astype(np.float32)
        Pi_mx = prod_q.mse_quantizer.Pi
        q_rot = mx.matmul(mx.array(query_np), mx.transpose(Pi_mx))

        scores = turboquant_mse_score_metal(
            q_rot, quantized.mse_indices, quantized.norms,
            prod_q.mse_quantizer.centroids, bits - 1
        )
        mx.eval(scores)
        assert np.array(scores).shape == (BH, N)
        assert np.all(np.isfinite(np.array(scores)))


class TestQJLScoreKernel:
    def test_matches_pytorch_reference(self):
        """Metal QJL score should match pure-PyTorch computation."""
        from turboquant_mac.backends.metal_kernels import (
            turboquant_mse_score_metal,
            turboquant_qjl_score_metal,
        )

        BH, N, D, bits = 4, 32, 64, 3  # N != D to avoid shape ambiguity
        rng = np.random.RandomState(42)

        prod_q, quantized, keys_np = _make_quantized_data(BH, N, D, bits)
        query_np = rng.randn(BH, D).astype(np.float32)

        # --- Metal combined path ---
        Pi_mx = prod_q.mse_quantizer.Pi
        S_mx = prod_q.S

        q_rot = mx.matmul(mx.array(query_np), mx.transpose(Pi_mx))
        q_sketch = mx.matmul(mx.array(query_np), mx.transpose(S_mx))

        mse_scores = turboquant_mse_score_metal(
            q_rot, quantized.mse_indices, quantized.norms,
            prod_q.mse_quantizer.centroids, bits - 1
        )
        mx.eval(mse_scores)

        combined_scores = turboquant_qjl_score_metal(
            q_sketch, quantized.qjl_signs, quantized.residual_norms,
            prod_q.qjl_scale, mse_scores
        )
        mx.eval(combined_scores)
        metal_np = np.array(combined_scores)

        # --- MLX Python reference (same quantized data, no Metal kernels) ---
        # attention_score expects (..., n_q, D), so reshape query to (BH, 1, D)
        query_3d = mx.array(query_np).reshape(BH, 1, D)
        ref_scores = prod_q.attention_score(query_3d, quantized)
        mx.eval(ref_scores)
        ref_np = np.array(ref_scores).squeeze(1)  # (BH, 1, N) -> (BH, N)

        # Compare Metal vs MLX Python — should match closely (same data)
        for bh in range(BH):
            corr = np.corrcoef(metal_np[bh], ref_np[bh])[0, 1]
            assert corr > 0.95, (
                f"BH={bh}: correlation={corr:.4f} (expected > 0.95)"
            )


class TestCombinedMetalAttention:
    def test_high_level_api(self):
        """turboquant_attention_score_metal should produce reasonable scores."""
        from turboquant_mac.backends.metal_kernels import turboquant_attention_score_metal

        BH, N, D, bits = 2, 128, 64, 3
        rng = np.random.RandomState(42)

        prod_q, quantized, keys_np = _make_quantized_data(BH, N, D, bits)
        query_np = rng.randn(BH, D).astype(np.float32)

        scores = turboquant_attention_score_metal(
            query=mx.array(query_np),
            mse_packed=quantized.mse_indices,
            qjl_signs=quantized.qjl_signs,
            norms=quantized.norms,
            residual_norms=quantized.residual_norms,
            Pi=prod_q.mse_quantizer.Pi,
            S=prod_q.S,
            centroids=prod_q.mse_quantizer.centroids,
            mse_bits=bits - 1,
            qjl_scale=prod_q.qjl_scale,
        )
        mx.eval(scores)
        scores_np = np.array(scores)

        assert scores_np.shape == (BH, N)
        assert np.all(np.isfinite(scores_np))
        # Scores should not be all zeros
        assert np.std(scores_np) > 0.01
