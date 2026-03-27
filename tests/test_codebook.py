"""Tests for Lloyd-Max codebook computation."""

import numpy as np
import pytest
from turboquant_mac.codebook import (
    beta_pdf,
    compute_lloyd_max_codebook,
    get_codebook,
    get_codebook_arrays,
)


class TestBetaPDF:
    def test_integrates_to_one(self):
        """The Beta PDF should integrate to 1 over [-1, 1]."""
        from scipy import integrate
        for d in [64, 128, 256]:
            val, _ = integrate.quad(lambda x: beta_pdf(np.array([x]), d)[0], -1, 1)
            assert abs(val - 1.0) < 1e-6, f"d={d}: integral = {val}"

    def test_symmetric(self):
        """The PDF should be symmetric around 0."""
        for d in [64, 128]:
            x = np.linspace(0.01, 0.99, 50)
            assert np.allclose(beta_pdf(x, d), beta_pdf(-x, d), atol=1e-10)

    def test_rejects_small_d(self):
        with pytest.raises(ValueError):
            beta_pdf(np.array([0.0]), d=2)


class TestLloydMaxCodebook:
    def test_mse_matches_paper(self):
        """MSE values should match the paper's Table 1."""
        # Paper values are total MSE (sum over d coordinates):
        # b=1 ~0.36, b=2 ~0.117, b=3 ~0.03, b=4 ~0.009
        expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
        for bits, expected_mse in expected.items():
            cb = get_codebook(128, bits)
            assert abs(cb["mse_total"] - expected_mse) < expected_mse * 0.25, (
                f"bits={bits}: MSE_total={cb['mse_total']:.4f}, expected ~{expected_mse}"
            )

    def test_centroid_symmetry(self):
        """Centroids should be approximately symmetric around 0."""
        for bits in [1, 2, 3, 4]:
            cb = get_codebook(128, bits)
            centroids = np.array(cb["centroids"])
            assert np.allclose(centroids, -centroids[::-1], atol=1e-4), (
                f"bits={bits}: centroids not symmetric"
            )

    def test_centroids_sorted(self):
        """Centroids should be sorted in ascending order."""
        for bits in [1, 2, 3, 4]:
            cb = get_codebook(128, bits)
            centroids = np.array(cb["centroids"])
            assert np.all(centroids[:-1] <= centroids[1:])

    def test_correct_count(self):
        """Should have 2^bits centroids and 2^bits + 1 boundaries."""
        for bits in [1, 2, 3, 4]:
            cb = get_codebook(128, bits)
            assert len(cb["centroids"]) == 2**bits
            assert len(cb["boundaries"]) == 2**bits + 1


class TestGetCodebookArrays:
    def test_returns_numpy(self):
        centroids, boundaries = get_codebook_arrays(128, 3)
        assert isinstance(centroids, np.ndarray)
        assert centroids.dtype == np.float32
        assert len(centroids) == 8  # 2^3

    def test_precomputed_codebooks_exist(self):
        """All pre-shipped codebooks should load without computation."""
        for d in [64, 128]:
            for bits in [1, 2, 3, 4]:
                cb = get_codebook(d, bits)
                assert cb["d"] == d
                assert cb["bits"] == bits
