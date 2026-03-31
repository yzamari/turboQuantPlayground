"""Tests for random rotation and QJL matrix generation."""

import numpy as np
import pytest
from turboquant_mac.rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    generate_wht_signs,
    apply_wht,
)


class TestRotationMatrix:
    def test_orthogonality(self):
        """Pi @ Pi^T should be approximately identity."""
        for d in [64, 128]:
            Pi = generate_rotation_matrix(d, seed=42)
            I_approx = Pi @ Pi.T
            assert np.allclose(I_approx, np.eye(d), atol=1e-5), (
                f"d={d}: max error = {np.max(np.abs(I_approx - np.eye(d)))}"
            )

    def test_norm_preservation(self):
        """Rotation should preserve vector norms."""
        d = 128
        Pi = generate_rotation_matrix(d, seed=42)
        rng = np.random.RandomState(123)
        x = rng.randn(100, d).astype(np.float32)
        y = x @ Pi.T

        norms_before = np.linalg.norm(x, axis=-1)
        norms_after = np.linalg.norm(y, axis=-1)
        assert np.allclose(norms_before, norms_after, atol=1e-4)

    def test_coordinate_distribution(self):
        """After rotating unit vectors, coordinates should have std ~ 1/sqrt(d)."""
        d = 128
        Pi = generate_rotation_matrix(d, seed=42)
        rng = np.random.RandomState(456)

        # Generate random unit vectors
        x = rng.randn(5000, d).astype(np.float32)
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        y = x @ Pi.T
        observed_std = np.std(y)
        expected_std = 1.0 / np.sqrt(d)
        assert abs(observed_std - expected_std) / expected_std < 0.1, (
            f"observed_std={observed_std:.4f}, expected ~{expected_std:.4f}"
        )

    def test_deterministic(self):
        """Same seed should produce identical matrices."""
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=42)
        assert np.array_equal(Pi1, Pi2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different matrices."""
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=99)
        assert not np.array_equal(Pi1, Pi2)

    def test_shape(self):
        for d in [64, 128, 256]:
            Pi = generate_rotation_matrix(d)
            assert Pi.shape == (d, d)
            assert Pi.dtype == np.float32


class TestQJLMatrix:
    def test_shape_and_dtype(self):
        S = generate_qjl_matrix(128, seed=12345)
        assert S.shape == (128, 128)
        assert S.dtype == np.float32

    def test_deterministic(self):
        S1 = generate_qjl_matrix(128, seed=12345)
        S2 = generate_qjl_matrix(128, seed=12345)
        assert np.array_equal(S1, S2)

    def test_approximately_gaussian(self):
        """Entries should be approximately N(0, 1)."""
        S = generate_qjl_matrix(256, seed=12345)
        assert abs(np.mean(S)) < 0.1
        assert abs(np.std(S) - 1.0) < 0.1


class TestWHTSigns:
    def test_shape_and_dtype(self):
        for d in [64, 128, 256]:
            signs = generate_wht_signs(d, seed=42)
            assert signs.shape == (d,)
            assert signs.dtype == np.float32

    def test_values_are_plus_minus_one(self):
        signs = generate_wht_signs(128, seed=42)
        assert set(signs.tolist()).issubset({-1.0, 1.0})

    def test_deterministic(self):
        s1 = generate_wht_signs(128, seed=42)
        s2 = generate_wht_signs(128, seed=42)
        assert np.array_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = generate_wht_signs(128, seed=42)
        s2 = generate_wht_signs(128, seed=99)
        assert not np.array_equal(s1, s2)

    def test_power_of_2_guard(self):
        with pytest.raises(ValueError, match="power of 2"):
            generate_wht_signs(100, seed=42)
        with pytest.raises(ValueError, match="power of 2"):
            generate_wht_signs(576, seed=42)

    def test_roughly_balanced(self):
        """Signs should be roughly half +1 and half -1."""
        signs = generate_wht_signs(256, seed=42)
        ratio = np.mean(signs > 0)
        assert 0.3 < ratio < 0.7


class TestWHTTransform:
    def test_roundtrip(self):
        """Forward then inverse should recover original."""
        d = 128
        signs = generate_wht_signs(d, seed=42)
        rng = np.random.RandomState(123)
        x = rng.randn(100, d).astype(np.float32)

        y = apply_wht(x, signs, inverse=False)
        x_recovered = apply_wht(y, signs, inverse=True)
        assert np.allclose(x, x_recovered, atol=1e-4), (
            f"Max roundtrip error: {np.max(np.abs(x - x_recovered))}"
        )

    def test_norm_preservation(self):
        """WHT should preserve vector norms."""
        d = 128
        signs = generate_wht_signs(d, seed=42)
        rng = np.random.RandomState(123)
        x = rng.randn(100, d).astype(np.float32)
        y = apply_wht(x, signs, inverse=False)

        norms_before = np.linalg.norm(x, axis=-1)
        norms_after = np.linalg.norm(y, axis=-1)
        assert np.allclose(norms_before, norms_after, atol=1e-4)

    def test_coordinate_distribution(self):
        """After WHT of unit vectors, coordinates should have std ~ 1/sqrt(d)."""
        d = 128
        signs = generate_wht_signs(d, seed=42)
        rng = np.random.RandomState(456)

        x = rng.randn(5000, d).astype(np.float32)
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        y = apply_wht(x, signs, inverse=False)
        observed_std = np.std(y)
        expected_std = 1.0 / np.sqrt(d)
        assert abs(observed_std - expected_std) / expected_std < 0.15, (
            f"observed_std={observed_std:.4f}, expected ~{expected_std:.4f}"
        )

    def test_different_dims(self):
        """Should work for various power-of-2 dimensions."""
        for d in [32, 64, 128, 256]:
            signs = generate_wht_signs(d, seed=42)
            rng = np.random.RandomState(42)
            x = rng.randn(10, d).astype(np.float32)
            y = apply_wht(x, signs, inverse=False)
            assert y.shape == x.shape
            x_back = apply_wht(y, signs, inverse=True)
            assert np.allclose(x, x_back, atol=1e-4)

    def test_single_vector(self):
        """Should work for a single vector (no batch dim)."""
        d = 64
        signs = generate_wht_signs(d, seed=42)
        x = np.random.RandomState(42).randn(d).astype(np.float32)
        y = apply_wht(x, signs, inverse=False)
        assert y.shape == (d,)
        x_back = apply_wht(y, signs, inverse=True)
        assert np.allclose(x, x_back, atol=1e-4)

    def test_batch_dims(self):
        """Should work with multiple batch dimensions."""
        d = 64
        signs = generate_wht_signs(d, seed=42)
        rng = np.random.RandomState(42)
        x = rng.randn(2, 3, 4, d).astype(np.float32)
        y = apply_wht(x, signs, inverse=False)
        assert y.shape == x.shape
        x_back = apply_wht(y, signs, inverse=True)
        assert np.allclose(x, x_back, atol=1e-4)
