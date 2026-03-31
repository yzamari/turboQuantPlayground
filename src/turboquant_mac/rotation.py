"""
Random rotation utilities for TurboQuant.

Two rotation modes:
  - QR: Full d×d orthogonal matrix via QR decomposition. Works for any d. O(d²) per vector.
  - WHT: Randomized Walsh-Hadamard Transform via butterfly. Requires d to be power of 2. O(d log d) per vector.

Uses NumPy for generation (CPU, one-time cost) then converts to backend arrays.
The same seed produces identical rotations across backends.
"""

import numpy as np


def generate_rotation_matrix(d: int, seed: int = 42) -> np.ndarray:
    """
    Generate a random orthogonal matrix Pi in R^{d x d} via QR decomposition.

    This is Algorithm 1 from the paper. For head_dim=128, this is a 128x128
    matrix = 64KB in float32, negligible.

    Returns numpy float32 array — caller converts to backend array type.
    """
    rng = np.random.RandomState(seed)
    G = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(G)

    # Ensure proper rotation (det = +1) by fixing signs
    diag_sign = np.sign(np.diag(R))
    Q = Q * diag_sign[np.newaxis, :]

    return Q.astype(np.float32)


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def generate_wht_signs(d: int, seed: int = 42) -> np.ndarray:
    """
    Generate random ±1 signs for the randomized Walsh-Hadamard Transform.

    The rotation is R = (1/√d) · H_d · diag(signs), where H_d is the
    Hadamard matrix applied via O(d log d) butterfly operations.

    Args:
        d: Dimension (must be a power of 2).
        seed: Random seed for reproducibility.

    Returns:
        signs: (d,) float32 array with entries in {-1, +1}.
    """
    if not _is_power_of_2(d):
        raise ValueError(f"WHT requires d to be a power of 2, got {d}")
    rng = np.random.RandomState(seed)
    return (2 * rng.randint(0, 2, size=d) - 1).astype(np.float32)


def apply_wht(x: np.ndarray, signs: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Apply randomized Walsh-Hadamard Transform via butterfly factorization.

    Forward:  y = (1/√d) · H · (signs ⊙ x)
    Inverse:  x = signs ⊙ ((1/√d) · H · y)

    Since the normalized Hadamard matrix is symmetric and self-inverse,
    applying forward then inverse recovers the original vector.

    Args:
        x: (..., d) input array. d must be a power of 2.
        signs: (d,) random ±1 signs from generate_wht_signs().
        inverse: If True, apply inverse transform.

    Returns:
        (..., d) transformed array.
    """
    d = x.shape[-1]
    orig_shape = x.shape

    if not inverse:
        # Forward: multiply by signs, then butterfly
        result = (x * signs).reshape(-1, d).copy()
    else:
        # Inverse: butterfly first, then multiply by signs
        result = x.reshape(-1, d).copy()

    # Hadamard butterfly: log2(d) stages
    h = 1
    while h < d:
        half_blocks = d // (2 * h)
        result = result.reshape(-1, half_blocks, 2, h)
        a = result[:, :, 0, :].copy()
        b = result[:, :, 1, :].copy()
        result[:, :, 0, :] = a + b
        result[:, :, 1, :] = a - b
        result = result.reshape(-1, d)
        h *= 2

    # Normalize
    result = result / np.sqrt(d)

    if inverse:
        result = result * signs

    return result.reshape(orig_shape).astype(np.float32)


def generate_qjl_matrix(d: int, seed: int = 12345) -> np.ndarray:
    """
    Generate the random projection matrix S in R^{d x d} for QJL.
    S has i.i.d. N(0,1) entries.

    Returns numpy float32 array — caller converts to backend array type.
    """
    rng = np.random.RandomState(seed)
    S = rng.randn(d, d).astype(np.float32)
    return S
