"""
Random rotation utilities for TurboQuant.

Generates rotation matrices via QR decomposition of random Gaussian matrices.
Uses NumPy for generation (CPU, one-time cost) then converts to backend arrays.
The same seed produces identical rotation matrices across backends.
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


def generate_qjl_matrix(d: int, seed: int = 12345) -> np.ndarray:
    """
    Generate the random projection matrix S in R^{d x d} for QJL.
    S has i.i.d. N(0,1) entries.

    Returns numpy float32 array — caller converts to backend array type.
    """
    rng = np.random.RandomState(seed)
    S = rng.randn(d, d).astype(np.float32)
    return S
