"""
Lloyd-Max codebook computation for TurboQuant.

After random rotation, each coordinate of a unit-norm vector follows:
    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
which is a scaled Beta distribution on [-1, 1].

For high d, this converges to N(0, 1/d).

We solve the continuous 1D k-means (Lloyd-Max) to find optimal centroids.
Pre-computed codebooks are cached as JSON files.
"""

import os
import json
import numpy as np
from scipy import integrate, special


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a single coordinate of a uniform random point on S^{d-1}."""
    if d <= 2:
        raise ValueError(f"Dimension d={d} too small, need d >= 3")
    log_const = (
        special.gammaln(d / 2.0)
        - 0.5 * np.log(np.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    log_val = log_const + exponent * np.log(1 - x**2)
    return np.exp(log_val)


def _conditional_mean(lo: float, hi: float, d: int) -> float:
    """E[X | lo < X < hi] under the Beta PDF on [-1, 1]."""
    num, _ = integrate.quad(lambda x: x * beta_pdf(np.array([x]), d)[0], lo, hi)
    den, _ = integrate.quad(lambda x: beta_pdf(np.array([x]), d)[0], lo, hi)
    if den < 1e-30:
        return (lo + hi) / 2.0
    return num / den


def _mse_cost(centroids: np.ndarray, d: int) -> float:
    """Compute MSE cost for a given set of sorted centroids."""
    n = len(centroids)
    boundaries = np.zeros(n + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    cost = 0.0
    for i in range(n):
        lo, hi = boundaries[i], boundaries[i + 1]
        c = centroids[i]
        val, _ = integrate.quad(
            lambda x: (x - c) ** 2 * beta_pdf(np.array([x]), d)[0], lo, hi
        )
        cost += val
    return cost


def compute_lloyd_max_codebook(d: int, bits: int, max_iter: int = 200, tol: float = 1e-12) -> dict:
    """
    Compute optimal Lloyd-Max codebook for the Beta distribution on [-1, 1].

    Args:
        d: dimension of the embedding space (e.g., head_dim = 128)
        bits: number of bits per coordinate (1, 2, 3, or 4)
        max_iter: max Lloyd-Max iterations
        tol: convergence tolerance

    Returns:
        dict with centroids, boundaries, mse_per_coord, mse_total, d, bits
    """
    n_clusters = 2**bits

    # Initialize centroids using quantiles of the distribution
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, 10000)
    pdf_vals = beta_pdf(x_grid, d)
    cdf_vals = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
    cdf_vals /= cdf_vals[-1]

    quantile_edges = np.linspace(0, 1, n_clusters + 1)
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        q_mid = (quantile_edges[i] + quantile_edges[i + 1]) / 2.0
        idx = min(np.searchsorted(cdf_vals, q_mid), len(x_grid) - 1)
        centroids[i] = x_grid[idx]

    # Lloyd-Max iterations
    prev_cost = float("inf")
    cost = float("inf")
    for _ in range(max_iter):
        boundaries = np.zeros(n_clusters + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_clusters - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        new_centroids = np.zeros(n_clusters)
        for i in range(n_clusters):
            new_centroids[i] = _conditional_mean(boundaries[i], boundaries[i + 1], d)

        cost = _mse_cost(new_centroids, d)
        centroids = new_centroids

        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

    # Final boundaries
    boundaries = np.zeros(n_clusters + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n_clusters - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    return {
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
        "mse_per_coord": float(cost),
        "mse_total": float(cost * d),
        "d": d,
        "bits": bits,
    }


# -- Codebook cache --
_CODEBOOK_CACHE: dict[tuple[int, int], dict] = {}
_CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "codebooks")


def get_codebook(d: int, bits: int) -> dict:
    """Get or compute a codebook, with on-disk caching."""
    key = (d, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    os.makedirs(_CODEBOOK_DIR, exist_ok=True)
    path = os.path.join(_CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            cb = json.load(f)
        _CODEBOOK_CACHE[key] = cb
        return cb

    print(f"[TurboQuant] Computing Lloyd-Max codebook for d={d}, bits={bits}...")
    cb = compute_lloyd_max_codebook(d, bits)
    with open(path, "w") as f:
        json.dump(cb, f, indent=2)
    print(f"[TurboQuant] MSE per coord = {cb['mse_per_coord']:.6e}, total MSE = {cb['mse_total']:.6f}")
    _CODEBOOK_CACHE[key] = cb
    return cb


def get_codebook_arrays(d: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Get codebook as numpy arrays (centroids, boundaries)."""
    cb = get_codebook(d, bits)
    centroids = np.array(cb["centroids"], dtype=np.float32)
    boundaries = np.array(cb["boundaries"], dtype=np.float32)
    return centroids, boundaries
