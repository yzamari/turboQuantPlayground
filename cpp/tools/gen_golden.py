"""
Golden corpus generator for the C++ parity test.

Runs the Python reference implementation (turboquant_mac, pytorch backend so
the result is deterministic on CPU) and dumps:
  - Pi, S    — exact rotation/QJL matrices (so C++ doesn't have to match
                numpy.linalg.qr, which it can't)
  - inputs   — deterministic float32 input vectors
  - query    — single deterministic query
  - ProdQuantized.{mse_indices, qjl_signs, norms, residual_norms}
  - attention_score(query, prod_quantized) output

Run from the repo root:
    venv/bin/python cpp/tools/gen_golden.py

Output files land in cpp/tests/golden/. Idempotent: existing files are
overwritten. A manifest.json with shapes + dtypes accompanies the .bin files
so the C++ test (parity_test.cpp) knows exactly what to load.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Make the package importable when running from the repo root with the venv.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turboquant_mac.backends import get_backend  # noqa: E402
from turboquant_mac.quantizer import TurboQuantProd  # noqa: E402
from turboquant_mac.rotation import (  # noqa: E402
    generate_qjl_matrix,
    generate_rotation_matrix,
)


CONFIGS: list[tuple[int, int]] = [
    (64, 2),
    (64, 3),
    (64, 4),
    (128, 2),
    (128, 3),
    (128, 4),
]
N_VECTORS = 32
N_QUERIES = 1
ROT_SEED = 42
QJL_SEED = ROT_SEED + 1000  # = 1042

GOLDEN_DIR = REPO_ROOT / "cpp" / "tests" / "golden"


def _to_numpy(t) -> np.ndarray:
    """Backend tensor (torch / numpy / mlx) -> numpy. Idempotent for ndarrays."""
    if isinstance(t, np.ndarray):
        return t
    if hasattr(t, "detach"):  # torch.Tensor
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def _write_f32(path: Path, arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    path.write_bytes(a.tobytes())


def _write_raw(path: Path, arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr)
    path.write_bytes(a.tobytes())


def _gen_one(d: int, bits: int, n: int, n_q: int) -> dict:
    """Generate the golden corpus for a single (d, bits) config and return its
    manifest entry (a dict)."""
    backend_name = "pytorch"
    B = get_backend(backend_name)

    # Pi and S — load from the Python reference using the documented seeds.
    pi_np = generate_rotation_matrix(d, seed=ROT_SEED).astype(np.float32)
    s_np = generate_qjl_matrix(d, seed=QJL_SEED).astype(np.float32)

    pi_path = GOLDEN_DIR / f"pi_d{d}_seed{ROT_SEED}.bin"
    s_path = GOLDEN_DIR / f"s_d{d}_seed{QJL_SEED}.bin"
    _write_f32(pi_path, pi_np)
    _write_f32(s_path, s_np)

    # Inputs (N, D) and query (1, D) — fixed seeds for reproducibility.
    inputs_np = np.random.RandomState(123).randn(n, d).astype(np.float32)
    query_np = np.random.RandomState(456).randn(n_q, d).astype(np.float32)

    inputs_path = GOLDEN_DIR / f"inputs_d{d}_n{n}.bin"
    query_path = GOLDEN_DIR / f"query_d{d}_n_q{n_q}.bin"
    _write_f32(inputs_path, inputs_np)
    _write_f32(query_path, query_np)

    # ProdQuantized via pytorch backend (deterministic on CPU).
    prod = TurboQuantProd(dim=d, bits=bits, seed=ROT_SEED, backend=backend_name)
    inputs_t = B.from_numpy(inputs_np)
    q = prod.quantize(inputs_t)

    mse_indices_np = _to_numpy(q.mse_indices)
    qjl_signs_np = _to_numpy(q.qjl_signs)
    norms_np = _to_numpy(q.norms).astype(np.float32, copy=False)
    residual_norms_np = _to_numpy(q.residual_norms).astype(np.float32, copy=False)

    # Convert mse_indices to a raw byte buffer matching the C++ packed layout.
    # For 3-bit packing the Python tensor is uint32 (10 vals/word); for 1/2/4-bit
    # it is uint8. In both cases .tobytes() on the contiguous numpy buffer gives
    # the exact bytes the C++ packer emits (little-endian uint32 on disk).
    mse_indices_bytes = np.ascontiguousarray(mse_indices_np).tobytes()
    qjl_signs_bytes = np.ascontiguousarray(
        qjl_signs_np.astype(np.uint8, copy=False)
    ).tobytes()

    mse_indices_path = GOLDEN_DIR / f"prod_b{bits}_d{d}.mse_indices.bin"
    qjl_signs_path = GOLDEN_DIR / f"prod_b{bits}_d{d}.qjl_signs.bin"
    norms_path = GOLDEN_DIR / f"prod_b{bits}_d{d}.norms.bin"
    residual_norms_path = GOLDEN_DIR / f"prod_b{bits}_d{d}.residual_norms.bin"
    scores_path = GOLDEN_DIR / f"prod_b{bits}_d{d}.scores.bin"

    mse_indices_path.write_bytes(mse_indices_bytes)
    qjl_signs_path.write_bytes(qjl_signs_bytes)
    _write_f32(norms_path, norms_np)
    _write_f32(residual_norms_path, residual_norms_np)

    # attention_score: query (1, D) vs prod (N, D) -> (1, N)
    query_t = B.from_numpy(query_np)
    scores = prod.attention_score(query_t, q)
    scores_np = _to_numpy(scores).astype(np.float32, copy=False).reshape(-1)
    _write_f32(scores_path, scores_np)

    # Sanity checks on file sizes.
    assert mse_indices_path.stat().st_size == mse_indices_np.nbytes
    assert qjl_signs_path.stat().st_size == qjl_signs_np.nbytes

    entry = {
        "d": d,
        "bits": bits,
        "N": n,
        "n_q": n_q,
        "rot_seed": ROT_SEED,
        "qjl_seed": QJL_SEED,
        "mse_bits": bits - 1,
        "pi": pi_path.name,
        "pi_shape": [d, d],
        "s": s_path.name,
        "s_shape": [d, d],
        "inputs": inputs_path.name,
        "inputs_shape": [n, d],
        "query": query_path.name,
        "query_shape": [n_q, d],
        "mse_indices": mse_indices_path.name,
        "mse_indices_shape": list(mse_indices_np.shape),
        "mse_indices_dtype": str(mse_indices_np.dtype),
        "mse_indices_nbytes": int(mse_indices_np.nbytes),
        "qjl_signs": qjl_signs_path.name,
        "qjl_signs_shape": list(qjl_signs_np.shape),
        "qjl_signs_dtype": "uint8",
        "qjl_signs_nbytes": int(qjl_signs_np.size),
        "norms": norms_path.name,
        "norms_shape": [n],
        "residual_norms": residual_norms_path.name,
        "residual_norms_shape": [n],
        "scores": scores_path.name,
        "scores_shape": [n_q, n],
    }

    print(
        f"[golden] d={d:>3} bits={bits} -> "
        f"mse_indices {tuple(mse_indices_np.shape)} {mse_indices_np.dtype}, "
        f"qjl_signs {tuple(qjl_signs_np.shape)}, "
        f"scores {tuple(scores_np.reshape(n_q, n).shape)}"
    )
    return entry


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "n_vectors": N_VECTORS,
        "n_queries": N_QUERIES,
        "rot_seed": ROT_SEED,
        "qjl_seed": QJL_SEED,
        "backend": "pytorch",
        "configs": [],
    }

    for d, bits in CONFIGS:
        entry = _gen_one(d, bits, N_VECTORS, N_QUERIES)
        manifest["configs"].append(entry)

    manifest_path = GOLDEN_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[golden] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
