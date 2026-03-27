"""
Benchmark: Quantization and dequantization throughput (vectors/sec).

Compares MLX (Apple Silicon GPU) vs PyTorch CPU backends across
different dimensions, bit widths, and batch sizes.
"""

import time
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
DIMS = [64, 128]
BITS = [2, 3, 4]
BATCH_SIZES = [100, 1_000, 5_000]
WARMUP = 2
RUNS = 5


def _detect_backends():
    backends = []
    try:
        import mlx.core  # noqa: F401
        backends.append("mlx")
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        backends.append("pytorch")
    except ImportError:
        pass
    return backends


def bench_quantize(backend_name, dim, bits, batch_size):
    """Return (quant_vecs_per_sec, dequant_vecs_per_sec)."""
    from turboquant_mac.quantizer import TurboQuantProd
    from turboquant_mac.backends import get_backend, reset_backend

    reset_backend()
    B = get_backend(backend_name)
    quantizer = TurboQuantProd(dim=dim, bits=bits, backend=backend_name)

    rng = np.random.RandomState(42)
    x_np = rng.randn(batch_size, dim).astype(np.float32)
    x = B.from_numpy(x_np)

    # Force any lazy init
    q = quantizer.quantize(x)
    if backend_name == "mlx":
        import mlx.core as mx
        mx.eval(q.mse_indices, q.qjl_signs, q.residual_norms, q.norms)

    # ── Warmup ───────────────────────────────────────────────────────
    for _ in range(WARMUP):
        q = quantizer.quantize(x)
        if backend_name == "mlx":
            mx.eval(q.mse_indices, q.qjl_signs, q.residual_norms, q.norms)

    # ── Quantize timing ──────────────────────────────────────────────
    quant_times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        q = quantizer.quantize(x)
        if backend_name == "mlx":
            mx.eval(q.mse_indices, q.qjl_signs, q.residual_norms, q.norms)
        t1 = time.perf_counter()
        quant_times.append(t1 - t0)

    # ── Dequantize timing ────────────────────────────────────────────
    dequant_times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        x_hat = quantizer.dequantize(q)
        if backend_name == "mlx":
            mx.eval(x_hat)
        t1 = time.perf_counter()
        dequant_times.append(t1 - t0)

    avg_quant = sum(quant_times) / len(quant_times)
    avg_dequant = sum(dequant_times) / len(dequant_times)

    q_vps = batch_size / avg_quant if avg_quant > 0 else float("inf")
    dq_vps = batch_size / avg_dequant if avg_dequant > 0 else float("inf")
    return q_vps, dq_vps


def main():
    backends = _detect_backends()
    if not backends:
        print("ERROR: No backends available (need mlx or torch).")
        return

    print("=" * 90)
    print("TurboQuant Quantization / Dequantization Throughput Benchmark")
    print("=" * 90)
    print(f"Backends: {', '.join(backends)}")
    print(f"Dims: {DIMS}  |  Bits: {BITS}  |  Batch sizes: {BATCH_SIZES}")
    print(f"Warmup: {WARMUP} runs  |  Timed: {RUNS} runs (averaged)")
    print()

    # Header
    header = f"{'dim':>5} {'bits':>5} {'batch':>7}"
    for bk in backends:
        header += f" | {bk + ' Q (v/s)':>16} {bk + ' DQ (v/s)':>16}"
    print(header)
    print("-" * len(header))

    for dim in DIMS:
        for bits in BITS:
            for batch in BATCH_SIZES:
                row = f"{dim:>5} {bits:>5} {batch:>7}"
                for bk in backends:
                    try:
                        q_vps, dq_vps = bench_quantize(bk, dim, bits, batch)
                        row += f" | {q_vps:>16,.0f} {dq_vps:>16,.0f}"
                    except Exception as e:
                        row += f" | {'ERROR':>16} {'ERROR':>16}"
                        print(f"  [warn] {bk} dim={dim} bits={bits} batch={batch}: {e}")
                print(row)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
