"""
Benchmark: Attention score computation latency at various sequence lengths.

Compares three paths:
  1. Metal kernel  — turboquant_attention_score_metal (fused GPU)
  2. MLX Python    — TurboQuantProd.attention_score (MLX lazy graph)
  3. PyTorch CPU   — TurboQuantProd.attention_score (eager CPU)
"""

import time
import math
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
BH = 8          # batch * heads (simulates 1 batch, 8 heads)
D = 128         # head dimension
BITS = 3        # quantization bits
WARMUP = 3
TIMED_RUNS = 10


def _has_backend(name):
    try:
        if name == "mlx":
            import mlx.core  # noqa: F401
            return True
        elif name == "pytorch":
            import torch  # noqa: F401
            return True
        elif name == "metal":
            from turboquant_mac.backends.metal_kernels import turboquant_attention_score_metal  # noqa: F401
            return True
    except ImportError:
        return False
    return False


def _bench_metal(N):
    """Benchmark Metal kernel path."""
    import mlx.core as mx
    from turboquant_mac.backends.metal_kernels import turboquant_attention_score_metal
    from turboquant_mac.quantizer import TurboQuantProd
    from turboquant_mac.backends import reset_backend

    reset_backend()
    quantizer = TurboQuantProd(dim=D, bits=BITS, backend="mlx")

    rng = np.random.RandomState(42)
    keys_np = rng.randn(BH, N, D).astype(np.float32)
    query_np = rng.randn(BH, D).astype(np.float32)

    keys = mx.array(keys_np)
    query = mx.array(query_np)

    # Quantize keys
    q_keys = quantizer.quantize(keys)
    mx.eval(q_keys.mse_indices, q_keys.qjl_signs, q_keys.residual_norms, q_keys.norms)

    Pi = quantizer.mse_quantizer.Pi
    S = quantizer.S
    centroids = quantizer.mse_quantizer.centroids
    qjl_scale = quantizer.qjl_scale

    # Warmup
    for _ in range(WARMUP):
        scores = turboquant_attention_score_metal(
            query, q_keys.mse_indices, q_keys.qjl_signs,
            q_keys.norms, q_keys.residual_norms,
            Pi, S, centroids, BITS - 1, qjl_scale,
        )
        mx.eval(scores)

    # Timed runs
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        scores = turboquant_attention_score_metal(
            query, q_keys.mse_indices, q_keys.qjl_signs,
            q_keys.norms, q_keys.residual_norms,
            Pi, S, centroids, BITS - 1, qjl_scale,
        )
        mx.eval(scores)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)


def _bench_mlx_python(N):
    """Benchmark MLX Python path (TurboQuantProd.attention_score)."""
    import mlx.core as mx
    from turboquant_mac.quantizer import TurboQuantProd
    from turboquant_mac.backends import reset_backend

    reset_backend()
    quantizer = TurboQuantProd(dim=D, bits=BITS, backend="mlx")

    rng = np.random.RandomState(42)
    keys_np = rng.randn(BH, N, D).astype(np.float32)
    query_np = rng.randn(BH, 1, D).astype(np.float32)

    keys = mx.array(keys_np)
    query = mx.array(query_np)

    q_keys = quantizer.quantize(keys)
    mx.eval(q_keys.mse_indices, q_keys.qjl_signs, q_keys.residual_norms, q_keys.norms)

    # Warmup
    for _ in range(WARMUP):
        scores = quantizer.attention_score(query, q_keys)
        mx.eval(scores)

    # Timed runs
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        scores = quantizer.attention_score(query, q_keys)
        mx.eval(scores)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)


def _bench_pytorch_cpu(N):
    """Benchmark PyTorch CPU path (TurboQuantProd.attention_score)."""
    from turboquant_mac.quantizer import TurboQuantProd
    from turboquant_mac.backends import reset_backend

    reset_backend()
    quantizer = TurboQuantProd(dim=D, bits=BITS, backend="pytorch")

    rng = np.random.RandomState(42)
    keys_np = rng.randn(BH, N, D).astype(np.float32)
    query_np = rng.randn(BH, 1, D).astype(np.float32)

    import torch
    keys = torch.from_numpy(keys_np)
    query = torch.from_numpy(query_np)

    q_keys = quantizer.quantize(keys)

    # Warmup
    for _ in range(WARMUP):
        _ = quantizer.attention_score(query, q_keys)

    # Timed runs
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        _ = quantizer.attention_score(query, q_keys)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times)


def main():
    has_metal = _has_backend("metal")
    has_mlx = _has_backend("mlx")
    has_pytorch = _has_backend("pytorch")

    paths = []
    if has_metal:
        paths.append(("Metal kernel", _bench_metal))
    if has_mlx:
        paths.append(("MLX Python", _bench_mlx_python))
    if has_pytorch:
        paths.append(("PyTorch CPU", _bench_pytorch_cpu))

    if not paths:
        print("ERROR: No backends available.")
        return

    print("=" * 100)
    print("TurboQuant Attention Score Latency Benchmark")
    print("=" * 100)
    print(f"Paths: {', '.join(n for n, _ in paths)}")
    print(f"BH={BH}, D={D}, bits={BITS}")
    print(f"Warmup: {WARMUP} runs  |  Timed: {TIMED_RUNS} runs (averaged)")
    print()

    # Collect results
    results = {}  # N -> {path_name: latency_ms}
    for N in SEQ_LENGTHS:
        results[N] = {}
        for name, fn in paths:
            try:
                avg_sec = fn(N)
                results[N][name] = avg_sec * 1000.0  # to ms
            except Exception as e:
                results[N][name] = None
                print(f"  [warn] {name} N={N}: {e}")

    # Print table
    path_names = [n for n, _ in paths]
    header = f"{'N':>7}"
    for name in path_names:
        header += f" | {name + ' (ms)':>16}"
    # Speedup columns: each path vs the slowest (PyTorch CPU if available)
    baseline_name = path_names[-1]  # last path as baseline
    for name in path_names[:-1]:
        header += f" | {name[:6] + ' speedup':>16}"
    print(header)
    print("-" * len(header))

    for N in SEQ_LENGTHS:
        row = f"{N:>7}"
        for name in path_names:
            val = results[N].get(name)
            if val is not None:
                row += f" | {val:>16.3f}"
            else:
                row += f" | {'ERROR':>16}"

        baseline_val = results[N].get(baseline_name)
        for name in path_names[:-1]:
            val = results[N].get(name)
            if val is not None and baseline_val is not None and val > 0:
                speedup = baseline_val / val
                row += f" | {speedup:>15.2f}x"
            else:
                row += f" | {'N/A':>16}"
        print(row)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
