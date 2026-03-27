"""
Benchmark: Memory footprint for realistic KV cache configurations.

Simulates a 32-layer transformer with 8 KV heads at D=128 and compares
FP16 baseline memory vs TurboQuant compressed memory for various
sequence lengths and bit widths.
"""

import numpy as np

from turboquant_mac.kv_cache import TurboQuantKVCache
from turboquant_mac.backends import get_backend, reset_backend

# ── Configuration ────────────────────────────────────────────────────
N_LAYERS = 32
N_KV_HEADS = 8
D = 128
BITS_LIST = [2, 3, 4]
SEQ_LENGTHS = [1_024, 4_096, 8_192, 16_384]
BUFFER_SIZE = 128


def _detect_backend():
    """Pick the best available backend."""
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        return "pytorch"
    except ImportError:
        pass
    raise ImportError("No backend available (need mlx or torch).")


def fp16_kv_bytes(seq_len, n_layers, n_heads, head_dim):
    """FP16 memory for both keys and values across all layers."""
    # keys + values, each: n_layers * n_heads * seq_len * head_dim * 2 bytes
    return 2 * n_layers * n_heads * seq_len * head_dim * 2


def measure_tq_memory(seq_len, bits, backend_name):
    """
    Create TurboQuantKVCache for each layer, prefill, and sum memory.

    Returns total bytes across all layers.
    """
    B = get_backend(backend_name)
    rng = np.random.RandomState(0)

    total_bytes = 0
    for layer_idx in range(N_LAYERS):
        cache = TurboQuantKVCache(
            head_dim=D,
            key_bits=bits,
            value_bits=min(bits, 2),  # value bits capped at 2 (common setting)
            value_group_size=32,
            buffer_size=BUFFER_SIZE,
            layer_idx=layer_idx,
            backend=backend_name,
        )

        # Generate random KV data (batch=1)
        keys_np = rng.randn(1, N_KV_HEADS, seq_len, D).astype(np.float32)
        values_np = rng.randn(1, N_KV_HEADS, seq_len, D).astype(np.float32)

        cache.prefill(B.from_numpy(keys_np), B.from_numpy(values_np))
        mem = cache.memory_bytes()
        total_bytes += mem["total"]

    return total_bytes


def main():
    backend_name = _detect_backend()
    reset_backend()

    print("=" * 90)
    print("TurboQuant KV Cache Memory Footprint Benchmark")
    print("=" * 90)
    print(f"Backend: {backend_name}")
    print(f"Model: {N_LAYERS} layers, {N_KV_HEADS} KV heads, D={D}")
    print(f"Buffer (unquantized recent tokens): {BUFFER_SIZE}")
    print(f"Value bits: min(key_bits, 2)")
    print()

    # Header
    header = f"{'seq_len':>9} {'bits':>5} {'FP16 (MB)':>12} {'TQ (MB)':>12} {'ratio':>10} {'savings':>10}"
    print(header)
    print("-" * len(header))

    for seq_len in SEQ_LENGTHS:
        fp16_bytes = fp16_kv_bytes(seq_len, N_LAYERS, N_KV_HEADS, D)
        fp16_mb = fp16_bytes / (1024 * 1024)

        for bits in BITS_LIST:
            try:
                reset_backend()
                tq_bytes = measure_tq_memory(seq_len, bits, backend_name)
                tq_mb = tq_bytes / (1024 * 1024)
                ratio = fp16_mb / tq_mb if tq_mb > 0 else float("inf")
                savings_pct = (1.0 - tq_mb / fp16_mb) * 100.0 if fp16_mb > 0 else 0.0
                print(
                    f"{seq_len:>9,} {bits:>5} {fp16_mb:>12.2f} {tq_mb:>12.2f} "
                    f"{ratio:>9.2f}x {savings_pct:>9.1f}%"
                )
            except Exception as e:
                print(f"{seq_len:>9,} {bits:>5} {fp16_mb:>12.2f} {'ERROR':>12} {'N/A':>10} {'N/A':>10}")
                print(f"  [warn] {e}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
