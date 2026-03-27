# turboquant-mac

KV cache compression for Apple Silicon -- a port of Google's TurboQuant (ICLR 2026) from NVIDIA Triton to MLX Metal kernels and PyTorch CPU.

## What is TurboQuant?

TurboQuant is a two-stage KV cache compression algorithm introduced by Zandieh et al. at ICLR 2026 (arXiv:2504.19874). It combines:

1. **PolarQuant** -- random orthogonal rotation + Lloyd-Max scalar quantization. After rotation, each coordinate follows a known Beta distribution, enabling optimal per-coordinate scalar quantization with zero calibration data.
2. **QJL (Quantized Johnson-Lindenstrauss)** -- 1-bit sign sketching of the residual error, producing an unbiased inner-product estimator for attention score computation.

Key properties:
- **6x memory compression** at 3-bit precision with zero accuracy loss
- **Data-oblivious** -- no calibration data, training, or fine-tuning required
- **Near-optimal** -- within 2.7x of the information-theoretic lower bound
- **Complements weight quantization** -- TurboQuant compresses KV caches at runtime, not model weights

## What this project does

The upstream implementation ([0xSero/turboquant](https://github.com/0xSero/turboquant)) relies on NVIDIA Triton kernels that only run on CUDA GPUs. This project ports the full algorithm to run natively on Apple Silicon:

- **MLX Metal backend** -- custom Metal shaders compiled via `mx.fast.metal_kernel()` for GPU-accelerated attention scoring directly from packed quantized data.
- **PyTorch CPU backend** -- reference implementation for learning, debugging, and non-GPU environments.
- **Backend abstraction layer** -- auto-detects MLX (preferred) or PyTorch at import time; all quantizer and KV cache code is backend-agnostic.
- **Pre-computed Lloyd-Max codebooks** -- cached as JSON for dimensions 64, 128, 576 at bit widths 1-4.
- **Educational notebooks** -- step-by-step scripts explaining each algorithm component.

## Benchmark Results (Apple Silicon)

### Attention Score Latency

Metal kernels compute attention scores directly from packed quantized data, avoiding full dequantization. Tested with BH=8 heads, D=128, 3-bit keys.

| Seq Length | Metal Kernel | MLX Python | PyTorch CPU | Metal Speedup |
|-----------|-------------|-----------|------------|---------------|
| 128 | 0.46 ms | 0.70 ms | 0.70 ms | 1.5x |
| 256 | 0.33 ms | 0.59 ms | 0.85 ms | 2.6x |
| 512 | 0.23 ms | 0.88 ms | 1.12 ms | 4.9x |
| 1,024 | 0.49 ms | 1.34 ms | 1.47 ms | 3.0x |
| 2,048 | 0.41 ms | 2.16 ms | 2.46 ms | 6.0x |
| 4,096 | 0.56 ms | 4.04 ms | 4.53 ms | **8.2x** |

Metal kernels scale sub-linearly while CPU scales linearly. The speedup grows with sequence length.

### Memory Compression

Simulated with 32 layers, 8 KV heads, D=128. Buffer of 128 recent unquantized tokens.

| Seq Length | Bits | FP16 Baseline | TurboQuant | Compression | Savings |
|-----------|------|--------------|------------|-------------|---------|
| 1,024 | 3-bit | 128 MB | 38 MB | 3.4x | 70.4% |
| 4,096 | 3-bit | 512 MB | 113 MB | 4.5x | 78.0% |
| 8,192 | 3-bit | 1,024 MB | 213 MB | 4.8x | 79.2% |
| 16,384 | 2-bit | 2,048 MB | 349 MB | **5.9x** | **82.9%** |
| 16,384 | 3-bit | 2,048 MB | 413 MB | 5.0x | 79.8% |

Compression ratio improves with longer sequences as the fixed-cost unquantized buffer is amortized.

### Dequantization Throughput

MLX leverages Metal GPU for dequantization (codebook lookup + matrix multiply).

| Config | MLX (Metal) | PyTorch CPU | MLX Speedup |
|--------|------------|-------------|-------------|
| d=64, 3-bit, 5K vectors | 8.0M vec/s | 5.0M vec/s | 1.6x |
| d=128, 3-bit, 5K vectors | 4.1M vec/s | 3.4M vec/s | 1.2x |

## Installation

Requires Python 3.11+.

```bash
# Clone and setup
git clone https://github.com/<your-user>/turboQuantPlayground.git
cd turboQuantPlayground
python -m venv venv && source venv/bin/activate

# MLX backend (Apple Silicon GPU acceleration):
pip install -e ".[mlx,dev]"

# OR PyTorch-only (no GPU):
pip install -e ".[torch,dev]"
```

## Quick Start

```python
from turboquant_mac import TurboQuantProd
from turboquant_mac.backends import get_backend
import numpy as np

B = get_backend()  # auto-detects MLX or PyTorch

# Create a 3-bit inner-product quantizer for head_dim=128
tq = TurboQuantProd(dim=128, bits=3)

# Simulate 512 key vectors
keys = B.from_numpy(np.random.randn(512, 128).astype(np.float32))
query = B.from_numpy(np.random.randn(1, 128).astype(np.float32))

# Quantize keys
q = tq.quantize(keys)

# Compute attention scores directly from quantized data
scores = tq.attention_score(query, q)   # shape: (1, 512)

# Check compression
fp16_bytes = 512 * 128 * 2
packed_bytes = (
    B.to_numpy(q.mse_indices).nbytes
    + B.to_numpy(q.qjl_signs).nbytes
    + B.to_numpy(q.residual_norms).nbytes
    + B.to_numpy(q.norms).nbytes
)
print(f"Compression ratio: {fp16_bytes / packed_bytes:.1f}x")
```

### Using the KV Cache

```python
from turboquant_mac import TurboQuantKVCache
from turboquant_mac.backends import get_backend
import numpy as np

B = get_backend()

cache = TurboQuantKVCache(head_dim=128, key_bits=3, value_bits=2, buffer_size=64)

# Prefill with 256 tokens (batch=1, heads=4)
keys = B.from_numpy(np.random.randn(1, 4, 256, 128).astype(np.float32))
values = B.from_numpy(np.random.randn(1, 4, 256, 128).astype(np.float32))
cache.prefill(keys, values)

# Decode step
query = B.from_numpy(np.random.randn(1, 4, 1, 128).astype(np.float32))
scores = cache.attention_scores(query)          # (1, 4, 1, 256)
weights = B.softmax(scores, dim=-1)
output = cache.attend(weights)                   # (1, 4, 1, 128)

print(f"Memory: {cache.memory_bytes()}")
```

### Using Metal Kernels Directly

```python
import mlx.core as mx
from turboquant_mac.backends.metal_kernels import turboquant_attention_score_metal

# After quantizing keys with TurboQuantProd...
scores = turboquant_attention_score_metal(
    query=query_flat,
    mse_packed=quantized.mse_indices,
    qjl_signs=quantized.qjl_signs,
    norms=quantized.norms,
    residual_norms=quantized.residual_norms,
    Pi=tq.mse_quantizer.Pi,
    S=tq.S,
    centroids=tq.mse_quantizer.centroids,
    mse_bits=tq.bits - 1,
    qjl_scale=tq.qjl_scale,
)
mx.eval(scores)
```

## Project Structure

```
turboQuantPlayground/
  pyproject.toml                     # Package config (pip install -e ".[mlx,dev]")
  README.md

  src/turboquant_mac/
    __init__.py                      # Public API exports
    codebook.py                      # Lloyd-Max codebook computation + JSON caching
    rotation.py                      # Random rotation (QR) and QJL matrix generation
    quantizer.py                     # TurboQuantMSE (Alg 1) + TurboQuantProd (Alg 2)
    kv_cache.py                      # Drop-in KV cache with key + value quantization
    codebooks/                       # 9 pre-computed codebook JSON files
    backends/
      __init__.py                    # Auto-detection and backend dispatch
      mlx_backend.py                 # MLX array ops (Apple Silicon Metal GPU)
      pytorch_backend.py             # PyTorch CPU array ops
      metal_kernels.py               # Metal shader compilation and dispatch
      metal/
        mse_score.py                 # Metal kernel: MSE attention scoring
        qjl_score.py                 # Metal kernel: QJL correction scoring

  tests/                             # 63 tests across 5 modules
    test_codebook.py                 # Beta PDF, Lloyd-Max, codebook loading
    test_rotation.py                 # Orthogonality, norm preservation, determinism
    test_quantizer.py                # Bit-packing, MSE distortion, inner product bias
    test_kv_cache.py                 # Prefill/decode, value quantization, memory
    test_metal_kernels.py            # Metal vs PyTorch reference comparison

  benchmarks/
    bench_quantize.py                # Quantization throughput (vectors/sec)
    bench_attention_score.py         # Score latency: Metal vs MLX vs CPU
    bench_memory.py                  # Memory footprint comparison

  notebooks/
    01_lloyd_max_codebook.py         # Visualize Beta PDF and codebook centroids
    02_rotation_and_quantization.py  # Step through the MSE quantization pipeline
    03_qjl_correction.py            # Demonstrate bias correction with QJL
    04_kv_cache_demo.py             # Full KV cache with needle-in-haystack test

  upstream/                          # READ-ONLY reference implementations
    turboquant-0xSero/               # 0xSero/turboquant (NVIDIA Triton, MIT)
    turboquant-pytorch/              # tonbistudio/turboquant-pytorch (PyTorch, MIT)
    turbo-quant/                     # RecursiveIntell/turbo-quant (Rust, MIT)
```

## Algorithm Overview

### Stage 1: PolarQuant (MSE-optimal, b-1 bits)

```
Input x -> normalize to unit sphere -> rotate: y = Pi @ x_unit
        -> quantize each y[j] via Lloyd-Max codebook -> bit-pack indices
        -> store indices + original norm
```

The rotation matrix Pi is a random orthogonal matrix (QR of Gaussian). After rotation, each coordinate follows a Beta distribution with a known, precomputable density. The Lloyd-Max codebook minimizes MSE for this distribution without needing calibration data.

### Stage 2: QJL Correction (1 bit)

```
Residual r = x - dequant(stage1)
          -> project: s = S @ r (random Gaussian matrix)
          -> store sign(s) as 1 bit per coordinate + ||r||
```

The combined inner product estimator is unbiased:
```
<q, k> ~ <q, k_mse> + ||r|| * sqrt(pi/2)/d * <S@q, sign(S@r)>
```

### Metal Kernel Optimization

Instead of dequantizing all keys (D-dimensional matmul per token), the Metal kernels:
1. Rotate the query forward: `q_rot = q @ Pi^T` (once per decode step)
2. Compute scores directly from packed indices: `score = sum_j q_rot[j] * centroid[idx[j]] * norm`
3. Add QJL correction from packed sign bits

This avoids materializing full-precision key vectors entirely.

## Backends

| | MLX Metal | PyTorch CPU |
|---|---|---|
| **Hardware** | Apple Silicon (M1-M4) | Any CPU |
| **GPU acceleration** | Metal via unified memory | None |
| **Install** | `pip install mlx` | `pip install torch` |
| **Custom kernels** | MSE + QJL Metal shaders | N/A |
| **Best for** | Inference on Mac | Learning, debugging, CI |
| **Auto-detected** | Yes (preferred) | Yes (fallback) |

Force a specific backend:

```python
tq_mlx = TurboQuantProd(dim=128, bits=3, backend="mlx")
tq_pt  = TurboQuantProd(dim=128, bits=3, backend="pytorch")
```

## Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

63 tests covering: codebook correctness, rotation properties, MSE distortion bounds, inner product unbiasedness, bit-packing roundtrips, KV cache operations, value group quantization, Metal kernel vs CPU parity, and memory savings.

## Benchmarks

```bash
python benchmarks/bench_attention_score.py   # Metal vs MLX vs CPU latency
python benchmarks/bench_memory.py            # Compression ratios
python benchmarks/bench_quantize.py          # Throughput (vectors/sec)
```

## Educational Notebooks

```bash
python notebooks/01_lloyd_max_codebook.py         # Beta PDF + codebook visualization
python notebooks/02_rotation_and_quantization.py  # MSE quantization pipeline
python notebooks/03_qjl_correction.py             # QJL bias correction demo
python notebooks/04_kv_cache_demo.py              # Full KV cache + needle-in-haystack
```

## References

- **Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026)
- **OpenReview**: [ICLR 2026 accepted paper](https://openreview.net/forum?id=tO3ASKZlok)
- **Google Research Blog**: [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **Companion papers**: [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617), [QJL (AAAI 2025)](https://arxiv.org/abs/2406.03482)
- **Upstream implementations**: [0xSero/turboquant](https://github.com/0xSero/turboquant) (Triton), [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) (PyTorch)

## License

MIT
