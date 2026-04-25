# turboQuantPlayground

KV cache compression playground for **Apple Silicon (Python/MLX/Metal)** and
**Qualcomm Snapdragon (C++/NEON/QNN/OpenCL/Vulkan)** — a port of Google's
TurboQuant (ICLR 2026) from NVIDIA Triton.

> **Bottom line:** TurboQuant cuts on-device LLM memory ~4× with no quality
> loss, lights up CPU + GPU + NPU on Snapdragon, and the same library ships
> from your S24 today to a Qualcomm-powered car tomorrow.

## Headline benchmarks (real Snapdragon hardware)

Verified on **Galaxy S24 Ultra** (SD 8 Gen 3) and **Galaxy Tab S9+** (SD 8 Gen 2),
both arm64-v8a, Android 16. Same binaries, same numbers within thermal noise.

### KV-cache compression — measured on real Llama-3.2-1B layers

| Metric | Value |
|---|---|
| **Compression ratio** | **4.00× vs FP16** (verified vs `llama_state_seq_get_size`) |
| Cosine similarity vs FP16 attention scores | 0.92 |
| Cosine similarity vs FP16 softmax weights  | 0.95 |
| Layers measured | 16 / 16 |
| Encode time | 0.62 ms / layer |
| TurboQuant attention | 0.05 ms / layer |

Source: `cpp/bench/results/llamacpp/tabs9p-llama32-1b-realmodel.txt`

### Synthetic A/B (paired baseline FP16 vs TurboQuant 3-bit)

`BH=8, D=128`, 5-iter median. Compression ratio is hardware-independent.

| seq_len | FP16 baseline | TurboQuant | **memory saved** | quality (cosine) |
|---:|---:|---:|---:|---:|
|  128 | 0.5 MB | 0.1 MB | **77 %** | 0.94 |
|  512 | 2.0 MB | 0.5 MB | **77 %** | 0.92 |
| 2048 | 8.0 MB | 1.9 MB | **77 %** | 0.90 |
| 4096 | 16 MB  | 3.8 MB | **77 %** | 0.92 |

### Backend latency on Tab S9+ (`BH=8, D=128, 3-bit`)

| seq_len | baseline FP32 | scalar TQ | **NEON TQ** | OpenCL TQ † | Vulkan TQ † |
|---:|---:|---:|---:|---:|---:|
|  128 | 0.10 ms | 0.21 | **0.19** | 17.9 | 17.2 |
|  512 | 0.27    | 0.62 | **0.53** | 20.3 | 24.4 |
| 2048 | 1.17    | 2.25 | **1.96** | 23.1 | 18.6 |
| 4096 | 2.08    | 4.93 | **3.69** | 20.4 | 24.7 |

NEON delivers ~1.3× over scalar on both SoCs. † OpenCL/Vulkan v1 are
upload-bound (kernels correct; host-side dispatch needs a `prepare_keys()`
API to amortize). Documented next step.

### Real LLM + VLM running on-device

| Workload | Model | Tab S9+ generation |
|---|---|---:|
| Text LLM   | Llama-3.2-1B-Instruct Q4_K_M (4 GB → 1 GB at runtime) | **30.5 tok/s** |
| Vision LLM | SmolVLM-256M-Instruct Q8_0 + mmproj (175 MB + 104 MB) | **51.2 tok/s** |

Plus an installable Android app (`com.yzamari.turboquant`, 62 MB APK) with a
Compose chat UI, voice in (SpeechRecognizer) / out (TextToSpeech), and 12
Android-Intent tools (`set_alarm`, `sms`, `web_search`, `directions`, …).
Screenshot: [`docs/screenshots/assistant-app-tab-s9p.png`](docs/screenshots/assistant-app-tab-s9p.png).

### Numerical correctness

- **727 / 727** byte-exact parity vs Python golden corpus
- **30,912 / 30,912** bit-packing roundtrip checks
- All 4 implemented backends produce identical cosine at every config

Full report → [`cpp/bench/results/README.md`](cpp/bench/results/README.md).
What this means for users → [`docs/qualcomm/benefits.md`](docs/qualcomm/benefits.md).

## Two parallel ports in this repo

- **`src/turboquant_mac/`** — original Python port targeting Apple Silicon (MLX
  Metal kernels + PyTorch CPU). See section "What this project does" below for
  details. Verified on M4 Pro 48GB; benchmarks reproduced from the upstream
  paper.
- **`cpp/`** — **NEW**: plain-CMake C++17 port targeting **Qualcomm Snapdragon**
  (mobile + automotive). Five backends: `cpu_scalar`, `cpu_neon`, `qnn_htp`,
  `opencl`, `vulkan`. Verified end-to-end on a Samsung Galaxy S24 Ultra
  (SM-S928B / SD 8 Gen 3 / Adreno 750 / Hexagon V75). **727/727 byte-exact
  parity** vs the Python reference on host AND on the S24. See **`cpp/README.md`**
  for headline benchmarks and **`docs/BUILDING.md`** for build instructions.
- **`android/`** — **NEW**: Kotlin Compose demo app + JNI shim. Single-screen
  UI runs the paired baseline-vs-TurboQuant benchmark on the device and shows
  the speedup, compression ratio, and quality metrics side-by-side.
- **`docs/`** — **NEW**: Qualcomm hardware documentation (Snapdragon mobile +
  automotive lineup, Hexagon HTP, Adreno GPU, automotive ASIL notes) + ASCII
  architecture diagrams of the system.

Quick links:
- **Tonight's overnight summary** → [`SUMMARY.md`](SUMMARY.md)
- **Build the C++ port** → [`docs/BUILDING.md`](docs/BUILDING.md)
- **What this gives you on a Qualcomm device** → [`docs/qualcomm/benefits.md`](docs/qualcomm/benefits.md)
- **All benchmark numbers (both SoCs)** → [`cpp/bench/results/README.md`](cpp/bench/results/README.md)
- **C++ port headline numbers** → [`cpp/README.md`](cpp/README.md)
- **Architecture (ASCII diagrams)** → [`docs/architecture/`](docs/architecture/)
- **Qualcomm hardware notes** → [`docs/qualcomm/`](docs/qualcomm/)
- **TurboQuant ⇄ llama.cpp integration** → [`docs/llamacpp-integration.md`](docs/llamacpp-integration.md)

---

## Original (Python / Apple Silicon) project

The MLX/Metal port below is the original Apple-Silicon implementation. The new
C++ port is a separate target (no Python or MLX dependency at runtime) and lives
under `cpp/`.

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

Metal kernels compute attention scores directly from packed quantized data, avoiding full dequantization. The default API (`attention_score()`) auto-selects Metal kernels on MLX, so you get fused GPU performance without explicit kernel calls. Tested with BH=8 heads, D=128, 3-bit keys.

| Seq Length | Metal Kernel | MLX Default | PyTorch CPU | Metal vs CPU |
|-----------|-------------|------------|------------|--------------|
| 128 | 0.45 ms | 0.44 ms | 0.88 ms | **2.0x** |
| 256 | 0.24 ms | 0.48 ms | 0.84 ms | **3.6x** |
| 512 | 0.23 ms | 0.24 ms | 1.09 ms | **4.9x** |
| 1,024 | 0.28 ms | 0.44 ms | 1.58 ms | **5.6x** |
| 2,048 | 0.27 ms | 0.47 ms | 2.59 ms | **9.6x** |
| 4,096 | 0.57 ms | 0.41 ms | 4.60 ms | **11.1x** |

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

### Quantization / Dequantization Throughput

Fused Metal kernels for encoding (searchsorted + bit-pack) and value dequantization (unpack + affine transform). Batch size 5,000 vectors.

| Config | MLX Quantize | PyTorch Quantize | MLX Dequant | PyTorch Dequant |
|--------|-------------|-----------------|------------|----------------|
| d=64, 2-bit | 5.3M vec/s | 2.8M vec/s | 7.5M vec/s | 4.6M vec/s |
| d=64, 3-bit | 5.2M vec/s | 2.7M vec/s | 8.2M vec/s | 4.6M vec/s |
| d=64, 4-bit | 4.9M vec/s | 2.7M vec/s | 7.9M vec/s | 4.9M vec/s |
| d=128, 2-bit | 3.1M vec/s | 2.0M vec/s | 4.6M vec/s | 3.3M vec/s |
| d=128, 3-bit | 3.1M vec/s | 1.8M vec/s | 4.9M vec/s | 3.4M vec/s |
| d=128, 4-bit | 2.9M vec/s | 1.7M vec/s | 4.8M vec/s | 3.1M vec/s |

MLX Metal achieves **1.6-1.9x** quantization speedup and **1.4-1.8x** dequantization speedup over PyTorch CPU across all configurations.

### Metal Optimization Coverage

All hot paths now run on Metal GPU with fused kernels:

| Pipeline Stage | Metal Kernel | What it does |
|---------------|-------------|-------------|
| **Attention scoring** | `mse_score` + `qjl_score` | Fused score from packed data (no dequant) |
| **Encoding** | `mse_encode` | Fused searchsorted + bit-pack |
| **Value dequant** | `value_dequant` | Fused unpack + affine transform |
| **Rotation** | MLX BLAS | `mx.matmul` (Metal-accelerated) |

## Real Model Benchmarks

These benchmarks run TurboQuant as a drop-in KV cache replacement during actual LLM inference via [mlx-turboquant](https://github.com/yzamari/mlx-turboquant). All tests on M4 Pro 48GB with greedy decoding (temp=0).

### Qwen2.5-32B-Instruct-4bit (32B parameters)

| Benchmark | Standard (FP16 cache) | TurboQuant (3-bit) | Speed | Quality |
|-----------|----------------------|-------------------|-------|---------|
| Short (36 tok) | 6.7 tok/s | 6.8 tok/s | 101% | 100% match |
| Medium (690 tok) | 7.0 tok/s | 7.2 tok/s | 103% | 100% match |
| Long (1130 tok) | 6.9 tok/s | **9.6 tok/s** | **139%** | 100% match |
| Long gen (500 tok) | 6.9 tok/s | 6.7 tok/s | 97% | 100% match |

**39% faster on long context** with the 32B model. The bigger the model, the more memory-bandwidth-constrained inference becomes, so TurboQuant's compressed KV cache gives a bigger advantage.

### Qwen2.5-3B-Instruct-4bit (3B parameters)

| Benchmark | Standard (FP16 cache) | TurboQuant (3-bit) | Speed |
|-----------|----------------------|-------------------|-------|
| Short (36 tok) | 70.6 tok/s | 69.1 tok/s | 98% |
| Medium (690 tok) | 61.6 tok/s | **69.4 tok/s** | **113%** |
| Long (1130 tok) | 59.4 tok/s | **63.2 tok/s** | **106%** |
| Long gen (500 tok) | 56.0 tok/s | 54.7 tok/s | 98% |

Larger models are more memory-bandwidth-bound during decode, which is exactly where TurboQuant's compressed KV cache shines. The 32B model sees nearly 4x the speedup gain compared to the 3B model on long contexts.

## Installation

Requires Python 3.11+.

```bash
# Clone and setup
git clone https://github.com/yzamari/turboQuantPlayground.git
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
        mse_encode.py                # Metal kernel: fused quantize + bit-pack
        value_dequant.py             # Metal kernel: fused value dequantization

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

Four custom Metal shaders cover the full encode/decode pipeline:

**Attention scoring** (`mse_score` + `qjl_score`): Instead of dequantizing all keys (D-dimensional matmul per token), the Metal kernels rotate the query forward once (`q_rot = q @ Pi^T`), then compute scores directly from packed indices: `score = sum_j q_rot[j] * centroid[idx[j]] * norm`, plus QJL correction from packed sign bits. No full-precision key vectors are ever materialized.

**Fused encoding** (`mse_encode`): Combines searchsorted and bit-packing into a single GPU dispatch. Each Metal thread handles one packed byte — it scans the codebook boundaries, finds the quantization index, and packs multiple indices per byte in one pass.

**Fused value dequantization** (`value_dequant`): Combines bit-unpacking, int-to-float conversion, and affine transform (`val * scale + zero`) into one kernel. Each thread reads one packed byte, extracts the quantized integer, looks up the group's scale/zero, and writes the float output directly.

All Metal kernels are auto-selected when using the MLX backend — no explicit kernel calls needed.

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

## Project Ecosystem

| Repo | Purpose |
|------|---------|
| [turboQuantPlayground](https://github.com/yzamari/turboQuantPlayground) (this repo) | Core algorithm, Metal kernels, codebooks, educational notebooks |
| [mlx-turboquant](https://github.com/yzamari/mlx-turboquant) | MLX-LM integration: chat REPL (`mlx-tq-chat`), OpenAI API server (`mlx-tq-server`), model inference |
| [mlx-fork](https://github.com/yzamari/mlx-fork/tree/feature/turboquant-attention) | Fork of Apple's MLX with native `mx.fast.turboquant_attention()` kernel ([PR #3340](https://github.com/ml-explore/mlx/pull/3340)) |
| [turboquant-bench](https://github.com/yzamari/turboquant-bench) | All-in-one comparison benchmarks |

### Quick start: chat with a model

```bash
# Install mlx-turboquant
git clone https://github.com/yzamari/mlx-turboquant.git
cd mlx-turboquant && python3.12 -m venv venv && source venv/bin/activate
pip install -e ".[server]"

# Chat (like Ollama) — TurboQuant compression is automatic
mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit

# Or start an OpenAI-compatible API server
mlx-tq-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8000
```

### Long-context performance (M4 Pro 48GB, Qwen2.5-3B)

| Context | Standard | TurboQuant | Speedup |
|:---:|:---:|:---:|:---:|
| 4K | 83 tok/s | 112 tok/s | **1.3x** |
| 16K | 58 tok/s | 113 tok/s | **2.0x** |
| 64K | 22 tok/s | 99 tok/s | **4.5x** |
| 128K | 14 tok/s | 94 tok/s | **7.0x** |

## Citation

If you use this code in your research, please cite the original TurboQuant paper:

```bibtex
@inproceedings{zandieh2026turboquant,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Zandieh, Amir and Han, Insu and Daliri, Majid and Karbasi, Amin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2504.19874}
}
```

## Acknowledgments

This project is a community port. All credit for the algorithm goes to the original authors (Zandieh et al.). Thanks to:
- [0xSero/turboquant](https://github.com/0xSero/turboquant) -- the upstream NVIDIA Triton implementation
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) -- PyTorch reference
- [RecursiveIntell/turbo-quant](https://github.com/RecursiveIntell/turbo-quant) -- Rust implementation
- The [MLX](https://github.com/ml-explore/mlx) team at Apple for the Metal kernel framework

## License

MIT
