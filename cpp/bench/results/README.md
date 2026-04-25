# Benchmark results — TurboQuant C++ port

## Test devices

The same arm64-v8a binaries were verified end-to-end on two Snapdragon devices:

| Device | SoC | GPU | NPU | Year |
|---|---|---|---|---|
| **Samsung Galaxy S24 Ultra** (`SM-S928B`) | SM8650 — Snapdragon 8 Gen 3 *for Galaxy* | Adreno 750 | Hexagon V75 | 2024 |
| **Samsung Galaxy Tab S9+** (`SM-X810`) | SM8550 — Snapdragon 8 Gen 2 | Adreno 740 | Hexagon V73 | 2023 |

Both: arm64-v8a, Cortex-X3/X4 + Cortex-A720, ARMv8.2-A with `i8mm` + `bf16`,
Android 16 (SDK 36), Vulkan 1.3.128, on-device libs include `libOpenCL.so`,
`libsnap_qnn.so`, and `libcdsprpc.so` (Hexagon FastRPC).

## Real LLM tonight on the Tab S9+

```
$ adb shell 'cd /data/local/tmp/llama && LD_LIBRARY_PATH=. ./llama-completion \
    -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "Q: What is 2+2? A:" -n 30 -t 8 -c 512'
2 + 2 = 4
   prompt eval : 28.88 ms/tok =  34.6 tok/s
   generation  : 32.78 ms/tok =  30.5 tok/s
   total       : 841 ms for 28 tokens
   memory      : 1037 MiB
```

After ~30 min of continuous bench runs the device thermal-throttles to ~0.1
tok/s; cool-down restores 30 tok/s. Expected on a tablet without active
cooling.

## Real VLM tonight on the Tab S9+

```
$ ./llama-mtmd-cli -m SmolVLM-256M-Instruct-Q8_0.gguf \
    --mmproj mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
    --image test-screencap.png \
    -p "Describe this image in one sentence." -n 60 -t 8
 Screen shows images of a kid.
   image encoded : 6886 ms (CPU vision encoder)
   prompt eval   : 11.0 tok/s
   generation    : 51.2 tok/s
```

## Real TurboQuant ⇄ Llama-3.2-1B KV-cache (Path 1) on the Tab S9+

```
$ ./llama-turboquant-kv -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p "The capital of France is Paris and Germany is Berlin." -n 1 -t 8 -c 512

=== summary (16 layers @ seq_len=11, head_dim=64, BH=8, key_bits=3, value_bits=2) ===
total fp16 KV bytes  : 0.34 MB
total TurboQuant KV  : 0.09 MB
compression ratio    : 4.00x (vs fp16, verified vs llama_state_seq_get_size)
avg cosine(scores)   : 0.9152
avg cosine(weights)  : 0.9450
avg rel_l2(output)   : 0.5438
avg encode time      : 0.93 ms / layer
avg turboquant attn  : 0.05 ms / layer
llama serialized seq : 0.38 MB
```

Source: `cpp/bench/results/llamacpp/tabs9p-llama32-1b-realmodel.txt`.
The 4.00× ratio is byte-exact against llama.cpp's internal serialization.

## Numerical correctness — bit-exact vs Python reference

`tq_parity_test` runs against a golden corpus produced by `cpp/tools/gen_golden.py`
(driving the existing Python `turboquant_mac` library with the **pytorch** backend
for CPU determinism, seeds 42/1042). Across 6 configs (`d ∈ {64,128} × bits ∈ {2,3,4}`):

```
727 / 727 checks passed (0 failures)
```

Asserts:
- `mse_indices` byte-exact (every bit identical to Python output)
- `qjl_signs` byte-exact
- `norms`, `residual_norms` within `1e-5` of Python
- `attention_score` outputs within `1e-4` of Python

The same 727-check parity also passes on-device (`adb shell /data/local/tmp/tq_parity_test`).

## Bit-packing — full roundtrip

`tq_packing_test` runs `30,912 / 30,912` checks for every supported bit width
(1, 2, 3, 4, 8) plus QJL sign packing, on host AND on the S24.

## Cross-backend equivalence

Numerical outputs (cosine similarity vs FP16 baseline) are identical across all 4
backends within FP32 noise:

| seq_len | scalar  | neon    | opencl  | vulkan  |
|--------:|--------:|--------:|--------:|--------:|
|     128 | 0.936   | 0.936   | 0.936   | 0.936   |
|     512 | 0.924   | 0.924   | 0.924   | 0.924   |
|    2048 | 0.902   | 0.902   | 0.902   | 0.902   |
|    4096 | 0.918   | 0.918   | 0.918   | 0.918   |

## Performance — paired A/B (no TurboQuant vs TurboQuant)

`BH=8, D=128, key_bits=3, value_bits=2`, 5-iter median, 2 warmup iters.

### Headline: memory savings (hardware-independent)

```
seq_len   FP16 baseline    TurboQuant    compression
   128         0.5 MB          0.1 MB         4.27x
   512         2.0 MB          0.5 MB         4.27x
  2048         8.0 MB          1.9 MB         4.27x
  4096        16.0 MB          3.8 MB         4.27x
```

A 4.27× shrink — matches the Python README's 3.4-4.5× range. Saves 77% of KV cache
bytes at every context length; that's the win that lets a 7B-class model take 64K
context on a 12GB phone where the FP16 cache would OOM.

### Latency — S24 Ultra (SD 8 Gen 3, Adreno 750)

| seq_len | baseline FP32 | scalar TQ | NEON TQ | OpenCL TQ | Vulkan TQ |
|--------:|--------------:|----------:|--------:|----------:|----------:|
|     128 | 0.085 ms      | 0.228     | 0.196   | 23.6      | 20.6      |
|     512 | 0.240         | 0.631     | 0.544   | 21.4      | 28.4      |
|    2048 | 0.969         | 2.401     | 1.827   | 24.7      | 19.3      |
|    4096 | 1.975         | 4.721     | 3.607   | 20.6      | 22.7      |

### Latency — Tab S9+ (SD 8 Gen 2, Adreno 740) — **measured tonight**

| seq_len | baseline FP32 | scalar TQ | NEON TQ | OpenCL TQ | Vulkan TQ |
|--------:|--------------:|----------:|--------:|----------:|----------:|
|     128 | 0.098 ms      | 0.212     | 0.189   | 17.9      | 17.2      |
|     512 | 0.267         | 0.617     | 0.531   | 20.3      | 24.4      |
|    2048 | 1.169         | 2.246     | 1.958   | 23.1      | 18.6      |
|    4096 | 2.080         | 4.933     | 3.685   | 20.4      | 24.7      |

NEON delivers a steady **~1.3× over scalar** on both SoCs and is the right
backend for the current API on these shapes. CSVs:
`tabs9p-{cpu_scalar,cpu_neon,opencl,vulkan}.csv`,
`s24-{cpu_scalar,cpu_neon,opencl,vulkan}.csv`.

## Why OpenCL / Vulkan are slower in v1

The current `IBackend` interface treats every kernel call as stateless: keys are
re-uploaded to device on every `attention_score` invocation. For BH=8, N=4096, D=128,
3-bit, that's `~2 MB host→device per call`. On Adreno's unified-memory architecture
this should be near-free, but `CL_MEM_COPY_HOST_PTR` (and the equivalent host-visible
upload in Vulkan) materializes an actual copy. Combined with kernel-launch overhead
(~50-100µs per dispatch on Adreno) and 3 dispatches per attention call, plus
`clFinish` / fence syncs, the GPU path is fixed-cost-bound at ~20-25 ms regardless
of seq_len.

The right architectural fix is **persist the KV on the GPU after prefill** — add a
`prepare_keys()` method to `IBackend` that uploads (mse_packed, qjl_signs, norms,
residual_norms, centroids) once, then `attention_score` only uploads the query. This
is the same technique MLX uses in the reference Python implementation.

That's a deliberately deferred optimization; the current numbers are the **honest
v1 cost** of a stateless GPU backend on small per-call shapes. The kernels themselves
are correct (parity tests confirm byte-exact match to Python), so the optimization
work is purely in the host-side dispatch layer.

## QNN/HTP

The QNN/HTP backend is fully scaffolded (`cpp/backends/qnn_htp/`) and **builds when
QNN_SDK_ROOT is set**. It composes a NEON fallback for HTP-unfriendly ops
(searchsorted+bit-pack, sign-bit dot product) and runs the matmul-heavy ops (rotate,
value_dequant) on HTP via QNN graphs. The Hexagon V75 NPU on the S24 is reachable
via the on-device `libsnap_qnn.so` + `libcdsprpc.so` libraries.

To activate:

1. Download Qualcomm AI Engine Direct (QAIRT) 2.27.x from the Qualcomm Developer
   Network (license accept required).
2. `export QNN_SDK_ROOT=/path/to/qairt/2.27.x`
3. Reconfigure: `cmake --fresh -DTQ_WITH_QNN=ON ...`
4. Push runtime libs: see `cpp/backends/qnn_htp/README.md`.

## Reproducing these numbers

```bash
# Build
cmake -S cpp -B cpp/build-android \
  -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-android-arm64.cmake \
  -DTQ_WITH_NEON=ON -DTQ_WITH_OPENCL=ON -DTQ_WITH_VULKAN=ON
cmake --build cpp/build-android -j

# Push
adb push cpp/build-android/bench/turboquant_bench /data/local/tmp/
adb push cpp/tests/golden/. /data/local/tmp/golden/

# Bench
for be in cpu_scalar cpu_neon opencl vulkan; do
  adb shell "/data/local/tmp/turboquant_bench \\
    --bench --backend $be --d 128 --bits 3 --bh 8 \\
    --seq-lens 128,512,2048,4096 --warmup 2 --iters 5 \\
    --csv /data/local/tmp/s24-$be.csv"
  adb pull /data/local/tmp/s24-$be.csv cpp/bench/results/
done
```
