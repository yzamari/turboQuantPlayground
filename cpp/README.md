# TurboQuant — C++ port for Qualcomm Snapdragon

Plain-CMake C++17 port of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026)
KV-cache compression, with five interchangeable backends — **portable scalar**,
**ARM NEON**, **Qualcomm Hexagon HTP** (via QNN), **Adreno OpenCL**, **Adreno Vulkan
compute** — designed so the same library ships unchanged from Snapdragon mobile (S24
Ultra, our verification target) to Snapdragon automotive (SA8155P / SA8295P / SA8775P).

## Status

| Phase | Status | What works |
|-------|--------|------------|
| P0  Skeleton + scalar reference + golden corpus + paired A/B bench | ✅ done | 727/727 byte-exact parity vs Python, 30,912/30,912 packing roundtrips |
| P1.a NEON CPU backend                                              | ✅ done | 1.3× over scalar on S24, automotive-portable (ARMv8.0-A baseline) |
| P1.b QNN/HTP NPU backend                                           | ✅ scaffolded | Builds when `QNN_SDK_ROOT` is set; gracefully skips otherwise |
| P2   Adreno OpenCL backend                                         | ✅ correct | Bit-exact parity; v1 perf upload-bound (see Limitations) |
| P3   Adreno Vulkan compute backend                                 | ✅ correct | Bit-exact parity; same v1 perf characteristics as OpenCL |
| P4   Android demo app                                              | ✅ scaffolded | Kotlin Compose + JNI; user runs `gradle wrapper && ./gradlew assembleDebug` |
| Docs / Qualcomm hardware notes / ASCII diagrams                    | 🚧 in flight | `docs/` folder being populated by parallel agent |

## Headline results — Samsung S24 Ultra (SD 8 Gen 3 / Adreno 750 / Hexagon V75)

### Memory compression — hardware-independent, the main win
```
seq_len   FP16 baseline    TurboQuant (3-bit)    compression
   128         0.5 MB              0.1 MB             4.27x
   512         2.0 MB              0.5 MB             4.27x
  2048         8.0 MB              1.9 MB             4.27x
  4096        16.0 MB              3.8 MB             4.27x
```

### Bit-width / quality trade-off (NEON, BH=8, D=128)
| bits | compression | cosine vs FP16 | use case |
|-----:|------------:|---------------:|----------|
|    2 |    4.92×    |      0.78      | aggressive (acceptable for some tasks) |
|    3 |    4.27×    |      0.91      | **recommended sweet spot** |
|    4 |    3.66×    |      0.98      | near-lossless |

### Numerical correctness
- **727 / 727** byte-exact checks vs Python golden corpus (`gen_golden.py`) on host **and** on the S24 Ultra
- **30,912 / 30,912** bit-packing roundtrip checks across 1/2/3/4/8-bit + QJL signs, on both targets
- All four implemented backends (scalar / NEON / OpenCL / Vulkan) produce **identical cosine similarity** at every (seq_len, bits) configuration

### Latency — paired A/B (`baseline / TurboQuant / speedup`, ms, BH=8 D=128 b=3)
| seq_len | baseline | scalar | NEON   | OpenCL  | Vulkan  |
|--------:|---------:|-------:|-------:|--------:|--------:|
|     128 | 0.085    | 0.228  | 0.196  | 23.6    | 20.6    |
|     512 | 0.240    | 0.631  | 0.544  | 21.4    | 28.4    |
|    2048 | 0.969    | 2.401  | 1.827  | 24.7    | 19.3    |
|    4096 | 1.975    | 4.721  | 3.607  | 20.6    | 22.7    |

NEON is the winner at these shapes, delivering steady ~1.3× over scalar.

## Architecture (one-paragraph)

`libturboquant` is a portable C++17 static library with **zero OS / vendor-SDK
dependency** — it includes only `<cstdint>`-style standard headers and an
`IBackend` virtual interface. Each backend lives in its own `cpp/backends/<name>/`
subdir and is the **only** place that touches its vendor SDK (QNN, OpenCL, Vulkan).
Codebooks ship as embedded byte arrays compiled in by CMake from the existing
Python project's JSON files; no filesystem at runtime. Build is plain CMake (≥
3.22) with toolchain files for Android arm64 (active), Linux aarch64 (stub for
SA8775P), QNX aarch64 (stub for SA8155P/SA8295P).

See `docs/architecture/` for ASCII diagrams of the layering, per-kernel data flow,
and KV-cache lifecycle.

## Build & verify

```bash
# Host (macOS / Linux): scalar backend only — used for CI parity
cmake -S cpp -B cpp/build-host
cmake --build cpp/build-host -j
ctest --test-dir cpp/build-host --output-on-failure   # 3/3 green: packing, smoke, parity

# Android arm64: all available backends
cmake -S cpp -B cpp/build-android \
  -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-android-arm64.cmake \
  -DTQ_WITH_NEON=ON -DTQ_WITH_OPENCL=ON -DTQ_WITH_VULKAN=ON
cmake --build cpp/build-android -j

# Push + run on connected S24
adb push cpp/build-android/bench/turboquant_bench /data/local/tmp/
adb push cpp/tests/golden /data/local/tmp/
adb shell '/data/local/tmp/tq_parity_test /data/local/tmp/golden'   # 727/727
adb shell '/data/local/tmp/turboquant_bench --bench --backend cpu_neon \
           --bits 3 --bh 8 --seq-lens 128,512,2048,4096 --warmup 2 --iters 5'
```

## Activating QNN/HTP

Requires the Qualcomm AI Engine Direct (QAIRT) SDK 2.27.x — license-walled,
download from Qualcomm Developer Network.

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.27.x
cmake --fresh -S cpp -B cpp/build-android-qnn \
  -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-android-arm64.cmake \
  -DTQ_WITH_NEON=ON -DTQ_WITH_QNN=ON
cmake --build cpp/build-android-qnn -j
# See cpp/backends/qnn_htp/README.md for runtime lib paths and ADSP_LIBRARY_PATH.
```

## Limitations & known follow-ups

1. **OpenCL/Vulkan v1 are upload-bound on small per-call shapes.** Each
   `attention_score` call re-uploads the keys (BH × N × packed_d bytes) to the GPU
   even though they don't change between calls. The kernels are correct (parity
   tests confirm) but the host-side dispatch layer needs a `prepare_keys()` API
   on `IBackend` so prefill uploads keys once and decode-time only uploads the
   query. This is the canonical "GPU KV cache" pattern; current bench numbers
   reflect the un-optimized v1.
2. **QNN/HTP** is wired but inactive without the SDK. The hybrid pipeline
   (rotate + value_dequant on HTP, mse_encode + qjl_score on NEON) is a v1 split;
   v2 should compose `mse_score` as a custom HVX UDO for max NPU utilization.
3. **3-bit u32 packing** path is implemented in scalar / NEON / OpenCL / Vulkan
   but has loose `<1e-4` tolerance vs Python at extreme `D` values; for D=128
   bits=3 the parity test passes byte-exact.
4. **Decode-time `append()`** on `TurboQuantKVCache` is a stub — current bench
   exercises prefill only. The buffer-flush mechanism is documented but not
   ported from the Python reference yet.

## Project layout

```
cpp/
  include/turboquant/           # public API: api.hpp, backend.hpp, types.hpp, codebook.hpp
  src/                          # algorithm core (portable C++17)
  backends/
    cpu_scalar/                 # P0 — pure C++17 reference
    cpu_neon/                   # P1.a — ARM NEON
    qnn_htp/                    # P1.b — Qualcomm QNN graph + Hexagon HTP (gated on SDK)
    opencl/                     # P2 — Adreno OpenCL (5 .cl kernels, dlopen libOpenCL.so)
    vulkan/                     # P3 — Adreno Vulkan compute (7 .comp shaders -> SPIR-V)
  cmake/                        # toolchain wrappers + embed_resource helper
  tools/gen_golden.py           # produces byte-exact golden corpus from Python lib
  tests/                        # packing_test, smoke_test, parity_test (727 checks)
  bench/                        # turboquant_bench CLI + baseline_kv_cache + bench_runner.hpp
  bench/results/                # CSV outputs from S24 Ultra runs

android/                        # Kotlin Compose demo app + JNI shim (P4)
docs/                           # Qualcomm hw notes, ASCII architecture diagrams
```
