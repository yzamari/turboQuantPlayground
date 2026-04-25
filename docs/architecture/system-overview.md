# System Overview

A single ASCII diagram of the entire system, top to bottom, with shapes and
data flow annotated at every layer. Read this before any other doc in
`architecture/`.

## 1. Big picture

```
                    ┌───────────────────────────────────────────────────────┐
                    │  Python reference  (src/turboquant_mac/)              │
                    │  • the literal porting spec for the C++ port          │
                    │  • runs on Apple Silicon via MLX/Metal                │
                    │  • produces the golden corpus via gen_golden.py       │
                    │  • Pi[D,D], S[D,D], codebooks/*.json                  │
                    └─────────────────────┬─────────────────────────────────┘
                                          │ tools/gen_golden.py
                                          │ → cpp/tests/golden/*.bin
                                          ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │            libturboquant — C++17 algorithm core (cpp/src/)                │
   │            no OS deps · no JNI · no dynamic alloc in hot path             │
   │                                                                           │
   │   ┌──────────────────────────────────────────────────────────────┐        │
   │   │ public API:                                                  │        │
   │   │   TurboQuantMSE              (encode-only)                   │        │
   │   │   TurboQuantProd             (encode + QJL residual)         │        │
   │   │   TurboQuantKVCache          (prefill + attention_scores +   │        │
   │   │                               attend, with buffer flush)     │        │
   │   │   IBackend                   (dispatch interface, below)     │        │
   │   └──────────────────────────────────────────────────────────────┘        │
   │   ┌──────────────────────────────────────────────────────────────┐        │
   │   │ implementation:                                              │        │
   │   │   codebook.cpp     parses embedded JSON / blob               │        │
   │   │   rotation.cpp     QR + WHT init  (matmul → backend)         │        │
   │   │   packing.cpp      1/2/3/4/8-bit + QJL signs (reference)     │        │
   │   │   quantizer.cpp    end-to-end encode/dequant                 │        │
   │   │   kv_cache.cpp     KV cache + recent-token buffer flush      │        │
   │   └──────────────────────────────────────────────────────────────┘        │
   │                                                                           │
   │     all five hot ops dispatched to IBackend (one method each):            │
   │                                                                           │
   │       rotate         in[N,D]   ·  Pi[D,D]                  →  out[N,D]    │
   │       mse_encode     rot[N,D]  ·  boundaries[2^b-1]        →  packed[..]  │
   │       mse_score      q_rot[BH,D] · packed · norms · cb     →  scores[BH,N]│
   │       qjl_score      q_sketch[BH,D] · signs · res_norms    →  scores[BH,N]│
   │       value_dequant  packed · scales · zeros               →  out[N,D]    │
   └───────────────────────────────┬───────────────────────────────────────────┘
                                   │ IBackend* (virtual dispatch, one vtable
                                   │           lookup, called per kernel)
                                   ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │     Backend dispatch + factory (cpp/src/backend_factory.cpp)              │
   │                                                                           │
   │     create_backend(BackendKind) → unique_ptr<IBackend>                    │
   │                                                                           │
   │     gates: TQ_WITH_CPU_SCALAR  TQ_WITH_NEON  TQ_WITH_QNN                  │
   │            TQ_WITH_OPENCL      TQ_WITH_VULKAN                             │
   └─────┬──────────────┬───────────────┬───────────────┬───────────────┬──────┘
         │              │               │               │               │
         ▼              ▼               ▼               ▼               ▼
   ┌───────────┐  ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
   │cpu_scalar │  │ cpu_neon  │  │  qnn_htp   │  │  opencl    │  │  vulkan    │
   │           │  │           │  │            │  │            │  │            │
   │portable   │  │ARMv8.0 +  │  │QNN SDK +   │  │OpenCL 1.2  │  │Vulkan 1.1  │
   │C++17      │  │NEON FP32  │  │HTP backend │  │+ cl_khr_fp16│  │compute     │
   │           │  │INT8 FMA   │  │(FP16 graph)│  │            │  │+ FP16 SSBOs│
   └─────┬─────┘  └─────┬─────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
         │              │              │               │               │
         ▼              ▼              ▼               ▼               ▼
       (host)    Cortex-X4 +      Hexagon V75 HTP    Adreno 750      Adreno 750
       any ARM  / Cortex-A720    cDSP + HVX + HMX    (Khronos        (Khronos
       /x86 CPU  NEON intrinsics  via libQnnHtp.so   ICD; dlopen     loader; volk
                                                     libOpenCL.so)   on Vulkan)
                                                                                
   Galaxy S24 Ultra (SM-S928B / SM8650 — Snapdragon 8 Gen 3 for Galaxy)
   Same layout (different revs) on Snapdragon 8 Gen 2, Snapdragon 8 Elite,
   and on automotive: SA8155P / SA8295P / SA8775P.
```

## 2. Above the core — host-specific layers

Three host environments wrap `libturboquant`. Each one is independent of
the others; the core has zero awareness of which host it's running under.

```
   ┌───────────────────────────────────────────────────────┐
   │         Android demo app (P4 — android/)              │
   │                                                       │
   │   Kotlin Compose UI                                   │
   │            │                                          │
   │            ▼                                          │
   │   libtq_jni.so   (turboquant_jni.cpp — only           │
   │                   Android-specific C++ in repo)       │
   │            │                                          │
   │            ▼                                          │
   │   libturboquant.a  (built as part of the same         │
   │                     Gradle/CMake build)               │
   └───────────────────────────────────────────────────────┘

   ┌───────────────────────────────────────────────────────┐
   │   turboquant_bench  (CLI — bench/bench_cli.cpp)       │
   │   adb-pushable on Android, runs natively on QNX/Linux │
   │                                                       │
   │   Same baseline_kv_cache.{hpp,cpp} A/B harness        │
   │   used by both the CLI and the JNI bench path.        │
   │                                                       │
   │   Links: libturboquant.a + selected backends          │
   └───────────────────────────────────────────────────────┘

   ┌───────────────────────────────────────────────────────┐
   │   ctest / GoogleTest  (cpp/tests/)                    │
   │     packing_test         bit-pack roundtrip           │
   │     parity_test          golden corpus vs backend     │
   │     cross_backend_test   pairwise on enabled backends │
   │                                                       │
   │   Runs on host (cpu_scalar) in CI                     │
   │   and on-device (every backend) per phase.            │
   └───────────────────────────────────────────────────────┘
```

## 3. Data shapes and dtypes at the layer boundaries

This is the contract the C++ port enforces. Every shape is in
**row-major** (C-style); the leftmost dim is the slowest-varying.

### Top of the core (public API → implementation)

```
   prefill input:
       keys   :  float32  [B, H, S, D]    or  [BH, S, D]   (B*H folded)
       values :  float32  [B, H, S, D]
   attention_scores input:
       query  :  float32  [B, H, n_q, D]   or  [BH, n_q, D]
   attention_scores output:
       scores :  float32  [B, H, n_q, S]   (softmax NOT applied — caller does it)
   attend input:
       weights:  float32  [B, H, n_q, S]   (softmaxed)
   attend output:
       out    :  float32  [B, H, n_q, D]
```

For the rest of the doc we use the folded form `BH = B*H`. On Snapdragon
8 Gen 3 our default sweep is `BH=8, D=128, S∈{128…4096}`.

### Core → IBackend (per-kernel)

```
   rotate(in, Pi, n=BH or BH*S, D)         in: f32[n,D] · Pi: f32[D,D] →  f32[n,D]
   mse_encode(rotated, boundaries, N, D, b) → packed bytes  size = N · ⌈D·b/8⌉
        ※ for b=3 the "byte" is uint32 with 10 vals/word (3*10 = 30 bits)
   mse_score(q_rot, packed, norms, cb, BH, N, D, b) → f32[BH,N]
   qjl_score(q_sketch, signs, res_norms, mse_in, BH, N, D, qjl_scale) → f32[BH,N]
   value_dequant(packed, scales, zeros, N, D, b, group_size)         → f32[N,D]
```

Numerical contract:

- Integer outputs (packed bytes / packed u32) are **bit-exact** vs the
  Python reference. Tested via `parity_test`.
- Float outputs are within `<1e-4` for FP32 paths, `<1e-3` for FP16 paths
  (HTP / FP16 GPU). Tested via `parity_test` and pairwise via
  `cross_backend_test`.

### IBackend → device

| Backend | What "the device" actually is | Memory model |
|---|---|---|
| `cpu_scalar` | the host CPU | shared address space; pointers in/out |
| `cpu_neon` | Cortex-X4/A720 | shared; same as scalar |
| `qnn_htp` | Hexagon V75 HTP | distinct cDSP memory; `qnn->tensorCreate` allocates HTP-resident buffers; `host→cDSP` copy is implicit on bind |
| `opencl` | Adreno 750 | distinct GPU memory; `clCreateBuffer` + `clEnqueueWrite/Read` |
| `vulkan` | Adreno 750 | distinct GPU memory; SSBOs with explicit memory barriers |

For the HTP / GPU paths, **per-call host↔device copy is the dominant cost
at small sequence lengths**. The architecture mitigates this by:

1. Allocating device tensors once at backend `init()`.
2. Keeping intermediate results on the device between kernel launches
   (e.g. `Q_rot` from `rotate` stays on HTP and feeds `mse_score` directly).
3. Pinning frequently-used tables (centroids, boundaries) in `__constant`
   memory at init time.

## 4. Strict layering rules (from the plan)

Reproduced here so you don't have to dig into the plan to find them:

1. **Algorithm core is OS-free.** No `<jni.h>`, `<android/log.h>`,
   `__ANDROID__` ifdefs in `cpp/src/` or `cpp/include/`.
2. **No dynamic allocation in hot paths.** Allocate at backend `init()`.
3. **CMake-only.** No NDK glue inside the core. NDK and QCC are toolchain
   files.
4. **No filesystem at runtime.** Codebooks ship as embedded byte arrays;
   OpenCL/Vulkan kernel sources also embedded.
5. **Vendor SDKs are isolated.** `cpp/backends/<name>/` is the *only* place
   that includes QNN / Hexagon / OpenCL / Vulkan headers.
6. **Backend gating per platform.** Each backend's `CMakeLists.txt`
   declares supported platforms.
7. **Determinism.** Bit-exact for INT, `<1e-4` for FP32, `<1e-3` for FP16.

If you find yourself wanting to break one of these for convenience, stop —
the rules exist because breaking them silently kills the QNX automotive
build.

## 5. What this diagram doesn't show

- **Cross-backend test:** `tests/cross_backend_test.cpp` runs every enabled
  backend on the same input and asserts pairwise tolerance. On the S24 in
  P1+ this means scalar + NEON + QNN/HTP all produce equivalent outputs.
- **The baseline KV cache:** `bench/baseline_kv_cache.{hpp,cpp}` is a plain
  FP16 cache used **only** for A/B comparison in the bench. It lives under
  `bench/`, not `src/`, so production never carries reference code.
- **The Python golden generator:** `tools/gen_golden.py` runs on the host
  against `src/turboquant_mac/` and dumps fixed-seed binary corpus to
  `cpp/tests/golden/`. It only runs at dev time, never in CI.

For the per-kernel data flow inside `IBackend`, see
[`data-flow.md`](data-flow.md).
For how the KV cache life cycle plays out on top of these kernels, see
[`kv-cache-flow.md`](kv-cache-flow.md).
