# Backend Comparison

The `IBackend` interface is implemented by five backends. They are
**alternative implementations** of the same algorithm, not a stack — the
factory picks one at runtime based on a name string and the build-time
flags. This page is a side-by-side comparison so you can pick the right
one for a given deployment.

For the per-kernel split inside the QNN/HTP backend (HTP for some ops,
NEON for others), see [`../qualcomm/hexagon-htp.md`](../qualcomm/hexagon-htp.md).

## 1. The five backends at a glance

| Backend | Where it runs | Peak (FP16, S24-class) | Per-launch overhead | Automotive support | Status in this repo |
|---|---|---|---|---|---|
| `cpu_scalar` | host CPU, any ISA | ~0.05 TFLOPS | ~0 µs | ✓ all (any C++17 toolchain) | **Done (P0)** — reference |
| `cpu_neon` | Cortex-X4/A720, any ARMv8.0+NEON | ~0.3 TFLOPS | ~0 µs | ✓ all (NDK / QCC / aarch64-gcc) | **In progress (P1.a)** |
| `qnn_htp` | Hexagon V75 HTP (cDSP+HVX+HMX) | ~6 TFLOPS (FP16, HMX) | 50–200 µs (graphExecute) | ✓ Cockpit Auto (V69+) | **In progress (P1.b)** — hybrid HTP+NEON |
| `opencl` | Adreno 750 | ~3 TFLOPS (FP16, sustained ~half) | 50–100 µs (NDRange) | ✓ Auto Adreno + QNX OCL | **Started (P2)** — kernels translated, integration WIP |
| `vulkan` | Adreno 750 (compute pipeline) | ~3 TFLOPS (FP16) | 50–100 µs (vkCmdDispatch) | ⚠ Linux Auto: yes; QNX: partial | **Started (P3)** — secondary GPU path |

Numbers are order-of-magnitude estimates from Qualcomm public material; do
not use as benchmarks. The CSVs under `cpp/bench/results/` are the source
of truth for real numbers.

## 2. Side-by-side: what each backend actually does

Each backend implements the same five `IBackend` methods. The
implementations differ in how each method gets to the metal.

### `cpu_scalar` — the reference

Pure C++17. Triple-nested loops. No intrinsics. The whole point is to be
the **golden truth on the host**; if any other backend disagrees with
`cpu_scalar` (within tolerance), the other backend is wrong by definition.

```
   rotate         → straight 3-loop SGEMM
   mse_encode     → searchsorted (linear scan) + bit-pack
   mse_score      → 2-loop dot of q_rot and centroid table
   qjl_score      → 2-loop dot, ±1 from packed sign byte
   value_dequant  → 2-loop unpack + fma
```

When to ship: **never to a phone**, but always to host CI. Also useful as
a sanity check when bringing up a new backend — disable everything else,
run cpu_scalar, get a baseline number, then enable the new backend and
confirm it agrees.

### `cpu_neon` — fast CPU path

ARMv8.0-A baseline + NEON intrinsics. Used as both the always-available
fallback and the v1 host for `mse_encode` / `qjl_score` (the bit-fiddling
ops the QNN backend forwards to NEON).

```
   rotate         → 4-lane SGEMV (decode) / SGEMM (prefill) with vfmaq_f32
   mse_encode     → vector compare against boundaries, bit-pack in scalar tail
   mse_score      → vector accumulator, scalar centroid lookup
   qjl_score      → vbslq_f32 to materialize ±1.0 from sign bits, vfmaq_f32 acc
   value_dequant  → vector unpack + fma per group
```

When it wins: small `N` (<256) where launch overhead of HTP/GPU dominates,
and on every SoC that doesn't have HTP at all (older Snapdragons,
SA8155P).

### `qnn_htp` — the strategic Qualcomm target

Hybrid: heavy ops on HTP via QNN graph, bit ops forwarded to a
co-resident `cpu_neon` instance. See
[`../qualcomm/hexagon-htp.md`](../qualcomm/hexagon-htp.md) §5 for the
per-op split.

```
   rotate         → HTP   (QNN MatMul, FP16 via HMX)
   mse_encode     → NEON  (forwarded to cpu_neon)
   mse_score      → HTP   (QNN Gather + MatMul, HMX)
   qjl_score      → NEON  (forwarded to cpu_neon)
   value_dequant  → HTP   (QNN Gather + Mul + Add)
```

When it wins: medium-to-long sequences (`N ≥ 512`) where HMX's matrix
throughput on `rotate` and `mse_score` swamps everything else. This is
the backend we ship to production on Snapdragon-class hardware.

### `opencl` — primary GPU

OpenCL 1.2 + `cl_khr_fp16`. One kernel per IBackend op; closely tracks
the existing Apple Metal kernels.

```
   rotate         → __kernel rotate_qkt   (NOT named "rotate" — see
                                            qualcomm/adreno-gpu.md §4.4)
   mse_encode     → fused searchsorted + bit-pack, one byte/word per thread
   mse_score      → one (BH, N) thread, inner loop over packed bytes
   qjl_score      → one (BH, N) thread, inner loop over sign bytes
   value_dequant  → one (N, coord) thread; group lookup of scale/zero
```

When it wins: SoCs without a working HTP path (older mobile, SA8155P
auto). Also wins when the GPU is otherwise idle and the HTP is busy
(unlikely for our workload but possible if running concurrent with
other AI features).

### `vulkan` — secondary GPU

Vulkan 1.1 compute pipelines, GLSL `comp` shaders compiled to SPIR-V at
build time and embedded as `const unsigned char[]`. Same kernel logic as
OpenCL but expressed with descriptor sets and push constants.

When it wins: OpenCL ICD bugs (the Adreno blob has had several over the
years), or automotive Linux deployments where Vulkan is healthier than
OpenCL.

## 3. Compile-time gating

Every backend can be turned off independently:

```
TQ_WITH_CPU_SCALAR   default ON     never disabled (it's the reference)
TQ_WITH_NEON         default OFF    on for arm64 builds in CI
TQ_WITH_QNN          default OFF    on for the on-device bench
TQ_WITH_OPENCL       default OFF    on for the on-device bench
TQ_WITH_VULKAN       default OFF    on for the on-device bench
```

This matters for **automotive** deployments: a safety-critical integrator
may certify only `cpu_scalar + cpu_neon + qnn_htp` and ship a build with
`-DTQ_WITH_OPENCL=OFF -DTQ_WITH_VULKAN=OFF`. They get a smaller binary,
no GPU dependencies, and no GPU code in their certified image.

## 4. When each backend wins — refer to bench results

For ranges below, see `cpp/bench/results/<device>-<backend>.csv` for the
authoritative numbers. The qualitative picture, expected from the plan's
P1 success criteria:

```
   seq_len   memory winner       latency winner
   ───────   ────────────────    ───────────────────────────
     128     all (compression    cpu_neon (HTP launch overhead
              still ≥3×)          dominates short workloads)
     512     all                 mixed: cpu_neon ≈ qnn_htp
                                  (transition point)
    1024     all                 qnn_htp     (HMX starts to pay off)
    2048     all                 qnn_htp     (clear win)
    4096+    all (≥4×)           qnn_htp     (commanding win, ~5-10×
                                  speedup vs FP16 baseline expected)
```

The plan's hard pass criteria for any GPU/NPU backend:

- `compression_ratio ≥ 3.0×` at `seq_len=1024, bits=3`
- `attn_score_cosine_sim ≥ 0.99`
- `attn_speedup ≥ 1.0×` at `seq_len ≥ 2048`

`cpu_neon` is allowed to be ≥ 0.7× speedup-ratio and still pass — its
win is memory, not compute.

## 5. Decision tree for picking a backend

```
   is the SoC Qualcomm with HTP (8 Gen 2 / 3 / Elite, SA8295P, SA8775P)?
   ┌─── yes ──► is the workload N ≥ 512 typically?
   │             ┌─── yes ──► qnn_htp                ← ship this
   │             └─── no  ──► cpu_neon               ← short ctx
   │
   └─── no  ──► is there a working OpenCL ICD?
                 ┌─── yes ──► opencl                 ← Adreno-only SoC
                 └─── no  ──► cpu_neon               ← portable fallback
```

Vulkan is the "OpenCL had a driver bug, try Vulkan" branch — same
decision otherwise.

## 6. What's NOT a backend

Two things in the codebase look like backends but aren't:

- **`bench/baseline_kv_cache.{hpp,cpp}`** — a plain FP16 KV cache used
  ONLY by the bench harness for the A/B comparison. It is not a
  TurboQuant backend; it's the *thing being compared against*.
- **`tools/gen_golden.py`** — runs the Python reference on the dev host
  to dump fixed-seed binary corpus. Pure offline tool; doesn't ship.

## 7. Where to look for each backend

```
cpp/backends/cpu_scalar/scalar_backend.cpp        ── P0 reference
cpp/backends/cpu_neon/neon_backend.cpp            ── P1.a
cpp/backends/qnn_htp/{qnn_loader,qnn_graph,htp_backend}.cpp
                                                  ── P1.b
cpp/backends/opencl/{opencl_backend.cpp, kernels/*.cl}
                                                  ── P2
cpp/backends/vulkan/{vulkan_backend.cpp, shaders/*.comp}
                                                  ── P3
```

For benchmark numbers across all five, see
[`../benchmarks/README.md`](../benchmarks/README.md).
