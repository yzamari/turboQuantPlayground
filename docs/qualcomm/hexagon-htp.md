# Hexagon HTP — Deep Dive

This is the most important Qualcomm doc in the repo. The whole reason we are
porting TurboQuant to C++ instead of leaving it on Apple Silicon is that
**Snapdragon's distinctive accelerator is the Hexagon NPU**. Adreno is good,
but every modern phone has a "good enough" GPU. The HTP is what makes a
Qualcomm SoC a Qualcomm SoC.

## 1. What "Hexagon HTP" actually means

The terminology has drifted. As of 2025:

- **Hexagon DSP** — the umbrella name for Qualcomm's VLIW DSP family. Same
  ISA family for ~15 years. Lives on every Snapdragon, in multiple instances:
  one **cDSP** (compute), one **aDSP** (audio), one **mDSP / sDSP** (sensor),
  sometimes one **modem DSP**. We only ever talk to the cDSP.
- **HVX** — *Hexagon Vector eXtensions*. The 1024-bit SIMD ISA bolted onto
  Hexagon. Two HVX contexts per cDSP on modern parts.
- **HMX** — *Hexagon Tensor Core* (sometimes called "Hexagon Matrix
  eXtensions"). A dedicated INT8/INT16 matrix-multiply unit. Introduced on
  V68 (Snapdragon 8 Gen 1), considerably enlarged on V73 / V75.
- **HTP** — *Hexagon Tensor Processor*. **The marketing/runtime name for
  the cDSP-plus-HMX combination acting as an NPU.** When the QNN SDK says
  "HTP backend", it means "we'll JIT a graph onto cDSP+HVX+HMX".

In practice, when we say "the NPU" or "HTP" in the rest of these docs we mean
exactly one thing: **the QNN-managed view of the cDSP+HVX+HMX**, on the
hardware revision of whichever Snapdragon we're on:

| Snapdragon | Hexagon rev | HMX | HVX width | First seen |
|---|---|---|---|---|
| 8 Gen 2 (SM8550) | **V73** | Yes (gen 2) | 1024-bit | 2022 |
| 8 Gen 3 (SM8650) ← us | **V75** | Yes (gen 3) | 1024-bit | 2023 |
| 8 Elite (SM8750) | V79 | Yes (HMX2) | 1024-bit | 2024 |
| Auto SA8155P | V66 (no HMX) | 1024-bit | 2019 |
| Auto SA8295P | V69 | Yes | 1024-bit | 2022 |
| Auto SA8775P | V73 | Yes (gen 2) | 1024-bit | 2024 |

The S24 Ultra in our drawer reports V75. The same QNN graph we build for V75
will run on V73 (auto SA8775P, mobile 8 Gen 2) at lower throughput and
probably without HMX-only ops.

## 2. HVX — the vector model

Every byte of "fast" code on Hexagon eventually goes through HVX. The model:

```
   ┌─────────────────────────────────────────────────┐
   │   HVX register file:  32 × 1024-bit registers   │
   │   (one register holds, e.g.,                    │
   │      128 × INT8                                 │
   │       64 × INT16                                │
   │       32 × INT32 / FP32-as-fixed                │
   │       64 × FP16   (V73+)                        │
   │   )                                             │
   └─────────────────────────────────────────────────┘
            │             │              │
            ▼             ▼              ▼
        Lane ops      Reduce ops     Pack/Unpack
       (vadd/vmpy)   (vrmpyacc...)  (vshuff/vlut)
```

Important properties for a KV-cache compression workload:

- **No native FP32.** HVX's "float" support is FP16 from V73 onward; FP32 is
  emulated and slow. **TurboQuant's hot path is naturally FP16-friendly**
  (the rotation matmul, the MSE score accumulation, the QJL dot product all
  tolerate FP16 with `<1e-3` tolerance), so this is not a problem — but it's
  why our QNN tolerance bound is `<1e-3`, not `<1e-4`.
- **INT8 / INT16 are first-class.** Quantized centroid lookups and packed-byte
  unpacks are exactly what HVX is built for.
- **HMX = matrix unit on top of HVX.** For the rotation matmul (`Q @ Pi^T`)
  and the dequant-then-matmul of `mse_score`, HMX is the entire reason this
  port is fast. Without HMX you fall back to HVX outer-product loops — still
  fast, but ~3-5× slower at FP16 GEMM.
- **No instruction-pointer divergence.** Hexagon is VLIW; the compiler emits
  packets of ≤4 instructions. Branchy code is a perf disaster. **This is why
  we keep `mse_encode` (which has a `searchsorted` loop with data-dependent
  count) on NEON — it's branchy by nature.**

## 3. The QNN SDK — what it is and what we use

**QNN (Qualcomm Neural Network) SDK**, formerly *Qualcomm AI Engine Direct*,
is the C/C++ runtime that gives us programmatic access to Hexagon HTP from
user space. It is:

- A **C ABI** (`QnnInterface_t` is the entry point — a vtable of function
  pointers). We dlopen `libQnnHtp.so` and `libQnnSystem.so` and resolve
  symbols.
- A **graph compiler.** You describe a static DAG of ops (`MatMul`, `Gather`,
  `ElementWiseAdd`, ...) and QNN compiles it ahead-of-time for the target
  HTP rev. Compilation is slow (seconds); execution is fast.
- A **set of backend `.so`s.** `libQnnHtp.so` is the HTP backend. There are
  also `libQnnGpu.so` (Adreno via OpenCL), `libQnnCpu.so` (NEON reference),
  `libQnnDsp.so` (older DSP without HMX). We exclusively target HTP.

### What we use from it

The minimum API surface for our use:

```
QnnInterface_t            // function-pointer vtable; dlsym this from libQnnSystem.so
qnn->backendCreate()      // make a backend handle for HTP
qnn->contextCreate()      // a memory + device context inside that backend
qnn->graphCreate()        // declare a new graph
qnn->graphAddNode()       // add ops (MatMul, Gather, ElementWiseAdd, ...)
qnn->graphFinalize()      // ahead-of-time compile to HTP
qnn->graphExecute()       // run with bound input/output tensors
qnn->tensorCreate()       // allocate device-visible tensors
```

All of this is wrapped in `cpp/backends/qnn_htp/qnn_loader.cpp` so the rest of
the C++ port only sees `IBackend::rotate(...)`, etc.

### A QNN graph is **not** a TF/ONNX graph

This is the single most important conceptual point. Coming from PyTorch /
ONNX you expect:

- Dynamic shapes. *Not in QNN.* All tensor dimensions must be known at
  `graphFinalize` time. Resizing means re-finalizing — measured at ~10–50 ms,
  so we cannot do it per token.
- Eager execution. *Not in QNN.* You build the graph once, finalize once,
  then execute many times.
- Op coverage parity with ONNX. *Not in QNN.* The HTP op set is a strict
  subset of ONNX. A few critical ops (e.g. `searchsorted`, custom bit
  packing) **are not available** — this is exactly why our pipeline is
  hybrid HTP + NEON.

What this means concretely for us:

```
   ┌────────────────────────────────────────────────────────┐
   │  Init time (once per (BH, N, D, bits) configuration):  │
   │                                                        │
   │   build graph: rotate (MatMul[D,D])                    │
   │   build graph: mse_score (Gather + MatMul)             │
   │   build graph: value_dequant (Gather + Mul + Add)      │
   │   finalize all three graphs                            │
   └────────────────────────────────────────────────────────┘
   ┌────────────────────────────────────────────────────────┐
   │  Per decode step (hot path):                           │
   │                                                        │
   │   bind Q, run rotate-graph,        get Q_rot           │
   │   bind mse_packed + Q_rot, run mse_score-graph         │
   │   on NEON: bit-unpack signs and run qjl_score          │
   │   add scores; softmax; ...                             │
   └────────────────────────────────────────────────────────┘
```

When `seq_len` (one of the graph's tensor dims) changes, we re-finalize. We
mitigate this by either (a) bucketing — finalize a graph for N=128, 256, 512,
1024, 2048, 4096 once at startup — or (b) by always padding up to the next
power of two. The bucketing strategy is the one the v1 plan picks because
it's lower memory at the cost of slower first-use after a power of two
boundary.

## 4. Custom HVX ops via UDO

QNN's stock op library doesn't include `searchsorted`, our 3-bit u32 packing,
or the QJL bit-unpack-then-fma. For maximum performance you can write a
**UDO** (User Defined Op):

- Write the kernel in C with `<hexagon_types.h>` and HVX intrinsics
  (`Q6_V_vlut32_VbVbVb_si`, `Q6_Vw_vmpyi_VwVw`, `Q6_V_vlsr_VuhR`...). It compiles
  with `hexagon-clang` from the Hexagon SDK.
- Wrap it as a UDO descriptor JSON. `qnn-op-package-generator` emits a
  `libQnnHtp_<MyOp>.so`.
- Register it at runtime via `qnn->backendRegisterOpPackage(...)`.

This gives you **bare-metal HVX performance** inside the QNN graph — the
op is just another node, no dispatch overhead between nodes, full graph-level
scheduling.

We've **deferred UDO to a follow-up project.** The v1 hybrid HTP+NEON path
gets us most of the way at much lower implementation cost. UDO becomes
attractive when:

- The NEON path is the bottleneck end-to-end (looking at `bench` numbers,
  not before).
- We have validated the algorithm on stock QNN ops first, so the UDO is just
  a perf optimization, not a correctness change.

## 5. Why hybrid HTP + NEON is the right v1 split

This is the kernel-by-kernel decision documented in the plan:

| Op | Where | Reason |
|---|---|---|
| `rotate` | **HTP** (QNN MatMul) | World-class FP16 GEMM via HMX. Biggest single op cost in the pipeline. |
| `mse_encode` (search + bit-pack) | **NEON** | searchsorted has a data-dependent loop count; bit-pack is byte-level — both are HTP-hostile. Tiny data, NEON is fast enough. |
| `mse_score` | **HTP** (Gather + MatMul) | Centroid table fits in HMX scratch; the inner loop is a pure GEMM after Gather. HMX wins by 3–5×. |
| `qjl_score` | **NEON** | Bit-unpack to ±1 then dot. Small data; NEON's `vbslq_f32` materializes signs in 1 cycle, faster than burning a re-finalize cost on a small graph. |
| `value_dequant` | **HTP** (Gather + Mul + Add) | Group-wise scale + zero is a textbook quantize-aware pattern. QNN supports it natively. |

Heuristic that produced this split:

- **Big regular GEMM with FP16-tolerant precision → HTP.** Always.
- **Bit fiddling with branches → NEON.** Always.
- **Small data with low arithmetic intensity → NEON** (HTP launch overhead
  swamps the kernel time).

If a future op breaks the heuristic — say, a 4 K context `qjl_score` becomes
GEMM-bound — we move it to HTP. The split is data-driven, not a religion.

## 6. Setting up the SDK

```
# Once, on dev machine:
1. Sign in at https://www.qualcomm.com/developer
2. Download "Qualcomm AI Engine Direct SDK" (current stable; we pin 2.27.x)
3. Accept license; unzip somewhere stable
4. export QNN_SDK_ROOT=/opt/qcom/qnn-2.27.0
5. CMake auto-detects via -DQNN_SDK_ROOT=$QNN_SDK_ROOT
```

On-device:

```
# Push the QNN runtime libs once per device:
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so       /data/local/tmp/qnn/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so    /data/local/tmp/qnn/
adb push $QNN_SDK_ROOT/lib/hexagon-v75/libQnnHtpV75Skel.so    /data/local/tmp/qnn/

# Then, before running our bench:
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=/data/local/tmp/qnn && \
           /data/local/tmp/turboquant_bench --check --backend qnn_htp'
```

`ADSP_LIBRARY_PATH` is the variable the FastRPC client uses to find the
Hexagon-side skel library. Forgetting it produces a baffling `RPC error
0x80000406` at first call.

## 7. Caveats and gotchas

1. **First QNN execute is slow** — typically 100–500 ms on cold start,
   because the graph blob has to be loaded onto the cDSP and HMX state has
   to be initialized. Always warmup at least 3 iterations before timing.
2. **Skel library version must match Hexagon revision exactly.**
   `libQnnHtpV75Skel.so` runs on V75 only. If you push the V73 skel to a V75
   device, you get cryptic `QnnError: 1004` at graph execute — the runtime
   doesn't validate the skel proactively.
3. **Static shapes are non-negotiable.** Don't write code that re-finalizes
   the graph per call. Bucket your sequence lengths.
4. **`libsnap_qnn.so` on Galaxy ≠ standard QNN.** The Galaxy "snap" variant
   is Samsung's wrapper. We target the **standard** QNN runtime libraries
   from the SDK and let them load the device-side HTP backend stub. Don't
   try to dlopen Samsung's variant — its ABI drifts between Galaxy firmware
   updates.
5. **The HTP frequency policy is governor-driven.** On a thermally throttled
   device the same graph can run 2× faster after a cold start than after 30
   seconds of sustained load. Bench in a thermally stable state and report
   the steady-state number, not the burst peak.

## 8. Further reading

- Qualcomm AI Engine Direct (QNN) SDK landing page:
  https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- Hexagon SDK landing page (Hexagon-clang, hexagon-llvm-link, sim):
  https://www.qualcomm.com/developer/software/hexagon-dsp-sdk
- Hexagon V75 architecture brief (HMX-gen-3 specifics):
  https://www.qualcomm.com/products/snapdragon-8-gen-3-mobile-platform
- For UDO authoring, the canonical reference is the SDK's own
  `examples/QNN/OpPackage/` directory — public docs are sparse, the SDK
  examples are the source of truth.
