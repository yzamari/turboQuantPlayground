# Path C — QNN SigLIP Port Design Document

> Investigation produced 2026-04-26 (night session). Surveys the existing
> QNN scaffold (`cpp/backends/qnn_htp/`) and the QAIRT export pipeline.
> Some Qualcomm-toolchain specifics are documented from publicly available
> SDK references — call-out flags any assumption that needs verification.

## Summary

SigLIP-base is a 96M-parameter ViT (12 transformer blocks, hidden_dim=768)
that bottlenecks Live VLM at 227 s per 480×640 frame on S24 Ultra CPU.
Porting it to Hexagon HTP via QNN should bring per-frame latency to
~50-100 ms. This is the **higher-payoff but higher-cost** Path C option:
estimated **5.5-8 weeks**, vs the parallel **mtmd-on-OpenCL design (~7 days)**.

The repo already has a QNN scaffold at `cpp/backends/qnn_htp/` accelerating
two TurboQuant kernels (`rotate()`, `value_dequant()`); SigLIP would be a
much larger graph but reuses the same loader, fallback chain, and CMake
gating. Hard constraints: macOS dev box cannot run the QAIRT converter
(Linux-only binary) — Docker or a Linux CI runner is mandatory.

## Current QNN scaffold survey

`cpp/backends/qnn_htp/`:

- **`qnn_htp_backend.cpp` (~194 lines)** — implements `IBackend` for
  `rotate()` + `value_dequant()` (8-bit only). Composes a CPU fallback via
  `cpu_fallback_` for everything else. Caches finalized graphs by shape
  (rotate_cache_, dequant_cache_) to amortize ~10-50 ms `graphFinalize`.
- **`qnn_graph.{hpp,cpp}`** — `RotateGraph` (single MatMul, Pi^T baked in
  as static weight) + `ValueDequantGraph` (Cast→Gather→Mul→Add). Default
  FP16 with `GraphOptions.use_fp32_fallback`. `MseScoreGraph` is a stub.
- **`qnn_loader.{hpp,cpp}`** — dlopens `libQnnHtp.so` / `libQnnSystem.so`.
  Probe order: `$LD_LIBRARY_PATH`, `/data/local/tmp/qnn_libs/` (dev push),
  `/vendor/lib64/` (Samsung "snap" runtimes). Returns `false` cleanly if
  QNN unavailable — caller falls back silently.
- **`README.md` (131 lines)** — SDK acquisition, on-device pushes, build
  flags, vendor runtime paths, numerical tolerance, CMake wiring.

Backend factory: `cpp/src/backend_factory.cpp:78-91` tries QnnHtp first,
then OpenCL, Vulkan, CpuNeon, CpuScalar. Compile-time dispatch gated by
`TQ_WITH_QNN`.

`IBackend` interface (`cpp/include/turboquant/backend.hpp:28-85`):

```cpp
virtual void rotate(const float* in, const float* Pi, int n, int D, float* out) = 0;
virtual void value_dequant(const uint8_t* packed, const float* scales, const float* zeros,
                           int N, int D, int bits, int group_size, float* out) = 0;
virtual void mse_encode(...) = 0;
virtual void mse_score(...) = 0;
virtual void qjl_score(...) = 0;
```

For SigLIP we add either a coarse single-call `siglip_forward()` or a
finer `siglip_block()` API — see proposed signatures in §JNI integration.

## SigLIP-base ops × HTP support matrix

| Op | Shape (typical) | HTP V73 (Tab S9+) | HTP V75 (S24 Ultra) | Notes |
|---|---|---|---|---|
| Patch Embed conv2d 16×16 | [1,3,224,224] → [1,768,14,14] | ✓ | ✓ | Standard QNN Conv2d, FP16 |
| LayerNorm | [seq, 768] | ✓ | ✓ | ElementWiseNorm or custom UDO |
| Q/K/V/out Linear | [seq, 768] → [seq, 768] | ✓ | ✓ | MatMul, HMX accelerates FP16 |
| Scaled-dot-product attn | Q/K/V [seq, h, dh] | ⚠ partial | ⚠ partial | Compose from MatMul + Reduce + Softmax. Mask broadcast may need custom logic |
| Softmax | [seq, seq] | ✓ | ✓ | Reduce + ElementWiseExp + Reduce, or custom UDO if perf-critical |
| GELU | [seq, mlp_dim] | ⚠ no native op | ⚠ no native op | Options: CPU fallback, polynomial approx (5-7 ElementWise), or custom HVX UDO |
| Residual add | [seq, 768] × 2 | ✓ | ✓ | ElementWiseAdd, ~2-5 cycles HVX |
| MLP up/down (Linear) | 768→3072→768 | ✓ | ✓ | MatMul |
| Final LayerNorm | [seq, 768] | ✓ | ✓ | Same as block-level |
| Global avg pool | [seq, 768] → [768] | ✓ | ✓ | Reduce |
| mmproj Linear | [768] → [llm_text_dim] | ✓ | ✓ | MatMul |

**Coverage gaps to plan around:**
- **GELU** — not native in core QNN ops on either V73 or V75. v1: keep MLP
  activation on CPU (NEON GELU, ~1-2 ms × 12 blocks = ~12-24 ms). Custom
  HVX UDO is a v2 optimisation.
- **Attention mask broadcast** — assumption: SigLIP is vision (no causal
  mask) and padding masks are static for fixed image size. Bake mask into
  graph constants; no dynamic-mask path needed for v1. **Verify in code
  review.**
- **V73 vs V75 op parity** — assumed identical for ops in this table; HMX
  gen 2 (V73) vs gen 3 (V75) differ in throughput, not op set. Validate
  on Tab S9+ before committing to "works on both."

## Export pipeline (HF → QNN .bin)

| Step | Command | Host |
|---|---|---|
| 1. Download | `git clone hf.co/openai/SigLIP-base-patch16-224` | Any |
| 2. PyTorch → ONNX | `torch.onnx.export(...)` (opset 13, no constant fold control) | **Linux** |
| 3. ONNX → QNN | `$QNN_SDK_ROOT/bin/qnn-onnx-converter --target_runtime htp` | **Linux only** (QAIRT bin is x86_64 Linux) |
| 4. (opt) INT8 quantize | calibrate w/ representative imgs; rerun converter | **Linux** |
| 5. Push to device | `adb push qnn_model/*.bin /data/local/tmp/` | macOS or Linux |

**macOS host limitation: QAIRT 2.27.x ships Linux-only.** Three viable
mitigations:

- **A. Docker on macOS** — wrap conversion in a container with QAIRT
  pre-installed. Likely need to build the image from a Qualcomm-provided
  base.
- **B. GitHub Actions Linux runner** — auto-convert on push of new SigLIP
  weights, upload `.bin` as artifact.
- **C. Linux pair box** — sync via Git/S3/rsync.

Recommendation: **B for CI + A for local iteration**. Document in
`docs/BUILDING.md` (currently does not cover the conversion step).

## Phased implementation plan

### Phase 1 — Scaffold + single-block parity (7-10 days)

- Test harness: extract one transformer block, save weights to `.safetensors`,
  write a Python FP32 reference.
- Wire QNN ops in `cpp/backends/qnn_htp/qnn_siglip.cpp` (NEW): LayerNorm,
  attention triple, MLP with CPU-fallback GELU.
- Shape-specialized graph cache by (hidden_dim, seq_len, num_heads).
- Acceptance: cosine ≥ 0.99 vs FP32 ref, abs error < 1e-3.

### Phase 2 — Full 12-block ViT + mmproj parity (10-14 days)

- Chain 12 blocks (single composed graph or 12 sub-graphs with pinned
  residuals).
- Add patch-embed Conv2d 16×16 head and final LayerNorm + global pool.
- Run real SigLIP-base weights through ONNX → QNN converter (Linux box /
  Docker).
- Latency profile on S24 Ultra (V75); FP16 tolerance tuning if any block
  drifts > 1e-3.
- Acceptance: end-to-end cosine ≥ 0.99 vs FP32 CPU baseline on 10 random
  test images; latency snapshot.

### Phase 3 — Android JNI integration (5-7 days)

- New class `QnnSigLipBackend` (init / load_mmproj / encode_image).
- `mtmd_jni.cpp:248-269` — conditional dispatch: QNN path or MTMD CPU
  fallback.
- `mtmd_jni.cpp:359-376` — replace `mtmd_helper_eval_chunks` call when
  QNN path is active.
- Wire Kotlin flag `useQnnVision` through `MtmdNative.kt`.
- End-to-end: image → QNN SigLIP → mmproj projection → llama_decode.

### Phase 4 — Device validation (5-7 days)

- S24 Ultra (V75): confirm 50-100 ms/frame, profile op-by-op.
- Tab S9+ (V73): identify any V73-specific op gaps; CPU-fallback the
  problem block if any. Tab S9+ not currently connected — phase blocked
  on device availability.
- A/B vs CPU baseline on power, thermals, perceived FPS.
- Update `cpp/backends/qnn_htp/README.md` with device matrix.

**Total: 27-38 working days + 10-15% contingency = ~5.5-8 weeks. Phases 1-2
can partially parallelise (graph-wiring vs export pipeline).**

## Android JNI integration point

Today (`android/app/src/main/cpp/mtmd_jni.cpp`):

- Lines **248-269**: `mtmd_init_from_file(mmprojPath, ...)` to set up the
  vision path with `vparams.use_gpu = false` (CPU SigLIP).
- Lines **359-376**: `mtmd_helper_eval_chunks(...)` runs the vision graph
  per frame.

After the port — conditional dispatch:

```cpp
struct VlmSession {
    // ... existing ...
    enum class VisionType { MTMD, QNN } vision_type;
    float image_embedding[/* model.text_dim */];
};

// In loadModel():
std::call_once(g_siglip_init, []() {
    g_qnn_siglip_backend = std::make_unique<QnnSigLipBackend>();
    if (!g_qnn_siglip_backend->init()) g_qnn_siglip_backend.reset();
});

if (g_qnn_siglip_backend && g_qnn_siglip_backend->load_mmproj(mmprojPath, sess->model)) {
    sess->vision_type = VisionType::QNN;
} else {
    sess->vision_type = VisionType::MTMD;
    sess->vision = mtmd_init_from_file(mmprojPath, sess->model, vparams);
}

// In describe():
if (sess->vision_type == VisionType::QNN) {
    rc = g_qnn_siglip_backend->encode_image(image, sess->image_embedding);
    // inject embedding into chunks for llama_decode
} else {
    rc = mtmd_helper_eval_chunks(...);
}
```

New files: `qnn_siglip_backend.{hpp,cpp}`, `cpp/backends/qnn_htp/qnn_siglip.{hpp,cpp}`.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| HTP V73 op gaps on Tab S9+ | High | Test V73 *early* (Phase 4 first day). Per-block CPU fallback if needed. |
| macOS host can't run converter | Medium | Mandate Docker (A) or GH Actions (B) at setup. Block branch merge until BUILDING.md has the recipe. |
| GELU not native | Medium | v1 CPU fallback (~12-24 ms total). Custom HVX UDO is v2 if needed. |
| mmproj graph scope unclear | Medium | Verify in Phase 1: is mmproj a separate `.gguf` linear, or fused into vision graph? Affects whether to add a 13th node or a separate graph. |
| FP16 numerical drift | Low | 0.99 cosine target with 1e-3 abs tolerance per HTP FP16 docs. Per-block FP32 fallback if any drift > tolerance. |
| Tab S9+ device unavailable | Medium | Phase 4 gated on device. Ship S24 Ultra-only first if needed. |

## Effort summary

| Phase | Effort |
|---|---|
| 1 — single-block parity | 7-10 d |
| 2 — full ViT + mmproj | 10-14 d |
| 3 — JNI integration | 5-7 d |
| 4 — device validation | 5-7 d |
| Conversion pipeline + docs | 2-3 d |
| Contingency 10-15% | — |
| **Total** | **~5.5-8 weeks** |

## Open questions for user

1. **Priority vs Path 2.1c (TurboQuant KV cache)?** — both are multi-week.
2. **mmproj fused or separate?** — verify in our SmolVLM `.gguf` files.
3. **Linux infrastructure** — Docker, GH Actions, or pair box for QAIRT?
4. **V73 abort tolerance** — if Tab S9+ lacks ops, degrade to CPU on that
   device only, or block the feature on full V73 support?
5. **Vision encoder lock-in** — SigLIP only, or design for DINOv2/CLIP swap?
6. **INT8 quantization scope?** — Phase 2 (needs calibration data) or v2.

## Comparison with `path-c-mtmd-opencl-design.md`

| Dimension | mtmd-on-OpenCL | QNN SigLIP port |
|---|---|---|
| Effort | ~1 week | ~5.5-8 weeks |
| Per-frame latency target | < 15 s | 50-100 ms |
| Speedup vs CPU | ~15× | ~2000× |
| Linux dev box required | No | Yes (or Docker) |
| Touches our QNN scaffold | No | Yes (extends it) |
| Upstream-friendly | Yes (could PR) | No (Qualcomm-specific) |
| Tab S9+ status | Adreno 740 deadlock — needs SoC bypass | V73 op-gap risk — same fallback story |
| Risk profile | Low (already half-wired) | High (custom UDO + toolchain dep) |

**Recommendation:** Land mtmd-on-OpenCL first as a 1-week win. Decide on
QNN SigLIP after seeing real OpenCL numbers — if the Adreno path lands
inside ~30 ms anyway, the 2000× HTP win matters less than the engineering
cost. If OpenCL lands at multi-second, QNN port becomes worth the 6-week
investment.
