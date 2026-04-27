# Path C — mtmd-on-OpenCL Design Document

> Investigation produced 2026-04-26 (night session). Based on static analysis
> of the pinned llama.cpp fork (`external/llama.cpp/`, commit `f454bd7e` =
> upstream tag `b8935`). Runtime deadlock diagnosis on Adreno 740 will require
> on-device profiling beyond what was inspected.

## Summary

The SigLIP vision tower in `mtmd` currently runs entirely on CPU even though
all of its operations have OpenCL kernels. The bug is **not** missing kernels —
it's routing. `clip_ctx` already initialises a GPU backend (via env var
`MTMD_BACKEND_DEVICE` or auto-detect of `GGML_BACKEND_DEVICE_TYPE_GPU`) and
hands both GPU and CPU backends to the scheduler. The vision graph is
constructed without backend affinity hints, and the scheduler defaults to
CPU placement. Three phases: (1) minimal patch + scheduler reordering /
explicit GPU hints (~2 days); (2) S24 Ultra Adreno 750 cosine + latency
validation (~3 days); (3) Adreno 740 deadlock diagnosis + SoC-aware bypass
(~2 days).

**Total: ~7 days, not multi-week.** That is materially smaller than the
parent handoff estimated.

## Code-level findings

### Vision graph build & dispatch

- `mtmd_helper_eval_chunks` → `mtmd_encode_chunk` (mtmd-helper.cpp:366) →
  `mtmd_encode` (mtmd.cpp:1036) → `clip_image_batch_encode`
  (clip.cpp:3125).
- Inside `clip_image_batch_encode`:
  - line 3139: scheduler reset.
  - line 3141: `clip_image_build_graph(ctx, imgs)` builds the ViT graph.
  - line 3140-3141: `ggml_backend_sched_alloc_graph` allocates / schedules.
  - line 3718: `ggml_backend_sched_graph_compute` runs it.

### Backend selection in `clip_ctx`

`clip.cpp:195-220` (constructor):

| Line | What it does |
|---|---|
| 203 | CPU backend always initialised: `ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, ...)` |
| 206-211 | If `ctx_params.use_gpu`: read `MTMD_BACKEND_DEVICE` env var, else `GGML_BACKEND_DEVICE_TYPE_GPU` then `GGML_BACKEND_DEVICE_TYPE_IGPU` |
| 204 | GPU backend pushed into `backend_ptrs` / `backend_buft` |
| 213 | CPU backend pushed *after* — becomes the default placement target |
| 217-220 | `ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), ...)` — scheduler sees both |

### Adreno detection

clip.cpp:222-238 already detects Adreno generation. Adreno 740 (Tab S9+
kalama) is recognised — useful for SoC-aware logic in Phase 3.

### Sync points relevant to the Adreno 740 deadlock

ggml/src/ggml-opencl/ggml-opencl.cpp lines 3920, 6351, 6636 contain
`clFlush`/`clFinish` calls. Adreno 740-specific paths exist around
line 226 (`ADRENO_GPU_GEN::A7X`).

## ggml-opencl coverage for SigLIP ops

All SigLIP operations have OpenCL kernels in the pinned fork:

| Op | Kernel | File:line |
|---|---|---|
| Conv2D (patch embed) | Supported | `GGML_OP_CONV_2D` clip.cpp:4207 |
| LayerNorm | Supported | `GGML_OP_NORM` clip.cpp:4187 |
| RMSNorm | Supported | `GGML_OP_RMS_NORM` clip.cpp:4189 |
| MHA (fused) | Supported | `GGML_OP_FLASH_ATTN_EXT` clip.cpp:4293 |
| MatMul (MLP) | Supported | `GGML_OP_MUL_MAT` clip.cpp:4219 |
| GELU activation | Supported | `GGML_OP_UNARY` clip.cpp:4147 |
| Residual add | Supported | `GGML_OP_ADD` clip.cpp:4127 |
| Softmax (attention) | Supported | `GGML_OP_SOFT_MAX` clip.cpp:4186 |

**No coverage gaps.** Routing is the only blocker.

## Proposed patch points

1. **Reorder `backend_ptrs` so GPU is index 0** — clip.cpp:217-220. Verify
   experimentally that the scheduler honours order rather than capability.
2. **Optional: explicit `ggml_set_output()` / backend-affinity hints** on
   key vision-graph nodes in `clip-graph.h`.
3. **`MTMD_VISION_GPU` / `MTMD_VISION_GPU_DISABLE` env vars** in clip.cpp
   constructor for kill-switch testing.
4. **Adreno 740 SoC bypass** in clip.cpp:222-238 — extend the gen check to
   force `use_gpu = false` for the vision tower on Adreno 740.
5. **Defensive `clWaitForEvents` after CONV_2D / FLASH_ATTN_EXT** for
   Adreno 740 in ggml-opencl.cpp around the existing A7X branch.

## Phased plan

### Phase 1 — Minimal GPU routing (~2 days)

1. Reorder `backend_ptrs` in clip.cpp:217-220 so GPU is primary.
2. Add `MTMD_VISION_GPU_DISABLE` env var for kill-switch.
3. Build with `-DGGML_OPENCL=ON` and verify `ggml_backend_sched_debug()`
   shows GPU as primary for vision graph nodes.

**Acceptance:** patch compiles, no functional regression on CPU-only path,
scheduler debug confirms GPU placement.

### Phase 2 — S24 Ultra (Adreno 750) validation (~3 days)

1. Build + push to S24 Ultra (`pineapple`).
2. SmolVLM-256M on a 480×640 frame: target < 15 s (baseline 227 s CPU).
3. Cosine similarity GPU vs CPU on 10 random images: target ≥ 0.999.
4. End-to-end VLM decode parity check on logits.

**Acceptance:**
- `loadModel` + image eval < 15 s.
- Cosine ≥ 0.999.
- No NaN / Inf in vision embeddings.
- LLM logits match CPU run (FP16 tolerance).

### Phase 3 — Adreno 740 deadlock workaround (~2 days)

1. Reduced repro: just CONV_2D + FLASH_ATTN_EXT on Tab S9+ kalama, run
   under `clProfilingCommandQueue` to find the hang point.
2. Extend the existing Adreno-gen branch in clip.cpp to disable GPU for
   the vision tower on Adreno 740 only — keep LLM decoder on GPU.
3. Validate Tab S9+ inference completes within 10 min timeout.

**Acceptance:**
- Tab S9+ no hang (NB: Tab S9+ not currently connected — schedule next time it is).
- S24 Ultra still uses GPU for vision tower.
- Documentation: Adreno 740 vision falls back to CPU, Adreno 750+ uses GPU.

## Risks and unknowns

- **Scheduler ordering may be ignored.** If the scheduler picks placement
  by capability rather than backend-list order, Patch 1 alone won't help —
  we'd need explicit backend-affinity hints (Patch 2). Verify before
  committing to the cheap path.
- **OpenCL kernel shape gaps.** Kernels exist but may not handle every
  shape SigLIP emits (esp. unusual conv strides, non-power-of-2 batch).
  Mitigate with NaN/Inf gates and CPU fallback.
- **Adreno 740 root cause unknown.** Could be ggml-opencl event-handling
  (missing `clReleaseEvent`?), MTMD-specific graph layout, or scheduler bug.
  Reduced repro is the cheapest first step.
- **GPU overhead at small batch.** Kernel launch + data copy may eat the
  speedup for tiny frames. Mitigate with a min-frame-size threshold.
- **Upstream drift.** Patches must stay minimal so they don't conflict
  when we eventually re-pin to a newer upstream tag.

## Effort estimate

| Phase | Days |
|---|---|
| 1 — minimal patch | 2 |
| 2 — S24 Ultra validation | 3 |
| 3 — Adreno 740 bypass | 2 |
| **Total** | **~7 days (1 week)** |

## Open follow-ups

- Verify scheduler order-vs-capability assumption with a 1-line printk
  patch before committing to the simple reorder.
- Tab S9+ is not currently connected — Phase 3 device validation deferred
  until it returns.
- This path competes with **Path C QNN SigLIP port** (separate design doc)
  — pick one based on user appetite for upstream-friendly patches vs raw
  HTP throughput.
