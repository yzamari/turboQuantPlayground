# Changelog

All notable changes to **turboQuantPlayground** are tracked here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions are commit-pinned, not semver — see the merge commit linked next
to each entry.

## [Unreleased]

### Deferred
- **P2.1c** — actual algorithmic K/V substitution in our llama.cpp fork.
  The integration shape (cparams flag, `llama_kv_cache_turboquant` class,
  `create_memory` factory branch, JNI/UI plumbing) is fixed and live;
  the algorithm itself slots in without further plumbing churn. Step-by-
  step in [`docs/path2-algorithm-playbook.md`](docs/path2-algorithm-playbook.md).
  Realistic ~3-4 weeks of focused work.
- **P2.1e** — full on-device acceptance test for `kvType=3` (cosine
  ≥0.90 vs FP16, RSS ≥3.5× drop at 8K context). Gated on P2.1c.

## 2026-04-26 — VLM live + Path 2 scaffold + persistent session

### Added
- **Live camera tab** ([`#11`](https://github.com/yzamari/turboQuantPlayground/pull/11),
  `61561c5`). New 4th tab in the Android assistant: CameraX `Preview` +
  `ImageAnalysis` (640×480, `KEEP_ONLY_LATEST`) feeds frames to the
  persistent VLM session through a single-flight `AtomicBoolean` gate.
  Steady-state ~1 FPS continuous on-device captioning for SmolVLM-256M
  on Adreno. Top stats chip + bottom streaming caption card. Pause /
  resume FAB and front/back camera switch.
- **Persistent in-process mtmd JNI session** ([`#10`](https://github.com/yzamari/turboQuantPlayground/pull/10),
  `a2f9fc1`). New `mtmd_jni.cpp` + `MtmdNative.kt`. Replaces the previous
  `Runtime.exec("libllama-mtmd-cli.so")` fork-per-image pattern: LLM +
  mmproj load once on first describe(), every subsequent image pays only
  encode + decode. ~3.4 s/image → ~0.8-1.2 s warm on Adreno (SmolVLM).
  Per-call stats exposed via `getStats()` JSON.
- **VLM image-input downscale** ([`#8`](https://github.com/yzamari/turboQuantPlayground/pull/8),
  `88d83e1`). Both gallery picker and camera capture paths now cap input
  at longest-side 1024 px (JPEG q=90) before the VLM. Cuts SigLIP /
  clip.cpp internal resize work ~10–20× vs raw 12 MP camera input.
  Logged under tag `VlmDownscale`.
- **Path 2 scaffold — vendor llama.cpp + source-build toggle**
  ([`#6`](https://github.com/yzamari/turboQuantPlayground/pull/6),
  `30f4dda`). Forked `ggml-org/llama.cpp` to
  [`yzamari/llama.cpp`](https://github.com/yzamari/llama.cpp) on branch
  `tq-main`, pinned at upstream tag `b8935` (commit `f454bd7e`).
  Submodule at `external/llama.cpp/` + sibling `external/OpenCL-Headers/`
  for the NDK build.
- **Path 2 scaffold — `kvType=3` end-to-end** (same PR + the prior
  scaffold work). New `llama_kv_cache_turboquant` derived class in the
  fork, `bool kv_turboquant` plumbed through `llama_context_params` →
  internal `llama_cparams` → `llama_memory_params` → `create_memory`
  factory. Wired in JNI (`llama_jni.cpp:149-185`) and a fourth radio
  *"TurboQuant native — PolarQuant + 1-bit QJL (Path 2, scaffold)"* in
  Settings → Vision (VLM) model.
- **Path 2 algorithm playbook** ([`#7`](https://github.com/yzamari/turboQuantPlayground/pull/7),
  `9f5ebe6`). New `docs/path2-algorithm-playbook.md` with step-by-step
  recipe for landing the still-deferred P2.1c work — captures the four
  design constraints (cross-library linkage, tensor-layout reconciliation,
  K/V write hooking, flash-attention path) with file paths + line numbers
  + tripwires.
- **Fresh S24 cpu_neon bench** ([`#13`](https://github.com/yzamari/turboQuantPlayground/pull/13),
  `b587187`). `cpp/bench/results/s24-cpu_neon-2026-04-26.csv` — full
  sweep at `BH=8, D=128, 3-bit, seq_len ∈ {128…4096}`. 4.27× compression
  + 0.90–0.94 cosine across the range, parity check `--check` green
  (smoke quantize/dequantize cosine 0.923).
- **README "What this fork adds"** ([`#14`](https://github.com/yzamari/turboQuantPlayground/pull/14),
  `8ad9a4e` + this entry). New comparison table near the top contrasting
  this repo against upstream Google TurboQuant + vanilla llama.cpp +
  generic on-device VLM apps.

### Changed
- **Build system: from-source is now the only path** ([`#7`](https://github.com/yzamari/turboQuantPlayground/pull/7),
  `9f5ebe6`). Removed the `BUILD_LLAMA_FROM_SOURCE` Gradle property toggle
  and the `jniLibs-prebuilt/` fallback dir. `./gradlew :app:assembleDebug`
  now builds llama.cpp from the pinned submodule unconditionally.
- **VLM default model documented** ([`#12`](https://github.com/yzamari/turboQuantPlayground/pull/12),
  `02b163c`). Pure docstring update on `VlmRunner.activeModel` — explains
  why the default stays SmolVLM-256M (~1 FPS continuous on Adreno) vs
  Qwen2.5-VL-3B (~0.2 FPS, 12× heavier; available as opt-in in Settings).

### Removed
- **Prebuilt llama+ggml `.so` blobs** in `android/app/src/main/jniLibs/`.
  Source-built libraries land here at build time. The `libOpenCL.so`
  ICD stub stays.

## 2026-04-25 — Adreno on chat decode + VLM picker

### Added
- **Adreno OpenCL backend in chat decode path** (`2fa48db`).
  llama.cpp built with `-DGGML_OPENCL=ON` and Qualcomm-tuned Adreno
  kernels. Llama-3.2-1B prompt-eval on Adreno 750: 34.6 → 192.4 tok/s
  (**5.6× faster** the moment the toggle flipped on).
- **VLM model picker** (`a47fb62`). Settings → Vision (VLM) model radio
  between SmolVLM-256M (small / English-leaning / fast) and
  Qwen2.5-VL-3B (multilingual / much better quality / slower).
- **KV compression + Adreno toggles ACTIVE in chat decode** (`095e9b6`).
  `kvType=1` (q4_0, the "TurboQuant cousin", same 4× compression as
  TurboQuant native) and `gpuLayers=99` Adreno offload both wired
  through `llama_jni.cpp:loadModel` so the user sees the savings live
  in chat, not just in the standalone bench.

## Earlier — foundation work

### 2026-04-24 / earlier — C++ core, on-device app, original Apple Silicon implementation
- **C++ port of TurboQuant** for Qualcomm Snapdragon (mobile + automotive)
  — plain CMake C++17, 4 backends (cpu_scalar, cpu_neon, opencl, vulkan)
  + qnn_htp scaffold, golden-vector parity tests vs the Python reference.
- **Path 1 standalone verifier** (`external/llama-turboquant-kv-tool/`)
  — loads a real GGUF, queries the model's KV geometry, runs libturboquant
  on shape-matched K/V. Reports the 4.0× compression / cosine 0.92 numbers
  on actual model layers.
- **Android assistant app** (`com.yzamari.turboquant`) — Compose UI, JNI
  bridge to libllama, voice in/out, 12 Android-Intent tools, camera +
  gallery image input. Installable on S24 Ultra and Tab S9+.
- **Original Python / Apple Silicon implementation** under `src/turboquant_mac/`
  — MLX/Metal kernels for `mse_score`, `qjl_score`, `mse_encode`,
  `value_dequant`. The C++ port is a literal translation of these.
