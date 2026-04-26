# Related work — On-device KV-cache compression and Snapdragon LLM stacks

> Survey of public projects adjacent to ours (a C++ TurboQuant port for Qualcomm
> Snapdragon mobile + automotive, plus an Android assistant app). Focused on
> what overlaps, what's different, and where our positioning is defensible.

## 1. KV-cache compression methods and implementations

### 1.1 KIVI — 2-bit asymmetric KV (ICML 2024)
- https://github.com/jy-yuan/KIVI
- PyTorch / CUDA, tuning-free. Per-channel quantization for K, per-token for V.
- **Our delta:** ours ships Snapdragon NEON / Hexagon HTP / Adreno OpenCL / Adreno Vulkan kernels. KIVI has no mobile path; PolarQuant matches/beats KIVI quality below 3 bits with simpler rotation math.

### 1.2 KVQuant — 10M-token context (NeurIPS 2024)
- https://neurips.cc/virtual/2024/poster/96936
- 3-bit KV with `<0.1 PPL` drop, scales LLaMA-7B to 1M-10M ctx on 8×A100.
- **Our delta:** ours targets a 12 GB phone, not 8×A100. PolarQuant + 1-bit QJL gives a sharper compression curve at <3 bits than KVQuant's NUF lookup.

### 1.3 llama.cpp native KV quantization (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0)
- https://github.com/ggml-org/llama.cpp
- Built-in `--cache-type-k` / `--cache-type-v` flags.
- **Our delta:** their lowest practical setting is Q4_0 (~3.8×). We hit TurboQuant TQ3 (~4.0× verified on real Llama-3.2-1B layers, cosine 0.92), with calibrated PolarQuant rotations rather than per-block min/max.

### 1.4 TurboQuant llama.cpp discussion #20969 — **closest sibling**
- https://github.com/ggml-org/llama.cpp/discussions/20969
- Multiple parallel ports of the same paper we implement (Metal, CUDA, CPU, Vulkan).
- **Our delta:** those forks are desktop-class (Apple Silicon, RTX 5090). **We are the only port targeting Snapdragon's four heterogeneous compute engines** (CPU NEON, HTP NPU via QNN, Adreno OpenCL, Adreno Vulkan compute), and the only one keeping Algorithm 2 (1-bit QJL) for sparse-V attention rather than discarding it.

### 1.5 vLLM / LMDeploy quantized KV
- INT8 / FP8 paths, server-side CUDA.
- **Our delta:** server stacks; no Hexagon, no Adreno. Our 3-bit + 1-bit modes are 2-3× denser than INT8.

## 2. Qualcomm-targeted LLM stacks

### 2.1 llama.cpp OpenCL Adreno backend (Qualcomm PR #10693) — **complementary**
- https://github.com/ggml-org/llama.cpp/pull/10693
- Qualcomm-engineered FP16 / Q4_0 weight-only Adreno GEMM path. KV stays FP16.
- **Our delta:** we add a TurboQuant KV layer on top of it, plus a Vulkan compute alternative for devices where OpenCL is gated. We **build on this**, not against it.

### 2.2 llama.cpp-npu (Hexagon HTP via FastRPC + HVX/HMX)
- https://github.com/haozixu/llama.cpp-npu — paper https://arxiv.org/html/2509.23324v1
- ~7K LOC, bypasses QNN, custom HTP-Ops library. Qwen / Llama <4B on SD8G2+ NPU.
- **Our delta:** we use the official QNN graph path (portable across SA8155P / SA8295P / SA8775P automotive parts) and add KV compression. They have neither Adreno nor automotive coverage and no KV compression at all.

### 2.3 MLC-LLM Adreno OpenCL backend
- https://github.com/mlc-ai/mlc-llm — TVM Unity → generated OpenCL kernels.
- **Our delta:** MLC compiles whole graphs; we are a focused KV layer that drops into llama.cpp without the TVM toolchain. MLC has no equivalent KV mode at our compression ratio.

### 2.4 ONNX Runtime QNN execution provider
- https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Microsoft / Qualcomm path to HTP for ONNX models.
- **Our delta:** ORT-QNN does not implement custom KV compression ops; we add HTP HVX kernels for PolarQuant rotation + bit-packed dequant.

### 2.5 ExecuTorch Qualcomm backend
- PyTorch ExecuTorch + Qualcomm AI Engine Direct.
- **Status:** existence confirmed as a path; couldn't verify a public TurboQuant/QJL KV mode in the search budget. Treat as unverified.

## 3. On-device personal assistants (chat + voice + vision)

### 3.1 PocketPal AI — **closest UX sibling**
- https://github.com/a-ghorbani/pocketpal-ai · https://play.google.com/store/apps/details?id=com.pocketpalai
- React Native + llama.cpp + Whisper. Chat, model swap, SmolVLM2-500M camera/gallery, voice transcription.
- **Our delta:** PocketPal uses stock llama.cpp KV (FP16 / Q8). Our Android app carries the same UX surface (SmolVLM-256M + Llama-3.2-1B) but adds the TurboQuant KV layer (longer context same RAM) plus a Hexagon HTP path for sustained throughput PocketPal does not have.

### 3.2 MLC-Chat (Android)
- https://github.com/mlc-ai/mlc-llm/tree/main/android
- MLC-compiled models on Adreno OpenCL.
- **Our delta:** no voice, no VLM-via-camera in the reference app; no KV compression layer.

## 4. Automotive Qualcomm LLM efforts (SA8155P / SA8295P / SA8775P) — **the open category**

The search budget did not surface a public open-source LLM stack targeting these
SoCs with KV compression. What exists is largely vendor marketing:

- Qualcomm Snapdragon Cockpit / Ride — generative-AI cockpit announcements, no public LLM repo.
- Cerence CaLLM / Chat Pro — proprietary automotive-grade LLM, no on-device code.
- Mobileye — focuses on perception, not LLMs.

**Honest take:** this is a defensible niche for us. SA8295P / SA8775P share the
HTP and Adreno IP we already target; the same `.a` files build with a toolchain
swap (we ship the stubs at `cpp/cmake/toolchain-{linux,qnx}-aarch64.cmake`).

## Where we sit on the map

| Dimension | Closest sibling | Our differentiator |
|---|---|---|
| Algorithm | llama.cpp #20969 forks | only port keeping Algorithm 2 (1-bit QJL) and only one targeting Snapdragon's 4 backends |
| Hexagon HTP | llama.cpp-npu | we use QNN graph mode (portable to automotive); they bypass QNN |
| Adreno OpenCL | Qualcomm PR #10693 | we build on it and add a KV-cache layer it doesn't have |
| Mobile UX | PocketPal AI | same UX, plus TurboQuant KV + HTP backend |
| Automotive | (no public OSS) | we are the candidate |

## Sources

All URLs above. Survey conducted 2026-04-26.
