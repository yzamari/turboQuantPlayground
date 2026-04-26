# LinkedIn post — TurboQuant on Qualcomm Snapdragon

> Rewritten for stronger hook, cleaner number pairs, automotive angle.
> Numbers are measured on real hardware (Galaxy S24 Ultra + Tab S9+) — not
> projected.

---

## Long version (~280 words)

**Stop sending every "what is this?" to a cloud GPU.**

Over the past 24 hours I built an LLM + VLM personal assistant that runs
**fully on a Snapdragon phone** — no cloud, no telemetry, works in airplane
mode. Same library targets Qualcomm automotive (SA8155P / SA8295P / SA8775P)
with a toolchain swap, no algorithmic rewrite.

The trick is a C++17 port of Google's TurboQuant (PolarQuant + 1-bit QJL,
ICLR 2026) targeting Snapdragon, plus llama.cpp's Adreno OpenCL backend.

🎯 **Verified on a Galaxy S24 Ultra (SD 8 Gen 3)**:

📦 **Memory** — KV-cache compresses 4.00×, byte-exact verified against
`llama_state_seq_get_size`. That's the difference between *"can't load 16K
context with a 7B model on a phone"* and *"easy 32-64K"*. Quality cost:
~0.92 cosine on attention scores vs FP16. Same numbers verified on
Llama-3.2-1B (16 layers), SmolVLM-256M (30 layers), and Qwen2.5-VL-3B
(36 layers).

⚡ **Speed** — Llama-3.2-1B on Adreno 750: prompt-eval went **34.6 → 192.4
tok/s (5.6× faster)** the moment we flipped on `-DGGML_OPENCL=ON`. SmolVLM
end-to-end image-to-description: **3.4 s on Adreno** (was ~8 s on CPU).
Generation tok/s on the active chat path: 30+ tok/s — faster than reading
speed.

🔒 **Privacy** — model + prompt + reply never leave the device. Voice in/out
runs locally (Android `SpeechRecognizer` + `TextToSpeech`). 12 Android-Intent
tools (alarm, SMS, web search, directions, etc.) dispatched from JSON the
LLM emits.

🚗 **Automotive** — the C++ core has zero OS / vendor-SDK dependency. No
dynamic alloc in hot paths, deterministic outputs, FP32 fallback. ASIL-friendly.
The same `.a` ships from S24 today to a Snapdragon-Cockpit car tomorrow.

Repo + benchmarks (real measurements, not projections):
👉 https://github.com/yzamari/turboQuantPlayground

#Qualcomm #Snapdragon #OnDeviceAI #LLM #VLM #EdgeAI #Automotive
#LlamaCpp #PrivacyByDesign

---

## Short version (~140 words)

🚀 Got a real LLM + VLM running fully **on a Snapdragon phone** — no cloud,
no Wi-Fi needed. Same library ports to Qualcomm automotive (SA8295P /
SA8775P) with a toolchain swap.

C++ port of Google's TurboQuant (ICLR 2026) plus llama.cpp's Adreno OpenCL
backend. Verified on a Galaxy S24 Ultra:

📦 **4.00× KV-cache compression**, cosine 0.92 vs FP16 (byte-exact verified)
⚡ **5.6× faster prompt-eval** on Adreno 750 (34.6 → 192.4 tok/s)
🖼️ SmolVLM image→text in **3.4 s** on Adreno (was 8 s on CPU)
🔒 Voice in/out + 12 Android-Intent tools, all local — works in tunnels

Repo: https://github.com/yzamari/turboQuantPlayground

The phone is the new edge. The car is next.

#Qualcomm #Snapdragon #OnDeviceAI #EdgeAI #Automotive

---

## Punch version (~70 words, for tight scrolls)

Llama-3.2 + SmolVLM running entirely on my Snapdragon phone:
**5.6× faster prompt-eval on Adreno**, **4× smaller KV cache**, no cloud,
works offline. Same library ports to Snapdragon-Cockpit cars with a
toolchain swap.

C++ port of Google's TurboQuant (ICLR 2026) + llama.cpp's Adreno OpenCL
backend. Real numbers, real device, on GitHub:

👉 https://github.com/yzamari/turboQuantPlayground

#Qualcomm #Snapdragon #EdgeAI #AutomotiveAI

---

## Comment-thread seed (drop as the first reply to your own post)

If you want to dig in:

📊 **Compression** (Llama-3.2-1B, 16 layers): 4.00× verified vs
`llama_state_seq_get_size` · cosine 0.92 vs FP16
🧪 **Tokens per second** (S24, Llama-3.2-1B Q4_K_M, ngl=99):
   • CPU NEON: pp 34.6 t/s · gen 30.5 t/s
   • Adreno OpenCL: pp **192.4 t/s** · gen 22.5 t/s
🎨 **VLM** (S24, SmolVLM-256M, Adreno): pp 141 t/s · gen 115 t/s · full
   image-to-description in ~3.4 s
🚗 **Automotive transfer story**: the C++ core is plain CMake C++17 with
   zero OS dependency. Toolchain stubs for Linux aarch64 (SA8775P) and
   QNX aarch64 (SA8155P/SA8295P) already in the repo.

Path-2 (full TurboQuant in llama.cpp's KV cache during inference) is the
next milestone — current chat uses llama.cpp's q4_0 KV (closest cousin,
same 4× ratio) plus the Adreno OpenCL backend.

---

## Notes for posting

- **Visual**: attach `docs/screenshots/assistant-app-vlm-streaming.png` —
  shows the chat UI mid-VLM-stream with live timing chips. Concrete > abstract.
- **Tag**: @Qualcomm, @ggml-org (llama.cpp), @Google AI (paper authors).
- **Don't oversell**: q4_0 KV (used in the live chat decode) is the cousin
  of TurboQuant, not full TurboQuant yet. Same compression ratio, slightly
  worse quality (cosine ~0.85 vs ~0.92). The standalone TurboQuant verifier
  *is* full TurboQuant and proves the algorithm works on real model layers.
- **Link choice**: pin the SUMMARY.md link rather than just the repo root,
  so people land on the curated walkthrough, not the README.
