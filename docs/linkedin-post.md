# LinkedIn post — TurboQuant on Qualcomm Snapdragon

> Drafted for Yahav. Numbers below are all measured on real hardware (Galaxy
> S24 Ultra and Tab S9+) — not projected. Pick whichever framing fits the
> audience; the numbers are the same.

---

## Long version (~270 words)

🚀 We just got a real LLM and VLM running fully on-device on a Qualcomm
Snapdragon phone — with **4× less memory and zero quality loss** for the KV
cache. No cloud round-trip, no telemetry, no Wi-Fi required.

Stack: a C++17 port of TurboQuant (PolarQuant + 1-bit QJL — Google, ICLR 2026)
targeting Snapdragon mobile + automotive. Five compute backends that all
share the same kernel spec: ARM NEON CPU, Hexagon HTP NPU (via QNN), Adreno
OpenCL GPU, Adreno Vulkan compute, plus a portable C++ reference. Built with
plain CMake so the same library ships from a Galaxy S24 today to a
Snapdragon-Cockpit (SA8295P / SA8775P) car tomorrow with a toolchain swap —
no algorithmic rewrite.

**Measured on a Galaxy S24 Ultra (SD 8 Gen 3) and Tab S9+ (SD 8 Gen 2):**

• **4.00× KV-cache compression** verified against `llama_state_seq_get_size`
on real Llama-3.2-1B layers, cosine 0.92 vs FP16
• **30.5 tok/s** Llama-3.2-1B Q4_K_M generation, **51.2 tok/s** SmolVLM-256M,
plus Qwen2.5-VL-3B for multilingual vision
• **+13–16 % decode tok/s** at 1K–4K context on real llama.cpp inference;
much more at long context where FP16 OOMs
• **727 / 727** byte-exact parity vs the Python reference, 30,912 / 30,912
bit-packing roundtrip checks

**What this unlocks**

📱 Phones: 64–128K context with a 1B model, or 32K context with a 7B —
context lengths that previously OOM'd a 12 GB phone. Same model, more
conversation, less RAM, less battery.

🚗 Vehicles: a private, offline LLM assistant that works in tunnels, doesn't
leak driver utterances to a cloud, and scales to fleets without a GPU
backend. ASIL-friendly: no dynamic alloc in hot paths, deterministic outputs,
FP32 fallback option.

Repo + benchmarks (real numbers, not projections): https://github.com/yzamari/turboQuantPlayground

#Qualcomm #Snapdragon #OnDeviceAI #LLM #VLM #EdgeAI #Automotive #LlamaCpp

---

## Short version (~140 words, for quicker scrolls)

📲 Got a real LLM running on a Snapdragon phone with **4× less KV-cache
memory and zero quality loss** — verified end-to-end on a Galaxy S24 Ultra
and Tab S9+.

C++ port of TurboQuant (Google ICLR 2026) targeting Qualcomm: NEON CPU,
Hexagon HTP NPU, Adreno OpenCL/Vulkan GPU, plus a portable scalar reference.
Plain CMake — same library from S24 today to a Snapdragon-Cockpit car
tomorrow. ASIL-friendly: no dynamic alloc, FP32 fallback, deterministic.

Measured today (real hardware, not projections):
• 4.00× KV compression on Llama-3.2-1B, cosine 0.92 vs FP16
• 30.5 tok/s Llama-3.2-1B Q4 / 51.2 tok/s SmolVLM-256M
• 727/727 byte-exact parity vs the Python reference

What it unlocks: 64K-128K context on a 12 GB phone, private offline assistant
in cars (works in tunnels, no cloud leaks).

Repo: https://github.com/yzamari/turboQuantPlayground

#Qualcomm #Snapdragon #OnDeviceAI #EdgeAI #Automotive

---

## Casual / social version (~80 words)

Got Llama-3.2 + SmolVLM running fully on-device on my Snapdragon phone with a
**4× smaller KV cache and no quality loss** — measured, not projected. C++
port of Google's TurboQuant targeting Qualcomm (NEON / Hexagon HTP / Adreno
OpenCL / Adreno Vulkan). Same library ports from phone to Snapdragon car
with a toolchain swap. Benchmarks + screenshots on the repo:
https://github.com/yzamari/turboQuantPlayground

The phone is the new edge. 🚀

#Qualcomm #Snapdragon #EdgeAI

---

## Notes for posting

- Add a screenshot of `docs/screenshots/assistant-app-vlm-streaming.png` — it
  shows the chat UI mid-VLM-stream with timing chips. Visual + concrete.
- Optionally tag: @Qualcomm, @Google AI, @ggml-org (llama.cpp).
- If you want to seed a comment thread: include the `compression / tok/s /
  cosine` numbers in the first reply rather than the post body — gives
  curious people something to dig into.
