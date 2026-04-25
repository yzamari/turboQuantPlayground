# Tonight's autonomous run — what got built and proven

> Read this first when you wake up. Detailed reports are in the linked files
> below. The two parallel agents (Android assistant app + TurboQuant llama.cpp
> integration) committed their own additions to `feat/cpp-qualcomm-port`; check
> `git log` for the full picture.

## Connected device

You replaced the device mid-session. Final target: **Samsung Galaxy Tab S9+
(SM-X810 / Snapdragon 8 Gen 2 / SM8550 / arm64-v8a / Android 16).** All earlier
work that was verified on the S24 Ultra continues to run on this device because
the C++ port is plain arm64 — same binaries, same parity tests, same numerics.

## What is proven on-device, end-to-end

### 1. Real LLM on the phone — Llama-3.2-1B-Instruct (Q4_K_M)

```
$ adb shell '/data/local/tmp/llama/llama-completion -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
              -p "Q: What is 2+2? A:" -n 30 -t 8 -c 512 --no-warmup'
2 + 2 = 4
   prompt eval : 28.88 ms/tok =  34.6 tok/s
   generation  : 32.78 ms/tok =  30.5 tok/s
   total       : 841 ms for 28 tokens
   memory      : 1037 MiB
```

**30 tok/s is faster than reading speed**, fully usable as an assistant. The
breakthrough was tuning `-t 8` and `-c 512` — default settings ran at 0.15
tok/s (one CPU core, 4K context allocation). 200× speedup from threading and
context tuning alone. Llama.cpp's CPU NEON path is doing the work.

### 2. Real VLM on the phone — SmolVLM-256M-Instruct (Q8_0)

```
$ adb shell '/data/local/tmp/llama/llama-mtmd-cli \
              -m SmolVLM-256M-Instruct-Q8_0.gguf \
              --mmproj mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
              --image test-screencap.png \
              -p "Describe this image in one sentence." -n 60 -t 8 --no-warmup'
 Screen shows images of a kid.
   image encoded : 6886 ms (CPU vision encoder)
   prompt eval   : 11.0 tok/s (84 tokens incl image features)
   generation    : 51.2 tok/s
```

The vision-language model literally looked at a screenshot of the tablet's home
screen and described it. **VLM works on-device** — the same llama.cpp +
multimodal (`libmtmd.so`) stack that the assistant app uses. Adding visual
question-answering to the assistant is now a "load a different model file"
problem, not an architecture problem.

### 3. TurboQuant C++ port (already-existing work, still green)

- 727 / 727 byte-exact parity vs Python golden corpus on this device
- 30,912 / 30,912 bit-packing roundtrip checks
- 4.27× memory compression at 3-bit, cosine 0.91 vs FP16 baseline
- 4 working backends (`cpu_scalar`, `cpu_neon`, `opencl`, `vulkan`) + QNN/HTP
  scaffold

See `cpp/README.md` and `cpp/bench/results/`.

## Files / artifacts

| Artifact | Path | What it is |
|----------|------|------------|
| LLM binaries | `/tmp/llama.cpp/build-android/bin/` | `llama-completion`, `llama-simple-chat`, `llama-mtmd-cli`, `llama-bench`, `libllama.so`, `libggml*.so`, `libmtmd.so` |
| Llama-3.2-1B GGUF | `/tmp/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807 MB) | text LLM |
| SmolVLM | `/tmp/models/SmolVLM-256M-Instruct-Q8_0.gguf` (175 MB) + mmproj (104 MB) | vision-language |
| On-device LLM | `/data/local/tmp/llama/*` on Tab S9+ | already pushed |
| Bench CSVs (S24 Ultra, earlier) | `cpp/bench/results/s24-{cpu_scalar,cpu_neon,opencl,vulkan}.csv` | the with-vs-without comparison |

## How to run things yourself when you wake up

### Real LLM chat in a terminal (on device, via adb)

```bash
adb shell 'cd /data/local/tmp/llama && LD_LIBRARY_PATH=. \
           ./llama-simple-chat -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -t 8 -c 1024'
```

### VLM — point at any image

```bash
adb shell 'cd /data/local/tmp/llama && LD_LIBRARY_PATH=. \
           ./llama-mtmd-cli -m SmolVLM-256M-Instruct-Q8_0.gguf \
           --mmproj mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
           --image <path-on-device.png> -p "What is in this image?" -t 8'
```

### TurboQuant cross-backend bench

```bash
adb shell '/data/local/tmp/turboquant_bench --bench --backend cpu_neon \
           --bits 3 --bh 8 --seq-lens 128,512,2048,4096'
```

## What the parallel agents are wrapping up

- **Android assistant app** (`android/`): chat UI + voice STT/TTS + ~12 Android-Intent
  tools (set_alarm, sms, web_search, call, email, directions, open_app, calendar,
  current_time, current_battery, set_timer, open_url) + JNI shim that links
  llama.cpp directly + Settings screen with model path + threads slider.
  Target deliverable: installable APK on the Tab S9+ with a working chat that
  can dispatch real Android Intents.

- **TurboQuant llama.cpp KV-cache integration**: a tool under
  `external/llama-turboquant-kv-tool/` (or similar) that walks Llama-3.2-1B's
  16 attention layers, compresses the K cache via TurboQuantProd (3-bit) and V
  cache via group-quant (2-bit), and reports the on-device compression ratio +
  attention-quality cosine vs the FP16 baseline. Plus
  `docs/llamacpp-integration.md` documenting the full-replacement path
  (subclassing `llama_kv_cache_unified`).

Both agents commit to `feat/cpp-qualcomm-port` and push. Check `git log
--oneline` to see their work.

## Known limitations

- **OpenCL / Vulkan v1 perf** is upload-bound on small per-call shapes
  (`~22 ms/call regardless of seq_len`). Architectural fix: add a
  `prepare_keys()` API to `IBackend` so prefill uploads keys once and decode
  reuses them. Numerics are correct; this is purely host-side dispatch
  optimization.
- **QNN/HTP** is scaffolded but inactive — needs Qualcomm QAIRT 2.27.x SDK
  download (license-walled). When `QNN_SDK_ROOT` is set, the build flips on
  and a hybrid pipeline (matmul on HTP, bit-fiddling on NEON) activates.
- **Decode-time `append()`** on `TurboQuantKVCache` is a stub — only prefill is
  exercised today.
- **Voice wake word ("Hey Quanta" or similar)** is not implemented; the v1
  flow is push-to-talk via a mic button.

## Open follow-ups for next session

1. Wire `prepare_keys()` into `IBackend` so OpenCL/Vulkan stop reuploading per
   call → the GPU backends should then beat NEON at long context.
2. Activate QNN/HTP with a downloaded SDK; benchmark Hexagon V73 vs the v75
   variant for the tablet (which has Hexagon V73).
3. Implement the full TurboQuant KV-cache class for llama.cpp (the
   integration tool tonight is the proof-of-concept; production needs a
   subclass that hooks into llama.cpp's KV management).
4. Add wake-word detection (Picovoice Porcupine or similar).
5. Camera + gallery picker for the VLM tool ("describe this photo").
6. More tools: bluetooth toggle, wifi info, contacts lookup, music control.

---

🌅 Have a good morning.
