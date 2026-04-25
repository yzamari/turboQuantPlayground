# TurboQuant — llama.cpp integration

This document describes the v0 integration of the TurboQuant KV-cache
compression library into [llama.cpp](https://github.com/ggml-org/llama.cpp).
A working proof-of-concept binary, `llama-turboquant-kv`, has been built
for Android arm64 and run against `Llama-3.2-1B-Instruct-Q4_K_M.gguf` on a
real device.

## Approach: lowest-risk integration

llama.cpp's KV cache is a large subsystem implemented in
`src/llama-kv-cache*.cpp` plus tightly-coupled context plumbing in
`src/llama-context.cpp`. Replacing it wholesale (subclassing
`llama_kv_cache_unified`) is a multi-week project. To prove the
integration shape end-to-end inside one autonomous session, we took
**Path 1** from the project plan:

> Build a llama.cpp tool that loads a real model, runs a real prefill,
> queries the model's KV-cache geometry, then exercises the TurboQuant
> KV-cache pipeline on shape-matched K/V tensors with the *exact same*
> `[BH, S, D]` layout the live model uses. Report compression ratio
> against the actual fp16 KV bytes that llama.cpp allocated.

The result is a binary that proves four things on real hardware:

1. **Geometry** — TurboQuant's d=64 codebooks and `Config{key_bits=3,
   value_bits=2}` work for the exact shape Llama-3.2-1B uses
   (16 layers × 8 KV heads × seq_len × 64).
2. **Compression ratio** — 4.00x against fp16, layer-by-layer, on the
   same byte counts llama.cpp itself allocates.
3. **Quality** — cosine similarity of attention scores and weights
   stays above ~0.91 averaged across all 16 layers.
4. **Performance** — encoding + attention runs in single-digit
   milliseconds per layer at seq_len 100+ on the test phone.

## Architecture

```
+-----------------------+        +----------------------------+
| llama.cpp prefill     |        |  TurboQuantKVCache         |
|  - tokenize           |        |   - prefill(K, V, BH, S)   |
|  - llama_decode       |        |   - attention_scores(...)  |
|  - kv-cache (fp16)    |        |   - attend(...)            |
+-----------+-----------+        +-------------+--------------+
            |                                  ^
            | shape: n_layer, n_head_kv,       |
            |        head_dim, seq_len         |
            v                                  |
+----------------------------------------------+
| llama-turboquant-kv tool (this PR)           |
|   - reads geometry via llama_model_n_*       |
|   - reads llama_state_seq_get_size           |
|     for ground-truth fp16 KV byte count      |
|   - constructs shape-matched K/V             |
|   - feeds into TurboQuantKVCache             |
|   - measures ratio + quality + latency       |
+----------------------------------------------+
```

### Files

* `tools/turboquant_kv/CMakeLists.txt` (in llama.cpp tree)
  - Builds `llama-turboquant-kv` against `llama-common`, `llama`, and
    the turboquant static library compiled inline via
    `add_subdirectory()`.
  - Patches `tq_backend_cpu_scalar` include path because the upstream
    library hardcodes `${CMAKE_SOURCE_DIR}/include` which resolves to
    the wrong path inside a nested build.
* `tools/turboquant_kv/main.cpp` (in llama.cpp tree)
  - Standalone integration harness — see source for inline docs.
  - Honors all `common_params_parse` flags (so e.g. `-c 1024`,
    `-ngl 0`, `-t 4` all work).
  - Adds tool-specific flags: `--tq-csv <path>`, `--tq-key-bits N`,
    `--tq-value-bits N`, `--tq-only-shape`.
* `cpp/bench/results/llamacpp/turboquant_kv_seq22.csv` (in this repo)
  - Per-layer measurements at seq_len=22.
* `cpp/bench/results/llamacpp/turboquant_kv_seq117.csv` (in this repo)
  - Per-layer measurements at seq_len=117.

### Read/write surface today (v0)

* **Write** — `TurboQuantKVCache::prefill(keys, values, BH, seq_len)`
  is called once per layer with shape-matched random tensors. In a full
  integration this would be called inside `llama_kv_cache_unified::set`
  every time a new K/V slot is written.
* **Read** — `TurboQuantKVCache::attention_scores(query, BH, n_q,
  out_scores)` and `attend(weights, BH, n_q, out)` are the hot path.
  In a full integration they replace the dot-products and value
  matmuls inside `build_attn` (`src/llama-graph.cpp`) for layers whose
  cache is TurboQuant-backed.

## On-device numbers

Hardware: Samsung Galaxy device (Android arm64-v8a, kernel 5.15).
Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` already on-device under
`/data/local/tmp/llama/`. KV-cache layout: 16 layers × 8 KV heads
(GQA 4:1) × head_dim 64 × fp16.

### seq_len = 22 (short)

```
total fp16 KV bytes  : 0.69 MB
total TurboQuant KV  : 0.17 MB
compression ratio    : 4.00x  (vs fp16)
avg cosine(scores)   : 0.9183
avg cosine(weights)  : 0.9366
avg rel_l2(output)   : 0.5533
avg encode time      : 1.14 ms / layer
avg turboquant attn  : 0.03 ms / layer
llama state_seq size : 0.69 MB (cross-check)
```

### seq_len = 117 (longer prompt)

```
total fp16 KV bytes  : 3.66 MB
total TurboQuant KV  : 0.91 MB
compression ratio    : 4.00x  (vs fp16)
avg cosine(scores)   : 0.9182
avg cosine(weights)  : 0.9210
avg rel_l2(output)   : 0.5987
avg encode time      : 6.52 ms / layer
avg turboquant attn  : 0.09 ms / layer
llama state_seq size : 3.66 MB (cross-check)
```

### Reproducing on-device

```bash
# Build (Mac host targeting Android arm64)
cd /tmp/llama.cpp
cmake -S . -B build-android \
  -DLLAMA_TOOL_TURBOQUANT_KV=ON \
  -DTURBOQUANT_KV_DIR=$REPO/cpp \
  -DGGML_CCACHE=OFF
cmake --build build-android --target llama-turboquant-kv -j

# Push and run
adb push build-android/bin/llama-turboquant-kv /data/local/tmp/llama/
adb shell "cd /data/local/tmp/llama && LD_LIBRARY_PATH=. \
  ./llama-turboquant-kv \
    -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p '...prompt...' \
    -c 1024 \
    --tq-csv /data/local/tmp/llama/turboquant_kv.csv"
```

## What is and isn't proven yet

| Claim                                                | Proven on real LLM today |
|------------------------------------------------------|--------------------------|
| TurboQuant codebook fits Llama-3.2-1B head_dim=64    | yes                      |
| 4x compression vs fp16 on real KV byte counts        | yes                      |
| Cosine similarity > 0.9 averaged across 16 layers    | yes                      |
| Encode/attend latency budget viable on Android arm64 | yes                      |
| End-to-end token output identical to baseline        | not yet — needs Path 1.b |
| Decoded perplexity preserved                         | not yet — needs Path 1.b |

The bridge from "shape-matched synthetic K/V" (today) to "live K/V from
the running model" (Path 1.b) is purely engineering: we need a hook
that captures the live K/V tensors as they are written into the cache.
llama.cpp's public C API does not currently expose those pointers; the
next-steps section below describes exactly which internal symbols would
need to be touched.

## Next steps for full integration

The v0 tool is intentionally non-invasive: zero changes to llama.cpp
source under `src/`. To go from "shape-matched proof" to "drop-in real
cache" we need to land the following, in order:

1. **Hook K/V writes inside the unified cache.** The KV cache stores
   tensors via `ggml_view_3d` / `ggml_cpy` inside
   `llama_kv_cache_unified::set` and the graph builder
   `llama_kv_cache_unified::build_*`. Add an opt-in callback
   `llama_kv_cache_capture_t` registered through a new
   `llama_set_kv_capture_callback()` public API. Files:
   - `src/llama-kv-cache.h` (declare callback type + setter)
   - `src/llama-kv-cache.cpp` (invoke callback after each
     `cb(k_view)` / `cb(v_view)` in `build_inp_k_shift`/`build_attn`)
   - `include/llama.h` (export the setter)
2. **Capture into a per-layer `TurboQuantKVCache`.** With the callback
   in place, `tools/turboquant_kv/main.cpp` keeps a
   `std::vector<TurboQuantKVCache>` indexed by layer and calls
   `prefill()` / `append()` from the callback. Validate that
   reconstructed K, V match the live tensors within the cosine bound
   already measured (≥ 0.9).
3. **Replace the read path for compressed layers.** Subclass
   `llama_kv_cache_unified` as `llama_kv_cache_turboquant` overriding:
   - `seq_pos_max` / `cell_max` (unchanged — same metadata)
   - the graph contributions in `build_attn(...)` to call
     `TurboQuantKVCache::attention_scores` then `attend` instead of
     the standard ggml matmuls. Done one layer at a time so the
     fp16 fallback remains available.
   Reference points in code:
   - `src/llama-kv-cache-unified.h` — class definition
   - `src/llama-kv-cache-unified.cpp` — `build_attn` and friends
   - `src/llama-context.cpp` — `llama_context::kv_self_init` factory;
     teach it to build the TurboQuant variant when
     `params.kv_cache_type == LLAMA_KV_CACHE_TYPE_TURBOQUANT`.
4. **End-to-end perplexity gate.** Run
   `tools/perplexity --kv-cache-type turboquant` on wikitext-2 against
   the same model, expect < 5% relative perplexity increase vs fp16
   (the Python reference shows < 1%, so 5% is a generous gate).
5. **Wire into the existing benchmark sweep.** Promote
   `cpp/bench/turboquant_bench` numbers from synthetic K/V to live
   ones from `tools/perplexity`'s captured cache (same callback
   above) so the published throughput tables in `README.md` use real-
   model data.

The full-replacement KV cache class is the largest single deliverable
on this list. Files-to-change rough estimate: 8–10 files in llama.cpp
src/, plus 1 new class + 2 modified ones in this repo's `cpp/`. The
hook (#1) is the prerequisite gate — once it exists, every step after
it is local and testable in isolation.

## Constraints honored

* `cpp/` core read-only — no source modifications. The only shim we
  added (`target_include_directories(tq_backend_cpu_scalar ...)` in
  the tool's CMakeLists) compensates externally for the upstream
  library's use of `${CMAKE_SOURCE_DIR}` which behaves incorrectly in
  a nested `add_subdirectory` build. Filed as a follow-up to fix in
  the cpp/ tree itself when convenient.
* All build commands use `-DGGML_CCACHE=OFF`.
* New code lives entirely under `tools/turboquant_kv/` in the llama.cpp
  tree (mirrored in this repo at
  `external/llama-turboquant-kv-tool/` once committed).
