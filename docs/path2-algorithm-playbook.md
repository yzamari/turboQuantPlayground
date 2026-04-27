# Path 2 Algorithm Playbook

> Step-by-step recipe for landing **P2.1c** — the actual TurboQuant K/V
> substitution in llama.cpp's hot decode path. Designed to be picked up cold:
> open this file, follow it top-to-bottom, end up with chat tokens running
> through `TurboQuantKVCache` instead of stock FP16.
>
> **Status (2026-04-26 night session).** Steps 1-6 are implemented and the
> host parity sweep is green (72/72 configs at cosine ≥ 0.85, libllama links
> clean with all three TurboQuant symbols exported). Branch `p2.1c/integrated`
> contains the full stack. **What's pending:** on-device verification on
> S24 Ultra (Llama-3.2-1B with kvType=3) and Step β (the memory win — KV
> backing-store override, 5-7 days). See `docs/handoff-2026-04-26-night.md`.

## Context

What's already done (committed on `main` via PR #6):

- `external/llama.cpp/` submodule pinned at `f454bd7e` (upstream tag `b8935`)
  on our fork's `tq-main` branch.
- `bool kv_turboquant` plumbed end-to-end through `llama_context_params` →
  `llama_cparams` → `llama_memory_params` → `create_memory` factory.
- `llama_kv_cache_turboquant` class (header-only, `src/llama-kv-cache-turboquant.h`)
  derives from `llama_kv_cache`. Currently a thin tag — `is_turboquant()` returns
  true; everything else delegates to the base.
- `kvType=3` wired in `llama_jni.cpp:149-185` and a fourth radio in
  `SettingsScreen.kt`. Verified live on Galaxy S24 Ultra: selecting it builds
  the new cache class, behaviour is FP16-equivalent.

**The one missing piece.** The cache class doesn't yet do anything different —
no compression, no custom attention. The work below is what makes it actually
TurboQuant.

## The four constraints (decide each before coding)

### 1. Cross-library linkage

`libturboquant.a` is currently linked into `libturboquant_jni.so`. To call
`turboquant::TurboQuantKVCache::attention_scores()` from inside a custom ggml
op (which lives in `libllama.so`), one of these must happen:

- **Option A — link `libturboquant` into `libllama.so`.** Add a CMake option
  in our fork's top-level `CMakeLists.txt`: `LLAMA_TURBOQUANT_BACKEND` (default
  `OFF`). When ON, `target_link_libraries(llama PRIVATE turboquant)`. Cleanest
  ABI but couples the libraries. Recommended for the in-tree, non-upstream path.
- **Option B — runtime callback registration via public C API.** Add to
  `include/llama.h`:
  ```c
  // Register a TurboQuant attention provider (PolarQuant + 1-bit QJL).
  // The function is invoked by the custom ggml op when a TurboQuant cache
  // is in use. Pass NULL to deregister. Caller owns the function pointer.
  typedef void (*llama_turboquant_attn_fn)(
      const float * q, const float * k, const float * v,
      int BH, int n_q, int n_kv, int D,
      float scale, const float * mask, float * out);

  LLAMA_API void llama_set_turboquant_attn_fn(llama_turboquant_attn_fn fn);
  ```
  The JNI layer registers a thunk that calls `libturboquant`. Cleaner
  layering — `libllama.so` stays vendor-agnostic — at the cost of one
  function pointer indirection per attention call. Recommended if we
  ever PR upstream.

**Recommended pick: Option B.** It keeps `libllama.so` algorithm-agnostic and
sets us up for an eventual upstream PR that adds the hook without imposing a
TurboQuant dependency on the upstream build.

### 2. Tensor layout reconciliation

`build_attn_mha` in `llama-graph.cpp:1932-2053` operates on ggml tensors with
these shapes (after RoPE, before mat-mul):

| Tensor | Shape (ggml `[ne0, ne1, ne2, ne3]`) | Meaning |
|---|---|---|
| Q | `[n_embd_head_v, n_head, n_tokens]` | per-token, per-head |
| K | `[n_embd_head_k, n_kv, n_head_kv]` | full cache K |
| V | `[n_embd_head_v, n_kv, n_head_kv]` (or transposed) | full cache V |

`turboquant::TurboQuantKVCache::attention_scores` expects:

```cpp
attention_scores(
    const float * q,      // [BH * n_q * D] flat, row-major
    int BH,               // batch * n_head_kv
    int n_q,              // tokens this call
    float * out_scores,   // [BH * n_q * n_kv]
    float scale)
```

**Reconciliation rules:**

- For Llama-3 (GQA): `n_head_kv=8`, `n_head=32`, ratio `4:1`. The q tensor's
  `n_head=32` heads fold into 8 KV groups. Inside the custom op, replicate
  Q `n_head/n_head_kv` times along the BH axis so it pairs with each K head.
- `D = n_embd_head_v = n_embd_head_k` for all current architectures. Sanity-
  check it at runtime; assert mismatch if a future model violates it.
- ggml stores `[D, n_kv, n_head_kv]` as contiguous F16/F32 in memory with
  D as the fastest-changing dim. To get the libturboquant layout
  `[BH, n_kv, D]`, we permute K from `[D, n_kv, n_head_kv]` to
  `[n_head_kv, n_kv, D]` then flatten the first two axes. Same for V.
- The mask tensor (causal + sequence mask) is `[n_kv, n_q]`. Add it to
  `out_scores` before softmax, exactly like `ggml_soft_max_ext` does.

### 3. K/V write path hooking

When tokens enter the cache, ggml builds copy ops via
`llama_kv_cache::cpy_k(...)` and `cpy_v(...)` at `llama-kv-cache.cpp:1196-1285`.
These return ggml tensors that get scheduled by the backend.

**Two options:**

- **Option α — keep ggml tensor cache, compress lazily.** Don't override
  `cpy_k`/`cpy_v`. K/V remain stored as F16 ggml tensors. The custom attention
  op compresses K/V *on-the-fly* when called. Simpler to land, but loses the
  4× memory win — RSS still grows linearly with `seq_len * n_embd_gqa`.
- **Option β — own the K/V backing store outside ggml.** Override `cpy_k`/
  `cpy_v` in `llama_kv_cache_turboquant` to extract `k_cur`/`v_cur` data,
  feed `TurboQuantKVCache::append()`, and skip the ggml `set_rows`. Memory
  win is real (4×), but
  - `memory_breakdown()`, `state_write()`, `state_read()` all need
    re-implementation,
  - shift / copy / div sequence operations that mutate cell positions need
    to mutate our compressed buffers correctly,
  - the attention op now reads from our buffers rather than ggml tensors,
    which means the tensor inputs to the custom op are *empty* placeholders.

**Recommended pick: Option α first**, then β as a follow-up. Option α gives
us a working algorithm (cosine-similarity validation, perplexity check) in
roughly 1 week. Option β doubles that to 2-3 weeks but is the actual win.

### 4. Flash attention path

`llama-graph.cpp:1972` uses `ggml_flash_attn_ext()` — a single fused op that
does Q@K.T → softmax → attn@V internally. About half of production
deployments use this (it's faster on GPU and uses less memory).

For Path 2:

- **Easiest:** when `cache->is_turboquant()`, force `cparams.flash_attn = false`
  in `llama_context.cpp` so we always hit the standard 3-op path. Document the
  perf hit in the radio's helper text. Simple, ships first.
- **Right thing:** write a `ggml_flash_attn_turboquant_ext()` that does the
  same fused work calling libturboquant. Big chunk of work, mainly to support
  the GPU/Metal/Vulkan backends.

**Recommended pick: easiest first.** Flash on TurboQuant is its own ticket
once Option α is shipped.

## Implementation steps (bottom-up)

After picking the four options above (B, α, then β; flash off for now), the
work goes in this order.

### Step 1 — extend libturboquant with a stateless attention entry-point

Currently `TurboQuantKVCache` is stateful (call `prefill` then later
`attention_scores`). For Option α we need a **single-call** entry that takes
Q, K, V and returns attention output. Add to `cpp/include/turboquant/api.hpp`:

```cpp
namespace turboquant {

// One-shot TurboQuant attention. Compresses K/V internally per call —
// no persistent cache state. Used as the bridge from llama.cpp's
// graph-time custom op to the algorithm.
//
//   q     : [BH * n_q  * D] row-major
//   k     : [BH * n_kv * D] row-major
//   v     : [BH * n_kv * D] row-major
//   mask  : [n_q * n_kv] row-major (additive, -INF for masked)
//   out   : [BH * n_q  * D] row-major
void attention_turboquant(
    const float * q, const float * k, const float * v,
    int BH, int n_q, int n_kv, int D,
    float scale, const float * mask, float * out,
    int key_bits   = 3,
    int value_bits = 2);

}  // namespace turboquant
```

Implementation lives in `cpp/src/attention_turboquant.cpp`. Internally:

1. PolarQuant K rotate + groupwise quantize.
2. Loop `n_q`: compute MSE scores, then QJL correction scores, sum.
3. Apply mask, scale, softmax.
4. Groupwise quantize V, dequantize while computing weighted sum.

Tested via `cpp/bench/bench_cli.cpp --check-stateless` against a synthetic
golden corpus (cosine ≥ 0.92 vs FP32 baseline at `BH=8, D=128, n_kv=512`).

### Step 2 — expose the C-API hook in our llama.cpp fork

Patch `external/llama.cpp/include/llama.h` (after the existing `llama_set_*`
declarations):

```c
typedef void (*llama_turboquant_attn_fn)(
    const float * q, const float * k, const float * v,
    int BH, int n_q, int n_kv, int D,
    float scale, const float * mask, float * out);

LLAMA_API void llama_set_turboquant_attn_fn(llama_turboquant_attn_fn fn);
```

Patch `external/llama.cpp/src/llama.cpp` (or wherever the global state lives —
search for `llama_set_abort_callback` for the pattern):

```cpp
static llama_turboquant_attn_fn g_tq_attn_fn = nullptr;

void llama_set_turboquant_attn_fn(llama_turboquant_attn_fn fn) {
    g_tq_attn_fn = fn;
}

// Internal accessor for the custom op.
llama_turboquant_attn_fn llama_get_turboquant_attn_fn() {
    return g_tq_attn_fn;
}
```

### Step 3 — substitute the attention triple in `build_attn_mha`

`external/llama.cpp/src/llama-graph.cpp:1932-2053`. Inside the standard-
attention branch (lines 1997-2053), guard with the cache type:

```cpp
const bool is_turboquant =
    dynamic_cast<const llama_kv_cache_turboquant *>(memory) != nullptr;

if (is_turboquant && !cparams.flash_attn) {
    // Replace the Q@K.T → soft_max_ext → attn@V triple with a
    // ggml_map_custom3 op that forwards to llama_get_turboquant_attn_fn().
    cur = ggml_map_custom3(
        ctx0,
        q,            // [n_embd_head_v, n_head, n_tokens]
        k,            // [n_embd_head_k, n_kv, n_head_kv]
        v,            // [n_embd_head_v, n_kv, n_head_kv]
        ggml_custom_op_turboquant_attn,
        /* n_tasks */ GGML_N_TASKS_MAX,
        /* userdata */ &cparams);   // expose mask, scale via userdata
} else {
    // ... existing code ...
}
```

The forward function (place in a new file `external/llama.cpp/ggml/src/ggml-custom-turboquant.c`, then add to the ggml `CMakeLists.txt`):

```c
static void ggml_custom_op_turboquant_attn(
    struct ggml_tensor * dst,
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    int ith, int nth, void * userdata) {
    if (ith != 0) return;  // single-thread for simplicity in first cut

    llama_turboquant_attn_fn fn = llama_get_turboquant_attn_fn();
    GGML_ASSERT(fn != NULL && "TurboQuant attention fn not registered");

    const int n_kv = k->ne[1];
    const int n_head_kv = k->ne[2];
    const int n_q = q->ne[2];
    const int n_head = q->ne[1];
    const int D = q->ne[0];
    const int BH = n_head_kv * (n_head / n_head_kv);  // GQA fold

    // Permute K/V from [D, n_kv, n_head_kv] to [n_head_kv, n_kv, D].
    // ... permutation buffer fill ...

    const float scale = *(const float *) userdata;  // TODO: pass mask too
    fn(q->data, k_perm, v_perm, BH, n_q, n_kv, D,
       scale, mask_data, dst->data);
}
```

### Step 4 — register the JNI thunk

In `android/app/src/main/cpp/llama_jni.cpp`, after `JNI_OnLoad` or in the
`loadModel` entry, register the thunk:

```cpp
extern "C" {
static void tq_attn_thunk(
    const float * q, const float * k, const float * v,
    int BH, int n_q, int n_kv, int D,
    float scale, const float * mask, float * out) {
    turboquant::attention_turboquant(
        q, k, v, BH, n_q, n_kv, D, scale, mask, out);
}
}

// In JNI_OnLoad:
llama_set_turboquant_attn_fn(tq_attn_thunk);
```

### Step 5 — force-disable flash attention for kvType=3

`android/app/src/main/cpp/llama_jni.cpp` case 3:

```cpp
case 3:
    cparams.kv_turboquant   = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    LOGI("loadModel: KV cache = TurboQuant (PolarQuant + 1-bit QJL)");
    break;
```

### Step 6 — validation

On-host (no device needed):
- `ctest -R turboquant_attention_parity` — golden parity for the new stateless
  entry vs FP32 reference. Cosine ≥ 0.92 at `BH=8, D=128, key_bits=3`.

On S24 Ultra:
- `adb install` the new APK.
- Load Llama-3.2-1B with kvType=3, send a 200-token prompt, generate 100
  tokens. Acceptance criteria mirror P2.1e:
  - Coherent output (no garbage).
  - First-token cosine vs FP16 ≥ 0.90 (debug-dump path).
  - No NaNs over 1000-token continuation.
- Compare prompt-eval / gen tok/s across kvType ∈ {0, 1, 3} at the same
  prompt. Expect TurboQuant within 20% of q4_0 baseline. Memory savings
  show up in **Step β** below, not yet in α.

## Step β — the memory win (follow-up)

After Step 6 lands and is green:

1. Override `llama_kv_cache_turboquant::cpy_k` / `cpy_v` to extract
   `k_cur`/`v_cur` data and call `TurboQuantKVCache::append()` instead of
   `ggml_set_rows`.
2. The custom op now reads from `TurboQuantKVCache` state (passed via
   userdata) instead of from K/V tensor inputs — those become empty
   placeholders.
3. Re-implement `memory_breakdown()` to return our compressed footprint.
4. Re-implement `state_write()` / `state_read()` to serialise our
   compressed state instead of ggml tensors.
5. Re-test: `dumpsys meminfo com.yzamari.turboquant` should show ≥ 3.5×
   drop in KV-cache RSS at `n_ctx=8192` vs kvType=0.

## Files touched (cheat-sheet)

| Phase | File | Change |
|---|---|---|
| Step 1 | `cpp/include/turboquant/api.hpp` | Add `attention_turboquant()` |
| Step 1 | `cpp/src/attention_turboquant.cpp` (new) | Implementation |
| Step 1 | `cpp/tests/parity_test.cpp` | Golden corpus check |
| Step 2 | `external/llama.cpp/include/llama.h` | `llama_set_turboquant_attn_fn` |
| Step 2 | `external/llama.cpp/src/llama.cpp` | Global state + accessor |
| Step 3 | `external/llama.cpp/src/llama-graph.cpp:1997` | Substitute attention triple |
| Step 3 | `external/llama.cpp/ggml/src/ggml-custom-turboquant.c` (new) | Forward fn |
| Step 3 | `external/llama.cpp/ggml/CMakeLists.txt` or src equivalent | Add new .c |
| Step 4 | `android/app/src/main/cpp/llama_jni.cpp` | Register thunk in JNI_OnLoad |
| Step 5 | `android/app/src/main/cpp/llama_jni.cpp:case 3` | Disable flash |
| Step 6 | `cpp/bench/bench_cli.cpp` | `--check-stateless` flag |
| Step β | `external/llama.cpp/src/llama-kv-cache-turboquant.h` | Override cpy_k/cpy_v + state_*/memory_breakdown |

## Estimated effort

- Step 1 (libturboquant stateless attention): **2-3 days** — algorithm port
  from existing `TurboQuantKVCache` impl, no new math.
- Step 2 (C-API hook): **half day** — pure plumbing.
- Step 3 (custom ggml op + graph substitution): **3-4 days** — most of
  the risk is here; tensor-permutation correctness matters.
- Step 4 (JNI thunk): **1 hour**.
- Step 5 (force flash off): **15 min**.
- Step 6 (on-device validation): **1 day** — assuming no surprises.
- Step β (true memory win, override `cpy_k`/`cpy_v` + state ops): **5-7 days**.

**Total: ~3 weeks for steps 1-6 (algorithm running, not yet memory-winning).
Add 1 week for step β (memory win). 4 weeks all-in.**

## Tripwires (things that bit us before — watch out)

- **GQA fold direction.** Llama-3.2-1B has `n_head_kv=8`, `n_head=32`. When
  permuting Q to pair with K heads, ratio is `n_head/n_head_kv = 4`. Off-by-
  factor-of-N here gives garbled but non-NaN output.
- **Mask additive vs multiplicative.** ggml's mask is added pre-softmax with
  `-INF` for masked positions. Don't multiply — softmax handles the sign.
- **F16 vs F32.** ggml tensors might be F16 depending on the model and
  `cparams.type_k/v`. The custom op should check `tensor->type` and
  upcast to F32 before calling `attention_turboquant`.
- **`ith == 0` only.** First cut is single-threaded inside the custom op.
  The dispatch system will still call it `nth` times — guard with `if (ith
  != 0) return;`. Multi-threading the op is its own optimisation pass.
- **Flash attention silent enable.** `cparams.flash_attn_type = AUTO` may
  enable flash even when we set the cache type. Force DISABLED for kvType=3.
- **Reverse-engineering K storage.** llama.cpp may store K with `v_trans`
  (transposed V). Check `cparams.flash_attn` — when ON, V is stored
  contiguously as `[n_kv, n_embd_head_v]` instead of transposed. Our
  custom op is for the non-flash path, so V should be `[D, n_kv,
  n_head_kv]` — but verify.

## When this is done

Update `README.md` "Headline benchmarks" with a third column:
`FP16 baseline | q4_0 (TurboQuant cousin) | TurboQuant native`.
Update `docs/linkedin-post.md` to drop the "Path-2 ... is the next
milestone" caveat. Tag `v0.2.0-path2`. Open upstream PR to
`ggml-org/llama.cpp` proposing `llama_set_turboquant_attn_fn` as a
generic hook (sell it as the mechanism for any third-party KV-quant
research, not specifically TurboQuant).
