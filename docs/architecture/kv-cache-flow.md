# KV Cache Life Cycle

How keys and values move through the TurboQuant KV cache from prefill, into
the on-device compressed storage, into the per-decode attention flow, and
back out via `attend`. The buffer-flush mechanism for recent (un-quantized)
tokens is included because that's the part most-often misunderstood.

This is the C++ port of `src/turboquant_mac/kv_cache.py:150-272` and
should match it semantically to within the FP16 / FP32 numerical tolerance.

## 1. The two storage tiers

A `TurboQuantKVCache` keeps **two** stores at any moment:

```
   ┌───────────────────────────────────────────────────────┐
   │  TurboQuantKVCache (cpp/src/kv_cache.cpp)             │
   │                                                       │
   │  ┌──────────────────────────────────────────────────┐ │
   │  │   compressed store  ("the quantized portion")    │ │
   │  │                                                  │ │
   │  │   keys:                                          │ │
   │  │     mse_packed     u8[BH, N_q, ⌈D·b/8⌉]          │ │
   │  │     norms          f32[BH, N_q]                  │ │
   │  │     qjl_signs      u8[BH, N_q, ⌈D/8⌉]            │ │
   │  │     res_norms      f32[BH, N_q]                  │ │
   │  │   values:                                        │ │
   │  │     packed_values  u8[BH, N_q, ⌈D·b_v/8⌉]        │ │
   │  │     scales/zeros   f32[BH, N_q, D/group_size]    │ │
   │  └──────────────────────────────────────────────────┘ │
   │                                                       │
   │  ┌──────────────────────────────────────────────────┐ │
   │  │   recent-token buffer ("the FP16 tail")          │ │
   │  │                                                  │ │
   │  │   k_recent  f16[BH, B_max, D]                    │ │
   │  │   v_recent  f16[BH, B_max, D]                    │ │
   │  │   B_cur     int (live count, 0..B_max)           │ │
   │  └──────────────────────────────────────────────────┘ │
   │                                                       │
   │  totals:                                              │
   │     N_total = N_q + B_cur                             │
   └───────────────────────────────────────────────────────┘
```

Why two tiers:

- The **compressed store** has the win: ~3-5× memory reduction and faster
  attention because `mse_score` over packed bytes is much cheaper than
  FP16 GEMM over full keys.
- The **recent buffer** is FP16 because the **most recent few tokens** are
  what the model attends to most often (recency bias in attention) and
  quantizing them adds error exactly where it hurts most. Keeping them in
  FP16 until they "age out" is the standard streaming-quantization trick.

The size `B_max` of the recent buffer is the `cfg.buffer_size` config
field. `0` (the v1 default) effectively disables the recent buffer — every
token is quantized at prefill time. Larger values (32, 128) trade a little
memory for accuracy on short sequences.

## 2. Prefill — keys and values arrive

```
  caller passes:
     keys[BH, N, D]   f32      ← the full sequence at prompt time
     values[BH, N, D] f32

   ┌──────────────────────────────────────────────────────────────────┐
   │  prefill(keys, values, BH, N):                                   │
   │                                                                  │
   │    1. rotate keys → key_rot[BH*N, D] = keys @ Pi^T               │
   │       (one IBackend::rotate call, prefill is the big one)        │
   │                                                                  │
   │    2. for each token: norm = ||key_rot||_2                       │
   │       store norms[BH, N]                                         │
   │       key_rot_unit = key_rot / norm                              │
   │                                                                  │
   │    3. mse_encode(key_rot_unit, boundaries, BH*N, D, b_key)       │
   │       → mse_packed[BH, N, ⌈D·b_key/8⌉]                           │
   │                                                                  │
   │    4. residual = key_rot_unit -                                  │
   │           dequant_centroids(mse_packed)                          │
   │       res_norm = ||residual||_2                                  │
   │       residual_unit = residual / res_norm                        │
   │       qjl_sketch = residual_unit @ S                             │
   │       qjl_signs = pack(sign(qjl_sketch))                         │
   │       store res_norms[BH, N]                                     │
   │                                                                  │
   │    5. for values: per-group min/max → scales, zeros              │
   │       qval = round((value - zero) / scale)  ∈ [0, 2^b_v-1]       │
   │       packed_values = bitpack(qval)                              │
   └──────────────────────────────────────────────────────────────────┘
```

After prefill: the compressed store has `N_q = N` quantized tokens, the
recent buffer is empty.

## 3. Decode — one token in, one token out (with no buffer)

This is the path on the v1 default `cfg.buffer_size = 0`. The recent
buffer is bypassed; every new token gets quantized immediately.

```
  Step:  caller passes Q[BH, n_q, D]   f32   (typically n_q = 1)

   ┌─────────────────────────────────────────────────────────────────────┐
   │  attention_scores(Q, BH, n_q, scale):                               │
   │                                                                     │
   │    1.  rotate Q → q_rot[BH, n_q, D] = Q @ Pi^T                      │
   │        rotate Q → q_sketch[BH, n_q, D] = Q @ S                      │
   │                                                                     │
   │    2.  scores_mse = mse_score(q_rot, mse_packed, norms, codebook)   │
   │        scores     = qjl_score(q_sketch, qjl_signs, res_norms,       │
   │                                  scores_mse, qjl_scale)             │
   │        ─── output:  scores[BH, n_q, N_q]   (pre-softmax)            │
   │                                                                     │
   │    3.  scores *= scale         (= 1 / sqrt(D))                      │
   │                                                                     │
   │  ─── caller does:  weights = softmax(scores)                        │
   │                                                                     │
   │  attend(weights, BH, n_q):                                          │
   │                                                                     │
   │    4.  V_full = value_dequant(packed_values, scales, zeros)         │
   │        out = weights @ V_full   (a GEMM on the backend)             │
   │        ─── output:  out[BH, n_q, D]   f32                           │
   └─────────────────────────────────────────────────────────────────────┘
```

If a new K/V pair from the just-emitted token is appended to the cache,
it goes through the same prefill sub-steps for that one token (`N=1`):
rotate, encode, residual-encode, value-quantize. Cost is sub-millisecond
on every backend.

## 4. Decode with a recent buffer (`buffer_size > 0`)

When `cfg.buffer_size = B_max > 0`, freshly-arrived tokens go into the
**FP16 recent buffer** first; they only get quantized when the buffer
fills and **flushes**.

```
  invariant on entry to attention_scores:
     compressed:   N_q tokens   in mse_packed/qjl_signs/etc.
     recent:       B_cur tokens in k_recent/v_recent (FP16)

  attention scoring becomes a TWO-PART sum:

     ┌────────────────────────────────────────────────┐
     │ part A — compressed (TurboQuant) tokens        │
     │                                                │
     │   scores_q = qjl_score(q_sketch, qjl_signs,   │
     │                  res_norms,                    │
     │                  mse_score(q_rot, mse_packed, │
     │                            norms, codebook))   │
     │   shape:  f32[BH, n_q, N_q]                   │
     └────────────────────────────────────────────────┘

     ┌────────────────────────────────────────────────┐
     │ part B — recent (FP16) tokens                  │
     │                                                │
     │   scores_r = Q @ k_recent^T          (fp16/fp32)
     │   shape:  f32[BH, n_q, B_cur]                 │
     └────────────────────────────────────────────────┘

     concatenate along last dim:
        scores[BH, n_q, N_q + B_cur] = [scores_q | scores_r]
```

Same on the `attend` side: the weights are split, the compressed half is
applied via `value_dequant @ ...`, the recent half via a direct
`weights_r @ v_recent`.

### The buffer flush

```
   pseudo:
      def append(k_new[BH,D], v_new[BH,D]):
          if B_cur < B_max:
              k_recent[:, B_cur, :] = fp16(k_new)
              v_recent[:, B_cur, :] = fp16(v_new)
              B_cur += 1
              return                 ← fast path

          ─── slow path: flush the entire recent buffer into compressed
          flush_keys   = k_recent[:, :B_max, :]   (BH, B_max, D)
          flush_values = v_recent[:, :B_max, :]   (BH, B_max, D)

          ─── re-run the prefill subpipeline on those flush tokens
          rotate, mse_encode, residual-encode, value-quantize
          append the resulting packed bytes / norms / scales / zeros
          to the compressed store at positions [N_q .. N_q + B_max).

          N_q += B_max
          B_cur = 0
          ─── now there's room
          k_recent[:, 0, :] = fp16(k_new)
          v_recent[:, 0, :] = fp16(v_new)
          B_cur = 1
```

The flush is the only place `mse_encode` runs during decode (other than
the trivial 1-token append in the `buffer_size = 0` mode). It's batched —
encoding `B_max` tokens at once is much more efficient than encoding one
at a time.

## 5. Visual summary — full life cycle

```
   t=0  prefill of length S=2048
   ──────────────────────────────────────────────────────────
                         compressed store   recent buffer
                         ───────────────────────────────
                         N_q = 2048         B_cur = 0


   t=1  decode 1 token, buffer_size = 0
   ──────────────────────────────────────────────────────────
                         compressed store   recent buffer
                         ───────────────────────────────
                         N_q = 2049         B_cur = 0


   t=1  decode 1 token, buffer_size = 32   (no flush yet)
   ──────────────────────────────────────────────────────────
                         compressed store   recent buffer
                         ───────────────────────────────
                         N_q = 2048         B_cur = 1   k_recent[0]


   t=33 (32 decodes later, the 33rd token triggers flush)
   ──────────────────────────────────────────────────────────
                         compressed store   recent buffer
                         ───────────────────────────────
                         N_q = 2080         B_cur = 1   k_recent[0]
                                            (the previous 32 got
                                            batch-encoded into the
                                            compressed store)
```

## 6. Where this lives in the codebase

| Component | File |
|---|---|
| `TurboQuantKVCache` C++ class | `cpp/src/kv_cache.cpp` |
| Public config struct | `cpp/include/turboquant/api.hpp` (`TurboQuantKVCache::Config`) |
| Python reference | `src/turboquant_mac/kv_cache.py:150-272` |
| Bench harness wiring | `cpp/bench/bench_runner.hpp` (sets `cfg.buffer_size = 0` for v1) |
| Baseline (FP16) for A/B | `cpp/bench/baseline_kv_cache.{hpp,cpp}` (does **not** have a buffer; it's just a raw FP16 cache) |

## 7. Memory accounting

For a sweep with `BH=8, D=128, N=4096, b_key=3, b_val=2, group_size=32`:

```
   Baseline (FP16):
      keys + values = 2 * BH * N * D * 2 bytes
                    = 2 * 8 * 4096 * 128 * 2
                    = 16 MB    per layer

   TurboQuant compressed:
      mse_packed     = BH * N * ⌈D*b_key/8⌉  (b=3 → ⌈128*3/8⌉ = 48 bytes; *)
                     = 8 * 4096 * 48            = 1.5 MB
      norms          = BH * N * 4 bytes         = 128 KB
      qjl_signs      = BH * N * (D/8)            = 8 * 4096 * 16 = 512 KB
      res_norms      = BH * N * 4 bytes          = 128 KB
      packed_values  = BH * N * ⌈D*b_v/8⌉ (b=2 → 32 bytes)
                     = 8 * 4096 * 32             = 1.0 MB
      scales+zeros   = BH * N * (D/32) * 2 * 4   = 8 * 4096 * 4 * 8 = 1.0 MB
                                                   ─────────────────
                                                   ≈ 4.3 MB

   compression ratio ≈ 16 / 4.3 ≈ 3.7×

   * for the b=3 u32 path, packed_d = ⌈D/10⌉ * 4 bytes =  ⌈128/10⌉ * 4 = 52 bytes
     so the 3-bit u32 path is slightly larger than the byte-packing math
     above. That overhead (8% per token) buys substantially faster mse_score
     and mse_encode on the GPU/HTP paths and is the right trade for the
     decode-bound regime; see qualcomm/hexagon-htp.md and qualcomm/adreno-gpu.md.
```

The plan's pass criterion of `compression_ratio ≥ 3.0×` at
`seq_len=1024, bits=3` is comfortably above this floor for every backend
because the per-token overhead (`norms`, `res_norms`, `scales`, `zeros`) is
amortized across many tokens and the per-token `D`-sized payload dominates
at long contexts.
