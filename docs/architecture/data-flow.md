# Per-Kernel Data Flow

The four hot kernels of the C++ port, one ASCII diagram each, with input
shapes / dtypes, the transformation in pseudo-prose, and output shapes.
These mirror the Python Metal sources under
`src/turboquant_mac/backends/metal/` (the literal porting spec) and the
`IBackend` method signatures in `cpp/include/turboquant/backend.hpp`.

If the Python source and these diagrams ever disagree, **the Python source
wins** — it's the authority. Update this doc.

## Conventions

- All arrays are row-major.
- `BH` = batched heads (folded `B * H`).
- `N` = number of KV tokens (sequence length).
- `D` = head dim (typically 128 for our sweep).
- `b` = bits per quantized value (typically 3 for keys, 2 for values).
- `VPB` = "values per byte" = `8 / eff_bits` for `b ∈ {1,2,4,8}`. For
  `b = 3`, we pack into `uint32` words at 10 values per word — see the
  per-kernel notes.

## 1. `mse_encode` — fused searchsorted + bit-pack

Mirrors `src/turboquant_mac/backends/metal/mse_encode.py`. Used at
**prefill** to compress new keys. One thread per (batch, byte/word).

```
  inputs:
     rotated     f32[N, D]              ← the keys, after Q@Pi^T
     boundaries  f32[2^b - 1]           ← interior decision boundaries from
                                          the codebook (no ±1 padding)
     N, D, b     ints                   ← shape + bits per value
                                          (kernel is templated on b)

  per-thread (b ∈ {1,2,4,8} fast path):
       byte_idx  = thread_position.x          ← in [0, ⌈D·b/8⌉)
       batch_idx = thread_position.y          ← in [0, N)
       packed_byte = 0
       for sub in 0..VPB-1:
           coord = byte_idx * VPB + sub
           if coord < D:
               val = rotated[batch_idx, coord]
               idx = count of (val >= boundaries[k] for k in 0..2^b-2)
                     ──── i.e. searchsorted(boundaries, val, right-inclusive)
               packed_byte |= (idx << (sub * eff_bits))   ← LSB-first within byte
       out[batch_idx, byte_idx] = packed_byte

  per-thread (b == 3 special path, uint32 packing):
       word_idx  = thread_position.x          ← in [0, ⌈D / 10⌉)
       packed = 0
       for sub in 0..9:
           coord = word_idx * 10 + sub
           if coord < D:
               idx = searchsorted(boundaries, rotated[batch_idx, coord])
               packed |= (idx << (sub * 3))         ← 30 of 32 bits used
       out_u32[batch_idx, word_idx] = packed

  output:
     packed      u8 [N, ⌈D · b / 8⌉]    for b ∈ {1,2,4,8}
                  or
                 u32[N, ⌈D / 10⌉]       for b == 3
```

Why this shape: the fused kernel writes one packed byte/word, so the
launch grid is `(⌈D·b/8⌉ or ⌈D/10⌉, N)` — small in the inner dim, wide in
the outer dim. Adreno work-group `(64, 1, 1)` is the sweet spot.

> **C++ port note.** The reference packing is in `cpp/src/packing.cpp` per
> the plan's "Critical Files to Reference" table, and the bit-pack is
> **little-endian within each byte** (LSB = element 0). The QJL sign-pack
> uses the same convention.

## 2. `mse_score` — fused dequant + dot product

Mirrors `src/turboquant_mac/backends/metal/mse_score.py`. Used during
**decode** to compute the MSE-quantized contribution to attention scores.
One thread per `(batch_head, token)`.

```
  inputs:
     q_rot       f32[BH, D]                   ← rotated query  (Q @ Pi^T)
     packed      u8 [BH, N, ⌈D·b/8⌉]          ← from mse_encode
                  or u32[BH, N, ⌈D/10⌉]       ← for b==3
     norms       f32[BH, N]                   ← original key vector norms
     centroids   f32[2^b]                     ← codebook decode table
     BH, N, D, b ints

  per-thread:
       n  = thread_position.x   ← which KV token
       bh = thread_position.y   ← which batch-head
       score = 0.0
       for byte_or_word_idx in 0..PACKED_D-1:
           p = packed[bh, n, byte_or_word_idx]
           for sub in 0..VPB-1 (or 0..9 for b==3):
               coord = byte_or_word_idx * VPB + sub
               if coord < D:
                   idx = (p >> (sub * eff_bits)) & MASK
                   score += q_rot[bh, coord] * centroids[idx]
       out[bh, n] = score * norms[bh, n]    ← rescale by original key norm

  output:
     scores      f32[BH, N]                   ← MSE-quantized attention
                                                contribution; the QJL kernel
                                                will ADD residual correction
```

This is the kernel where **HMX (Hexagon Tensor Core) wins big.** The
inner double loop is exactly an INT-indexed gather + FP16 GEMM. Our QNN
graph models it as `Gather(centroids, indices) → MatMul(., q_rot)`, and
HMX runs that in a single packet. On NEON the same loop is a 4-lane
FMA inner loop; on Adreno it's a similarly tight loop with the centroid
table in `__constant` memory.

## 3. `qjl_score` — bit-unpack signs and accumulate residual

Mirrors `src/turboquant_mac/backends/metal/qjl_score.py`. Used during
**decode** immediately after `mse_score`; adds the QJL residual correction
to the existing MSE scores.

```
  inputs:
     q_sketch       f32[BH, D]              ← Q @ S  (the QJL sketch
                                              of the query, separate
                                              rotation from Pi)
     signs          u8 [BH, N, ⌈D/8⌉]       ← packed sign bits from
                                              QJL encode (1 → +1, 0 → -1)
     res_norms      f32[BH, N]              ← per-token residual norms
                                              from QJL encode
     mse_scores_in  f32[BH, N]              ← output of mse_score above
     BH, N, D       ints
     qjl_scale      f32                     ← sqrt(pi/2) / D   (compile-time)

  per-thread:
       n  = thread_position.x
       bh = thread_position.y
       dot = 0.0
       for byte_idx in 0..⌈D/8⌉-1:
           p = signs[bh, n, byte_idx]
           for bit in 0..7:
               coord = byte_idx * 8 + bit
               if coord < D:
                   sign_bit = (p >> bit) & 1
                   sign_val = (sign_bit == 1) ? +1.0f : -1.0f
                   dot += q_sketch[bh, coord] * sign_val
       out[bh, n] = mse_scores_in[bh, n] + dot * res_norms[bh, n] * qjl_scale

  output:
     scores      f32[BH, N]                 ← MSE + QJL combined attention
                                              scores (still pre-softmax)
```

We keep this kernel on **NEON**, not HTP, in v1. Reasons:

1. The bit-unpack-then-fma is `vbslq_f32` on NEON — one instruction
   materializes ±1.0 from a sign byte. HVX can also do it but only via
   a UDO; not worth a custom op for a small data block.
2. `D = 128`, so each thread does 16 byte-loads + 128 fma. Total work is
   ~few µs on NEON. Sending it to HTP costs more in graph-launch overhead
   than the kernel takes.

> **Sign mapping correctness.** Bit `1` → `+1.0f`, bit `0` → `-1.0f`.
> NOT the reverse. Listed in the plan's "Critical constants" section
> precisely because it is the most common porting bug.

## 4. `value_dequant` — extract + dequant + affine

Mirrors `src/turboquant_mac/backends/metal/value_dequant.py`. Used during
**decode** when the softmaxed weights are applied to the value cache. One
thread per `(batch, coord)`.

```
  inputs:
     packed     u8 [N, ⌈D · b_v / 8⌉]      ← group-quantized values
                                            (b_v typically 2 → VPB=4)
     scales     f32[N, D / group_size]    ← per-group quant scale
     zeros      f32[N, D / group_size]    ← per-group quant zero (asymmetric)
     N, D       ints
     b_v        bits per value            (kernel templated on b_v)
     group_size ints                      (typically 32)

  per-thread:
       coord     = thread_position.x      ← in [0, D)
       batch_idx = thread_position.y      ← in [0, N)
       byte_idx  = coord / VPB
       sub       = coord % VPB
       group_idx = coord / group_size

       packed_byte = packed[batch_idx, byte_idx]
       qval = (packed_byte >> (sub * eff_bits)) & MASK
       result = (float)qval * scales[batch_idx, group_idx]
                          + zeros [batch_idx, group_idx]
       out[batch_idx, coord] = result

  output:
     values    f32[N, D]                  ← reconstructed values; fed into
                                            the weighted sum that produces
                                            the attention output
```

Group quant trade-off: smaller `group_size` (e.g. 32) → better accuracy,
more `scales`/`zeros` overhead. We use 32 by default. The cost is
`N * (D / 32) * 2 * sizeof(float)` extra bytes per token — for D=128, that's
8 floats × 8 bytes = 32 bytes/token, on top of the 32 packed bytes
(`128 * 2 / 8 = 32`). So the FP16-equivalent compression ratio is
`128 * 2 / (32 + 32) = 4×` for value cache at b=2.

## 5. The fifth op: `rotate`

Not a Metal kernel of its own (it's an `mx.matmul` in the Python path),
but `IBackend::rotate` is a separate method because **on Hexagon HTP this
is the single most expensive op** and we want to dispatch it explicitly.

```
  inputs:
     in   f32[n, D]      ← the K or V tensor to rotate
     Pi   f32[D, D]      ← rotation matrix (QR-orthogonal + WHT init)
     n    int            ← varies: BH (during decode) or BH*N (during prefill)
     D    int

  computation:
     out = in @ Pi^T

  output:
     out  f32[n, D]      ← rotated tensor
```

Backend-specific notes:

| Backend | Implementation |
|---|---|
| `cpu_scalar` | triple-nested loop, FP32 |
| `cpu_neon` | 4-lane SGEMV with `vfmaq_f32`; for `n=1` (decode) it's a single SGEMV; for `n=BH*N` (prefill) it's a SGEMM |
| `qnn_htp` | QNN graph with single `MatMul` op; FP16 input/output via `cast` ops at the boundary |
| `opencl` | `__kernel rotate_qkt(...)` — **not** named `rotate` because OpenCL reserves that identifier (see `qualcomm/adreno-gpu.md` § 4.4) |
| `vulkan` | `comp` shader; subgroup reductions for the inner-product accumulation |

## 6. How the kernels chain at decode time

The full decode-time chain looks like this (one decode step, single
query token):

```
  Q[BH, 1, D]
        │
        ▼
   ┌─────────┐  ┌─────────┐
   │ rotate  │  │ rotate  │     (runs twice: Q@Pi^T and Q@S, on the same
   │ Q@Pi^T  │  │ Q@S     │      backend; output of #1 becomes q_rot,
   └─────────┘  └─────────┘      output of #2 becomes q_sketch)
        │            │
        ▼            ▼
     q_rot[BH,D]   q_sketch[BH,D]
        │            │
        │            │
        ▼            │
  ┌──────────┐       │
  │mse_score │       │     ─── reads packed mse_keys[BH,N,...], norms, codebook
  └──────────┘       │
        │            │
        ▼            ▼
  scores[BH,N] ─► ┌──────────┐
                  │qjl_score │    ─── reads packed signs[BH,N,⌈D/8⌉],
                  │          │        res_norms, scale; ADDS to scores
                  └──────────┘
                       │
                       ▼
                 scores'[BH,N]   ─── pre-softmax attention scores
                       │
                       ▼
                  softmax (host)
                       │
                       ▼
                 weights[BH,N]
                       │
                       ▼
                ┌───────────────┐
                │ value_dequant │  ─── reads packed values, scales, zeros
                └───────────────┘
                       │
                       ▼
                 V_dequant[N,D]
                       │
                       ▼
                weighted sum (host or as a GEMM on backend)
                       │
                       ▼
                 out[BH, 1, D]
```

For the prefill side and the recent-token buffer, see
[`kv-cache-flow.md`](kv-cache-flow.md).
