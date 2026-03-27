#!/usr/bin/env python3
"""
04 — TurboQuant KV Cache: End-to-End Demo
==========================================

Full demo of the TurboQuantKVCache class:

  1. Create a cache with realistic LLM dimensions
  2. Prefill with KV pairs (simulating prompt processing)
  3. Append decode tokens (simulating generation)
  4. Compute attention scores
  5. Show memory savings: FP16 vs TurboQuant at 2/3/4 bits
  6. Needle-in-a-haystack test: insert a distinctive vector and verify
     it receives the highest attention score

Run:
    python notebooks/04_kv_cache_demo.py
"""

import sys, os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turboquant_mac import TurboQuantKVCache
from turboquant_mac.backends import get_backend

SEPARATOR = "=" * 72
B = get_backend()

# -------------------------------------------------------------------------
# Configuration (realistic LLM dimensions)
# -------------------------------------------------------------------------
BATCH = 1
N_HEADS = 8
HEAD_DIM = 128
SEQ_LEN = 512       # prompt length
N_DECODE = 10       # generation tokens
BUFFER_SIZE = 64    # recent tokens kept unquantized

rng = np.random.RandomState(42)

# -------------------------------------------------------------------------
# 1.  Create and prefill the KV cache
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 1: Create and prefill KV cache")
print(SEPARATOR)
print()
print(f"  Config: batch={BATCH}, n_heads={N_HEADS}, head_dim={HEAD_DIM}")
print(f"  Prompt length:  {SEQ_LEN} tokens")
print(f"  Buffer size:    {BUFFER_SIZE} tokens (kept in FP32)")
print(f"  Key bits: 3  (2-bit MSE + 1-bit QJL)")
print(f"  Value bits: 2 (asymmetric group quantization)")
print()

cache = TurboQuantKVCache(
    head_dim=HEAD_DIM,
    key_bits=3,
    value_bits=2,
    value_group_size=32,
    buffer_size=BUFFER_SIZE,
    layer_idx=0,
)

# Generate prefill KV states
keys_np = rng.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)
values_np = rng.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)

cache.prefill(B.from_numpy(keys_np), B.from_numpy(values_np))

print(f"  After prefill:")
print(f"    Total seq length:    {cache.get_seq_length()}")
print(f"    Quantized tokens:    {SEQ_LEN - BUFFER_SIZE}")
print(f"    Buffer tokens:       {BUFFER_SIZE}")
print(f"    key_quantized:       {'present' if cache.key_quantized is not None else 'None'}")
print(f"    value_quantized:     {'present' if cache.value_quantized is not None else 'None'}")
print()

# -------------------------------------------------------------------------
# 2.  Append decode tokens
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 2: Append decode tokens (generation phase)")
print(SEPARATOR)
print()

for i in range(N_DECODE):
    new_key = B.from_numpy(rng.randn(BATCH, N_HEADS, 1, HEAD_DIM).astype(np.float32))
    new_val = B.from_numpy(rng.randn(BATCH, N_HEADS, 1, HEAD_DIM).astype(np.float32))
    cache.append(new_key, new_val)

print(f"  Appended {N_DECODE} decode tokens.")
print(f"  Total seq length now: {cache.get_seq_length()}")
print(f"  Buffer size: {cache.key_buffer.shape[-2]} tokens")
print()

# -------------------------------------------------------------------------
# 3.  Compute attention scores
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 3: Compute attention scores")
print(SEPARATOR)
print()

query_np = rng.randn(BATCH, N_HEADS, 1, HEAD_DIM).astype(np.float32)
query_t = B.from_numpy(query_np)

scores = cache.attention_scores(query_t)
scores_np = B.to_numpy(scores)

print(f"  Query shape:  (1, {N_HEADS}, 1, {HEAD_DIM})")
print(f"  Scores shape: {scores_np.shape}  (batch, heads, n_q, seq_len)")
print()

# Show score statistics for head 0
s = scores_np[0, 0, 0, :]  # (seq_len + n_decode,)
print(f"  Head 0 score statistics:")
print(f"    min   = {np.min(s):.4f}")
print(f"    max   = {np.max(s):.4f}")
print(f"    mean  = {np.mean(s):.4f}")
print(f"    std   = {np.std(s):.4f}")
print()

# Softmax attention weights
scores_shifted = scores_np - np.max(scores_np, axis=-1, keepdims=True)
attn_weights = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=-1, keepdims=True)
print(f"  Attention weight statistics (head 0, softmax):")
print(f"    max weight    = {np.max(attn_weights[0, 0, 0]):.6f}")
print(f"    min weight    = {np.min(attn_weights[0, 0, 0]):.6f}")
print(f"    entropy       = {-np.sum(attn_weights[0,0,0] * np.log(attn_weights[0,0,0] + 1e-20)):.2f}")
print(f"    (uniform would be {np.log(scores_np.shape[-1]):.2f})")
print()

# -------------------------------------------------------------------------
# 4.  Memory savings table
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 4: Memory savings — FP16 vs TurboQuant")
print(SEPARATOR)
print()

total_tokens = cache.get_seq_length()
fp16_per_token = 2 * HEAD_DIM * 2  # key + value, each head_dim * 2 bytes (FP16)
fp16_total = fp16_per_token * total_tokens * N_HEADS * BATCH

mem = cache.memory_bytes()
tq_total = mem["total"]

print(f"  Scenario: {total_tokens} tokens, {N_HEADS} heads, head_dim={HEAD_DIM}")
print()
print(f"  {'Method':<28}  {'Bytes':>12}  {'KB':>8}  {'Compression':>12}")
print(f"  {'-'*28}  {'-'*12}  {'-'*8}  {'-'*12}")
print(f"  {'FP16 (baseline)':<28}  {fp16_total:>12,}  {fp16_total/1024:>8.1f}  {'1.0x':>12}")
print(f"  {'TQ (3b key + 2b val)':<28}  {tq_total:>12,}  {tq_total/1024:>8.1f}  "
      f"{fp16_total/max(tq_total,1):>11.1f}x")
print()

# Show breakdown
print(f"  Breakdown:")
print(f"    Quantized keys:   {mem['quantized_keys']:>10,} bytes")
print(f"    Quantized values: {mem['quantized_values']:>10,} bytes")
print(f"    FP32 buffer:      {mem['buffer']:>10,} bytes")
print()

# Theoretical savings at different bit widths and sequence lengths
print(f"  Theoretical savings at different configurations:")
print(f"  (for {N_HEADS} heads x {HEAD_DIM} head_dim, {BATCH} batch)")
print()
print(f"  {'seq_len':>8}  {'FP16 KB':>8}  {'TQ-2b KB':>9}  {'TQ-3b KB':>9}  {'TQ-4b KB':>9}")
print(f"  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")

for seq_len in [512, 1024, 4096, 16384, 65536]:
    fp16_kb = seq_len * N_HEADS * HEAD_DIM * 2 * 2 / 1024  # KV * FP16
    # Approximate TQ bytes per token per head:
    #   key: b bits/coord packed + norms overhead
    #   value: v bits/coord packed + group scales
    for key_bits, label in [(2, "TQ-2b"), (3, "TQ-3b"), (4, "TQ-4b")]:
        # Key: MSE at (key_bits-1) bits + QJL 1 bit = key_bits bits/coord
        #   packed: key_bits * HEAD_DIM / 8 bytes + ~4 bytes norms
        #   value: 2 bits * HEAD_DIM / 8 + group overhead
        key_bytes = (key_bits * HEAD_DIM) / 8 + 4  # packed indices + norms
        val_bytes = (2 * HEAD_DIM) / 8 + (HEAD_DIM // 32) * 4  # packed + scales/zeros
        tq_bytes_per_token = key_bytes + val_bytes
        n_quant = max(0, seq_len - BUFFER_SIZE)
        n_buffer = min(seq_len, BUFFER_SIZE)
        tq_kb = (n_quant * N_HEADS * tq_bytes_per_token +
                 n_buffer * N_HEADS * HEAD_DIM * 2 * 2) / 1024
        if key_bits == 2:
            line = f"  {seq_len:>8}  {fp16_kb:>8.1f}"
        line += f"  {tq_kb:>9.1f}"
    print(line)

print()

# -------------------------------------------------------------------------
# 5.  Full attention output
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 5: Full attention output")
print(SEPARATOR)
print()

# Compute softmax weights
attn_t = B.from_numpy(attn_weights.astype(np.float32))
output = cache.attend(attn_t)
output_np = B.to_numpy(output)

print(f"  Output shape: {output_np.shape}  (batch, heads, n_q, head_dim)")
print(f"  Output norm (head 0): {np.linalg.norm(output_np[0, 0, 0]):.4f}")
print()

# -------------------------------------------------------------------------
# 6.  Needle-in-a-haystack test
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 6: Needle-in-a-haystack test")
print(SEPARATOR)
print()
print("Insert a distinctive 'needle' key that is highly correlated with the")
print("query.  After quantization, verify it still gets the highest attention.")
print()

# Create a fresh cache
cache2 = TurboQuantKVCache(
    head_dim=HEAD_DIM, key_bits=3, value_bits=2,
    value_group_size=32, buffer_size=32, layer_idx=1,
)

# Generate haystack
haystack_len = 256
haystack_keys = rng.randn(BATCH, N_HEADS, haystack_len, HEAD_DIM).astype(np.float32)
haystack_vals = rng.randn(BATCH, N_HEADS, haystack_len, HEAD_DIM).astype(np.float32)

# Create a query and a needle that's highly correlated with it
needle_query = rng.randn(BATCH, N_HEADS, 1, HEAD_DIM).astype(np.float32)
# Needle = scaled query + small noise (very high correlation)
needle_key = needle_query * 5.0 + rng.randn(BATCH, N_HEADS, 1, HEAD_DIM).astype(np.float32) * 0.1
needle_val = rng.randn(BATCH, N_HEADS, 1, HEAD_DIM).astype(np.float32)

# Insert needle at a random position in the haystack
needle_pos = 73  # arbitrary position in the quantized region
all_keys = np.concatenate([
    haystack_keys[:, :, :needle_pos, :],
    needle_key,
    haystack_keys[:, :, needle_pos:, :],
], axis=2)
all_vals = np.concatenate([
    haystack_vals[:, :, :needle_pos, :],
    needle_val,
    haystack_vals[:, :, needle_pos:, :],
], axis=2)

cache2.prefill(B.from_numpy(all_keys), B.from_numpy(all_vals))

# Compute attention scores
scores2 = cache2.attention_scores(B.from_numpy(needle_query))
scores2_np = B.to_numpy(scores2)

# Check if needle gets highest score across all heads
print(f"  Haystack: {haystack_len} random tokens")
print(f"  Needle at position: {needle_pos}")
print(f"  Total tokens: {all_keys.shape[2]}")
print()

all_passed = True
for head in range(N_HEADS):
    head_scores = scores2_np[0, head, 0, :]
    top_pos = np.argmax(head_scores)
    needle_score = head_scores[needle_pos]
    max_score = head_scores[top_pos]

    # Check top-5 positions
    top5_pos = np.argsort(-head_scores)[:5]
    needle_in_top5 = needle_pos in top5_pos
    needle_rank = np.where(np.argsort(-head_scores) == needle_pos)[0][0] + 1

    status = "PASS" if needle_rank <= 3 else "WARN"
    if needle_rank > 5:
        status = "FAIL"
        all_passed = False

    print(f"  Head {head}: needle_rank={needle_rank:>3}, "
          f"needle_score={needle_score:>8.2f}, "
          f"max_score={max_score:>8.2f}, "
          f"in_top5={needle_in_top5}  [{status}]")

print()
if all_passed:
    print("  RESULT: All heads found the needle in top-5 positions.")
    print("  TurboQuant preserves attention ranking even after quantization!")
else:
    print("  RESULT: Some heads did not rank the needle in top-5.")
    print("  (This can happen at low bit widths with random data; in real LLM")
    print("  data the correlations are stronger and retrieval is better.)")
print()

# -------------------------------------------------------------------------
# 7.  Score correlation analysis
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 7: Score correlation — quantized vs FP32")
print(SEPARATOR)
print()

# Compute true FP32 attention scores for comparison
# (Using the same key data, but unquantized)
true_scores_np = np.matmul(
    needle_query,
    all_keys.transpose(0, 1, 3, 2)
) / math.sqrt(HEAD_DIM)  # (batch, heads, 1, seq_len)

# Quantized scores (already have them, but need to apply scale)
quant_scores_np = scores2_np  # already scaled in attention_scores()

for head in [0, 3, 7]:
    fp32_s = true_scores_np[0, head, 0, :]
    quant_s = quant_scores_np[0, head, 0, :]
    corr = np.corrcoef(fp32_s, quant_s)[0, 1]
    print(f"  Head {head}: Pearson correlation(FP32, TQ) = {corr:.6f}")

print()
print("Correlation close to 1.0 means the ranking of tokens is well preserved.")
print()
print("Done!  All sections completed successfully.")
print(SEPARATOR)
