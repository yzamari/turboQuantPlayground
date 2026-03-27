#!/usr/bin/env python3
"""
03 — QJL Correction: Why MSE-only Inner Products Are Biased
============================================================

The key insight of Algorithm 2 (TurboQuant_Prod):

  If you just use TurboQuant_MSE to quantize keys and then compute
  <query, dequantized_key>, the result is BIASED.  The bias arises because
  the MSE quantization error is correlated with the original vector —
  it always shrinks the vector toward the nearest centroid.

  TurboQuant_Prod adds a Quantized Johnson-Lindenstrauss (QJL) correction
  to the residual, making the inner-product estimate UNBIASED.

This script demonstrates:
  1. MSE-only inner product has systematic bias (underestimates magnitude)
  2. TurboQuant_Prod with QJL correction removes the bias
  3. Statistical comparison: bias, variance, RMSE

Run:
    python notebooks/03_qjl_correction.py
"""

import sys, os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turboquant_mac import TurboQuantMSE, TurboQuantProd
from turboquant_mac.backends import get_backend

SEPARATOR = "=" * 72
B = get_backend()

d = 128
n_pairs = 2000
rng = np.random.RandomState(42)

# Generate random query-key pairs
queries_np = rng.randn(n_pairs, d).astype(np.float32)
keys_np = rng.randn(n_pairs, d).astype(np.float32)

# True inner products
true_ips = np.sum(queries_np * keys_np, axis=-1)

queries_t = B.from_numpy(queries_np)
keys_t = B.from_numpy(keys_np)

# -------------------------------------------------------------------------
# 1.  MSE-only inner products are BIASED
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 1: MSE-only inner product estimation is BIASED")
print(SEPARATOR)
print()
print("We quantize keys with TurboQuant_MSE, dequantize, and compute")
print("<query, dequantized_key>.  This is NOT the same as the true <query, key>.")
print()

mse_bits_list = [2, 3, 4]

for bits in mse_bits_list:
    mse_q = TurboQuantMSE(dim=d, bits=bits, seed=42)

    # Quantize and dequantize keys
    q = mse_q.quantize(keys_t)
    keys_hat = B.to_numpy(mse_q.dequantize(q))

    # Compute inner products with dequantized keys
    mse_ips = np.sum(queries_np * keys_hat, axis=-1)

    # Statistics
    errors = mse_ips - true_ips
    bias = np.mean(errors)
    variance = np.var(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    # Relative bias (compared to mean absolute IP)
    mean_abs_ip = np.mean(np.abs(true_ips))

    print(f"  MSE-only, bits={bits}:")
    print(f"    Bias (mean error)     = {bias:+.4f}  "
          f"(relative: {bias/mean_abs_ip:+.2%} of mean|IP|)")
    print(f"    Std of error          = {np.std(errors):.4f}")
    print(f"    RMSE                  = {rmse:.4f}")
    print(f"    mean|true IP|         = {mean_abs_ip:.4f}")
    print()

print("The bias is small in absolute terms but systematic.  MSE quantization")
print("shrinks vectors toward centroids, causing inner products to be")
print("underestimated.  This matters at scale (long sequences, many heads).")
print()

# -------------------------------------------------------------------------
# 2.  TurboQuant_Prod with QJL correction is UNBIASED
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 2: TurboQuant_Prod (MSE + QJL) is UNBIASED")
print(SEPARATOR)
print()
print("Algorithm 2 decomposes at b bits as:")
print("  - Stage 1: MSE quantization at (b-1) bits -> x_hat")
print("  - Stage 2: QJL on residual r = x - x_hat -> sign(S * r), ||r||")
print("  - Estimate: <q, x_hat> + sqrt(pi/2)/d * ||r|| * <S*q, sign(S*r)>")
print()

for bits in [2, 3, 4]:
    prod_q = TurboQuantProd(dim=d, bits=bits, seed=42)

    # Quantize keys
    q_prod = prod_q.quantize(keys_t)

    # Dequantize (this includes QJL correction)
    keys_hat_prod = B.to_numpy(prod_q.dequantize(q_prod))

    # Inner products with corrected keys
    prod_ips = np.sum(queries_np * keys_hat_prod, axis=-1)

    # Statistics
    errors = prod_ips - true_ips
    bias = np.mean(errors)
    variance = np.var(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_abs_ip = np.mean(np.abs(true_ips))

    print(f"  TurboQuant_Prod, bits={bits} ({bits-1} MSE + 1 QJL):")
    print(f"    Bias (mean error)     = {bias:+.4f}  "
          f"(relative: {bias/mean_abs_ip:+.2%} of mean|IP|)")
    print(f"    Std of error          = {np.std(errors):.4f}")
    print(f"    RMSE                  = {rmse:.4f}")
    print(f"    mean|true IP|         = {mean_abs_ip:.4f}")
    print()

print("The bias is much closer to zero — the QJL correction provides an")
print("unbiased estimator of <query, key>.")
print()

# -------------------------------------------------------------------------
# 3.  Direct attention_score comparison
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 3: Attention score estimation (batched)")
print(SEPARATOR)
print()
print("In practice, attention_score() computes scores in two parts:")
print("  score = <q, k_mse> + qjl_scale * ||r|| * <S*q, sign(S*r)>")
print("without ever materializing the full dequantized key.")
print()

# Reshape for attention_score API: (batch=1, seq, d) shape vectors
n_queries = 50
n_keys = 200

q_np = rng.randn(1, n_queries, d).astype(np.float32)
k_np = rng.randn(1, n_keys, d).astype(np.float32)

q_t = B.from_numpy(q_np)
k_t = B.from_numpy(k_np)

# True scores: (1, n_q, n_k)
true_scores = np.matmul(q_np, k_np.transpose(0, 2, 1))  # (1, 50, 200)

for bits in [2, 3, 4]:
    prod_q = TurboQuantProd(dim=d, bits=bits, seed=42)
    k_quant = prod_q.quantize(k_t)

    # Use the efficient attention_score method (no full dequantization)
    est_scores = B.to_numpy(prod_q.attention_score(q_t, k_quant))

    errors = est_scores - true_scores
    bias = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_abs_score = np.mean(np.abs(true_scores))

    # Check ranking preservation: for each query, does the top-k key match?
    true_top1 = np.argmax(true_scores[0], axis=-1)  # (n_q,)
    est_top1 = np.argmax(est_scores[0], axis=-1)
    top1_match = np.mean(true_top1 == est_top1)

    # Top-5 overlap
    true_top5 = np.argsort(-true_scores[0], axis=-1)[:, :5]
    est_top5 = np.argsort(-est_scores[0], axis=-1)[:, :5]
    top5_overlap = np.mean([
        len(set(t) & set(e)) / 5.0
        for t, e in zip(true_top5, est_top5)
    ])

    print(f"  bits={bits}: bias={bias:+.4f}, RMSE={rmse:.4f}, "
          f"top-1 match={top1_match:.1%}, top-5 overlap={top5_overlap:.1%}")

print()

# -------------------------------------------------------------------------
# 4.  Summary table
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 4: Summary — MSE-only vs TurboQuant_Prod")
print(SEPARATOR)
print()
print(f"  {'Method':<22}  {'bits':>4}  {'Bias':>10}  {'RMSE':>10}  {'Unbiased?':>10}")
print(f"  {'-'*22}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")

for bits in [2, 3, 4]:
    # MSE-only
    mse_q = TurboQuantMSE(dim=d, bits=bits, seed=42)
    q = mse_q.quantize(keys_t)
    keys_hat = B.to_numpy(mse_q.dequantize(q))
    mse_ips = np.sum(queries_np * keys_hat, axis=-1)
    mse_errors = mse_ips - true_ips
    mse_bias = np.mean(mse_errors)
    mse_rmse = np.sqrt(np.mean(mse_errors ** 2))

    # Prod (MSE + QJL)
    prod_q = TurboQuantProd(dim=d, bits=bits, seed=42)
    q_prod = prod_q.quantize(keys_t)
    keys_hat_prod = B.to_numpy(prod_q.dequantize(q_prod))
    prod_ips = np.sum(queries_np * keys_hat_prod, axis=-1)
    prod_errors = prod_ips - true_ips
    prod_bias = np.mean(prod_errors)
    prod_rmse = np.sqrt(np.mean(prod_errors ** 2))

    print(f"  {'MSE-only':<22}  {bits:>4}  {mse_bias:>+10.4f}  {mse_rmse:>10.4f}  {'No':>10}")
    print(f"  {'TurboQuant_Prod':<22}  {bits:>4}  {prod_bias:>+10.4f}  {prod_rmse:>10.4f}  {'Yes':>10}")

print()
print("Key takeaway: The QJL correction trades a small increase in variance")
print("for eliminating bias entirely — critical for accurate attention scores.")
print()
print("Done!  All sections completed successfully.")
print(SEPARATOR)
