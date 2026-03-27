#!/usr/bin/env python3
"""
02 — Random Rotation and Coordinate Quantization
==================================================

Step-by-step walkthrough of Algorithm 1 (TurboQuant_MSE):

  1. Generate a random orthogonal matrix Pi via QR decomposition
  2. Verify orthogonality (Pi @ Pi^T == I)
  3. Rotate random vectors and observe the coordinate distribution narrows
  4. Quantize each coordinate using the Lloyd-Max codebook
  5. Dequantize (look up centroids, rotate back) and measure reconstruction error
  6. Show MSE distortion at each bit width

Run:
    python notebooks/02_rotation_and_quantization.py
"""

import sys, os, warnings
import numpy as np

# Suppress numpy overflow warnings when computing det of large float32 matrices
warnings.filterwarnings("ignore", message=".*encountered in det.*", category=RuntimeWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turboquant_mac.rotation import generate_rotation_matrix
from turboquant_mac.codebook import get_codebook
from turboquant_mac import TurboQuantMSE
from turboquant_mac.backends import get_backend

SEPARATOR = "=" * 72
B = get_backend()

# -------------------------------------------------------------------------
# 1.  Generate a rotation matrix and verify orthogonality
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 1: Generate rotation matrix Pi (d=128)")
print(SEPARATOR)
print()

d = 128
Pi = generate_rotation_matrix(d, seed=42)
print(f"  Pi shape: {Pi.shape}, dtype: {Pi.dtype}")

# Check orthogonality: Pi @ Pi^T should be close to I
I_approx = Pi @ Pi.T
max_off_diag = np.max(np.abs(I_approx - np.eye(d)))
print(f"  max|Pi @ Pi^T - I| = {max_off_diag:.2e}")

# Check determinant (should be +/-1 for orthogonal matrix)
# Compute in float64 to avoid overflow for large d
det = np.linalg.det(Pi.astype(np.float64))
print(f"  det(Pi) = {det:+.6f}")

if max_off_diag < 1e-4:
    print("  --> PASS: Pi is orthogonal")
else:
    print("  --> FAIL: Pi is NOT orthogonal")
if abs(abs(det) - 1.0) < 1e-4:
    print("  --> PASS: |det| = 1 (proper rotation/reflection)")
print()

# -------------------------------------------------------------------------
# 2.  Rotate random vectors, observe coordinate distribution
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 2: How rotation affects coordinate distribution")
print(SEPARATOR)
print()

rng = np.random.RandomState(123)
n_samples = 5000

# Generate random unit vectors
x_raw = rng.randn(n_samples, d).astype(np.float32)
x_unit = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)

print("Before rotation (unit vectors, all coordinates):")
print(f"  mean  = {np.mean(x_unit):+.6f}  (expected ~0)")
print(f"  std   = {np.std(x_unit):.6f}  (expected ~{1/np.sqrt(d):.6f} = 1/sqrt({d}))")
print(f"  min   = {np.min(x_unit):+.6f}")
print(f"  max   = {np.max(x_unit):+.6f}")
print()

# Apply rotation
y = x_unit @ Pi.T

print("After rotation (y = x @ Pi^T, all coordinates):")
print(f"  mean  = {np.mean(y):+.6f}  (should still be ~0)")
print(f"  std   = {np.std(y):.6f}  (should still be ~{1/np.sqrt(d):.6f})")
print(f"  min   = {np.min(y):+.6f}")
print(f"  max   = {np.max(y):+.6f}")
print()

# Show that norms are preserved
norms_before = np.linalg.norm(x_unit, axis=-1)
norms_after = np.linalg.norm(y, axis=-1)
norm_err = np.max(np.abs(norms_before - norms_after))
print(f"  Norm preservation: max|norm_before - norm_after| = {norm_err:.2e}")
print(f"  --> Rotation preserves norms (isometry)")
print()

# Show coordinate histogram statistics for a single coordinate
print("Coordinate y[:, 0] histogram (first coordinate after rotation):")
coord0 = y[:, 0]
hist, edges = np.histogram(coord0, bins=20)
print(f"  mean={np.mean(coord0):+.6f}, std={np.std(coord0):.6f}")
print(f"  Histogram (20 bins): min_count={hist.min()}, max_count={hist.max()}")
print()

# -------------------------------------------------------------------------
# 3.  Quantize coordinates using the codebook
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 3: Quantize rotated coordinates with Lloyd-Max codebook")
print(SEPARATOR)
print()

cb = get_codebook(d, 3)  # 3-bit
centroids = np.array(cb["centroids"])
boundaries = np.array(cb["boundaries"])

# Pick one sample vector for illustration
y_sample = y[0]
print(f"  Sample vector y[0] (first 10 coords): {y_sample[:10].tolist()}")

# Quantize: find nearest centroid per coordinate
indices = np.searchsorted(boundaries[1:-1], y_sample)
y_quantized = centroids[indices]

print(f"  Indices (first 10):          {indices[:10].tolist()}")
print(f"  Quantized (first 10):        {y_quantized[:10].tolist()}")
print(f"  Original - quantized (first 10): {(y_sample[:10] - y_quantized[:10]).tolist()}")
print()

# Per-coordinate MSE
per_coord_sq_err = (y_sample - y_quantized) ** 2
print(f"  Sample MSE per coord: {np.mean(per_coord_sq_err):.6e}")
print(f"  Expected MSE/coord:   {cb['mse_per_coord']:.6e}")
print()

# -------------------------------------------------------------------------
# 4.  Full round-trip with TurboQuantMSE
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 4: Full round-trip: quantize -> dequantize via TurboQuantMSE")
print(SEPARATOR)
print()

# Use the TurboQuantMSE class (handles rotation, codebook, packing/unpacking)
for bits in [1, 2, 3, 4]:
    quant = TurboQuantMSE(dim=d, bits=bits, seed=42)

    # Create backend-compatible tensor from random vectors
    x_np = rng.randn(200, d).astype(np.float32)
    x_tensor = B.from_numpy(x_np)

    # Quantize
    q = quant.quantize(x_tensor)

    # Dequantize
    x_hat = quant.dequantize(q)
    x_hat_np = B.to_numpy(x_hat)

    # Compute MSE
    mse = np.mean((x_np - x_hat_np) ** 2)
    # Normalized MSE (relative to vector energy)
    rel_mse = mse / np.mean(x_np ** 2)

    # Also check a single vector's error
    diff_sample = x_np[0] - x_hat_np[0]
    cos_sim = np.dot(x_np[0], x_hat_np[0]) / (
        np.linalg.norm(x_np[0]) * np.linalg.norm(x_hat_np[0]) + 1e-10
    )

    print(f"  bits={bits}: MSE = {mse:.6f}, relative MSE = {rel_mse:.6f}, "
          f"cos_sim(sample) = {cos_sim:.6f}")

print()
print("Observation: MSE drops roughly 4x for each additional bit (as expected).")
print()

# -------------------------------------------------------------------------
# 5.  Compression ratio
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 5: Compression ratio")
print(SEPARATOR)
print()

print(f"  Original: each vector = {d} coords x 16 bits (FP16) = {d*2} bytes")
print(f"  Plus norm: 2 bytes (FP16)")
print()
print(f"  {'bits':>4}  {'bytes/vector':>14}  {'compression':>12}  {'MSE':>12}")
print(f"  {'----':>4}  {'-'*14}  {'-'*12}  {'-'*12}")

fp16_bytes = d * 2  # 256 bytes for d=128

for bits in [1, 2, 3, 4]:
    cb = get_codebook(d, bits)
    # Bit-packed data + 2 bytes for norm
    packed_bytes = (d * bits + 7) // 8 + 2
    ratio = fp16_bytes / packed_bytes
    print(f"  {bits:>4}  {packed_bytes:>14}  {ratio:>11.1f}x  {cb['mse_per_coord']:>12.6e}")

print()
print("Done!  All sections completed successfully.")
print(SEPARATOR)
