#!/usr/bin/env python3
"""
01 — Lloyd-Max Codebook for TurboQuant
=======================================

This script walks through the mathematical foundation of TurboQuant:

  After rotating a unit vector by a random orthogonal matrix, each coordinate
  follows a Beta-type distribution on [-1, 1] whose shape depends on the
  dimension d.  The Lloyd-Max algorithm finds the *optimal* set of quantization
  centroids (and decision boundaries) that minimize mean-squared error for
  this distribution.

  We visualize the distribution, print centroid tables, and compare the
  achieved MSE to the theoretical bound from the paper:

      MSE_per_coord <= sqrt(3) * pi / 2  *  (1 / 4^b)   [Zandieh et al.]

Run:
    python notebooks/01_lloyd_max_codebook.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turboquant_mac.codebook import beta_pdf, get_codebook, compute_lloyd_max_codebook

SEPARATOR = "=" * 72

# -------------------------------------------------------------------------
# 1.  Visualize the Beta distribution for different dimensions
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 1: The coordinate distribution after random rotation")
print(SEPARATOR)
print()
print("When a unit vector in R^d is multiplied by a random orthogonal matrix,")
print("each coordinate follows a Beta-type PDF on [-1, 1]:")
print()
print("  f(x) = Gamma(d/2) / [sqrt(pi) * Gamma((d-1)/2)] * (1 - x^2)^((d-3)/2)")
print()
print("As d grows, this concentrates around 0 with std ~ 1/sqrt(d).")
print()

x_grid = np.linspace(-0.3, 0.3, 1000)

fig, ax = plt.subplots(figsize=(8, 5))
for d, color in [(64, "tab:blue"), (128, "tab:orange"), (256, "tab:green")]:
    pdf_vals = beta_pdf(x_grid, d)
    std = 1.0 / np.sqrt(d)
    ax.plot(x_grid, pdf_vals, color=color, linewidth=2,
            label=f"d={d}  (std ~ {std:.4f})")
ax.set_xlabel("Coordinate value x")
ax.set_ylabel("Density f(x)")
ax.set_title("Coordinate Distribution After Random Rotation")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "01_beta_pdf.png"), dpi=120)
print("  [Saved plot: notebooks/01_beta_pdf.png]")
print()

# -------------------------------------------------------------------------
# 2.  Show Lloyd-Max centroids for d=128
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 2: Lloyd-Max centroids for d=128")
print(SEPARATOR)
print()
print("The Lloyd-Max algorithm finds optimal quantization levels that")
print("minimize expected MSE for the distribution above.")
print()

for bits in [1, 2, 3, 4]:
    cb = get_codebook(128, bits)
    centroids = cb["centroids"]
    mse = cb["mse_per_coord"]
    n = len(centroids)
    print(f"  bits={bits}  ({n} levels)")
    print(f"    centroids : {[f'{c:+.6f}' for c in centroids]}")
    print(f"    MSE/coord : {mse:.6e}")
    print()

# -------------------------------------------------------------------------
# 3.  Compare MSE to theoretical bound
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 3: MSE vs. theoretical bound")
print(SEPARATOR)
print()
print("Paper bound:  MSE_per_coord <= sqrt(3) * pi / 2  *  (1 / 4^b)")
print("(This bound is for the high-rate regime; 1-bit may exceed it.)")
print()
print(f"  {'bits':>4}  {'Measured MSE':>14}  {'Bound':>14}  {'Ratio':>8}")
print(f"  {'----':>4}  {'-'*14}  {'-'*14}  {'-'*8}")

theoretical_constant = np.sqrt(3) * np.pi / 2.0

for bits in [1, 2, 3, 4]:
    cb = get_codebook(128, bits)
    mse = cb["mse_per_coord"]
    bound = theoretical_constant * (1.0 / 4**bits)
    ratio = mse / bound
    print(f"  {bits:>4}  {mse:>14.6e}  {bound:>14.6e}  {ratio:>8.3f}")

print()
print("Ratios <= 1.0 confirm that the Lloyd-Max codebook meets the bound.")
print("(1-bit may slightly exceed it because the high-rate approximation")
print("is less tight at very low bit widths.)")
print()

# -------------------------------------------------------------------------
# 4.  Visualize quantization of a coordinate value
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 4: How a single coordinate gets quantized")
print(SEPARATOR)
print()

cb = get_codebook(128, 3)  # 3-bit example
centroids = np.array(cb["centroids"])
boundaries = np.array(cb["boundaries"])

# Pick a few example values and show what they quantize to
example_vals = np.array([-0.12, -0.05, 0.0, 0.04, 0.10, 0.15])

print("  Using 3-bit codebook (d=128): 8 centroids")
print(f"  Decision boundaries: {[f'{b:+.4f}' for b in boundaries]}")
print()
print(f"  {'Value':>8}  {'Bin':>4}  {'Centroid':>10}  {'Error':>10}")
print(f"  {'-'*8}  {'-'*4}  {'-'*10}  {'-'*10}")

for val in example_vals:
    idx = np.searchsorted(boundaries[1:-1], val)
    centroid = centroids[idx]
    error = val - centroid
    print(f"  {val:>+8.4f}  {idx:>4d}  {centroid:>+10.6f}  {error:>+10.6f}")

print()

# Visualization: PDF + centroids + boundaries
fig, ax = plt.subplots(figsize=(9, 5))
x_plot = np.linspace(-0.25, 0.25, 2000)
pdf_vals = beta_pdf(x_plot, 128)
ax.fill_between(x_plot, pdf_vals, alpha=0.15, color="steelblue")
ax.plot(x_plot, pdf_vals, color="steelblue", linewidth=1.5, label="PDF (d=128)")

# Draw boundaries as vertical dashed lines
for b in boundaries[1:-1]:
    ax.axvline(b, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

# Draw centroids as colored dots on the x-axis
for i, c in enumerate(centroids):
    ax.plot(c, 0, "o", color="red", markersize=8, zorder=5)
    ax.annotate(f"c{i}", (c, 0), textcoords="offset points",
                xytext=(0, -15), ha="center", fontsize=7, color="red")

ax.set_xlabel("Coordinate value x")
ax.set_ylabel("Density")
ax.set_title("3-bit Lloyd-Max Quantization (d=128)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "01_quantization_grid.png"), dpi=120)
print("  [Saved plot: notebooks/01_quantization_grid.png]")
print()

# -------------------------------------------------------------------------
# 5.  How MSE scales with dimension
# -------------------------------------------------------------------------
print(SEPARATOR)
print("SECTION 5: MSE across dimensions (d=64, 128, 256) at 3 bits")
print(SEPARATOR)
print()
print("The per-coordinate MSE depends on d because the distribution shape")
print("changes (narrower for larger d, so quantization is easier).")
print()
print(f"  {'d':>5}  {'MSE/coord':>14}  {'Total MSE (d coords)':>22}")
print(f"  {'---':>5}  {'-'*14}  {'-'*22}")

for d in [64, 128, 256]:
    cb = get_codebook(d, 3)
    print(f"  {d:>5}  {cb['mse_per_coord']:>14.6e}  {cb['mse_total']:>22.6f}")

print()
print("Done!  All sections completed successfully.")
print(SEPARATOR)
