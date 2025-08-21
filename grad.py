#!/usr/bin/env python3
"""
Compute Sobel gradient of CHM and save as grayscale PNG.

Input:
  chm1.tif

Output:
  chm_gradient.png  (normalized 0–1, grayscale)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import sobel

# paths
CHM_TIF = Path("chm1.tif")
OUT_PNG = Path("chm_gradient.png")

def read_chm(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, out_dtype="float32", masked=True)
    return arr.filled(np.nan)

def robust_norm(a: np.ndarray, lo=2, hi=98) -> np.ndarray:
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype="float32")
    p_lo, p_hi = np.nanpercentile(a[finite], [lo, hi])
    p_hi = max(p_hi, p_lo + 1e-6)
    out = (np.clip(a, p_lo, p_hi) - p_lo) / (p_hi - p_lo)
    return np.nan_to_num(out).astype("float32")

def sobel_grad_norm(chm: np.ndarray) -> np.ndarray:
    chm_filled = np.nan_to_num(chm, nan=np.nanmedian(chm[np.isfinite(chm)]) if np.isfinite(chm).any() else 0.0)
    dx = sobel(chm_filled, axis=1, mode="nearest")
    dy = sobel(chm_filled, axis=0, mode="nearest")
    mag = np.hypot(dx, dy)
    return robust_norm(mag)

def main():
    if not CHM_TIF.exists():
        raise FileNotFoundError(f"Missing {CHM_TIF}")

    chm = read_chm(CHM_TIF)
    grad01 = sobel_grad_norm(chm)

    plt.imsave(OUT_PNG, grad01, cmap="gray", vmin=0, vmax=1)
    print(f"[OK] Saved → {OUT_PNG}")

if __name__ == "__main__":
    main()