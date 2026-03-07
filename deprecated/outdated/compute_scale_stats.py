#!/usr/bin/env python3
"""
Compute global scaling stats for SAM RGB inputs and CHM/COST aux channels.
Stats are computed from TRAIN chips only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_percentiles(p_str: str) -> tuple[float, float]:
    """Internal helper function."""
    parts = [float(p.strip()) for p in p_str.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("percentiles must be two comma-separated values (e.g., 2,98)")
    return parts[0], parts[1]


def _sample_values(arr: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Internal helper function."""
    flat = np.asarray(arr, np.float32).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return flat
    if max_n <= 0 or flat.size <= max_n:
        return flat
    idx = rng.choice(flat.size, size=max_n, replace=False)
    return flat[idx]


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Compute global scale stats for SAM + StarDist.")
    pa.add_argument("--chips-root", type=Path, required=True)
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default is all under chips-root")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--rgb-sample-max", type=int, default=20000)
    pa.add_argument("--aux-sample-max", type=int, default=20000)
    pa.add_argument("--rgb-percentiles", type=str, default="2,98")
    pa.add_argument("--aux-percentiles", type=str, default="2,98")
    pa.add_argument("--seed", type=int, default=0)
    args = pa.parse_args()

    rgb_p_lo, rgb_p_hi = _parse_percentiles(args.rgb_percentiles)
    aux_p_lo, aux_p_hi = _parse_percentiles(args.aux_percentiles)
    rng = np.random.default_rng(args.seed)

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
        size_dirs = [args.chips_root / s for s in sizes]
    else:
        size_dirs = sorted(p for p in args.chips_root.iterdir() if p.is_dir() and p.name.isdigit())

    rgb_samples = None
    chm_samples = []
    cost_samples = []

    for size_dir in size_dirs:
        train_dir = size_dir / "chips_train"
        img_dir = train_dir / "img"
        if not img_dir.exists():
            continue
        img_paths = sorted(img_dir.glob("*.npy"))
        if not img_paths:
            continue

        if rgb_samples is None:
            first = np.load(img_paths[0], mmap_mode="r")
            if first.ndim != 3:
                raise ValueError(f"Unexpected image shape: {first.shape}")
            rgb_samples = [[] for _ in range(first.shape[0])]

        for p in img_paths:
            img = np.load(p, mmap_mode="r")
            if img.ndim != 3:
                continue
            for c in range(min(img.shape[0], len(rgb_samples))):
                vals = _sample_values(img[c], args.rgb_sample_max, rng)
                if vals.size:
                    rgb_samples[c].append(vals)

        chm_dir = train_dir / "chm"
        cost_dir = train_dir / "cost"
        for p in sorted(chm_dir.glob("*.npy")):
            arr = np.load(p, mmap_mode="r")
            vals = _sample_values(arr, args.aux_sample_max, rng)
            if vals.size:
                chm_samples.append(vals)
        for p in sorted(cost_dir.glob("*.npy")):
            arr = np.load(p, mmap_mode="r")
            vals = _sample_values(arr, args.aux_sample_max, rng)
            if vals.size:
                cost_samples.append(vals)

    if not rgb_samples:
        raise RuntimeError("No training RGB chips found to compute stats.")

    rgb_stats = []
    for c_vals in rgb_samples:
        if not c_vals:
            rgb_stats.append({"lo": 0.0, "hi": 1.0})
            continue
        vals = np.concatenate(c_vals)
        lo, hi = np.percentile(vals, (rgb_p_lo, rgb_p_hi))
        rgb_stats.append({"lo": float(lo), "hi": float(hi)})

    if chm_samples:
        chm_vals = np.concatenate(chm_samples)
        chm_lo, chm_hi = np.percentile(chm_vals, (aux_p_lo, aux_p_hi))
    else:
        chm_lo, chm_hi = 0.0, 1.0

    if cost_samples:
        cost_vals = np.concatenate(cost_samples)
        cost_lo, cost_hi = np.percentile(cost_vals, (aux_p_lo, aux_p_hi))
    else:
        cost_lo, cost_hi = 0.0, 1.0

    out = {
        "sam_rgb_u8": {"per_channel": rgb_stats},
        "chm_01": {"lo": float(chm_lo), "hi": float(chm_hi)},
        "cost_01": {"lo": float(cost_lo), "hi": float(cost_hi)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ wrote {args.out}")


if __name__ == "__main__":
    main()
