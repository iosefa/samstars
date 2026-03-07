#!/usr/bin/env python3
"""
Compute global per-band percentile scaling for SAM chips.
Outputs a JSON file with fixed low/high per band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _iter_chip_paths(chips_root: Path, sizes):
    """Internal helper function."""
    for sz in sizes:
        img_dir = chips_root / sz / "chips_train" / "img"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*.npy")):
            yield p


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Compute global per-band scaling for SAM chips.")
    pa.add_argument("--chips-root", type=Path, required=True)
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes (default: all numeric dirs)")
    pa.add_argument("--percentiles", type=str, default="2,98")
    pa.add_argument("--sample-per-chip", type=int, default=0, help="Sample N pixels per chip per band (0 = all)")
    pa.add_argument("--bands", type=str, default="7,5,3", help="Band labels for metadata only")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = sorted([p.name for p in args.chips_root.iterdir() if p.is_dir() and p.name.isdigit()])

    lo_p, hi_p = [float(x.strip()) for x in args.percentiles.split(",")]
    rng = np.random.default_rng(args.seed)

    samples = None
    n_chips = 0
    for p in _iter_chip_paths(args.chips_root, sizes):
        arr = np.load(p, mmap_mode="r")
        if samples is None:
            samples = [[] for _ in range(arr.shape[0])]
        for c in range(arr.shape[0]):
            flat = np.asarray(arr[c], np.float32).ravel()
            finite = np.isfinite(flat)
            flat = flat[finite]
            if flat.size == 0:
                continue
            if args.sample_per_chip and flat.size > args.sample_per_chip:
                idx = rng.choice(flat.size, size=args.sample_per_chip, replace=False)
                flat = flat[idx]
            samples[c].append(flat)
        n_chips += 1

    if not samples:
        raise SystemExit("No chip data found.")

    lows, highs = [], []
    for c, parts in enumerate(samples):
        if not parts:
            lows.append(0.0)
            highs.append(1.0)
            continue
        vals = np.concatenate(parts)
        lo, hi = np.nanpercentile(vals, [lo_p, hi_p])
        lows.append(float(lo))
        highs.append(float(hi))

    out = {
        "percentiles": [lo_p, hi_p],
        "bands": [int(b) for b in args.bands.split(",") if b.strip()],
        "low": lows,
        "high": highs,
        "sizes": sizes,
        "n_chips": n_chips,
        "sample_per_chip": int(args.sample_per_chip),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"✓ wrote {args.out}")


if __name__ == "__main__":
    main()
