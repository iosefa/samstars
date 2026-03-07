#!/usr/bin/env python3
"""
Read per-size metrics.csv files and plot a chosen metric vs. training size.

Defaults target Detectron outputs under v2/models_detectron/sizes/<size>/detectron/train_logs/metrics.csv
and plots segm.AP.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_metric(csv_path: Path, metric: str):
    """Read metric."""
    vals = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            k, v = row
            if k == "metric":
                continue
            try:
                vals[k] = float(v)
            except ValueError:
                continue
    return vals.get(metric)


def main():
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Plot metric vs training size from metrics.csv files.")
    pa.add_argument("--root", type=Path, default=Path("v2/models_detectron/sizes"), help="Root containing per-size subdirs")
    pa.add_argument("--metric", default="segm.AP", help="Metric key to plot (e.g., segm.AP, segm.AP50, bbox.AP)")
    pa.add_argument("--output", type=Path, default=None, help="Output plot path (defaults to <root>/metrics_<metric>.png)")
    args = pa.parse_args()

    sizes = []
    values = []
    for size_dir in sorted(args.root.iterdir()):
        if not size_dir.is_dir():
            continue
        try:
            n = int(size_dir.name)
        except ValueError:
            continue
        csv_path = size_dir / "detectron" / "train_logs" / "metrics.csv"
        if not csv_path.exists():
            continue
        val = read_metric(csv_path, args.metric)
        if val is None:
            continue
        sizes.append(n)
        values.append(val)

    if not sizes:
        raise SystemExit(f"No metrics found under {args.root} for metric '{args.metric}'")

    sizes, values = zip(*sorted(zip(sizes, values)))

    plt.figure(figsize=(8, 4.5))
    plt.plot(sizes, values, marker="o")
    plt.xlabel("Training size (chips)")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} vs training size")
    plt.grid(True, alpha=0.3)

    out_path = args.output or args.root / f"metrics_{args.metric.replace('.', '_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"✓ wrote plot to {out_path}")


if __name__ == "__main__":
    main()
