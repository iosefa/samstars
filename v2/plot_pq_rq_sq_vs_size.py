#!/usr/bin/env python3
"""
Plot PQ/RQ/SQ vs training size using summary_iou.csv from summarize_fixed_val_iou.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    pa = argparse.ArgumentParser(description="Plot PQ/RQ/SQ vs training size.")
    pa.add_argument("--summary-csv", type=Path, required=True)
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--x", type=str, default="size", choices=["size", "n_train_instances"])
    args = pa.parse_args()

    df = pd.read_csv(args.summary_csv)
    if args.x not in df.columns:
        raise SystemExit(f"Missing column '{args.x}' in {args.summary_csv}")

    df = df.sort_values(args.x)
    x = df[args.x].astype(float).values

    plt.figure(figsize=(7, 4))
    plt.plot(x, df["pq"].values, marker="o", label="PQ")
    plt.plot(x, df["rq"].values, marker="o", label="RQ")
    plt.plot(x, df["sq"].values, marker="o", label="SQ")
    plt.xlabel(args.x)
    plt.ylabel("metric")
    plt.title("PQ/RQ/SQ vs training size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
