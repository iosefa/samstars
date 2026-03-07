#!/usr/bin/env python3
# DEPRECATED: use plot_ap50_pq_line_2panel.py
"""
Plot PQ/RQ/SQ from two summary CSVs on the same axes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _load_df(path: Path, xcol: str, n_points: int | None) -> pd.DataFrame:
    """Internal helper function."""
    df = pd.read_csv(path)
    if xcol not in df.columns:
        raise SystemExit(f"Missing column '{xcol}' in {path}")
    df = df.sort_values(xcol)
    if n_points is not None:
        df = df.head(n_points)
    return df


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Plot PQ/RQ/SQ from two models.")
    pa.add_argument("--summary-a", type=Path, required=True)
    pa.add_argument("--summary-b", type=Path, required=True)
    pa.add_argument("--label-a", type=str, default=None)
    pa.add_argument("--label-b", type=str, default=None)
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--x", type=str, default="size", choices=["size", "n_train_instances"])
    pa.add_argument("--n-points", type=int, default=None, help="Keep only the first N points (after sorting by x)")
    args = pa.parse_args()

    label_a = args.label_a or args.summary_a.stem
    label_b = args.label_b or args.summary_b.stem

    df_a = _load_df(args.summary_a, args.x, args.n_points)
    df_b = _load_df(args.summary_b, args.x, args.n_points)

    x_a = df_a[args.x].astype(float).values
    x_b = df_b[args.x].astype(float).values

    plt.figure(figsize=(7, 4))
    plt.plot(x_a, df_a["pq"].values, label=f"{label_a} PQ")
    plt.plot(x_a, df_a["rq"].values, label=f"{label_a} RQ")
    plt.plot(x_a, df_a["sq"].values, label=f"{label_a} SQ")
    plt.plot(x_b, df_b["pq"].values, label=f"{label_b} PQ")
    plt.plot(x_b, df_b["rq"].values, label=f"{label_b} RQ")
    plt.plot(x_b, df_b["sq"].values, label=f"{label_b} SQ")

    plt.xlabel(args.x)
    plt.ylabel("metric")
    plt.title("PQ/RQ/SQ comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(left=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
