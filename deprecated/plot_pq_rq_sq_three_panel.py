#!/usr/bin/env python3
# DEPRECATED: use plot_ap50_pq_line_2panel.py
"""
Three-panel plot of PQ/RQ/SQ vs training instances for two models.
Panel 1: SAM-Star metrics
Panel 2: Detectron metrics
Panel 3: Overlay comparison (both models, all three metrics)

Note: This script draws raw curves (no smoothing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ("pq", "rq", "sq")
# Okabe-Ito colorblind-friendly palette (avoid red/green)
COLORS = {"pq": "#0072B2", "rq": "#E69F00", "sq": "#CC79A7"}


def _load_df(path: Path, xcol: str, n_points: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if xcol not in df.columns:
        raise SystemExit(f"Missing column '{xcol}' in {path}")
    df = df.sort_values(xcol)
    if n_points is not None:
        df = df.head(n_points)
    return df


def _prep_xy(df: pd.DataFrame, xcol: str, metric: str, origin: bool):
    x = df[xcol].astype(float).values
    y = df[metric].astype(float).values
    if origin:
        x = [0.0] + list(x)
        y = [0.0] + list(y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(np.unique(x)) < len(x):
        tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).mean()
        x = tmp["x"].values
        y = tmp["y"].values
    return x, y


def main() -> None:
    pa = argparse.ArgumentParser(description="Publication-ready 3-panel PQ/RQ/SQ curves for two models.")
    pa.add_argument("--summary-a", type=Path, required=True, help="SAM-Star summary_iou.csv")
    pa.add_argument("--summary-b", type=Path, required=True, help="Detectron summary_iou.csv")
    pa.add_argument("--summary-c", type=Path, default=None, help="Optional third summary_iou.csv")
    pa.add_argument("--label-a", type=str, default="sam-star")
    pa.add_argument("--label-b", type=str, default="detectron")
    pa.add_argument("--label-c", type=str, default="detectron-mc")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--x", type=str, default="n_train_instances", choices=["size", "n_train_instances"])
    pa.add_argument("--n-points", type=int, default=None, help="Keep only first N points after sorting by x")
    pa.add_argument("--origin", action="store_true", help="Prepend (0,0) to each curve")
    pa.add_argument("--combined-only", action="store_true", help="Plot only the combined overlay panel")
    pa.add_argument("--show-points", action="store_true", help="Show point markers on each curve")
    pa.add_argument("--x-max", type=float, default=None, help="Optional max x-axis value")
    pa.add_argument("--title", type=str, default="PQ/RQ/SQ vs training instances")
    args = pa.parse_args()

    df_a = _load_df(args.summary_a, args.x, args.n_points)
    df_b = _load_df(args.summary_b, args.x, args.n_points)
    df_c = _load_df(args.summary_c, args.x, args.n_points) if args.summary_c else None

    if args.combined_only:
        fig, ax_c = plt.subplots(1, 1, figsize=(6, 4), sharey=True)
        axes = [ax_c]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        ax_a, ax_b, ax_c = axes

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    def _plot_panel(ax, df, label):
        for metric in METRICS:
            x, y = _prep_xy(df, args.x, metric, args.origin)
            ax.plot(
                x,
                y,
                color=COLORS[metric],
                label=metric.upper(),
                linewidth=2,
                marker="o" if args.show_points else None,
                markersize=3 if args.show_points else None,
            )
        ax.set_title(label)
        ax.set_xlabel(args.x)
        ax.legend(frameon=False)

    if not args.combined_only:
        _plot_panel(ax_a, df_a, args.label_a)
        _plot_panel(ax_b, df_b, args.label_b)

    # Comparison overlay
    for metric in METRICS:
        xa, ya = _prep_xy(df_a, args.x, metric, args.origin)
        xb, yb = _prep_xy(df_b, args.x, metric, args.origin)
        ax_c.plot(
            xa,
            ya,
            color=COLORS[metric],
            linewidth=2,
            label=f"{metric.upper()} {args.label_a}",
            marker="o" if args.show_points else None,
            markersize=3 if args.show_points else None,
        )
        ax_c.plot(
            xb,
            yb,
            color=COLORS[metric],
            linewidth=2,
            linestyle="--",
            label=f"{metric.upper()} {args.label_b}",
            marker="o" if args.show_points else None,
            markersize=3 if args.show_points else None,
        )
        if df_c is not None:
            xc, yc = _prep_xy(df_c, args.x, metric, args.origin)
            ax_c.plot(
                xc,
                yc,
                color=COLORS[metric],
                linewidth=2,
                linestyle=":",
                label=f"{metric.upper()} {args.label_c}",
                marker="o" if args.show_points else None,
                markersize=3 if args.show_points else None,
            )
    ax_c.set_title("comparison")
    ax_c.set_xlabel(args.x)
    ax_c.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlim(left=0)
        if args.x_max is not None:
            ax.set_xlim(right=args.x_max)

    if not args.combined_only:
        ax_a.set_ylabel("metric")
        fig.suptitle(args.title, y=1.02)
    else:
        ax_c.set_ylabel("metric")
        ax_c.set_title(args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
