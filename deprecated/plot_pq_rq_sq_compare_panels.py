#!/usr/bin/env python3
# DEPRECATED: use plot_ap50_pq_line_2panel.py
"""
Panel plot comparing PQ/RQ/SQ between two models + per-model metric panels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    return x, y


def _smooth_y(y, window: int, origin: bool):
    if window <= 1:
        return y
    y_sm = pd.Series(y, dtype="float64").rolling(
        window=window, center=True, min_periods=1
    ).mean().values
    if len(y_sm):
        # Keep the first point anchored (origin or first observed value).
        y_sm[0] = 0.0 if origin else y[0]
    return y_sm


def _diff_interp(xa, ya, xb, yb):
    xa = np.asarray(xa, dtype=float)
    ya = np.asarray(ya, dtype=float)
    xb = np.asarray(xb, dtype=float)
    yb = np.asarray(yb, dtype=float)
    if len(xa) == 0 or len(xb) == 0:
        return np.array([]), np.array([])
    x_min = max(float(xa.min()), float(xb.min()))
    x_max = min(float(xa.max()), float(xb.max()))
    if x_max < x_min:
        return np.array([]), np.array([])
    x_base = np.unique(np.concatenate([xa, xb]))
    x_base = x_base[(x_base >= x_min) & (x_base <= x_max)]
    if len(x_base) == 0:
        return np.array([]), np.array([])
    ya_i = np.interp(x_base, xa, ya)
    yb_i = np.interp(x_base, xb, yb)
    return x_base, ya_i - yb_i


def main() -> None:
    pa = argparse.ArgumentParser(description="Compare PQ/RQ/SQ across two models in panels.")
    pa.add_argument("--summary-a", type=Path, required=True)
    pa.add_argument("--summary-b", type=Path, required=True)
    pa.add_argument("--label-a", type=str, default="detectron")
    pa.add_argument("--label-b", type=str, default="sam-stardist")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--x", type=str, default="n_train_instances", choices=["size", "n_train_instances"])
    pa.add_argument("--n-points", type=int, default=None, help="Keep only the first N points (after sorting by x)")
    pa.add_argument("--origin", action="store_true", help="Prepend (0,0) to each curve")
    pa.add_argument("--smooth-window", type=int, default=1, help="Moving average window (>=1). 1 disables.")
    args = pa.parse_args()

    df_a = _load_df(args.summary_a, args.x, args.n_points)
    df_b = _load_df(args.summary_b, args.x, args.n_points)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    ax_sq, ax_rq, ax_pq = axes[0]
    ax_sq_d, ax_rq_d, ax_pq_d = axes[1]
    ax_a, ax_b, ax_all = axes[2]

    for ax in (ax_pq, ax_rq, ax_sq, ax_a, ax_b, ax_all):
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    for ax in (ax_sq_d, ax_rq_d, ax_pq_d):
        ax.grid(True, alpha=0.3)

    # Compare panels
    x_max = 0.0
    for metric, ax in [("sq", ax_sq), ("rq", ax_rq), ("pq", ax_pq)]:
        xa, ya = _prep_xy(df_a, args.x, metric, args.origin)
        xb, yb = _prep_xy(df_b, args.x, metric, args.origin)
        ya = _smooth_y(ya, args.smooth_window, args.origin)
        yb = _smooth_y(yb, args.smooth_window, args.origin)
        ax.plot(xa, ya, label=args.label_a)
        ax.plot(xb, yb, label=args.label_b)
        ax.set_title(f"{metric.upper()} comparison")
        ax.set_xlabel(args.x)
        ax.set_ylabel(metric.upper())
        ax.legend()
        x_max = max(x_max, max(xa, default=0.0), max(xb, default=0.0))

    # Difference panels (A - B)
    diff_axes = {"sq": ax_sq_d, "rq": ax_rq_d, "pq": ax_pq_d}
    diff_max = 0.0
    for metric in ("sq", "rq", "pq"):
        xa, ya = _prep_xy(df_a, args.x, metric, args.origin)
        xb, yb = _prep_xy(df_b, args.x, metric, args.origin)
        ya = _smooth_y(ya, args.smooth_window, args.origin)
        yb = _smooth_y(yb, args.smooth_window, args.origin)
        xd, yd = _diff_interp(xa, ya, xb, yb)
        ax = diff_axes[metric]
        ax.plot(xd, yd, label=f"{args.label_a} - {args.label_b}")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{metric.upper()} diff")
        ax.set_xlabel(args.x)
        ax.set_ylabel("Δ")
        if len(yd):
            diff_max = max(diff_max, float(np.nanmax(np.abs(yd))))
        x_max = max(x_max, float(xd.max()) if len(xd) else 0.0)
        ax.legend()

    # Per-model panels
    for label, df, ax in [(args.label_a, df_a, ax_a), (args.label_b, df_b, ax_b)]:
        x = df[args.x].astype(float).values
        if args.origin:
            x = [0.0] + list(x)
        x_max = max(x_max, max(x, default=0.0))
        for metric in ("pq", "rq", "sq"):
            y = df[metric].astype(float).values
            if args.origin:
                y = [0.0] + list(y)
            y = _smooth_y(y, args.smooth_window, args.origin)
            ax.plot(x, y, label=metric.upper())
        ax.set_title(f"{label} metrics")
        ax.set_xlabel(args.x)
        ax.set_ylabel("metric")
        ax.legend()

    if x_max <= 0:
        x_max = 1.0
    for ax in (ax_pq, ax_rq, ax_sq, ax_a, ax_b, ax_all, ax_sq_d, ax_rq_d, ax_pq_d):
        ax.set_xlim(0, x_max)
    if diff_max > 0:
        for ax in (ax_sq_d, ax_rq_d, ax_pq_d):
            ax.set_ylim(-diff_max, diff_max)

    # All metrics for both models
    for metric in ("pq", "rq", "sq"):
        xa, ya = _prep_xy(df_a, args.x, metric, args.origin)
        xb, yb = _prep_xy(df_b, args.x, metric, args.origin)
        ya = _smooth_y(ya, args.smooth_window, args.origin)
        yb = _smooth_y(yb, args.smooth_window, args.origin)
        ax_all.plot(xa, ya, label=f"{args.label_a} {metric.upper()}")
        ax_all.plot(xb, yb, label=f"{args.label_b} {metric.upper()}", linestyle="--")
    ax_all.set_title("All metrics (both models)")
    ax_all.set_xlabel(args.x)
    ax_all.set_ylabel("metric")
    ax_all.legend()

    fig.suptitle("PQ/RQ/SQ comparison", y=0.98)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
