#!/usr/bin/env python3
# DEPRECATED: use plot_ap50_pq_line_2panel.py
"""
Plot PQ/RQ/SQ vs training size using summary_iou.csv from summarize_fixed_val_iou.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    pa = argparse.ArgumentParser(description="Plot PQ/RQ/SQ vs training size.")
    pa.add_argument("--summary-csv", type=Path, required=True)
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--x", type=str, default="size", choices=["size", "n_train_instances"])
    pa.add_argument("--smooth-window", type=int, default=1, help="Moving average window (>=1). 1 disables.")
    pa.add_argument(
        "--smooth-method",
        type=str,
        default="auto",
        choices=["auto", "none", "moving", "pchip"],
        help="Smoothing method. auto uses moving if smooth-window>1.",
    )
    pa.add_argument("--smooth-points", type=int, default=200, help="Point count for pchip curve.")
    args = pa.parse_args()

    df = pd.read_csv(args.summary_csv)
    if args.x not in df.columns:
        raise SystemExit(f"Missing column '{args.x}' in {args.summary_csv}")

    df = df.sort_values(args.x)
    x = df[args.x].astype(float).values

    def _with_origin(y):
        if len(x) == 0:
            return x, y
        return [0.0] + list(x), [0.0] + list(y)

    def _prep_xy(y):
        xx, yy = _with_origin(y)
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        order = np.argsort(xx)
        xx, yy = xx[order], yy[order]
        if len(np.unique(xx)) < len(xx):
            tmp = pd.DataFrame({"x": xx, "y": yy}).groupby("x", as_index=False).mean()
            xx = tmp["x"].values
            yy = tmp["y"].values
        return xx, yy

    method = args.smooth_method
    if method == "auto":
        method = "moving" if args.smooth_window > 1 else "none"

    if method == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator
        except Exception as exc:
            print(f"[warn] pchip unavailable ({exc}); falling back to moving average")
            method = "moving"

    def _smooth(y):
        if method != "moving":
            return None
        if args.smooth_window <= 1:
            return None
        y_sm = pd.Series(y, dtype="float64").rolling(
            window=args.smooth_window, center=True, min_periods=1
        ).mean().values
        if len(y_sm):
            y_sm[0] = 0.0
        return y_sm

    plt.figure(figsize=(7, 4))
    xpq, ypq = _prep_xy(df["pq"].values)
    xrq, yrq = _prep_xy(df["rq"].values)
    xsq, ysq = _prep_xy(df["sq"].values)

    sm_pq = _smooth(ypq)
    sm_rq = _smooth(yrq)
    sm_sq = _smooth(ysq)

    if method == "pchip":
        x_max = float(xpq.max()) if len(xpq) else 0.0
        x_dense = np.linspace(0.0, x_max, args.smooth_points) if x_max > 0 else xpq
        f_pq = PchipInterpolator(xpq, ypq, extrapolate=False)
        f_rq = PchipInterpolator(xrq, yrq, extrapolate=False)
        f_sq = PchipInterpolator(xsq, ysq, extrapolate=False)
        ypq = f_pq(x_dense)
        yrq = f_rq(x_dense)
        ysq = f_sq(x_dense)
        if len(ypq):
            ypq[0] = 0.0
            yrq[0] = 0.0
            ysq[0] = 0.0
        plt.plot(x_dense, ypq, label="PQ")
        plt.plot(x_dense, yrq, label="RQ")
        plt.plot(x_dense, ysq, label="SQ")
    else:
        if sm_pq is not None:
            ypq = sm_pq
            yrq = sm_rq
            ysq = sm_sq
        plt.plot(xpq, ypq, label="PQ")
        plt.plot(xrq, yrq, label="RQ")
        plt.plot(xsq, ysq, label="SQ")
    plt.xlabel(args.x)
    plt.ylabel("metric")
    plt.title("PQ/RQ/SQ vs training size")
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
