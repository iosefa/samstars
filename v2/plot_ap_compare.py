#!/usr/bin/env python3
"""
Single-panel AP comparison vs training instances for two models.
Optionally include AP50 and AP75 in the same plot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Okabe-Ito colorblind-friendly palette
MODEL_STYLE = {
    "a": {"ls": "-", "color": "#0072B2"},
    "b": {"ls": "--", "color": "#E69F00"},
    "c": {"ls": ":", "color": "#009E73"},
}
METRIC_COLORS = {"AP": "#0072B2", "AP50": "#E69F00", "AP75": "#CC79A7"}


def _load_df(path: Path, xcol: str, metric: str, n_points: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (xcol, metric):
        if col not in df.columns:
            raise SystemExit(f"Missing column '{col}' in {path}")
    df = df.sort_values(xcol)
    if n_points is not None:
        df = df.head(n_points)
    return df


def _prep_xy(
    df: pd.DataFrame,
    xcol: str,
    metric: str,
    origin: bool,
    scale_100: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = df[xcol].astype(float).values
    y = df[metric].astype(float).values
    if scale_100:
        y = y / 100.0
    if origin:
        x = [0.0] + list(x)
        y = [0.0] + list(y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(np.unique(x)) < len(x):
        tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).mean()
        x = tmp["x"].values
        y = tmp["y"].values
    return x, y


def _metric_suffix(metric: str) -> str:
    if metric.endswith("AP50"):
        return "AP50"
    if metric.endswith("AP75"):
        return "AP75"
    if metric.endswith("AP"):
        return "AP"
    return metric


def main() -> None:
    pa = argparse.ArgumentParser(description="Plot AP vs training instances for two models.")
    pa.add_argument("--summary-a", type=Path, required=True, help="Summary CSV for model A")
    pa.add_argument("--summary-b", type=Path, required=True, help="Summary CSV for model B")
    pa.add_argument("--summary-c", type=Path, default=None, help="Optional summary CSV for model C")
    pa.add_argument("--label-a", type=str, default="sam-star")
    pa.add_argument("--label-b", type=str, default="detectron2")
    pa.add_argument("--label-c", type=str, default="model-c")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--x", type=str, default="n_train_instances", choices=["size", "n_train_instances"])
    pa.add_argument(
        "--metric",
        type=str,
        default="segm_AP",
        help="Metric to plot (e.g., segm_AP, segm_AP50, segm_AP75, bbox_AP)",
    )
    pa.add_argument("--include-ap50", action="store_true", help="Also plot AP50 for the same task")
    pa.add_argument("--include-ap75", action="store_true", help="Also plot AP75 for the same task")
    pa.add_argument("--n-points", type=int, default=None, help="Keep only first N points after sorting by x")
    pa.add_argument("--origin", action="store_true", help="Prepend (0,0) to each curve")
    pa.add_argument("--show-points", action="store_true", help="Show point markers on each curve")
    pa.add_argument("--scale-100", action="store_true", help="Divide AP values by 100 for 0..1 scale")
    pa.add_argument("--fill-under", action="store_true", help="Lightly fill under each curve")
    pa.add_argument("--fill-alpha", type=float, default=0.08, help="Alpha for fill-under shading")
    pa.add_argument("--x-label", type=str, default=None, help="Override x-axis label")
    pa.add_argument("--y-label", type=str, default=None, help="Override y-axis label")
    pa.add_argument("--x-max", type=float, default=None, help="Optional max x-axis value")
    pa.add_argument("--y-max", type=float, default=None, help="Optional max y-axis value")
    pa.add_argument("--title", type=str, default="Average Precision vs training size")
    args = pa.parse_args()

    if args.metric.endswith("AP50") or args.metric.endswith("AP75"):
        metrics = [args.metric]
    elif args.metric.endswith("AP"):
        metrics = [args.metric]
        if args.include_ap50:
            metrics.append(args.metric.replace("AP", "AP50"))
        if args.include_ap75:
            metrics.append(args.metric.replace("AP", "AP75"))
    else:
        raise SystemExit("--metric must end with AP/AP50/AP75 (e.g., segm_AP, segm_AP50).")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.grid(True, alpha=0.3)
    for metric in metrics:
        df_a = _load_df(args.summary_a, args.x, metric, args.n_points)
        df_b = _load_df(args.summary_b, args.x, metric, args.n_points)
        df_c = _load_df(args.summary_c, args.x, metric, args.n_points) if args.summary_c else None
        xa, ya = _prep_xy(df_a, args.x, metric, args.origin, args.scale_100)
        xb, yb = _prep_xy(df_b, args.x, metric, args.origin, args.scale_100)
        suffix = _metric_suffix(metric)
        color = METRIC_COLORS.get(suffix, "#0072B2")

        ax.plot(
            xa,
            ya,
            color=color,
            linewidth=2,
            linestyle=MODEL_STYLE["a"]["ls"],
            marker="o" if args.show_points else None,
            markersize=3 if args.show_points else None,
            label=f"{suffix} {args.label_a}",
        )
        if args.fill_under:
            ax.fill_between(xa, 0, ya, color=color, alpha=args.fill_alpha, linewidth=0)
        ax.plot(
            xb,
            yb,
            color=color,
            linewidth=2,
            linestyle=MODEL_STYLE["b"]["ls"],
            marker="o" if args.show_points else None,
            markersize=3 if args.show_points else None,
            label=f"{suffix} {args.label_b}",
        )
        if args.fill_under:
            ax.fill_between(xb, 0, yb, color=color, alpha=args.fill_alpha, linewidth=0)
        if df_c is not None:
            xc, yc = _prep_xy(df_c, args.x, metric, args.origin, args.scale_100)
            ax.plot(
                xc,
                yc,
                color=color,
                linewidth=2,
                linestyle=MODEL_STYLE["c"]["ls"],
                marker="o" if args.show_points else None,
                markersize=3 if args.show_points else None,
                label=f"{suffix} {args.label_c}",
            )
            if args.fill_under:
                ax.fill_between(xc, 0, yc, color=color, alpha=args.fill_alpha, linewidth=0)
    ax.set_title(args.title)
    if args.x_label:
        x_label = args.x_label
    else:
        x_label = "Number of training crowns" if args.x == "n_train_instances" else "Training size"
    if args.y_label:
        y_label = args.y_label
    else:
        y_label = "Average Precision" + (" (0..1)" if args.scale_100 else "")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0)
    if args.x_max is not None:
        ax.set_xlim(right=args.x_max)
    if args.y_max is not None:
        ax.set_ylim(top=args.y_max)
    elif args.scale_100:
        ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
