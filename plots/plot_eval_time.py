#!/usr/bin/env python3
"""
Plot eval-time comparisons (seconds per km^2 and per crown).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

BASE_FONT = 13
TITLE_FONT = 14
LABEL_FONT = 12
TICK_FONT = 11


def _color_for(label: str) -> str:
    label_l = label.lower()
    if "sam" in label:
        return "#4C78A8"  # blue
    if "detectron" in label:
        return "#F58518"  # orange
    return "#54A24B"  # green fallback


def main() -> None:
    pa = argparse.ArgumentParser(description="Plot eval-time comparisons from summary CSV.")
    pa.add_argument("--csv", type=Path, required=True)
    pa.add_argument("--size", type=str, required=True)
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--title", type=str, default="Evaluation time (normalized)")
    pa.add_argument(
        "--stat",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Statistic to plot when *_median/mean columns exist.",
    )
    pa.add_argument(
        "--error-bars",
        action="store_true",
        help="Show error bars using min/max columns when available.",
    )
    pa.add_argument(
        "--inset-boxplots",
        action="store_true",
        help="Add inset boxplots using per-run timings when available.",
    )
    args = pa.parse_args()

    df = pd.read_csv(args.csv)
    size_str = str(args.size).lstrip("0")
    if size_str == "":
        size_str = "0"
    df["size_str"] = df["size"].astype(str).str.lstrip("0")
    df.loc[df["size_str"] == "", "size_str"] = "0"
    df = df[df["size_str"] == size_str]
    if df.empty:
        raise SystemExit(f"No rows for size {args.size} in {args.csv}")

    # keep only combined SAM-StarDist and Detectron2 eval
    keep = df[df["stage"].isin(["combined_eval", "eval"])]

    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.titlesize": TITLE_FONT,
        "axes.labelsize": LABEL_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    if f"sec_per_km2_{args.stat}" in df.columns:
        sec_km_col = f"sec_per_km2_{args.stat}"
        sec_crown_col = f"sec_per_crown_{args.stat}"
    else:
        sec_km_col = "sec_per_km2"
        sec_crown_col = "sec_per_crown"

    metrics = [
        (sec_km_col, "Seconds per km²"),
        (sec_crown_col, "Seconds per crown"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        sub = keep.dropna(subset=[col]).copy()
        if sub.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        labels = []
        vals = []
        colors = []
        runs = []
        for r in sub.itertuples():
            label = str(r.method) if r.stage in ("combined_eval", "eval") else f"{r.method} ({r.stage})"
            labels.append(label)
            vals.append(getattr(r, col))
            colors.append(_color_for(label))
            run_vals = None
            if getattr(r, "seconds_runs", None):
                try:
                    sec_runs = json.loads(r.seconds_runs)
                    if col.startswith("sec_per_km2"):
                        run_vals = [v / r.area_km2 for v in sec_runs] if r.area_km2 else None
                    elif col.startswith("sec_per_crown"):
                        run_vals = [v / r.n_crowns for v in sec_runs] if r.n_crowns else None
                    else:
                        run_vals = sec_runs
                except Exception:
                    run_vals = None
            runs.append(run_vals)
        yerr = None
        if args.error_bars and f"{col}_min" in sub.columns and f"{col}_max" in sub.columns:
            lows = sub[f"{col}_min"].to_numpy()
            highs = sub[f"{col}_max"].to_numpy()
            yerr = [vals - lows, highs - vals]
        ax.bar(labels, vals, color=colors, edgecolor="#333333", yerr=yerr, capsize=4)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=0)

        if args.inset_boxplots and any(r is not None and len(r) for r in runs):
            inset = inset_axes(ax, width="45%", height="30%", loc="upper left", borderpad=1)
            box_data = [r if r is not None else [] for r in runs]
            inset.boxplot(box_data, labels=None, patch_artist=True, showfliers=False)
            inset.set_xticks([])
            inset.tick_params(axis="y", labelsize=8)
            inset.set_title("runs", fontsize=9)

    fig.suptitle(args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
