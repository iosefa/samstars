#!/usr/bin/env python3
# DEPRECATED: use plot_ap50_density_height_boxplots_2panel.py
"""
Boxplots comparing AP50 across height and density bins for two models.
Expects per-chip CSVs with columns: model, bin, ap50.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Okabe-Ito palette
MODEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def _load(csv_path: Path, model_order: list[str] | None):
    """Internal helper function."""
    df = pd.read_csv(csv_path)
    if "model" not in df.columns or "bin" not in df.columns or "ap50" not in df.columns:
        raise SystemExit(f"Expected columns model/bin/ap50 in {csv_path}")
    if model_order:
        df = df[df["model"].isin(model_order)]
        df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
        df = df.sort_values(["bin", "model"])
    return df


def _boxplot(ax, df: pd.DataFrame, title: str, ylabel: str, show_counts: bool, count_mode: str):
    """Internal helper function."""
    bins = list(df["bin"].dropna().unique())
    models = list(df["model"].dropna().unique())
    if not bins or not models:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    width = 0.8 / max(len(models), 1)
    positions = []
    data = []
    labels = []
    colors = []

    if show_counts:
        if count_mode == "crowns" and "n_gt" in df.columns:
            if "chip" in df.columns:
                df_counts = df.drop_duplicates(subset=["chip", "bin"])
                bin_counts = df_counts.groupby("bin")["n_gt"].sum()
            else:
                bin_counts = df.groupby("bin")["n_gt"].sum()
        else:
            if "chip" in df.columns:
                bin_counts = df.groupby("bin")["chip"].nunique()
            else:
                bin_counts = df.groupby("bin").size()
                if len(models) > 1:
                    bin_counts = (bin_counts / len(models)).round().astype(int)
        bin_counts = {k: int(v) for k, v in bin_counts.items()}
    else:
        bin_counts = {}

    for i, bin_label in enumerate(bins):
        for j, model in enumerate(models):
            vals = df.loc[(df["bin"] == bin_label) & (df["model"] == model), "ap50"].astype(float).dropna().values
            pos = i + (j - (len(models) - 1) / 2) * width
            positions.append(pos)
            data.append(vals)
            labels.append(f"{bin_label}\n{model}")
            colors.append(MODEL_COLORS[j % len(MODEL_COLORS)])

    bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(range(len(bins)))
    if show_counts:
        xticklabels = [f"{b}\n(n={bin_counts.get(b, 0)})" for b in bins]
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels(bins)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    # legend
    handles = []
    for j, model in enumerate(models):
        handles.append(plt.Line2D([0], [0], color=MODEL_COLORS[j % len(MODEL_COLORS)], lw=6, label=model))
    ax.legend(handles=handles, frameon=False)


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Boxplot AP50 by height/density bins for two models.")
    pa.add_argument("--height-csv", type=Path, required=True)
    pa.add_argument("--density-csv", type=Path, required=True)
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--model-order", type=str, default=None, help="Comma-separated model order")
    pa.add_argument("--y-max", type=float, default=1.0)
    pa.add_argument("--title", type=str, default="AP50 by height/density")
    pa.add_argument("--show-counts", action="store_true", help="Annotate bin labels with chip counts")
    pa.add_argument(
        "--count-mode",
        type=str,
        choices=["chips", "crowns"],
        default="chips",
        help="When --show-counts, annotate with chip counts or summed GT crown counts.",
    )
    args = pa.parse_args()

    model_order = None
    if args.model_order:
        model_order = [m.strip() for m in args.model_order.split(",") if m.strip()]

    df_h = _load(args.height_csv, model_order)
    df_d = _load(args.density_csv, model_order)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _boxplot(axes[0], df_h, "Height bins", "AP50", args.show_counts, args.count_mode)
    _boxplot(axes[1], df_d, "Density bins", "AP50", args.show_counts, args.count_mode)
    axes[0].set_ylim(0, args.y_max)
    axes[1].set_ylim(0, args.y_max)
    fig.suptitle(args.title, y=1.02)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
