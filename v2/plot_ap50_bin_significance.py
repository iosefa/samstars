#!/usr/bin/env python3
"""
Plot per-chip boxplots with optional significance brackets from nonparam tests.

Inputs:
  - per-chip CSV (model/bin/chip/metric)
  - bin_pairwise CSV (from run_nonparametric_bin_tests.py)
  - method_within_bin CSV (from run_nonparametric_bin_tests.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Match color scheme used elsewhere
MODEL_COLORS = {
    "sam-star": "#0072B2",
    "detectron2": "#E69F00",
    "detectron": "#E69F00",
    "detectron2-full": "#009E73",
    "sam-star+xgb": "#CC79A7",
    "detectron2+xgb": "#E69F00",
}
MODEL_KEYS = sorted(MODEL_COLORS.keys(), key=len, reverse=True)


def _base_model_name(label: str) -> str:
    for key in MODEL_KEYS:
        if label.startswith(key):
            return key
    return label.split()[0]


def _model_color(label: str) -> str:
    base = _base_model_name(label)
    return MODEL_COLORS.get(base, "#000000")


def _add_bracket(ax, x1: float, x2: float, y: float, text: str, height: float):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", lw=1)
    ax.text((x1 + x2) / 2.0, y + height, text, ha="center", va="bottom", fontsize=9)


def _format_p(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def main() -> None:
    pa = argparse.ArgumentParser(description="Boxplot with nonparam significance brackets.")
    pa.add_argument("--per-chip-csv", required=True)
    pa.add_argument("--metric", default="ap50")
    pa.add_argument("--bin-pairwise-csv", default=None)
    pa.add_argument("--method-within-bin-csv", default=None)
    pa.add_argument("--p-col", default="p_adj_holm")
    pa.add_argument("--model-order", default=None, help="Comma-separated model order")
    pa.add_argument("--bin-order", default=None, help="Comma-separated bin order")
    pa.add_argument("--title", default=None)
    pa.add_argument("--out", required=True)
    pa.add_argument("--show-p", action="store_true", help="Annotate p-values (default)")
    args = pa.parse_args()

    df = pd.read_csv(args.per_chip_csv)
    if args.metric not in df.columns:
        raise SystemExit(f"Missing metric column '{args.metric}' in {args.per_chip_csv}")

    model_order = None
    if args.model_order:
        model_order = [m.strip() for m in args.model_order.split(",") if m.strip()]
        df = df[df["model"].isin(model_order)]
    bin_order = None
    if args.bin_order:
        bin_order = [b.strip() for b in args.bin_order.split(",") if b.strip()]
        df = df[df["bin"].isin(bin_order)]

    bins = bin_order or list(df["bin"].dropna().unique())
    models = model_order or list(df["model"].dropna().unique())

    if not bins or not models:
        raise SystemExit("No data after filtering; check --model-order/--bin-order.")

    # Positions: bins on x, models clustered within each bin
    gap = 1.0
    positions = []
    labels = []
    pos_map = {}
    pos = 0.0
    for b in bins:
        for i, m in enumerate(models):
            positions.append(pos + i)
            labels.append(b if i == 0 else "")
            pos_map[(b, m)] = pos + i
        pos += len(models) + gap

    data = []
    for b in bins:
        for m in models:
            vals = df[(df["bin"] == b) & (df["model"] == m)][args.metric].astype(float).to_numpy()
            data.append(vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    for patch, b_m in zip(box["boxes"], [(b, m) for b in bins for m in models]):
        patch.set_facecolor(_model_color(b_m[1]))
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.6)

    ax.set_xticks([pos_map[(b, models[0])] for b in bins])
    ax.set_xticklabels(bins, rotation=0)
    ax.set_xlabel("Bin")
    ax.set_ylabel(args.metric.upper())
    if args.title:
        ax.set_title(args.title)

    # Legend
    legend_handles = [Patch(facecolor=_model_color(m), edgecolor="#333333", label=m, alpha=0.6) for m in models]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor="white", framealpha=1.0)

    # Brackets
    y_all = np.concatenate([d for d in data if len(d)]) if any(len(d) for d in data) else np.array([0.0])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_range = y_max - y_min if y_max > y_min else 0.1
    base_offset = y_range * 0.05
    step = y_range * 0.08
    current_y = y_max + base_offset

    if args.bin_pairwise_csv:
        bdf = pd.read_csv(args.bin_pairwise_csv)
        for _, row in bdf.iterrows():
            b1, b2 = row["bin_a"], row["bin_b"]
            m = row["model"]
            if (b1, m) not in pos_map or (b2, m) not in pos_map:
                continue
            pval = float(row.get(args.p_col, row.get("p", 1.0)))
            label = _format_p(pval) if args.show_p else ""
            x1, x2 = pos_map[(b1, m)], pos_map[(b2, m)]
            _add_bracket(ax, x1, x2, current_y, label, height=step * 0.2)
            current_y += step

    if args.method_within_bin_csv:
        mdf = pd.read_csv(args.method_within_bin_csv)
        for _, row in mdf.iterrows():
            b = row["bin"]
            m1, m2 = row["model_a"], row["model_b"]
            if (b, m1) not in pos_map or (b, m2) not in pos_map:
                continue
            pval = float(row.get(args.p_col, row.get("p", 1.0)))
            label = _format_p(pval) if args.show_p else ""
            x1, x2 = pos_map[(b, m1)], pos_map[(b, m2)]
            _add_bracket(ax, x1, x2, current_y, label, height=step * 0.2)
            current_y += step

    ax.set_ylim(y_min, current_y + step)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
