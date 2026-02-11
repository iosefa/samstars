#!/usr/bin/env python3
# STATUS: active
"""
Bar chart of panoptic metrics (SQ/RQ/PQ) per model at a single size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Okabe-Ito palette (consistent per model)
MODEL_COLORS = {
    "sam-star": "#0072B2",
    "detectron2": "#E69F00",
    "detectron": "#E69F00",
    "detectron2-full": "#009E73",
    "sam-star+xgb": "#CC79A7",
    "detectron2+xgb": "#E69F00",
}
MODEL_KEYS = sorted(MODEL_COLORS.keys(), key=len, reverse=True)


def _model_color(label: str) -> str:
    for key in MODEL_KEYS:
        if label.startswith(key):
            return MODEL_COLORS[key]
    return "#000000"


def _load_row(path: Path, size: str) -> pd.Series:
    df = pd.read_csv(path)
    if "size" not in df.columns:
        raise SystemExit(f"{path} missing 'size' column")
    df["size"] = df["size"].astype(str).str.zfill(4)
    size = str(size).zfill(4)
    row = df.loc[df["size"] == size]
    if row.empty:
        raise SystemExit(f"{path} has no row for size {size}")
    return row.iloc[0]


def main() -> None:
    pa = argparse.ArgumentParser(description="Bar chart of SQ/RQ/PQ per model at one size.")
    pa.add_argument("--summary-csvs", type=str, required=True, help="Comma-separated summary_iou CSVs")
    pa.add_argument("--labels", type=str, required=True, help="Comma-separated labels for each CSV")
    pa.add_argument("--size", type=str, required=True)
    pa.add_argument(
        "--sizes-per-csv",
        type=str,
        default=None,
        help="Optional comma-separated sizes aligned to --summary-csvs.",
    )
    pa.add_argument("--metrics", type=str, default="sq,rq,pq", help="Comma list of metrics (sq,rq,pq)")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--title", type=str, default=None)
    pa.add_argument("--annotate", action="store_true")
    pa.add_argument("--y-max", type=float, default=1.05)
    args = pa.parse_args()

    paths = [Path(p) for p in args.summary_csvs.split(",") if p.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if len(paths) != len(labels):
        raise SystemExit("--summary-csvs and --labels must have the same length")

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    valid = {"sq", "rq", "pq"}
    if not metrics or any(m not in valid for m in metrics):
        raise SystemExit("--metrics must be a comma list of sq,rq,pq")

    sizes_per_csv = None
    if args.sizes_per_csv:
        sizes_per_csv = [s.strip() for s in args.sizes_per_csv.split(",") if s.strip()]
        if len(sizes_per_csv) != len(paths):
            raise SystemExit("--sizes-per-csv must match --summary-csvs length")

    rows = []
    for idx, (path, label) in enumerate(zip(paths, labels)):
        size = sizes_per_csv[idx] if sizes_per_csv else args.size
        row = _load_row(path, size)
        rows.append((label, row))

    x = np.arange(len(metrics), dtype=float)
    n_models = len(labels)
    width = 0.8 / max(n_models, 1)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for i, (label, row) in enumerate(rows):
        vals = [float(row.get(m, 0.0)) for m in metrics]
        pos = x + (i - (n_models - 1) / 2) * width
        bars = ax.bar(pos, vals, width=width * 0.95, color=_model_color(label), label=label)
        if args.annotate:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, args.y_max)
    ax.grid(True, axis="y", alpha=0.3)
    title = args.title or f"Panoptic metrics (size {str(args.size).zfill(4)})"
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=1.0, facecolor="white", edgecolor="black", loc="upper left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
