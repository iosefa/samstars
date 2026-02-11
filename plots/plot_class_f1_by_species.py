#!/usr/bin/env python3
"""
Plot model-level F1 (macro) and per-class F1 side by side for one size.

Expected per-class CSV columns: size, class, f1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Okabe-Ito palette (consistent per model)
MODEL_COLORS = {
    "sam-star": "#0072B2",
    "sam-stars": "#0072B2",
    "detectron2": "#E69F00",
    "detectron": "#E69F00",
    "detectron2-full": "#009E73",
    "sam-star+xgb": "#0072B2",
    "sam-stars+xgb": "#0072B2",
    "detectron2+xgb": "#E69F00",
}
MODEL_KEYS = sorted(MODEL_COLORS.keys(), key=len, reverse=True)

CLASS_ID_TO_NAME = {
    1: "METPOL",
    2: "GREROB",
    3: "ACAKOA",
    4: "DIOSAN",
    5: "PROPAL",
    6: "OTHER",
}
CLASS_ORDER = ["METPOL", "GREROB", "ACAKOA", "DIOSAN", "PROPAL", "OTHER"]


def _model_color(label: str) -> str:
    for key in MODEL_KEYS:
        if label.startswith(key):
            return MODEL_COLORS[key]
    return "#000000"


def _load_per_class(path: Path, size: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("size", "class", "f1"):
        if col not in df.columns:
            raise SystemExit(f"{path} missing required column '{col}'")
    size = str(size).zfill(4)
    df["size"] = df["size"].astype(str).str.zfill(4)
    df = df[df["size"] == size].copy()
    if df.empty:
        raise SystemExit(f"No rows for size {size} in {path}")
    df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["class"]).copy()
    df["class"] = df["class"].astype(int)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce").fillna(0.0).astype(float)
    for col in ("precision", "recall", "support"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    pa = argparse.ArgumentParser(description="Bar chart of per-class F1 by method for one size.")
    pa.add_argument("--per-class-csvs", type=str, required=True, help="Comma-separated per-class CSVs")
    pa.add_argument("--labels", type=str, required=True, help="Comma-separated labels for each CSV")
    pa.add_argument("--size", type=str, required=True, help="Default size to plot (e.g., 0064)")
    pa.add_argument(
        "--sizes-per-csv",
        type=str,
        default=None,
        help="Optional comma-separated sizes aligned to --per-class-csvs.",
    )
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--title", type=str, default=None)
    pa.add_argument("--rotate", type=int, default=20, help="Rotate x tick labels (degrees)")
    args = pa.parse_args()

    paths = [Path(p) for p in args.per_class_csvs.split(",") if p.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if len(paths) != len(labels):
        raise SystemExit("--per-class-csvs and --labels must have the same length")

    size_by_label = {}
    if args.sizes_per_csv:
        size_list = [s.strip() for s in args.sizes_per_csv.split(",") if s.strip()]
        if len(size_list) != len(labels):
            raise SystemExit("--sizes-per-csv must match --per-class-csvs length")
        size_by_label = {label: size for label, size in zip(labels, size_list)}

    # Enforce requested label order when present
    desired_order = ["sam-star+xgb", "detectron2+xgb", "detectron2-full"]
    if all(lbl in labels for lbl in desired_order):
        label_to_path = {lbl: path for lbl, path in zip(labels, paths)}
        labels = [lbl for lbl in desired_order if lbl in label_to_path]
        paths = [label_to_path[lbl] for lbl in labels]

    dfs = []
    for path, label in zip(paths, labels):
        size = size_by_label.get(label, args.size)
        df = _load_per_class(path, size)
        df["label"] = label
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    macro_f1 = {}
    for label in labels:
        df = all_df[all_df["label"] == label].copy()
        macro_f1[label] = float(df["f1"].mean()) if not df.empty else 0.0

    classes = sorted(all_df["class"].unique().tolist())
    class_names = [CLASS_ID_TO_NAME.get(c, str(c)) for c in classes]
    if all(name in CLASS_ORDER for name in class_names):
        ordered = [(name, c) for name, c in zip(class_names, classes)]
        ordered.sort(key=lambda x: CLASS_ORDER.index(x[0]))
        class_names = [n for n, _ in ordered]
        classes = [c for _, c in ordered]
    n_methods = len(labels)
    width = 0.8 / max(n_methods, 1)
    x = np.arange(len(classes), dtype=float)

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
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]})

    # Left: model-level macro F1
    x_left = np.arange(len(labels), dtype=float)
    y_left = [macro_f1.get(label, 0.0) for label in labels]
    bars_left = ax_left.bar(
        x_left,
        y_left,
        color=[_model_color(label) for label in labels],
        width=0.6,
    )
    ax_left.set_xticks(x_left)
    ax_left.set_xticklabels(labels, rotation=args.rotate, ha="right")
    ax_left.set_ylabel("F1 score")
    ax_left.set_ylim(0, 1.05)
    ax_left.set_title("Model-level F1")
    ax_left.grid(True, axis="y", alpha=0.3)
    for bar in bars_left:
        height = bar.get_height()
        ax_left.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    bar_containers = []
    for i, label in enumerate(labels):
        df = all_df[all_df["label"] == label]
        f1_map = {int(c): float(f) for c, f in zip(df["class"], df["f1"])}
        y = [f1_map.get(c, 0.0) for c in classes]
        pos = x + (i - (n_methods - 1) / 2) * width
        bars = ax_right.bar(
            pos,
            y,
            width=width * 0.95,
            color=_model_color(label),
            label=label,
        )
        bar_containers.append(bars)

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(class_names, rotation=args.rotate, ha="right")
    ax_right.set_ylabel("F1 score")
    ax_right.set_ylim(0, 1.05)
    ax_right.set_title("Per-class F1")
    ax_right.grid(True, axis="y", alpha=0.3)
    ax_right.legend(frameon=True, framealpha=1.0, facecolor="white", edgecolor="black", loc="lower left")

    # Label bars
    for bars in bar_containers:
        for bar in bars:
            height = bar.get_height()
            ax_right.annotate(
                f"{height:.2f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or f"F1 scores (size {str(args.size).zfill(4)})"
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
