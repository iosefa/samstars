#!/usr/bin/env python3
# STATUS: active
"""
Two-panel boxplot figure:
  1) Density-bin AP50 boxplots (two sizes, two models)
  2) Height-bin AP50 boxplots (two sizes, two models)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# Okabe-Ito palette (consistent per model)
MODEL_COLORS = {
    "sam-star": "#0072B2",
    "detectron2": "#E69F00",
    "detectron": "#E69F00",
    "detectron2-full": "#009E73",
    "sam-star+xgb": "#CC79A7",
    "detectron2+xgb": "#D55E00",
}
MODEL_KEYS = sorted(MODEL_COLORS.keys(), key=len, reverse=True)
HATCH_COLOR = "#666666"


def _load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _base_model_name(label: str) -> str:
    for key in MODEL_KEYS:
        if label.startswith(key):
            return key
    return label.split()[0]


def _model_color(label: str) -> str:
    base = _base_model_name(label)
    return MODEL_COLORS.get(base, "#000000")


def _load_counts_map(path: Path, count_col: str) -> dict[str, int]:
    df = pd.read_csv(path)
    if "size" not in df.columns or count_col not in df.columns:
        raise SystemExit(f"{path} must include size and {count_col} columns")
    return {str(s).zfill(4): int(n) for s, n in zip(df["size"], df[count_col])}


def _load_box_csvs(paths: list[Path], size_labels: list[str], model_order: list[str] | None) -> pd.DataFrame:
    frames = []
    for path, label in zip(paths, size_labels):
        df = _load_df(path)
        if "model" not in df.columns or "bin" not in df.columns or "ap50" not in df.columns:
            raise SystemExit(f"Expected columns model/bin/ap50 in {path}")
        df = df.copy()
        df["base_model"] = df["model"].astype(str)
        df["model"] = df["model"].astype(str) + f" {label}"
        df["size_label"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if model_order:
        out = out[out["model"].isin(model_order)]
        out["model"] = pd.Categorical(out["model"], categories=model_order, ordered=True)
        out = out.sort_values(["bin", "model"])
    return out


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _add_bracket(ax, x1: float, x2: float, y: float, text: str, height: float):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#333333", lw=1)
    if text:
        ax.text((x1 + x2) / 2.0, y + height, text, ha="center", va="bottom", fontsize=9, color="#333333")


def _boxplot(
    ax,
    df: pd.DataFrame,
    title: str,
    show_counts: bool,
    count_mode: str,
    bin_pairwise: pd.DataFrame | None = None,
    method_within: pd.DataFrame | None = None,
    p_col: str = "p_adj_holm",
):
    bins = list(df["bin"].dropna().unique())
    models = list(df["base_model"].dropna().unique())
    size_labels = list(df["size_label"].dropna().unique())
    if not bins or not models or not size_labels:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    n_groups = len(models) * len(size_labels)
    width = 0.8 / max(n_groups, 1)
    positions = []
    data = []
    colors = []
    edges = []
    hatches = []
    pos_map = {}

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

    show_size = len(size_labels) > 1
    size_hatch_map = {label: ("//" if i == 1 else "") for i, label in enumerate(size_labels)}
    if not show_size:
        size_hatch_map = {label: "" for label in size_labels}

    for i, bin_label in enumerate(bins):
        combo_idx = 0
        for size_label in size_labels:
            for model in models:
                vals = (
                    df.loc[
                        (df["bin"] == bin_label)
                        & (df["size_label"] == size_label)
                        & (df["base_model"] == model),
                        "ap50",
                    ]
                    .astype(float)
                    .dropna()
                    .values
                )
                pos = i + (combo_idx - (n_groups - 1) / 2) * width
                positions.append(pos)
                pos_map[(bin_label, size_label, model)] = pos
                data.append(vals)
                colors.append(_model_color(model))
                edges.append(HATCH_COLOR if size_hatch_map.get(size_label) else _model_color(model))
                hatches.append(size_hatch_map.get(size_label, None))
                combo_idx += 1

    bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=False)
    for patch, face, edge, hatch in zip(bp["boxes"], colors, edges, hatches):
        patch.set_facecolor(face)
        patch.set_alpha(0.6)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.2)
        if hatch:
            patch.set_hatch(hatch)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(range(len(bins)))
    if show_counts:
        xticklabels = [f"{b}\n(n={bin_counts.get(b, 0)})" for b in bins]
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels(bins)
    ax.set_title(title)
    ax.set_ylabel("AP50")
    ax.grid(True, axis="y", alpha=0.3)

    combo_handles = []
    for size_label in size_labels:
        for model in models:
            combo_handles.append(
                Patch(
                    facecolor=_model_color(model),
                    edgecolor=HATCH_COLOR if size_hatch_map.get(size_label) else _model_color(model),
                    hatch=size_hatch_map.get(size_label, ""),
                    alpha=0.6,
                    label=f"{model} {size_label}" if show_size else f"{model}",
                )
            )
    legend = ax.legend(
        handles=combo_handles,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        title="size" if show_size else None,
        loc="upper right",
    )

    # Significance brackets (asterisks only)
    if bin_pairwise is None and method_within is None:
        return
    if len(size_labels) != 1:
        print("[warn] significance brackets assume a single size_label; skipping")
        return
    # Try to keep brackets below the legend box.
    try:
        ax.figure.canvas.draw()
        bbox = legend.get_window_extent(renderer=ax.figure.canvas.get_renderer())
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        ylim = ax.get_ylim()
        legend_bottom_data = ylim[0] + bbox_axes.y0 * (ylim[1] - ylim[0])
    except Exception:
        legend_bottom_data = None
    size_label = size_labels[0]
    y_all = np.concatenate([d for d in data if len(d)]) if any(len(d) for d in data) else np.array([0.0])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_range = y_max - y_min if y_max > y_min else 0.1
    current_y = y_max + y_range * 0.05
    if legend_bottom_data is not None:
        current_y = min(current_y, legend_bottom_data - y_range * 0.05)
    step = y_range * 0.08

    if bin_pairwise is not None and not bin_pairwise.empty:
        for _, row in bin_pairwise.iterrows():
            b1, b2 = row.get("bin_a"), row.get("bin_b")
            model = row.get("model")
            if (b1, size_label, model) not in pos_map or (b2, size_label, model) not in pos_map:
                continue
            pval = float(row.get(p_col, row.get("p", 1.0)))
            stars = _p_to_stars(pval)
            if not stars:
                continue
            x1 = pos_map[(b1, size_label, model)]
            x2 = pos_map[(b2, size_label, model)]
            _add_bracket(ax, x1, x2, current_y, stars, height=step * 0.2)
            current_y += step

    if method_within is not None and not method_within.empty:
        for _, row in method_within.iterrows():
            b = row.get("bin")
            m1, m2 = row.get("model_a"), row.get("model_b")
            if (b, size_label, m1) not in pos_map or (b, size_label, m2) not in pos_map:
                continue
            pval = float(row.get(p_col, row.get("p", 1.0)))
            stars = _p_to_stars(pval)
            if not stars:
                continue
            x1 = pos_map[(b, size_label, m1)]
            x2 = pos_map[(b, size_label, m2)]
            _add_bracket(ax, x1, x2, current_y, stars, height=step * 0.2)
            current_y += step


def main() -> None:
    pa = argparse.ArgumentParser(description="Two-panel density/height AP50 boxplots.")
    pa.add_argument("--density-csvs", type=str, required=True, help="Comma-separated per-chip density CSVs")
    pa.add_argument("--height-csvs", type=str, required=True, help="Comma-separated per-chip height CSVs")
    pa.add_argument("--size-labels", type=str, required=True, help="Comma-separated size labels (e.g., 0048,0250)")
    pa.add_argument(
        "--size-label-mode",
        type=str,
        choices=["size", "n_train"],
        default="size",
        help="Use size labels or replace with n_train crown counts.",
    )
    pa.add_argument(
        "--train-counts-csv",
        type=Path,
        default=None,
        help="CSV with size and count column (required for --size-label-mode n_train).",
    )
    pa.add_argument(
        "--train-counts-col",
        type=str,
        default="n_train_unique",
        help="Column name in --train-counts-csv for size labels.",
    )
    pa.add_argument("--box-model-order", type=str, default=None, help="Comma-separated model order")
    pa.add_argument("--show-counts", action="store_true")
    pa.add_argument("--count-mode", type=str, choices=["chips", "crowns"], default="chips")
    pa.add_argument("--density-bin-pairwise", type=Path, default=None)
    pa.add_argument("--density-method-within", type=Path, default=None)
    pa.add_argument("--height-bin-pairwise", type=Path, default=None)
    pa.add_argument("--height-method-within", type=Path, default=None)
    pa.add_argument("--p-col", type=str, default="p_adj_holm")
    pa.add_argument("--title", type=str, default="Density/height effects")
    pa.add_argument("--out", type=Path, required=True)
    args = pa.parse_args()

    density_paths = [Path(p) for p in args.density_csvs.split(",") if p.strip()]
    height_paths = [Path(p) for p in args.height_csvs.split(",") if p.strip()]
    size_labels = [s.strip() for s in args.size_labels.split(",") if s.strip()]
    if len(density_paths) != len(height_paths) or len(density_paths) != len(size_labels):
        raise SystemExit("density/height CSV counts must match size-labels")

    if args.size_label_mode == "n_train":
        if args.train_counts_csv is None:
            raise SystemExit("--train-counts-csv is required for --size-label-mode n_train")
        counts_map = _load_counts_map(args.train_counts_csv, args.train_counts_col)
        new_labels = []
        for s in size_labels:
            key = str(s).zfill(4)
            n = counts_map.get(key)
            if n is None:
                new_labels.append(s)
            else:
                new_labels.append(f"(n={n})")
        size_labels = new_labels

    model_order = None
    if args.box_model_order:
        model_order = [m.strip() for m in args.box_model_order.split(",") if m.strip()]

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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_den, ax_hgt = axes

    df_den = _load_box_csvs(density_paths, size_labels, model_order)
    den_pair = pd.read_csv(args.density_bin_pairwise) if args.density_bin_pairwise else None
    den_within = pd.read_csv(args.density_method_within) if args.density_method_within else None
    _boxplot(
        ax_den,
        df_den,
        "Density bins",
        args.show_counts,
        args.count_mode,
        bin_pairwise=den_pair,
        method_within=den_within,
        p_col=args.p_col,
    )
    ax_den.set_ylim(0, 1)

    df_hgt = _load_box_csvs(height_paths, size_labels, model_order)
    hgt_pair = pd.read_csv(args.height_bin_pairwise) if args.height_bin_pairwise else None
    hgt_within = pd.read_csv(args.height_method_within) if args.height_method_within else None
    _boxplot(
        ax_hgt,
        df_hgt,
        "Height bins",
        args.show_counts,
        args.count_mode,
        bin_pairwise=hgt_pair,
        method_within=hgt_within,
        p_col=args.p_col,
    )
    ax_hgt.set_ylim(0, 1)

    if args.title and args.title.strip():
        fig.suptitle(args.title, y=1.02)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
