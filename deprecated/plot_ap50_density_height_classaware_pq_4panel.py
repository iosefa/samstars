#!/usr/bin/env python3
# DEPRECATED: use plot_ap50_pq_line_2panel.py + plot_ap50_density_height_boxplots_2panel.py
"""
Four-panel figure:
  1) AP50 vs training instances (two models)
  2) Density-bin AP50 boxplots (two sizes, two models)
  3) Height-bin AP50 boxplots (two sizes, two models)
  4) Class-aware PQ/RQ/SQ vs training instances (three models)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Okabe-Ito colorblind-friendly palette (consistent per model)
MODEL_COLORS = {
    "sam-star": "#0072B2",
    "detectron2": "#E69F00",
    "detectron": "#E69F00",
    "detectron2-full": "#009E73",
    "sam-star+xgb": "#CC79A7",
    "detectron2+xgb": "#D55E00",
}
MODEL_KEYS = sorted(MODEL_COLORS.keys(), key=len, reverse=True)
SIZE_COLORS = ["#56B4E9", "#F0E442", "#999999", "#CC79A7"]
HATCH_COLOR = "#666666"


def _load_df(path: Path) -> pd.DataFrame:
    """Internal helper function."""
    return pd.read_csv(path)


def _prep_curve(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    origin: bool,
    scale_100: bool,
    counts_map: dict[str, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal helper function."""
    if ycol not in df.columns:
        raise SystemExit(f"Missing column '{ycol}' in {df}")
    if counts_map is not None and "size" in df.columns:
        x = df["size"].astype(str).str.zfill(4).map(counts_map).astype(float).values
    else:
        if xcol not in df.columns:
            raise SystemExit(f"Missing column '{xcol}' in {df}")
        x = df[xcol].astype(float).values
    y = df[ycol].astype(float).values
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


def _base_model_name(label: str) -> str:
    """Internal helper function."""
    for key in MODEL_KEYS:
        if label.startswith(key):
            return key
    return label.split()[0]


def _model_color(label: str) -> str:
    """Internal helper function."""
    base = _base_model_name(label)
    return MODEL_COLORS.get(base, "#000000")


def _load_counts_map(path: Path, count_col: str) -> dict[str, int]:
    """Internal helper function."""
    df = pd.read_csv(path)
    if "size" not in df.columns or count_col not in df.columns:
        raise SystemExit(f"{path} must include size and {count_col} columns")
    return {str(s).zfill(4): int(n) for s, n in zip(df["size"], df[count_col])}


def _load_box_csvs(paths: list[Path], size_labels: list[str], model_order: list[str] | None) -> pd.DataFrame:
    """Internal helper function."""
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


def _boxplot(ax, df: pd.DataFrame, title: str, show_counts: bool, count_mode: str):
    """Internal helper function."""
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
    labels = []
    colors = []
    edges = []
    hatches = []

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

    size_hatch_map = {label: ("//" if i == 1 else "") for i, label in enumerate(size_labels)}

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
                data.append(vals)
                labels.append(f"{bin_label}\n{model} {size_label}")
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
                    label=f"{model} {size_label}",
                )
            )
    ax.legend(
        handles=combo_handles,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        title="model × size",
        loc="upper right",
    )


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Four-panel AP50 + density/height + class-aware PQ figure.")
    pa.add_argument("--ap-summary-a", type=Path, required=True)
    pa.add_argument("--ap-summary-b", type=Path, required=True)
    pa.add_argument("--ap-label-a", type=str, default="sam-star")
    pa.add_argument("--ap-label-b", type=str, default="detectron2")
    pa.add_argument("--ap-metric", type=str, default="segm_AP50")
    pa.add_argument("--ap-scale-100", action="store_true")
    pa.add_argument("--ap-x-max", type=float, default=10000.0)

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
        help="Column name in --train-counts-csv for size labels (default: n_train_unique).",
    )
    pa.add_argument(
        "--x-counts-csv",
        type=Path,
        default=None,
        help="CSV with size and count column for x-axis (defaults to --train-counts-csv).",
    )
    pa.add_argument(
        "--x-counts-col",
        type=str,
        default="n_train_unique",
        help="Column name in --x-counts-csv for x-axis (default: n_train_unique).",
    )
    pa.add_argument("--box-model-order", type=str, default=None, help="Comma-separated model order (after size label)")
    pa.add_argument("--show-counts", action="store_true")
    pa.add_argument(
        "--count-mode",
        type=str,
        choices=["chips", "crowns"],
        default="chips",
    )

    pa.add_argument("--pq-summary-a", type=Path, required=True)
    pa.add_argument("--pq-summary-b", type=Path, required=True)
    pa.add_argument("--pq-summary-c", type=Path, required=True)
    pa.add_argument("--pq-label-a", type=str, default="detectron-mc")
    pa.add_argument("--pq-label-b", type=str, default="sam-star+xgb")
    pa.add_argument("--pq-label-c", type=str, default="detectron+xgb")
    pa.add_argument("--pq-x-max", type=float, default=None)
    pa.add_argument(
        "--pq-metrics",
        type=str,
        default="pq",
        help="Comma-separated metrics to plot in PQ panel (pq,rq,sq). Default: pq",
    )
    pa.add_argument("--origin", action="store_true", help="Prepend (0,0) to curves")

    pa.add_argument("--xcol", type=str, default="n_train_instances")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--title", type=str, default="AP50 and class-aware PQ")
    args = pa.parse_args()

    density_paths = [Path(p) for p in args.density_csvs.split(",") if p.strip()]
    height_paths = [Path(p) for p in args.height_csvs.split(",") if p.strip()]
    size_labels = [s.strip() for s in args.size_labels.split(",") if s.strip()]
    if len(density_paths) != len(height_paths) or len(density_paths) != len(size_labels):
        raise SystemExit("density/height CSV counts must match size-labels")

    counts_map_labels = None
    if args.size_label_mode == "n_train":
        counts_source = args.train_counts_csv or args.x_counts_csv
        if args.train_counts_csv is not None:
            count_col = args.train_counts_col
        else:
            count_col = args.x_counts_col
        if counts_source is None:
            raise SystemExit("--train-counts-csv is required for --size-label-mode n_train")
        counts_map_labels = _load_counts_map(counts_source, count_col)
        new_labels = []
        for s in size_labels:
            key = str(s).zfill(4)
            n = counts_map_labels.get(key)
            if n is None:
                new_labels.append(s)
            else:
                new_labels.append(f"(n={n})")
        size_labels = new_labels

    model_order = None
    if args.box_model_order:
        model_order = [m.strip() for m in args.box_model_order.split(",") if m.strip()]

    counts_map_x = None
    counts_source = args.x_counts_csv or args.train_counts_csv
    count_col = args.x_counts_col
    if counts_source is not None:
        counts_map_x = _load_counts_map(counts_source, count_col)

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_ap = axes[0, 0]
    ax_den = axes[0, 1]
    ax_hgt = axes[1, 0]
    ax_pq = axes[1, 1]

    # Panel 1: AP50 curves
    df_ap_a = _load_df(args.ap_summary_a)
    df_ap_b = _load_df(args.ap_summary_b)
    xa, ya = _prep_curve(df_ap_a, args.xcol, args.ap_metric, args.origin, args.ap_scale_100, counts_map_x)
    xb, yb = _prep_curve(df_ap_b, args.xcol, args.ap_metric, args.origin, args.ap_scale_100, counts_map_x)
    ax_ap.plot(xa, ya, color=_model_color(args.ap_label_a), lw=2, label=args.ap_label_a)
    ax_ap.plot(xb, yb, color=_model_color(args.ap_label_b), lw=2, ls="--", label=args.ap_label_b)
    ax_ap.set_title("AP50 vs training size")
    if counts_map_x is not None and args.xcol == "n_train_instances":
        ax_ap.set_xlabel("Unique training crowns (pre-augmentation)")
    else:
        ax_ap.set_xlabel(
            "Number of training crowns (with augmentations)" if args.xcol == "n_train_instances" else args.xcol
        )
    ax_ap.set_ylabel("Average Precision (AP50)")
    ax_ap.set_xlim(left=0, right=args.ap_x_max)
    ax_ap.set_ylim(0, 1)
    ax_ap.grid(True, alpha=0.3)
    ax_ap.legend(frameon=True, framealpha=1.0, facecolor="white", edgecolor="black", fontsize=10)

    # Panel 2: Density boxplots
    df_den = _load_box_csvs(density_paths, size_labels, model_order)
    _boxplot(ax_den, df_den, "Density bins", args.show_counts, args.count_mode)
    ax_den.set_ylim(0, 1)

    # Panel 3: Height boxplots
    df_hgt = _load_box_csvs(height_paths, size_labels, model_order)
    _boxplot(ax_hgt, df_hgt, "Height bins", args.show_counts, args.count_mode)
    ax_hgt.set_ylim(0, 1)

    # Panel 4: Class-aware PQ/RQ/SQ curves
    df_pq_a = _load_df(args.pq_summary_a)
    df_pq_b = _load_df(args.pq_summary_b)
    df_pq_c = _load_df(args.pq_summary_c)
    metric_list = [m.strip().lower() for m in args.pq_metrics.split(",") if m.strip()]
    valid_metrics = {"pq", "rq", "sq"}
    if not metric_list or any(m not in valid_metrics for m in metric_list):
        raise SystemExit("--pq-metrics must be a comma list of pq,rq,sq")
    metric_styles = {"pq": "-", "rq": "--", "sq": ":"}
    metric_label = ", ".join(m.upper() for m in metric_list)

    if len(metric_list) == 1:
        x1, y1 = _prep_curve(df_pq_a, args.xcol, metric_list[0], args.origin, False, counts_map_x)
        x2, y2 = _prep_curve(df_pq_b, args.xcol, metric_list[0], args.origin, False, counts_map_x)
        x3, y3 = _prep_curve(df_pq_c, args.xcol, metric_list[0], args.origin, False, counts_map_x)
        ax_pq.plot(x1, y1, color=_model_color(args.pq_label_a), lw=2, label=args.pq_label_a)
        ax_pq.plot(x2, y2, color=_model_color(args.pq_label_b), lw=2, ls="--", label=args.pq_label_b)
        ax_pq.plot(x3, y3, color=_model_color(args.pq_label_c), lw=2, ls=":", label=args.pq_label_c)
        ax_pq.set_title(f"Class-aware {metric_label} vs training size")
        ax_pq.set_ylabel(metric_list[0].upper())
    else:
        for label, df in [
            (args.pq_label_a, df_pq_a),
            (args.pq_label_b, df_pq_b),
            (args.pq_label_c, df_pq_c),
        ]:
            for metric in metric_list:
                x, y = _prep_curve(df, args.xcol, metric, args.origin, False, counts_map_x)
                ax_pq.plot(
                    x,
                    y,
                    color=_model_color(label),
                    lw=2 if metric == "pq" else 1.6,
                    ls=metric_styles[metric],
                    label=f"{label} {metric.upper()}",
                )
        ax_pq.set_title(f"Class-aware {metric_label} vs training size")
        ax_pq.set_ylabel("Metric")
    if counts_map_x is not None and args.xcol == "n_train_instances":
        ax_pq.set_xlabel("Unique training crowns (pre-augmentation)")
    else:
        ax_pq.set_xlabel(
            "Number of training crowns (with augmentations)" if args.xcol == "n_train_instances" else args.xcol
        )
    ax_pq.set_xlim(left=0)
    if args.pq_x_max is not None:
        ax_pq.set_xlim(right=args.pq_x_max)
    ax_pq.set_ylim(0, 1)
    ax_pq.grid(True, alpha=0.3)
    ax_pq.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        ncol=2 if len(metric_list) > 1 else 1,
    )

    fig.suptitle(args.title, y=1.02)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
