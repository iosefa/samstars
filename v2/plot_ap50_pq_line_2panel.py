#!/usr/bin/env python3
# STATUS: active
"""
Two-panel line figure:
  1) AP50 (or AP metric) vs training instances (two models)
  2) Class-aware PQ/RQ/SQ vs training instances (three models)
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
    "sam-star+xgb": "#0072B2",
    "detectron2+xgb": "#E69F00",
}
MODEL_KEYS = sorted(MODEL_COLORS.keys(), key=len, reverse=True)


def _load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_counts_map(path: Path, count_col: str) -> dict[str, int]:
    df = pd.read_csv(path)
    if "size" not in df.columns or count_col not in df.columns:
        raise SystemExit(f"{path} must include size and {count_col} columns")
    return {str(s).zfill(4): int(n) for s, n in zip(df["size"], df[count_col])}


def _base_model_name(label: str) -> str:
    for key in MODEL_KEYS:
        if label.startswith(key):
            return key
    return label.split()[0]


def _model_color(label: str) -> str:
    base = _base_model_name(label)
    return MODEL_COLORS.get(base, "#000000")


def _prep_curve(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    origin: bool,
    scale_100: bool,
    counts_map: dict[str, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
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


def _ap_ylabel(metric: str) -> str:
    label = metric.upper()
    if label.endswith("AP50"):
        return "Average Precision (AP50)"
    if label.endswith("AP75"):
        return "Average Precision (AP75)"
    return "Average Precision (AP)"


def main() -> None:
    pa = argparse.ArgumentParser(description="Two-panel AP and class-aware PQ/RQ/SQ line figure.")
    pa.add_argument("--ap-summary-a", type=Path, required=True)
    pa.add_argument("--ap-summary-b", type=Path, required=True)
    pa.add_argument("--ap-label-a", type=str, default="sam-star")
    pa.add_argument("--ap-label-b", type=str, default="detectron2")
    pa.add_argument("--ap-metric", type=str, default="segm_AP50")
    pa.add_argument("--ap-scale-100", action="store_true")
    pa.add_argument("--ap-x-max", type=float, default=10000.0)

    pa.add_argument("--pq-summary-a", type=Path, required=True)
    pa.add_argument("--pq-summary-b", type=Path, required=True)
    pa.add_argument("--pq-summary-c", type=Path, required=True)
    pa.add_argument("--pq-label-a", type=str, default="detectron2-full")
    pa.add_argument("--pq-label-b", type=str, default="sam-star+xgb")
    pa.add_argument("--pq-label-c", type=str, default="detectron2+xgb")
    pa.add_argument("--pq-metrics", type=str, default="pq", help="Comma list of pq,rq,sq")
    pa.add_argument("--pq-x-max", type=float, default=None)

    pa.add_argument("--xcol", type=str, default="n_train_instances")
    pa.add_argument("--x-counts-csv", type=Path, default=None)
    pa.add_argument("--x-counts-col", type=str, default="n_train_unique")
    pa.add_argument("--origin", action="store_true")
    pa.add_argument("--title", type=str, default="AP and class-aware metrics vs training size")
    pa.add_argument("--out", type=Path, required=True)
    args = pa.parse_args()

    counts_map_x = None
    if args.x_counts_csv is not None:
        counts_map_x = _load_counts_map(args.x_counts_csv, args.x_counts_col)

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
    ax_ap, ax_pq = axes

    # AP curve panel
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
    ax_ap.set_ylabel(_ap_ylabel(args.ap_metric))
    ax_ap.set_xlim(left=0, right=args.ap_x_max)
    ax_ap.set_ylim(0, 1.2)
    ax_ap.set_yticks(np.linspace(0, 1.0, 6))
    ax_ap.grid(True, alpha=0.3)
    ax_ap.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        loc="upper center",
        ncol=2,
    )

    # PQ/RQ/SQ panel
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
        ax_pq.set_ylabel("Metric")
    ax_pq.set_title(f"Class-aware {metric_label} vs training size")
    if counts_map_x is not None and args.xcol == "n_train_instances":
        ax_pq.set_xlabel("Unique training crowns (pre-augmentation)")
    else:
        ax_pq.set_xlabel(
            "Number of training crowns (with augmentations)" if args.xcol == "n_train_instances" else args.xcol
        )
    ax_pq.set_xlim(left=0)
    if args.pq_x_max is not None:
        ax_pq.set_xlim(right=args.pq_x_max)
    ax_pq.set_ylim(0, 1.2)
    ax_pq.set_yticks(np.linspace(0, 1.0, 6))
    ax_pq.grid(True, alpha=0.3)
    ax_pq.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        loc="upper center",
        ncol=2 if len(metric_list) > 1 else 1,
    )

    fig.suptitle(args.title, y=1.02)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
