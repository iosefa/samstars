#!/usr/bin/env python3
"""
Figure 10: Compare OBIA (SAM–StarDist–XGBoost) vs CNN on Set C.

Inputs: two macro CSVs produced by section3_3.py (or your own),
which may use either 'overall_SQ' or 'overall_SQ_weighted'.
We’ll robustly read PQ/SQ/RQ (+ precision/recall/F1 if present) and TP/FP/FN if present.

Usage:
  python section3_3_figures.py \
    --obia_macro setC_xgb_metrics_macro.csv \
    --cnn_macro  setC_cnn_metrics_macro.csv \
    --out_fig    fig10_compare_methods.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick(row: pd.Series, *candidates, default=np.nan):
    """Return first existing column value from candidates."""
    for c in candidates:
        if c in row.index:
            return row[c]
    return default


def load_macro(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    # use first row
    row = df.iloc[0]
    # normalize column names (strip spaces)
    row.index = [str(c).strip() for c in row.index]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obia_macro", required=True, help="CSV with OBIA macro metrics (Set C)")
    ap.add_argument("--cnn_macro",  required=True, help="CSV with CNN macro metrics (Set C)")
    ap.add_argument("--out_fig", default="fig10_compare_methods.png")
    args = ap.parse_args()

    ob = load_macro(Path(args.obia_macro))
    cn = load_macro(Path(args.cnn_macro))

    # Pull metrics with robust fallbacks
    def extract_triplet(row):
        PQ = float(pick(row, "overall_PQ", "PQ"))
        SQ = float(pick(row, "overall_SQ", "overall_SQ_weighted", "SQ"))
        RQ = float(pick(row, "overall_RQ", "RQ"))
        return PQ, SQ, RQ

    def extract_prf(row):
        P = float(pick(row, "macro_precision", "precision", "overall_precision", default=np.nan))
        R = float(pick(row, "macro_recall", "recall", "overall_recall", default=np.nan))
        F = float(pick(row, "macro_F1", "macro_f1", "F1", "f1", default=np.nan))
        return P, R, F

    def extract_counts(row):
        TP = pick(row, "TP", default=np.nan)
        FP = pick(row, "FP", default=np.nan)
        FN = pick(row, "FN", default=np.nan)
        return TP, FP, FN

    ob_PQ, ob_SQ, ob_RQ = extract_triplet(ob)
    cn_PQ, cn_SQ, cn_RQ = extract_triplet(cn)

    ob_P, ob_R, ob_F1 = extract_prf(ob)
    cn_P, cn_R, cn_F1 = extract_prf(cn)

    ob_TP, ob_FP, ob_FN = extract_counts(ob)
    cn_TP, cn_FP, cn_FN = extract_counts(cn)

    # ----------------- Plot -----------------
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })

    metrics = ["PQ", "SQ", "RQ"]
    ob_vals  = [ob_PQ, ob_SQ, ob_RQ]
    cn_vals  = [cn_PQ, cn_SQ, cn_RQ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b1 = ax.bar(x - width/2, ob_vals, width, label="SAM–StarDist–XGB")
    b2 = ax.bar(x + width/2, cn_vals, width, label="CNN baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Figure 10. Set C semantic panoptic quality (IoU ≥ 0.50)")

    # annotate bars
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.legend(loc="upper left", frameon=False, ncol=1)

    # Optional side text: precision/recall/F1 and TP/FP/FN if present
    # Build small text box on the right
    def fmt_counts(TP, FP, FN):
        if np.isnan(TP) or np.isnan(FP) or np.isnan(FN):
            return "TP/FP/FN: n/a"
        return f"TP/FP/FN: {int(TP)}/{int(FP)}/{int(FN)}"

    def fmt_prf(P, R, F):
        if np.isnan(P) or np.isnan(R) or np.isnan(F):
            return "P/R/F1: n/a"
        return f"P/R/F1: {P:.3f}/{R:.3f}/{F:.3f}"

    text_ob = "OBIA\n" + fmt_prf(ob_P, ob_R, ob_F1) + "\n" + fmt_counts(ob_TP, ob_FP, ob_FN)
    text_cn = "CNN\n"  + fmt_prf(cn_P, cn_R, cn_F1) + "\n" + fmt_counts(cn_TP, cn_FP, cn_FN)

    # Place text box to the right of bars
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5)
    ax.text(1.05, 0.70, text_ob, transform=ax.transAxes, fontsize=10, va="top", bbox=props)
    ax.text(1.05, 0.30, text_cn, transform=ax.transAxes, fontsize=10, va="top", bbox=props)

    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {args.out_fig}")


if __name__ == "__main__":
    main()