#!/usr/bin/env python3
"""
Figure 7 — CNN vs XGBoost classification

Panel (a): Per-class F1 side-by-side bars comparing CNN (dual-encoder) vs XGBoost

Defaults assume:
- CNN report CSV at manuscript/data/set_b/cnn_report_set_b.csv (sklearn report-style)
- XGB metrics CSV at classification_metrics_by_class.csv (from classify.py)
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Defaults
# --------------------------
DEFAULT_CNN_REPORT = "manuscript/data/set_b/cnn_report_set_b.csv"
DEFAULT_XGB_METRICS = "classification_metrics_by_class.csv"
DEFAULT_OUT_FIG = "figure7.png"
DEFAULT_LABELS_JSON = "shap_class_names.json"  # optional, maps indices->names

CLASS_CODE_TO_NAME = {0:"METPOL",1:"GREROB",2:"ACAKOA",3:"DIOSAN",4:"PROPAL",5:"OTHER"}
CLASS_ORDER = ["METPOL", "GREROB", "ACAKOA", "DIOSAN", "PROPAL", "OTHER"]


def load_labels(labels_json_path: Path | None):
    if labels_json_path and Path(labels_json_path).exists():
        try:
            with open(labels_json_path, "r") as f:
                js = json.load(f)
            names = js.get("class_names", [])
            if names:
                return names
        except Exception:
            pass
    return [CLASS_CODE_TO_NAME[i] for i in sorted(CLASS_CODE_TO_NAME.keys())]


def tidy_metrics_any(metrics_csv: Path):
    """Return per-class precision/recall/f1 DataFrame with 'class_name'.
    Supports either sklearn-classification_report CSV (transposed dict)
    or the XGB metrics CSV produced by classify.py.
    """
    df = pd.read_csv(metrics_csv)

    # Try to detect sklearn report layout: often has columns like 'label', 'precision','recall','f1-score','support'
    # and class rows named '0','1','2',... with 'accuracy','macro avg' rows we should drop.
    if 'label' in df.columns and 'f1-score' in df.columns:
        # drop aggregate rows
        agg = set(["accuracy","macro avg","weighted avg","micro avg"]) \
              | set(["accuracy","macro","weighted","micro"]) \
              | set(["MACRO","Weighted","Macro"])
        d1 = df[~df['label'].astype(str).str.lower().isin(agg)].copy()
        # map label -> class_name
        def to_name(x):
            sx = str(x).strip()
            if sx.isdigit():
                return CLASS_CODE_TO_NAME.get(int(sx), sx)
            return sx
        d1['class_name'] = d1['label'].apply(to_name)
        # normalize columns
        d1 = d1.rename(columns={'f1-score':'f1'})
        keep = ['class_name']
        for col in ['precision','recall','f1','support']:
            if col in d1.columns:
                keep.append(col)
        return d1[keep].reset_index(drop=True)

    # Otherwise, assume XGB metrics CSV: has a class column under various names
    cname = None
    for cand in ["class_name","class","label","Class","ClassName","class_index"]:
        if cand in df.columns:
            cname = cand; break
    if cname is None:
        raise ValueError(f"Could not find class column in {metrics_csv}.")

    def norm(val):
        if pd.isna(val): return None
        if isinstance(val,(int,np.integer)) or (isinstance(val,str) and val.strip().isdigit()):
            return CLASS_CODE_TO_NAME.get(int(val), str(val))
        return str(val)
    df['class_name'] = df[cname].apply(norm)

    # map metric columns
    colmap={}
    for m in ["precision","recall","f1","support"]:
        for cand in [m,m.capitalize(),m.upper(),"f1-score","F1-score","F1"]:
            if cand in df.columns:
                colmap[m]=cand; break
    keep=['class_name'] + [c for k,c in colmap.items() if k in ("precision","recall","f1")]
    if 'support' in colmap: keep.append(colmap['support'])
    out=df[keep].rename(columns={
        colmap.get('precision','precision'):'precision',
        colmap.get('recall','recall'):'recall',
        colmap.get('f1','f1'):'f1',
        **({colmap['support']:'support'} if 'support' in colmap else {})
    })
    return out.reset_index(drop=True)


def ensure_order(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    if set(order).issubset(set(df['class_name'])):
        return df.set_index('class_name').loc[order].reset_index()
    return df


def plot_figure(cnn_df: pd.DataFrame, xgb_df: pd.DataFrame, out_path: Path,
                cnn_label: str = "CNN (ResNet-34, dual-pool)", stats_lines: list[str] | None = None,
                stats_header: str = "TP/FP/FN"):
    # Order both by CLASS_ORDER
    cnn_df = ensure_order(cnn_df, CLASS_ORDER)
    xgb_df = ensure_order(xgb_df, CLASS_ORDER)

    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.8))

    # Panel (a): F1 by class (CNN vs XGB)
    classes = cnn_df['class_name'].tolist()
    x = np.arange(len(classes))
    width = 0.35
    col_xgb = "#1f77b4"  # blue
    col_cnn = "#ff7f0e"  # orange
    b1 = ax.bar(x - width/2, xgb_df['f1'].values, width, label='XGB (XGBoost)', color=col_xgb, alpha=0.9)
    b2 = ax.bar(x + width/2, cnn_df['f1'].values, width, label=cnn_label, color=col_cnn, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.20)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel('F1 score')
    ax.set_title('F1 by class: XGB vs CNN', fontsize=15)
    ax.legend(frameon=False, loc='upper right')
    ax.yaxis.grid(True, linestyle='-', alpha=0.2)
    # annotate values
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 6),
                        textcoords='offset points', ha='center', va='bottom', fontsize=10, clip_on=False)

    # Compact top-left box with summary lines if provided (e.g., F1 + TP/FP/FN)
    if stats_lines:
        txt = "\n".join([stats_header, *stats_lines])
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray', alpha=0.85))


    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cnn_report', default=DEFAULT_CNN_REPORT)
    ap.add_argument('--xgb_metrics', default=DEFAULT_XGB_METRICS)
    ap.add_argument('--labels_json', default=DEFAULT_LABELS_JSON)
    ap.add_argument('--out_fig', default=str(Path('manuscript/figures/output')/DEFAULT_OUT_FIG))
    ap.add_argument('--cnn_label', default='CNN (ResNet-34, dual-pool)', help='Label to use for the CNN method in legend/box')
    args = ap.parse_args()

    # load names (optional)
    _ = load_labels(Path(args.labels_json) if args.labels_json else None)

    cnn_df = tidy_metrics_any(Path(args.cnn_report))
    xgb_df = tidy_metrics_any(Path(args.xgb_metrics))

    # ensure both have same classes and order
    if set(CLASS_ORDER).issubset(set(cnn_df['class_name'])):
        cnn_df = cnn_df.set_index('class_name').loc[CLASS_ORDER].reset_index()
    if set(CLASS_ORDER).issubset(set(xgb_df['class_name'])):
        xgb_df = xgb_df.set_index('class_name').loc[CLASS_ORDER].reset_index()

    # Build lines combining macro F1 and TP/FP/FN from optional confusion matrices
    def tp_fp_fn_from_conf(csv_path: Path):
        try:
            df = pd.read_csv(csv_path, index_col=0)
            M = df.values.astype(float)
            tp = int(np.trace(M))
            total = int(M.sum())
            fp = total - tp
            fn = total - tp
            return tp, fp, fn
        except Exception:
            return None

    xgb_conf = Path(args.xgb_metrics).with_name('manuscript/data/set_b/xgb_confusion_set_b.csv')
    cnn_conf = Path(args.cnn_report).with_name('manuscript/data/set_b/cnn_confusion_set_b.csv')
    # Macro F1s from per-class tables
    try:
        f1_xgb = float(pd.to_numeric(xgb_df['f1'], errors='coerce').mean())
    except Exception:
        f1_xgb = float('nan')
    try:
        f1_cnn = float(pd.to_numeric(cnn_df['f1'], errors='coerce').mean())
    except Exception:
        f1_cnn = float('nan')

    lines = []
    tpf = tp_fp_fn_from_conf(xgb_conf)
    if tpf:
        lines.append(f"XGBoost: F1 {f1_xgb:.2f} | {tpf[0]}/{tpf[1]}/{tpf[2]}")
    else:
        lines.append(f"XGBoost: F1 {f1_xgb:.2f}")
    tpf2 = tp_fp_fn_from_conf(cnn_conf)
    if tpf2:
        lines.append(f"{args.cnn_label}: F1 {f1_cnn:.2f} | {tpf2[0]}/{tpf2[1]}/{tpf2[2]}")
    else:
        lines.append(f"{args.cnn_label}: F1 {f1_cnn:.2f}")

    plot_figure(cnn_df, xgb_df, Path(args.out_fig),
                cnn_label=args.cnn_label, stats_lines=lines, stats_header="Macro F1 | TP/FP/FN")


if __name__ == '__main__':
    main()
