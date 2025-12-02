#!/usr/bin/env python3
"""
Supplementary Figure — Confusion matrices for Set B classification

Generates:
- supp_confusion_xgb.png (XGBoost confusion matrix)
- supp_confusion_cnn.png (CNN confusion matrix, if available)
- supp_confusion_side_by_side.png (side-by-side panel)

Inputs (defaults):
- Root: classification_confusion_matrix.csv (XGB)
- manuscript/data/set_b/cnn_confusion_set_b.csv (CNN) [optional]
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2].parent

XGB_CONF = ROOT / "manuscript/data/set_b/xgb_confusion_set_b.csv"
CNN_CONF = Path("manuscript/data/set_b/cnn_confusion_set_b.csv")
RF_CONF  = Path("manuscript/data/set_b/rf_confusion_set_b.csv")
MLP_CONF = Path("manuscript/data/set_b/mlp_confusion_set_b.csv")

CLASS_ORDER = ["METPOL","GREROB","ACAKOA","DIOSAN","PROPAL","OTHER"]
CLASS_CODE_TO_NAME = {0:"METPOL",1:"GREROB",2:"ACAKOA",3:"DIOSAN",4:"PROPAL",5:"OTHER"}


def _to_class_names(vals) -> list[str]:
    out = []
    for v in vals:
        s = str(v).strip()
        if s.isdigit():
            name = CLASS_CODE_TO_NAME.get(int(s), s)
        else:
            name = s
        out.append(name)
    return out


def read_conf(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a confusion matrix CSV and return (matrix, labels) in canonical order.
    Supports:
      1) Named headers (METPOL..OTHER)
      2) Numeric headers 0–5 or 1–6
      3) Headerless 6x6 numeric matrix
    """
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        df = None

    if df is None or df.shape[1] != 6 or df.shape[0] != 6:
        # Try flexible fallback (headerless or header row 0..5)
        df2 = pd.read_csv(path, header=None)
        if df2.shape[1] == 6:
            # If first row is 0..5 headers, drop it
            first = list(df2.iloc[0].astype(str)) if df2.shape[0] >= 1 else []
            if first == ['0','1','2','3','4','5'] and df2.shape[0] >= 7:
                df2 = df2.iloc[1:7]
            # If more than 6 rows, take last 6
            if df2.shape[0] > 6:
                df2 = df2.tail(6)
            if df2.shape == (6, 6):
                df2.index = CLASS_ORDER
                df2.columns = CLASS_ORDER
                return df2.values.astype(float), CLASS_ORDER
        # If still not 6x6, raise
        raise ValueError(f"Unexpected confusion matrix shape in {path}")

    # With headers/index: normalize to class names
    df.index = _to_class_names(df.index)
    df.columns = _to_class_names(df.columns)
    # Also handle 1–6 numeric codes by mapping code-1
    def norm_name(s):
        st = str(s).strip()
        if st.isdigit():
            k = int(st)
            if k in range(0,6):
                return CLASS_CODE_TO_NAME.get(k, st)
            if k in range(1,7):
                return CLASS_CODE_TO_NAME.get(k-1, st)
        return st
    df.index = [norm_name(s) for s in df.index]
    df.columns = [norm_name(s) for s in df.columns]

    # Reindex to canonical order; if any class missing, fill with zeros
    df = df.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
    return df.values.astype(float), CLASS_ORDER


def plot_conf(M: np.ndarray, labels: list[str], title: str, out_png: Path):
    # Publication style
    mpl.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=300)
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # annotate
    vmax = M.max() if M.size else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = int(M[i, j])
            ax.text(j, i, str(val), ha='center', va='center',
                    color='black' if val < 0.6*vmax else 'white', fontsize=12)
    # Compact colorbar to the right of the axis
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def load_report_any(path: Path) -> tuple[dict, float]:
    """Parse a sklearn-style classification report CSV and return (per-class f1 dict, macro f1)."""
    df = pd.read_csv(path)
    per = {}
    macro = float('nan')
    f1col = 'f1-score' if 'f1-score' in df.columns else ('f1' if 'f1' in df.columns else None)
    if 'label' in df.columns and f1col:
        for _, row in df.iterrows():
            lab = str(row['label']).strip()
            low = lab.lower()
            if low == 'macro avg':
                macro = float(row[f1col])
            elif low in {'accuracy','weighted avg','micro avg'}:
                continue
            else:
                name = CLASS_CODE_TO_NAME.get(int(lab), lab) if lab.isdigit() else lab
                if name in CLASS_ORDER:
                    per[name] = float(row[f1col])
    return per, macro


def main():
    if XGB_CONF.exists():
        Mx, labels = read_conf(XGB_CONF)
        plot_conf(Mx, labels, "XGBoost — Confusion Matrix (Set B)", OUT_DIR/"supp_confusion_xgb.png")
    else:
        print(f"[WARN] Missing {XGB_CONF}")

    Mc, labels_c = None, None
    if CNN_CONF.exists():
        Mc, labels_c = read_conf(CNN_CONF)
        plot_conf(Mc, labels_c, "CNN — Confusion Matrix (Set B)", OUT_DIR/"supp_confusion_cnn.png")
    else:
        print(f"[WARN] Missing {CNN_CONF}")

    # RF
    Mrf, labels_rf = None, None
    if RF_CONF.exists():
        Mrf, labels_rf = read_conf(RF_CONF)
        plot_conf(Mrf, labels_rf, "Random Forest — Confusion Matrix (Set B)", OUT_DIR/"supp_confusion_rf.png")
    else:
        print(f"[WARN] Missing {RF_CONF}")

    # MLP
    Mmlp, labels_mlp = None, None
    if MLP_CONF.exists():
        Mmlp, labels_mlp = read_conf(MLP_CONF)
        plot_conf(Mmlp, labels_mlp, "MLP — Confusion Matrix (Set B)", OUT_DIR/"supp_confusion_mlp.png")
    else:
        print(f"[WARN] Missing {MLP_CONF}")

    # Side-by-side panel explicitly disabled per request

    # ---- Save per-class F1 summary table across classifiers ----
    rows = []
    # XGB per-class from root metrics if present
    xgb_metrics = ROOT / "classification_metrics_by_class.csv"
    if xgb_metrics.exists():
        per, macro = load_report_any(xgb_metrics)
        rows.append({'Classifier':'XGB','MacroF1': macro, **{k: per.get(k, np.nan) for k in CLASS_ORDER}})
    # MLP
    mlp_rep = Path("manuscript/data/set_b/mlp_report_set_b.csv")
    if mlp_rep.exists():
        per, macro = load_report_any(mlp_rep)
        rows.append({'Classifier':'MLP','MacroF1': macro, **{k: per.get(k, np.nan) for k in CLASS_ORDER}})
    # RF
    rf_rep = Path("manuscript/data/set_b/rf_report_set_b.csv")
    if rf_rep.exists():
        per, macro = load_report_any(rf_rep)
        rows.append({'Classifier':'RF','MacroF1': macro, **{k: per.get(k, np.nan) for k in CLASS_ORDER}})
    # CNN
    cnn_rep = Path("manuscript/data/set_b/cnn_report_set_b.csv")
    if cnn_rep.exists():
        per, macro = load_report_any(cnn_rep)
        rows.append({'Classifier':'CNN','MacroF1': macro, **{k: per.get(k, np.nan) for k in CLASS_ORDER}})

    if rows:
        cols = ['Classifier','MacroF1'] + CLASS_ORDER
        pd.DataFrame(rows)[cols].to_csv(OUT_DIR/"supp_table_classifiers_f1.csv", index=False)
        print(f"[OK] Saved {OUT_DIR/'supp_table_classifiers_f1.csv'}")

    # 2x2 grid if all four available
    if all([p.exists() for p in (XGB_CONF, CNN_CONF, RF_CONF, MLP_CONF)]):
        mpl.rcParams.update({
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 12,
        })
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.8), dpi=300)
        confs = [
            (Mx, '(a) XGBoost', labels),
            (Mc, '(b) CNN', labels_c),
            (Mrf, '(c) Random Forest', labels_rf),
            (Mmlp, '(d) MLP', labels_mlp),
        ]
        for ax, (M, title, labels0) in zip(axes.flatten(), confs):
            im = ax.imshow(M, cmap='Blues')
            ax.set_xticks(range(len(labels0)))
            ax.set_xticklabels(labels0, rotation=40, ha='right')
            ax.set_yticks(range(len(labels0)))
            ax.set_yticklabels(labels0)
            ax.set_title(title)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            # annotate values inside cells
            vmax = M.max() if M.size else 1
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = int(M[i, j])
                    ax.text(j, i, str(val), ha='center', va='center',
                            color=('black' if val < 0.6 * vmax else 'white'), fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = OUT_DIR/"supp_confusion_all_models.png"
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
