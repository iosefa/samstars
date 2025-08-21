#!/usr/bin/env python3
"""
Create:
- Figure 9 (two panels):
  A) per-class precision/recall/F1 (grouped bars)
  B) mean |SHAP| per modality (size-normalized)

- Table 2 CSV: per-class metrics (+ macro row).

Run with defaults if you saved files via the classify script.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Defaults
# --------------------------
DEFAULT_METRICS_CSV = "classification_metrics_by_class.csv"
DEFAULT_MODALITY_MEAN = "shap_rollup_modality_MEAN.csv"
DEFAULT_OUT_FIG = "fig8_classification_and_shap.png"
DEFAULT_OUT_TABLE = "table2_classification_metrics.csv"
DEFAULT_LABELS_JSON = "shap_class_names.json"  # optional

DEFAULT_CLASS_ORDER = ["METPOL", "GREROB", "ACAKOA", "DIOSAN", "PROPAL", "OTHER"]
CLASS_CODE_TO_NAME = {0:"METPOL",1:"GREROB",2:"ACAKOA",3:"DIOSAN",4:"PROPAL",5:"OTHER"}

# --------------------------
# Helpers
# --------------------------
def load_labels(labels_json_path: Path | None):
    if labels_json_path and Path(labels_json_path).exists():
        with open(labels_json_path, "r") as f:
            js = json.load(f)
        names = js.get("class_names", [])
        if names:
            return names
    return [CLASS_CODE_TO_NAME[i] for i in sorted(CLASS_CODE_TO_NAME.keys())]

def tidy_metrics(metrics_csv: Path, class_order: list[str], _names: list[str] | None):
    df = pd.read_csv(metrics_csv)

    # find class column
    cname = None
    for cand in ["class_name","class","label","Class","ClassName","class_index"]:
        if cand in df.columns:
            cname = cand; break
    if cname is None:
        raise ValueError(f"Could not find class column in {metrics_csv}.")

    # normalize class names
    def norm(val):
        if pd.isna(val): return None
        if isinstance(val,(int,np.integer)) or (isinstance(val,str) and val.strip().isdigit()):
            return CLASS_CODE_TO_NAME.get(int(val), str(val))
        return str(val)
    df["class_name"] = df[cname].apply(norm)

    # map metric columns
    colmap={}
    for m in ["precision","recall","f1","support"]:
        for cand in [m,m.capitalize(),m.upper(),"f1-score","F1-score","F1"]:
            if cand in df.columns:
                colmap[m]=cand; break
    missing=[m for m in ["precision","recall","f1"] if m not in colmap]
    if missing: raise ValueError(f"Missing metric columns {missing} in {metrics_csv}.")

    # drop aggregate rows if present
    drop_vals={"accuracy","macro avg","weighted avg","micro avg","macro","weighted"}
    df=df[~df["class_name"].astype(str).str.lower().isin(drop_vals)].copy()

    keep=["class_name", colmap["precision"], colmap["recall"], colmap["f1"]]
    if "support" in colmap: keep.append(colmap["support"])
    df=df[keep].rename(columns={
        colmap["precision"]:"precision",
        colmap["recall"]:"recall",
        colmap["f1"]:"f1",
        **({colmap["support"]:"support"} if "support" in colmap else {})
    })

    # order classes if possible
    if set(class_order).issubset(set(df["class_name"])):
        order_map={n:i for i,n in enumerate(class_order)}
        df["__o"]=df["class_name"].map(order_map)
        df=df.sort_values("__o").drop(columns="__o")

    macro=pd.DataFrame({
        "class_name":["MACRO"],
        "precision":[df["precision"].mean()],
        "recall":[df["recall"].mean()],
        "f1":[df["f1"].mean()],
        **({"support":[df["support"].sum()]} if "support" in df.columns else {})
    })
    return df.reset_index(drop=True), macro

def plot_fig(class_df: pd.DataFrame, shap_modality_mean_csv: Path, out_path: Path):
    """
    Panel A: grouped bars (precision/recall/F1) by class with extra intra-group spacing
    Panel B: mean |SHAP| per modality (bar)
    """
    # Read SHAP modality MEAN
    mod = pd.read_csv(shap_modality_mean_csv)
    if "modality" not in mod.columns:
        for cand in ["group", "modality"]:
            if cand in mod.columns:
                if cand != "modality":
                    mod = mod.rename(columns={cand: "modality"})
                break
    if "mean_abs_shap_mean" not in mod.columns:
        for cand in ["mean_abs_shap", "mean_abs_shap_sum", "value", "importance"]:
            if cand in mod.columns:
                mod = mod.rename(columns={cand: "mean_abs_shap_mean"})
                break

    pref_order = ["lidar structure", "lidar intensity", "Spectral (WV3)"]
    # normalize modality labels to your lowercase lidar style
    mod["modality"] = mod["modality"].replace({
        "LiDAR structure": "lidar structure",
        "LiDAR intensity": "lidar intensity"
    })
    if set(pref_order).issubset(set(mod["modality"])):
        mod["__o"] = mod["modality"].map({n:i for i,n in enumerate(pref_order)})
        mod = mod.sort_values("__o").drop(columns="__o")

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

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios":[1.5,1]})

    # -------- Panel A: grouped bars with extra INTRA-group spacing --------
    ax = axes[0]
    cls = class_df["class_name"].tolist()

    # group-to-group spacing
    x = np.arange(len(cls)) * 1   # spread groups a bit more

    # bar width and intra-group gap
    width = 0.25                     # a bit narrower bars
    intra_gap = 0.22                 # extra space BETWEEN the three bars in a group

    # bar center offsets within a group (left, center, right)
    d = width * (1.0 + intra_gap)    # controls intra-group spacing
    offs = np.array([-d, 0.0, +d])

    barsP = ax.bar(x + offs[0], class_df["precision"].values, width, label="Precision")
    barsR = ax.bar(x + offs[1], class_df["recall"].values,    width, label="Recall")
    barsF = ax.bar(x + offs[2], class_df["f1"].values,        width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(cls)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("A) Species classification (hold-out set)")

    # Value labels with a little vertical padding
    def annotate(ax_, bars):
        for b in bars:
            h = b.get_height()
            ax_.annotate(f"{h:.2f}",
                         xy=(b.get_x() + b.get_width()/2, h),
                         xytext=(0, 4),
                         textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=10, clip_on=False)
    for bars in (barsP, barsR, barsF):
        annotate(ax, bars)

    # Legend below panel A with padding
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    # -------- Panel B: mean |SHAP| per modality --------
    ax2 = axes[1]
    mod_plot = mod.copy()
    keep = ["lidar structure","lidar intensity","Spectral (WV3)"]
    if set(keep).issubset(set(mod_plot["modality"])):
        mod_plot = mod_plot.set_index("modality").loc[keep].reset_index()

    bars = ax2.bar(mod_plot["modality"], mod_plot["mean_abs_shap_mean"])
    for i, (lab, val) in enumerate(zip(mod_plot["modality"], mod_plot["mean_abs_shap_mean"])):
        ax2.annotate(f"{val:.3f}", xy=(i, val), xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, clip_on=False)

    # Slight tilt to avoid collisions with panel boundary
    ax2.set_xticklabels(mod_plot["modality"], rotation=12, ha="right")
    ax2.set_ylabel("Mean |SHAP| per feature")
    ax2.set_title("B) Mean feature contribution by modality")
    ax2.set_ylim(0, max(mod_plot["mean_abs_shap_mean"].max()*1.18, 0.01))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure → {out_path}")

def write_table2(class_df: pd.DataFrame, macro_row: pd.DataFrame, out_csv: Path):
    out=class_df.copy()
    for c in ["precision","recall","f1"]:
        out[c]=out[c].astype(float).round(2)
    if "support" in out.columns:
        out["support"]=out["support"].astype(int)
    macro=macro_row.copy()
    for c in ["precision","recall","f1"]:
        macro[c]=macro[c].astype(float).round(2)
    if "support" in macro.columns:
        macro["support"]=macro["support"].astype(int)
    out=pd.concat([out,macro], ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote Table 2 CSV → {out_csv}")

# --------------------------
# CLI
# --------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default=DEFAULT_METRICS_CSV)
    ap.add_argument("--modality_shap_mean_csv", default=DEFAULT_MODALITY_MEAN)
    ap.add_argument("--out_fig", default=DEFAULT_OUT_FIG)
    ap.add_argument("--out_table", default=DEFAULT_OUT_TABLE)
    ap.add_argument("--labels_json", default=DEFAULT_LABELS_JSON)
    args=ap.parse_args()

    _=load_labels(Path(args.labels_json) if args.labels_json else None)

    class_df, macro = tidy_metrics(Path(args.metrics_csv), DEFAULT_CLASS_ORDER, _)
    if set(DEFAULT_CLASS_ORDER).issubset(set(class_df["class_name"])):
        class_df=class_df.set_index("class_name").loc[DEFAULT_CLASS_ORDER].reset_index()

    plot_fig(class_df, Path(args.modality_shap_mean_csv), Path(args.out_fig))
    write_table2(class_df, macro, Path(args.out_table))

if __name__=="__main__":
    main()