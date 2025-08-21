#!/usr/bin/env python3
"""
Section 3.3 – Set C semantic evaluation + Figure 10 (OBIA vs CNN), no CLI args.

What it does:
  1) Loads Set C reference crowns (with class labels) and two prediction layers:
        - OBIA (SAM–StarDist–XGBoost) polygons with predicted_class
        - CNN polygons with pred_id (0-indexed) or predicted_class
  2) Auto-aligns class codes (handles 0/1-based; adds +1 when using pred_id).
  3) Computes semantic PQ on IoU ≥ 0.50 (overall + per-class) and exports CSVs.
  4) Builds Figure 10 comparing PQ/SQ/RQ for the two methods (bars + side stats).

Edit the paths in the CONFIG block if needed, then:  python section3_3_allinone.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt

# ======================
# CONFIG (adjust paths)
# ======================
REF_GPKG        = Path("area51_labelled_validated_subset1.gpkg")  # Set C reference crowns
REF_LAYER       = None                                            # autodetect polygon layer if None
REF_CLASS_COL   = "class"                                         # 1..6

OBIA_GPKG       = Path("area51_predicted_subset1.gpkg")           # OBIA predictions
OBIA_LAYER      = None                                            # autodetect polygon layer if None
OBIA_CLASS_COLS = ["predicted_class", "pred_class", "class"]      # search order

CNN_GPKG        = Path("predicted_cnn_ss1.gpkg")                  # CNN predictions
CNN_LAYER       = None                                            # autodetect polygon layer if None
CNN_CLASS_COLS  = ["predicted_class", "pred_class", "pred_id"]    # search order, 'pred_id' is 0-indexed

IOU_THR         = 0.50

# Output names
OUT_OBIA_PREFIX = "setC_xgb"
OUT_CNN_PREFIX  = "setC_cnn"
OUT_FIG10       = "fig10_compare_methods.png"

# ======================
# Utilities
# ======================
def list_layers(path: Path):
    try:
        return fiona.listlayers(path)
    except Exception:
        return []

def pick_polygon_layer(path: Path):
    layers = list_layers(path)
    if not layers:
        return None
    for ly in layers:
        with fiona.open(path, layer=ly) as src:
            if src.schema["geometry"] in ("Polygon", "MultiPolygon"):
                return ly
    return layers[0]

def read_layer(path: Path, layer=None):
    lyr = layer or pick_polygon_layer(path)
    if lyr is None:
        raise FileNotFoundError(f"No usable layers in {path}")
    return gpd.read_file(path, layer=lyr)

def require_cols(df, names, where=""):
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {where or 'GeoDataFrame'}. Present: {list(df.columns)}")

def ensure_same_crs(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if a.crs is None or b.crs is None:
        raise RuntimeError("Both GeoDataFrames must have a valid CRS.")
    return b if a.crs == b.crs else b.to_crs(a.crs)

def greedy_match(df_pairs, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df_pairs.empty:
        return df_pairs
    out = df_pairs.sort_values(iou_col, ascending=False)
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out

def compute_counts_and_pq(n_pred, n_val, pairs_df, iou_col="iou"):
    TP = int(len(pairs_df))
    FP = int(n_pred - TP)
    FN = int(n_val  - TP)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    SQ        = float(pairs_df[iou_col].mean()) if TP else 0.0
    RQ        = TP / (TP + 0.5*FP + 0.5*FN) if (TP + FP + FN) else 0.0
    PQ        = SQ * RQ
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, SQ=SQ, RQ=RQ, PQ=PQ)

def pick_first_col(gdf: gpd.GeoDataFrame, candidates):
    for c in candidates:
        if c in gdf.columns:
            return c
    raise KeyError(f"None of the candidate class columns found: {candidates}")

def normalize_class_series(s: pd.Series) -> pd.Series:
    """Coerce to Int64, drop NaN."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def auto_align_classes(pred_s: pd.Series, ref_s: pd.Series) -> pd.Series:
    """
    Align predicted classes to reference (typically 1..6).
    Rules:
      - If the predicted column is named 'pred_id', assume it's 0-indexed and add +1.
      - Else, if min(pred)=0 and min(ref)>=1 and max(pred)+1==max(ref), shift +1.
      - Else, leave as-is.
    """
    pred = normalize_class_series(pred_s).copy()
    ref  = normalize_class_series(ref_s).copy()
    if pred.name == "pred_id":
        return pred + 1
    if pred.dropna().size and ref.dropna().size:
        pmin, pmax = int(pred.min()), int(pred.max())
        rmin, rmax = int(ref.min()),  int(ref.max())
        if pmin == 0 and rmin >= 1 and (pmax + 1 == rmax or (rmax - rmin) == (pmax - pmin)):
            return pred + 1
    return pred

def compute_intersections_semantic(pred: gpd.GeoDataFrame,
                                   ref: gpd.GeoDataFrame,
                                   iou_thr: float,
                                   pred_class_col: str,
                                   ref_class_col: str):
    """Return: per-class metrics DF, macro row (dict), pairs table, intersections GDF."""
    # IDs
    pred = pred.reset_index(drop=True).copy()
    pred["pid"] = pred.index
    ref  = ref.reset_index(drop=True).copy()
    ref["vid"]  = ref.index

    # same CRS
    ref = ensure_same_crs(pred, ref)

    # candidate pairs
    cand = gpd.sjoin(pred[["pid","geometry"]], ref[["vid","geometry"]], how="inner", predicate="intersects")
    if cand.empty:
        raise RuntimeError("No overlaps between predictions and reference.")

    # exact IoU
    pred_sub = pred.loc[cand["pid"], ["pid","geometry"]].rename(columns={"geometry":"geom_p"})
    ref_sub  = ref.loc[cand["vid"],  ["vid","geometry"]].rename(columns={"geometry":"geom_v"})
    m = cand.drop(columns=["index_right"]).merge(pred_sub, on="pid").merge(ref_sub, on="vid")
    inter = m.apply(lambda r: r["geom_p"].intersection(r["geom_v"]), axis=1)
    a_int = inter.area
    a_uni = (m["geom_p"].area + m["geom_v"].area - a_int).replace(0, np.nan)
    m["iou"] = (a_int / a_uni).fillna(0.0)

    # keep IoU ≥ thr and add classes
    keep = m[m["iou"] >= iou_thr][["pid","vid","iou"]]
    keep = (keep
            .merge(pred[["pid", pred_class_col]].rename(columns={pred_class_col:"pred_cls"}), on="pid", how="left")
            .merge(ref[["vid",  ref_class_col]].rename(columns={ref_class_col:"ref_cls"}),   on="vid", how="left"))

    # Greedy 1–1 matches per class
    classes = sorted(pd.unique(ref[ref_class_col].dropna().astype(int)))
    rows = []
    all_pairs = []
    for c in classes:
        preds_c = pred[pred[pred_class_col] == c][["pid"]]
        refs_c  = ref[ref[ref_class_col]   == c][["vid"]]
        cand_c  = keep[(keep["pred_cls"] == c) & (keep["ref_cls"] == c)][["pid","vid","iou"]]
        match_c = greedy_match(cand_c)
        stats   = compute_counts_and_pq(preds_c.shape[0], refs_c.shape[0], match_c)
        rows.append({"class": c, **stats})
        match_c = match_c.copy()
        match_c["class"] = c
        all_pairs.append(match_c)

    per_class = pd.DataFrame(rows).sort_values("PQ", ascending=False)
    pairs_df  = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=["pid","vid","iou","class"])

    # macro + overall (TP-weighted SQ)
    TP = int(per_class["TP"].sum())
    FP = int(per_class["FP"].sum())
    FN = int(per_class["FN"].sum())
    overall_SQ_weighted = float((per_class["SQ"] * per_class["TP"]).sum() / TP) if TP else 0.0
    overall_RQ = TP / (TP + 0.5*FP + 0.5*FN) if (TP + FP + FN) else 0.0
    overall_PQ = overall_SQ_weighted * overall_RQ

    macro = {
        "macro_precision": float(per_class["precision"].mean()) if not per_class.empty else np.nan,
        "macro_recall":    float(per_class["recall"].mean())    if not per_class.empty else np.nan,
        "macro_F1":        float((2*per_class["precision"]*per_class["recall"]/(per_class["precision"]+per_class["recall"]).replace([np.inf, -np.inf], np.nan)).mean())
                           if not per_class.empty else np.nan,
        "macro_SQ":        float(per_class["SQ"].mean())        if not per_class.empty else np.nan,
        "macro_RQ":        float(per_class["RQ"].mean())        if not per_class.empty else np.nan,
        "macro_PQ":        float(per_class["PQ"].mean())        if not per_class.empty else np.nan,
        "overall_SQ_weighted": overall_SQ_weighted,
        "overall_RQ": overall_RQ,
        "overall_PQ": overall_PQ,
        "TP": TP, "FP": FP, "FN": FN,
        "iou_threshold": IOU_THR,
    }

    # intersections GeoDataFrame
    inter_gdf = gpd.GeoDataFrame(
        keep.merge(pairs_df[["pid","vid","iou"]], on=["pid","vid","iou"], how="left"),
        geometry=inter, crs=pred.crs
    )

    return per_class, macro, pairs_df, inter_gdf

def evaluate_method(pred_path: Path, pred_layer, class_candidates, out_prefix: str,
                    ref_path: Path, ref_layer, ref_class_col: str):
    """Loads data, aligns classes, evaluates, saves CSV/GeoPackage, returns macro Series."""
    # Load
    ref  = read_layer(ref_path,  ref_layer)
    pred = read_layer(pred_path, pred_layer)

    # columns
    require_cols(ref,  [ref_class_col, "geometry"], where="reference")
    pred_class_col = pick_first_col(pred, class_candidates)
    require_cols(pred, [pred_class_col, "geometry"], where=f"pred ({pred_path.name})")

    # normalize classes
    ref_cls = normalize_class_series(ref[ref_class_col])
    pred_cls_aligned = auto_align_classes(pred[pred_class_col].rename(pred_class_col), ref_cls)

    ref  = ref.assign(ref_cls = ref_cls).dropna(subset=["ref_cls"]).copy()
    pred = pred.assign(pred_cls = pred_cls_aligned).dropna(subset=["pred_cls"]).copy()
    ref["ref_cls"]   = ref["ref_cls"].astype(int)
    pred["pred_cls"] = pred["pred_cls"].astype(int)

    # evaluate
    per_class, macro, pairs_df, inter_gdf = compute_intersections_semantic(
        pred, ref, IOU_THR, "pred_cls", "ref_cls"
    )

    # save
    per_class.to_csv(f"{out_prefix}_metrics_per_class.csv", index=False)
    pd.DataFrame([macro]).to_csv(f"{out_prefix}_metrics_macro.csv", index=False)
    pairs_df.to_csv(f"{out_prefix}_pairs.csv", index=False)

    inter_gpkg = f"{out_prefix}_intersections.gpkg"
    try:
        Path(inter_gpkg).unlink(missing_ok=True)
    except Exception:
        pass
    if not inter_gdf.empty:
        inter_gdf.to_file(inter_gpkg, driver="GPKG")

    print(f"[OK] {out_prefix}: overall_PQ={macro['overall_PQ']:.3f}, SQ={macro['overall_SQ_weighted']:.3f}, RQ={macro['overall_RQ']:.3f} "
          f"(TP={macro['TP']}, FP={macro['FP']}, FN={macro['FN']})")
    return pd.Series(macro)

def load_macro(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty: raise ValueError(f"{path} is empty.")
    row = df.iloc[0].copy()
    row.index = [str(c).strip() for c in row.index]
    return row

def pick(row: pd.Series, *candidates, default=np.nan):
    for c in candidates:
        if c in row.index:
            return row[c]
    return default

def make_fig10(ob_macro_csv: Path, cnn_macro_csv: Path, out_fig: str):
    ob = load_macro(ob_macro_csv)
    cn = load_macro(cnn_macro_csv)

    def extract_triplet(row):
        PQ = float(pick(row, "overall_PQ", "PQ"))
        SQ = float(pick(row, "overall_SQ_weighted", "overall_SQ", "SQ"))
        RQ = float(pick(row, "overall_RQ", "RQ"))
        return PQ, SQ, RQ

    def extract_prf(row):
        P = float(pick(row, "macro_precision", "precision", "overall_precision", default=np.nan))
        R = float(pick(row, "macro_recall",    "recall",    "overall_recall",    default=np.nan))
        F = float(pick(row, "macro_F1", "macro_f1", "F1", "f1", default=np.nan))
        return P, R, F

    def extract_counts(row):
        TP = pick(row, "TP", default=np.nan)
        FP = pick(row, "FP", default=np.nan)
        FN = pick(row, "FN", default=np.nan)
        return TP, FP, FN

    ob_PQ, ob_SQ, ob_RQ = extract_triplet(ob)
    cn_PQ, cn_SQ, cn_RQ = extract_triplet(cn)
    ob_P, ob_R, ob_F1   = extract_prf(ob)
    cn_P, cn_R, cn_F1   = extract_prf(cn)
    ob_TP, ob_FP, ob_FN = extract_counts(ob)
    cn_TP, cn_FP, cn_FN = extract_counts(cn)

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
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    b1 = ax.bar(x - width/2, [ob_PQ, ob_SQ, ob_RQ], width, label="SAM–StarDist–XGB")
    b2 = ax.bar(x + width/2, [cn_PQ, cn_SQ, cn_RQ], width, label="CNN baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.legend(loc="upper right", frameon=False, ncol=1)  # moved legend


    def fmt_counts(TP, FP, FN):
        if np.isnan(TP) or np.isnan(FP) or np.isnan(FN):
            return "TP/FP/FN: n/a"
        return f"TP/FP/FN: {int(TP)}/{int(FP)}/{int(FN)}"

    def fmt_prf(P, R, F):
        if np.isnan(P) or np.isnan(R) or np.isnan(F):
            return "P/R/F1: n/a"
        return f"P/R/F1: {P:.3f}/{R:.3f}/{F:.3f}"

    text_ob = "SAM–StarDist–XGB\n" + fmt_prf(ob_P, ob_R, ob_F1) + "\n" + fmt_counts(ob_TP, ob_FP, ob_FN)
    text_cn = "CNN baseline\n" + fmt_prf(cn_P, cn_R, cn_F1) + "\n" + fmt_counts(cn_TP, cn_FP, cn_FN)

    # Place text boxes in the top left inside the plot
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5)

    # OBIA (SAM–StarDist–XGB)
    ax.text(0.02, 0.95, text_ob,
            transform=ax.transAxes, fontsize=9, va="top", ha="left", bbox=props)
    # CNN
    ax.text(0.02, 0.80, text_cn,
            transform=ax.transAxes, fontsize=9, va="top", ha="left", bbox=props)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_fig}")

# ======================
# Run (no CLI needed)
# ======================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # OBIA
    obia_macro = evaluate_method(
        pred_path=OBIA_GPKG, pred_layer=OBIA_LAYER, class_candidates=OBIA_CLASS_COLS,
        out_prefix=OUT_OBIA_PREFIX,
        ref_path=REF_GPKG, ref_layer=REF_LAYER, ref_class_col=REF_CLASS_COL
    )
    # CNN
    cnn_macro = evaluate_method(
        pred_path=CNN_GPKG, pred_layer=CNN_LAYER, class_candidates=CNN_CLASS_COLS,
        out_prefix=OUT_CNN_PREFIX,
        ref_path=REF_GPKG, ref_layer=REF_LAYER, ref_class_col=REF_CLASS_COL
    )

    # Figure 10
    make_fig10(
        Path(f"{OUT_OBIA_PREFIX}_metrics_macro.csv"),
        Path(f"{OUT_CNN_PREFIX}_metrics_macro.csv"),
        OUT_FIG10
    )