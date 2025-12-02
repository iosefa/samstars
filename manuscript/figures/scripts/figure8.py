#!/usr/bin/env python3
"""
Class-aware segmentation metrics (Set C)

Inputs (hard-coded; relative to this script):
  - validation:  manuscript/data/set_c/set_c_validation.gpkg (expects 'class' field)
  - SAM–StarDist: manuscript/data/set_c/set_c_sam_predicted.gpkg (class in 'predicted_class')
  - CNN:          manuscript/data/set_c/set_c_cnn_predicted_classified.gpkg (class in 'predicted_class')
  - DeepForest:   manuscript/data/set_c/set_c_df_predicted.gpkg  (class in 'class')

Outputs (under manuscript/figures/output):
  - figure8.png (or name provided via --out-name)
  - figure8_overall_metrics.csv
  - TP/FP/FN GPKGs for each method in output/diags/ (e.g., figure8_sam_tp_pred.gpkg)

Notes:
  - IoU threshold defaults to 0.50 for PQ/SQ/RQ (class-aware); override with --iou-thr.
  - A prediction can only match a reference of the same class.
  - For DeepForest only, IoU is computed using validation bboxes (envelopes) vs. DF predictions (polygons), as in Figure 6.
  - This figure renders a single grouped bar plot (SQ, RQ, PQ) for the selected methods.
  - By default, DeepForest is excluded per session notes; use --include-df to add it.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Tuple

try:
    import fiona
except Exception:
    fiona = None

# ---------------- Config (hard-coded paths) ----------------
IOU_THR = 0.50

ROOT = Path(__file__).resolve().parents[2]  # manuscript
SETC = ROOT / "data/set_c"
REF_PATH = SETC / "set_c_validation.gpkg"
SAM_PATH = SETC / "set_c_sam_predicted.gpkg"
CNN_PATH = SETC / "set_c_cnn_predicted_classified.gpkg"
DF_PATH  = SETC / "set_c_df_predicted.gpkg"

OUT_DIR = ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure8.png"
DIAG_DIR = OUT_DIR / "diags"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Helpers ----------------
def list_layers(path: Path):
    if fiona is None:
        return []
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


def ensure_metric_crs(gdf: gpd.GeoDataFrame, target=None):
    if target is None:
        if (not gdf.crs) or gdf.crs.is_geographic:
            return gdf.to_crs(6635)
        return gdf
    else:
        if (not gdf.crs) or (gdf.crs != target):
            return gdf.to_crs(target)
    return gdf


def greedy_match(df_pairs: pd.DataFrame, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df_pairs is None or df_pairs.empty:
        return df_pairs
    out = df_pairs.sort_values(iou_col, ascending=False)
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out


def compute_counts_and_pq(n_pred: int, n_val: int, pairs_df: pd.DataFrame, iou_col="iou") -> Tuple[float, float, float]:
    TP = int(len(pairs_df)) if pairs_df is not None else 0
    FP = int(n_pred - TP)
    FN = int(n_val - TP)
    SQ = float(pairs_df[iou_col].mean()) if TP else 0.0
    RQ = float(TP / (TP + 0.5 * FP + 0.5 * FN)) if (TP + FP + FN) else 0.0
    PQ = float(SQ * RQ)
    return PQ, SQ, RQ


def load_gpkg_polygons(path: Path):
    lyr = pick_polygon_layer(path)
    try:
        if lyr is None:
            gdf = gpd.read_file(path, engine="fiona")
        else:
            gdf = gpd.read_file(path, layer=lyr, engine="fiona")
    except Exception as e:
        raise FileNotFoundError(f"Unable to read {path}: {e}")
    return gdf


def _standardize_class(series: pd.Series) -> pd.Series:
    """Return canonical class labels as strings so pred/ref compare reliably.
    - Trim whitespace, lowercase for non-numeric.
    - For numeric values close to integers, cast to integer strings (e.g., 1.0 -> '1').
    - For other numerics, keep a compact string representation.
    """
    s = series.astype(object)
    # Start with lowercase trimmed strings
    out = s.astype(str).str.strip().str.lower()
    # Try numeric coercion
    with np.errstate(all='ignore'):
        num = pd.to_numeric(out, errors='coerce')
    mask_num = num.notna() & np.isfinite(num)
    mask_int_like = mask_num & (np.abs(num - np.round(num)) < 1e-6)
    out.loc[mask_int_like] = num.loc[mask_int_like].round().astype(np.int64).astype(str)
    mask_float_like = mask_num & (~mask_int_like)
    out.loc[mask_float_like] = num.loc[mask_float_like].round(6).astype(str)
    return out


def _to_int_series(series: pd.Series) -> pd.Series:
    try:
        s = pd.to_numeric(series, errors='coerce').astype('Int64')
        return s
    except Exception:
        return pd.Series([None]*len(series), dtype='Int64')


def intersections_and_metrics_classaware(
    preds: gpd.GeoDataFrame,
    validation: gpd.GeoDataFrame,
    pred_class_field: str,
    ref_class_field: str = "class",
    ref_use_bbox: bool = False,
):
    preds = preds.reset_index(drop=True).copy(); preds["pid"] = preds.index
    validation = validation.reset_index(drop=True).copy(); validation["vid"] = validation.index
    validation = ensure_metric_crs(validation, target=preds.crs)

    # Clean invalids
    try:
        if not preds.geometry.is_valid.all():
            preds["geometry"] = preds.geometry.buffer(0)
        if not validation.geometry.is_valid.all():
            validation["geometry"] = validation.geometry.buffer(0)
    except Exception:
        pass

    # Prepare classes; attempt numeric alignment like section3_3
    preds["_cls_pred_raw"] = preds.get(pred_class_field)
    validation["_cls_ref_raw"] = validation.get(ref_class_field)
    p_num = _to_int_series(preds["_cls_pred_raw"])  # Int64 or NaNs
    r_num = _to_int_series(validation["_cls_ref_raw"])  # Int64 or NaNs

    # Geometry for validation side (optionally bbox)
    vgeo = validation[["vid", "geometry"]].copy()
    if ref_use_bbox:
        vgeo = vgeo.copy(); vgeo["geometry"] = vgeo.geometry.envelope

    # Intersections
    inter = gpd.overlay(preds[["pid", "geometry"]], vgeo, how="intersection")
    if inter.empty:
        preds["iou_best"] = 0.0
        validation["iou_best_from_pred"] = 0.0
        pairs_all = inter
        pairs_50 = inter
        PQ = SQ = RQ = 0.0
        return inter, preds, validation, PQ, SQ, RQ, pairs_all, pairs_50

    # IoU fields
    ap = preds.set_index("pid").geometry.area
    av = vgeo.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(lower=0.0, upper=1.0)

    # Class alignment: if numeric classes are available, try to align pred → ref
    if p_num.notna().any() and r_num.notna().any():
        # Merge numeric classes into intersections
        inter = inter.merge(preds[["pid"]].assign(pred_cls=p_num), on="pid", how="left")
        inter = inter.merge(validation[["vid"]].assign(ref_cls=r_num), on="vid", how="left")
        # Build counts between pred_cls and ref_cls for mapping (ignore NaNs)
        tbl = inter.dropna(subset=["pred_cls", "ref_cls"])[["pred_cls","ref_cls"]]
        mapping = {}
        if not tbl.empty:
            p_vals = sorted(tbl["pred_cls"].astype(int).unique().tolist())
            r_vals = sorted(tbl["ref_cls"].astype(int).unique().tolist())
            # Simple shift if obvious (pred starts at 0, ref starts at >=1)
            if p_vals and r_vals and min(p_vals) == 0 and min(r_vals) >= 1 and len(p_vals) == len(r_vals):
                shift = min(r_vals) - min(p_vals)
                mapping = {pi: pi + shift for pi in p_vals}
            else:
                # Greedy mapping by frequency
                counts = {
                    int(pi): {int(rj): 0 for rj in r_vals}
                    for pi in p_vals
                }
                for a, b in tbl.values:
                    a = int(a); b = int(b)
                    counts[a][b] = counts[a].get(b, 0) + 1
                assigned = set()
                for pi in p_vals:
                    cand = sorted(counts[pi].items(), key=lambda kv: (-kv[1], kv[0]))
                    for rj, _cnt in cand:
                        if rj not in assigned:
                            mapping[pi] = rj; assigned.add(rj); break
                    if pi not in mapping and cand:
                        mapping[pi] = cand[0][0]
        # Apply mapping to pred_cls
        inter["pred_cls_map"] = inter["pred_cls"].apply(lambda v: mapping.get(int(v), np.nan) if pd.notna(v) else np.nan)
        # Filter to matching classes
        inter = inter.loc[(inter["pred_cls_map"].notna()) & (inter["ref_cls"].notna()) & (inter["pred_cls_map"] == inter["ref_cls"]), ["pid","vid","iou"]].copy()
    else:
        # Fall back to standardized string equality
        preds["cls_pred"] = _standardize_class(preds["_cls_pred_raw"])
        validation["cls_ref"] = _standardize_class(validation["_cls_ref_raw"])
        inter = inter.merge(preds[["pid","cls_pred"]], on="pid", how="left")
        inter = inter.merge(validation[["vid","cls_ref"]], on="vid", how="left")
        inter = inter.loc[inter["cls_pred"] == inter["cls_ref"], ["pid","vid","iou"]].copy()

    # Best IoU per pred/ref for diagnostics
    if inter.empty:
        preds["iou_best"] = 0.0
        validation["iou_best_from_pred"] = 0.0
        pairs_all = inter
        pairs_50 = inter
        PQ = SQ = RQ = 0.0
        return inter, preds, validation, PQ, SQ, RQ, pairs_all, pairs_50

    preds["iou_best"] = inter.groupby("pid")["iou"].max().reindex(preds["pid"], fill_value=0).values
    validation["iou_best_from_pred"] = inter.groupby("vid")["iou"].max().reindex(validation["vid"], fill_value=0).values

    pairs_all = greedy_match(inter.loc[inter["iou"] > 0, ["pid", "vid", "iou"]])
    pairs_50  = greedy_match(inter.loc[inter["iou"] >= IOU_THR, ["pid", "vid", "iou"]])
    PQ, SQ, RQ = compute_counts_and_pq(preds.shape[0], validation.shape[0], pairs_50)
    return inter, preds, validation, PQ, SQ, RQ, pairs_all, pairs_50


def parse_args():
    p = argparse.ArgumentParser(description="Figure 9 — Class-aware SQ/RQ/PQ (Set C)")
    p.add_argument("--iou-thr", type=float, default=0.50, help="IoU threshold for PQ/SQ/RQ (default: 0.50)")
    p.add_argument("--out-name", type=str, default="figure8.png", help="Output image filename under figures/output/")
    p.add_argument("--include-df", action="store_true", help="Include DeepForest in the comparison")
    return p.parse_args()


def main():
    args = parse_args()
    global IOU_THR, OUT_FIG
    IOU_THR = float(args.iou_thr)
    OUT_FIG = OUT_DIR / args.out_name

    # 1) Load reference and predictions
    ref = ensure_metric_crs(load_gpkg_polygons(REF_PATH))
    sam = ensure_metric_crs(load_gpkg_polygons(SAM_PATH), target=ref.crs)
    cnn = ensure_metric_crs(load_gpkg_polygons(CNN_PATH), target=ref.crs)
    dfo = ensure_metric_crs(load_gpkg_polygons(DF_PATH),  target=ref.crs)

    # 2) Prepare reference (no clipping)
    ref_use = ref.reset_index(drop=True)
    ref_use["vid"] = ref_use.index

    # 3) Intersections & metrics per method (class-aware; DF uses bbox on ref side)
    out = {}
    methods = [
        ("SAM–StarDist", sam, "predicted_class", False),
        ("CNN (ResNet-34, dual-pool)", cnn, "predicted_class", False),
    ]
    if args.include_df:
        methods.append(("DeepForest", dfo, "class", True))
    for label, preds, pcls, use_bbox in methods:
        inter, preds2, ref2, PQ, SQ, RQ, pairs_all, pairs_50 = intersections_and_metrics_classaware(
            preds, ref_use, pred_class_field=pcls, ref_class_field="class", ref_use_bbox=use_bbox
        )
        out[label] = dict(preds=preds2, ref=ref2, inter=inter, PQ=PQ, SQ=SQ, RQ=RQ, pairs_all=pairs_all, pairs_50=pairs_50)

    # 4) Single bar plot: SQ, RQ, PQ per method
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    metrics = ["SQ", "RQ", "PQ"]
    labels = [m[0] for m in methods]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][:len(labels)]
    x = np.arange(len(metrics))
    m = len(labels)
    bar_w = 0.75 / m
    for i, (lab, col) in enumerate(zip(labels, colors)):
        vals = [out[lab][k] for k in metrics]
        centers = x + (i - (m-1)/2) * bar_w
        bars = ax.bar(centers, vals, width=bar_w, label=lab, color=col, alpha=0.85)
        # Annotate values above bars
        for cx, v in zip(centers, vals):
            ax.annotate(f"{v:.2f}", xy=(cx, v), xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Semantic Panoptic SQ, RQ, PQ (Set C; IoU ≥ 0.5)")
    ax.legend(frameon=False, loc="upper right")

    # 4b) Single compact top-left box listing TP/FP/FN per method
    lines = ["TP/FP/FN"]
    for lab in labels:
        TP = int(len(out[lab]["pairs_50"]))
        FP = int(len(out[lab]["preds"]) - TP)
        FN = int(len(out[lab]["ref"]) - TP)
        lines.append(f"{lab}: {TP}/{FP}/{FN}")
    txt = "\n".join(lines)
    props = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, linewidth=0.4)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8.5, va="top", ha="left", bbox=props)

    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")

    # 5) Save CSV summaries
    rows = []
    n_ref = int(len(ref_use))
    for lab in labels:
        TP = int(len(out[lab]["pairs_50"]))
        FP = int(len(out[lab]["preds"]) - TP)
        FN = int(n_ref - TP)
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(dict(method=lab,
                         SQ=out[lab]["SQ"], RQ=out[lab]["RQ"], PQ=out[lab]["PQ"],
                         TP=TP, FP=FP, FN=FN,
                         precision=precision, recall50=recall, f1=f1,
                         iou_thr=IOU_THR))
    pd.DataFrame(rows).to_csv(OUT_DIR / "figure8_overall_metrics.csv", index=False)
    print(f"[OK] Saved {OUT_DIR / 'figure8_overall_metrics.csv'}")

    # 6) Export TP/FP/FN GeoPackages per method (for diagnostics)
    prefix_map = {
        "SAM–StarDist": "figure8_sam",
        "CNN (ResNet-34, dual-pool)": "figure8_cnn",
        "DeepForest": "figure8_df",
    }
    for lab in labels:
        prefix = prefix_map.get(lab, f"figure8_{lab.lower().split()[0]}")
        preds2 = out[lab]["preds"].copy()
        ref2 = out[lab]["ref"].copy()
        pairs50 = out[lab]["pairs_50"]
        tp_pids = set(pairs50["pid"].astype(int).tolist()) if pairs50 is not None and not pairs50.empty else set()
        tp_vids = set(pairs50["vid"].astype(int).tolist()) if pairs50 is not None and not pairs50.empty else set()
        tp_pred = preds2.loc[preds2["pid"].astype(int).isin(tp_pids)] if len(preds2) else preds2.iloc[0:0]
        fp_pred = preds2.loc[~preds2["pid"].astype(int).isin(tp_pids)] if len(preds2) else preds2.iloc[0:0]
        fn_ref  = ref2.loc[~ref2["vid"].astype(int).isin(tp_vids)] if len(ref2) else ref2.iloc[0:0]
        tp_pred.to_file(DIAG_DIR / f"{prefix}_tp_pred.gpkg", driver="GPKG")
        fp_pred.to_file(DIAG_DIR / f"{prefix}_fp_pred.gpkg", driver="GPKG")
        fn_ref.to_file(DIAG_DIR / f"{prefix}_fn_ref.gpkg", driver="GPKG")
    print(f"[OK] Exported TP/FP/FN GPKGs to {DIAG_DIR}")


if __name__ == "__main__":
    main()
