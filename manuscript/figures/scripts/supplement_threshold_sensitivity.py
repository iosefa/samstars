#!/usr/bin/env python3
"""
Supplement — IoU threshold sensitivity (0.3 vs 0.5) for Set A (class-agnostic)
and Set C (class-aware).

Outputs:
- figures/output/supp_threshold_sensitivity_seta.csv
- figures/output/supp_threshold_sensitivity_setc.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2]

# Set A
SET_A = ROOT / "data/set_a"
SET_A_REF = SET_A / "validation_crowns.gpkg"
SET_A_SAM = SET_A / "sam_crowns.gpkg"
SET_A_CNN = SET_A / "detectron_crowns.gpkg"
SET_A_DF  = SET_A / "deepforest_crowns.gpkg"

# Set C
SET_C = ROOT / "data/set_c"
SET_C_REF = SET_C / "set_c_validation.gpkg"
SET_C_SAM = SET_C / "set_c_sam_predicted.gpkg"
SET_C_CNN = SET_C / "set_c_cnn_predicted_classified.gpkg"
SET_C_DF  = SET_C / "set_c_df_predicted.gpkg"

OUT_DIR = ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_metric_crs(gdf: gpd.GeoDataFrame, target=None):
    if target is None:
        if (not gdf.crs) or gdf.crs.is_geographic:
            return gdf.to_crs(6635)
        return gdf
    else:
        if (not gdf.crs) or (gdf.crs != target):
            return gdf.to_crs(target)
    return gdf


def greedy_match(df_pairs, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df_pairs is None or df_pairs.empty:
        return df_pairs
    out = df_pairs.sort_values(iou_col, ascending=False)
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out


def compute_counts_and_pq(n_pred, n_val, pairs_df, iou_col="iou"):
    TP = int(len(pairs_df)) if pairs_df is not None else 0
    FP = int(n_pred - TP)
    FN = int(n_val - TP)
    SQ = float(pairs_df[iou_col].mean()) if TP else 0.0
    RQ = float(TP / (TP + 0.5 * FP + 0.5 * FN)) if (TP + FP + FN) else 0.0
    PQ = float(SQ * RQ)
    return PQ, SQ, RQ, TP, FP, FN


def class_agnostic_metrics(preds: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, thr: float):
    preds = preds.reset_index(drop=True).copy(); preds["pid"] = preds.index
    ref = ref.reset_index(drop=True).copy(); ref["vid"] = ref.index
    ref = ensure_metric_crs(ref, target=preds.crs)
    try:
        if not preds.geometry.is_valid.all(): preds["geometry"] = preds.geometry.buffer(0)
        if not ref.geometry.is_valid.all(): ref["geometry"] = ref.geometry.buffer(0)
    except Exception: pass
    inter = gpd.overlay(preds[["pid","geometry"]], ref[["vid","geometry"]], how="intersection")
    if inter.empty:
        return 0.0, 0.0, 0.0, 0, len(preds), len(ref)
    ap = preds.set_index("pid").geometry.area
    av = ref.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(0.0, 1.0)
    pairs = greedy_match(inter.loc[inter["iou"] >= thr, ["pid","vid","iou"]])
    return compute_counts_and_pq(preds.shape[0], ref.shape[0], pairs)


def class_aware_metrics(preds: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, thr: float, pred_class_field: str):
    preds = preds.reset_index(drop=True).copy(); preds["pid"] = preds.index
    ref = ref.reset_index(drop=True).copy(); ref["vid"] = ref.index
    ref = ensure_metric_crs(ref, target=preds.crs)
    try:
        if not preds.geometry.is_valid.all(): preds["geometry"] = preds.geometry.buffer(0)
        if not ref.geometry.is_valid.all(): ref["geometry"] = ref.geometry.buffer(0)
    except Exception: pass
    inter = gpd.overlay(preds[["pid","geometry", pred_class_field]], ref[["vid","geometry","class"]], how="intersection")
    if inter.empty:
        return 0.0, 0.0, 0.0, 0, len(preds), len(ref)
    ap = preds.set_index("pid").geometry.area
    av = ref.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(0.0, 1.0)
    # match only same class
    inter = inter.loc[inter[pred_class_field].astype(str) == inter["class"].astype(str), ["pid","vid","iou"]]
    pairs = greedy_match(inter.loc[inter["iou"] >= thr, ["pid","vid","iou"]])
    return compute_counts_and_pq(preds.shape[0], ref.shape[0], pairs)


def main():
    # Set A
    refA = ensure_metric_crs(gpd.read_file(SET_A_REF))
    samA = ensure_metric_crs(gpd.read_file(SET_A_SAM), target=refA.crs)
    cnnA = ensure_metric_crs(gpd.read_file(SET_A_CNN), target=refA.crs)
    dfA  = ensure_metric_crs(gpd.read_file(SET_A_DF),  target=refA.crs)

    rowsA = []
    for thr in (0.30, 0.50):
        for lab, g in (("SAM–StarDist", samA), ("CNN", cnnA), ("DeepForest", dfA)):
            PQ, SQ, RQ, TP, FP, FN = class_agnostic_metrics(g, refA, thr)
            precision = TP/(TP+FP) if (TP+FP) else 0.0
            recall = TP/(TP+FN) if (TP+FN) else 0.0
            f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
            rowsA.append(dict(set="SetA", method=lab, iou_thr=thr, PQ=PQ, SQ=SQ, RQ=RQ, TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1))
    pd.DataFrame(rowsA).to_csv(OUT_DIR/"supp_threshold_sensitivity_seta.csv", index=False)

    # Set C
    refC = ensure_metric_crs(gpd.read_file(SET_C_REF))
    samC = ensure_metric_crs(gpd.read_file(SET_C_SAM), target=refC.crs)
    cnnC = ensure_metric_crs(gpd.read_file(SET_C_CNN), target=refC.crs)
    rowsC = []
    for thr in (0.30, 0.50):
        for lab, g, cls in (("SAM–StarDist", samC, "predicted_class"), ("CNN", cnnC, "predicted_class")):
            PQ, SQ, RQ, TP, FP, FN = class_aware_metrics(g, refC, thr, cls)
            precision = TP/(TP+FP) if (TP+FP) else 0.0
            recall = TP/(TP+FN) if (TP+FN) else 0.0
            f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
            rowsC.append(dict(set="SetC", method=lab, iou_thr=thr, PQ=PQ, SQ=SQ, RQ=RQ, TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1))
    pd.DataFrame(rowsC).to_csv(OUT_DIR/"supp_threshold_sensitivity_setc.csv", index=False)
    print("[OK] Saved threshold sensitivity CSVs")


if __name__ == "__main__":
    main()

