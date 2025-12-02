#!/usr/bin/env python3
"""
Supplement — Per-plot PQ/SQ/RQ for Set A (class-agnostic)

Reads six validation GPKGs (setA_validation1..6.gpkg) and intersects with
SAM, CNN, DeepForest predictions to produce per-plot metrics at IoU ≥ 0.5.

Output: figures/output/supp_seta_perplot_metrics.csv
Columns: plot_id, method, PQ, SQ, RQ, TP, FP, FN, precision, recall50, f1
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

IOU_THR = 0.50

ROOT = Path(__file__).resolve().parents[2]  # manuscript
DATA = ROOT / "data/set_a"
OUT_DIR = ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "supp_seta_perplot_metrics.csv"


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


def intersections_and_metrics(preds: gpd.GeoDataFrame, validation: gpd.GeoDataFrame):
    preds = preds.reset_index(drop=True).copy(); preds["pid"] = preds.index
    validation = validation.reset_index(drop=True).copy(); validation["vid"] = validation.index
    validation = ensure_metric_crs(validation, target=preds.crs)
    try:
        if not preds.geometry.is_valid.all():
            preds["geometry"] = preds.geometry.buffer(0)
        if not validation.geometry.is_valid.all():
            validation["geometry"] = validation.geometry.buffer(0)
    except Exception:
        pass
    inter = gpd.overlay(preds[["pid","geometry"]], validation[["vid","geometry"]], how="intersection")
    if inter.empty:
        return 0.0, 0.0, 0.0, 0, len(preds), len(validation)
    ap = preds.set_index("pid").geometry.area
    av = validation.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(0.0, 1.0)
    pairs_50 = greedy_match(inter.loc[inter["iou"] >= IOU_THR, ["pid","vid","iou"]])
    PQ, SQ, RQ, TP, FP, FN = compute_counts_and_pq(preds.shape[0], validation.shape[0], pairs_50)
    return PQ, SQ, RQ, TP, FP, FN


def load_poly(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def main():
    sam = ensure_metric_crs(load_poly(DATA/"sam_crowns.gpkg"))
    cnn = ensure_metric_crs(load_poly(DATA/"detectron_crowns.gpkg"), target=sam.crs)
    df  = ensure_metric_crs(load_poly(DATA/"deepforest_crowns.gpkg"), target=sam.crs)

    rows = []
    for i in range(1, 7):
        ref = ensure_metric_crs(load_poly(DATA/f"setA_validation{i}.gpkg"), target=sam.crs)
        # Restrict predictions to ref bbox for speed
        minx, miny, maxx, maxy = ref.total_bounds
        bbox = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=ref.crs)
        def clip_to(g):
            try:
                return g.clip(bbox)
            except Exception:
                return g
        for lab, preds in (("SAM–StarDist", clip_to(sam)),
                           ("CNN", clip_to(cnn)),
                           ("DeepForest", clip_to(df))):
            PQ, SQ, RQ, TP, FP, FN = intersections_and_metrics(preds, ref)
            precision = TP / (TP + FP) if (TP + FP) else 0.0
            recall = TP / (TP + FN) if (TP + FN) else 0.0
            f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
            rows.append(dict(plot_id=i, method=lab, PQ=PQ, SQ=SQ, RQ=RQ,
                              TP=TP, FP=FP, FN=FN, precision=precision, recall50=recall, f1=f1))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
