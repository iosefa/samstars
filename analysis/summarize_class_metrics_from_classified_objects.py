#!/usr/bin/env python3
"""
Summarize class-aware metrics from classified objects vs ground-truth crowns.

Uses a fixed validation manifest to load GT crowns (with class field) and
classified_objects.gpkg per size (with predicted_class).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def _parse_sizes(sizes_arg: str | None, classified_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(classified_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "classified_objects.gpkg").exists():
            sizes.append(p.name)
    return sizes


def _load_gt(manifest_path: Path, class_field: str) -> gpd.GeoDataFrame:
    pairs = json.load(open(manifest_path))
    gdfs = []
    for rec in pairs:
        gpkg = Path(rec["gpkg_path"])
        if not gpkg.exists():
            continue
        gdf = gpd.read_file(gpkg)
        gdfs.append(gdf)
    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    gt = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    if class_field not in gt.columns:
        raise RuntimeError(f"Ground-truth crowns missing '{class_field}' field.")
    gt[class_field] = pd.to_numeric(gt[class_field], errors="coerce").astype("Int64")
    gt = gt.dropna(subset=[class_field]).copy()
    gt[class_field] = gt[class_field].astype(int)
    return gt


def _load_preds(path: Path, pred_field: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if pred_field not in gdf.columns:
        raise RuntimeError(f"Predictions missing '{pred_field}' field: {path}")
    gdf[pred_field] = pd.to_numeric(gdf[pred_field], errors="coerce").astype("Int64")
    gdf = gdf.dropna(subset=[pred_field]).copy()
    gdf[pred_field] = gdf[pred_field].astype(int)
    return gdf


def _match_by_iou(preds: gpd.GeoDataFrame, gt: gpd.GeoDataFrame, iou_thr: float) -> pd.DataFrame:
    preds = preds.reset_index(drop=True).copy()
    gt = gt.reset_index(drop=True).copy()
    preds["pid"] = preds.index
    gt["gid"] = gt.index

    if preds.crs and gt.crs and preds.crs != gt.crs:
        gt = gt.to_crs(preds.crs)
    elif preds.crs and gt.crs is None:
        gt = gt.set_crs(preds.crs)

    pairs = gpd.sjoin(
        preds[["pid", "geometry"]],
        gt[["gid", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if pairs.empty:
        return pairs

    pairs = pairs.merge(gt[["gid", "geometry"]], on="gid", suffixes=("", "_gt"))
    inter = pairs.geometry.intersection(pairs.geometry_gt)
    inter_area = inter.area
    pred_area = pairs.geometry.area
    gt_area = pairs.geometry_gt.area
    pairs["iou"] = inter_area / (pred_area + gt_area - inter_area + 1e-12)

    pairs = pairs.sort_values("iou", ascending=False)
    best = pairs.drop_duplicates("gid")
    best = best[best["iou"] >= iou_thr].copy()
    return best


def _report_dict(y_true, y_pred, labels) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )


def main() -> None:
    pa = argparse.ArgumentParser(description="Summarize class-aware metrics from classified objects.")
    pa.add_argument("--val-manifest", type=Path, required=True, help="detectron_manifest.json for val chips")
    pa.add_argument("--classified-root", type=Path, required=True, help="Root with <size>/classified_objects.gpkg")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect")
    pa.add_argument("--gt-class-field", type=str, default="class")
    pa.add_argument("--pred-class-field", type=str, default="predicted_class")
    pa.add_argument("--iou-thr", type=float, default=0.5)
    pa.add_argument("--out-csv", type=Path, default=None)
    pa.add_argument("--per-class-csv", type=Path, default=None, help="Optional per-class metrics CSV")
    args = pa.parse_args()

    sizes = _parse_sizes(args.sizes, args.classified_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure classified_objects.gpkg exists.")

    gt = _load_gt(args.val_manifest, args.gt_class_field)
    if gt.empty:
        raise RuntimeError("Ground-truth crowns are empty.")

    labels = sorted(gt[args.gt_class_field].unique().tolist())

    summary_rows = []
    per_class_rows = []

    for sz in sizes:
        pred_path = args.classified_root / sz / "classified_objects.gpkg"
        if not pred_path.exists():
            print(f"[skip] missing classified objects: {pred_path}")
            continue

        preds = _load_preds(pred_path, args.pred_class_field)
        if preds.empty:
            print(f"[skip] empty predictions: {pred_path}")
            continue

        best = _match_by_iou(preds, gt, args.iou_thr)
        n_gt = len(gt)
        n_pred = len(preds)
        n_matched = len(best)

        if best.empty:
            summary_rows.append({
                "size": sz,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "n_matched": 0,
                "match_rate": 0.0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
            })
            print(f"[ok] {sz}: no matches (iou>={args.iou_thr})")
            continue

        y_true = gt.loc[best["gid"], args.gt_class_field].astype(int).values
        y_pred = preds.loc[best["pid"], args.pred_class_field].astype(int).values

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        match_rate = n_matched / n_gt if n_gt else 0.0

        summary_rows.append({
            "size": sz,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_matched": n_matched,
            "match_rate": match_rate,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })
        print(f"[ok] {sz}: match_rate={match_rate:.3f} acc={acc:.3f} macro_f1={macro_f1:.3f}")

        if args.per_class_csv:
            rpt = _report_dict(y_true, y_pred, labels)
            for cls in labels:
                entry = rpt.get(str(cls), None)
                if entry is None:
                    # sklearn may use int keys
                    entry = rpt.get(cls, None)
                if entry is None:
                    continue
                per_class_rows.append({
                    "size": sz,
                    "class": cls,
                    "precision": entry.get("precision", 0.0),
                    "recall": entry.get("recall", 0.0),
                    "f1": entry.get("f1-score", 0.0),
                    "support": entry.get("support", 0),
                })

    df = pd.DataFrame(summary_rows)
    out_csv = args.out_csv or (args.classified_root / "summary_class_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    if args.per_class_csv and per_class_rows:
        per_df = pd.DataFrame(per_class_rows)
        per_df.to_csv(args.per_class_csv, index=False)
        print(f"wrote {args.per_class_csv}")


if __name__ == "__main__":
    main()
