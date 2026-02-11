#!/usr/bin/env python3
"""
Summarize per-class F1 (and precision/recall) for multiclass Detectron2 outputs.

Matches predictions to GT by IoU, then computes classification metrics on matched pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score


def _parse_sizes(sizes_arg: str | None, preds_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(preds_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "val_preds.gpkg").exists():
            sizes.append(p.name)
    return sizes


def _parse_sizes_per_chip(preds_root: Path, per_chip_subdir: str) -> list[str]:
    sizes = []
    for p in sorted(preds_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / per_chip_subdir).exists():
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
    pa = argparse.ArgumentParser(description="Per-class metrics from multiclass Detectron2 outputs.")
    pa.add_argument("--val-manifest", type=Path, required=True, help="detectron_manifest.json for val chips")
    pa.add_argument("--preds-root", type=Path, required=True, help="Root with <size>/val_preds.gpkg")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect")
    pa.add_argument("--gt-class-field", type=str, default="class")
    pa.add_argument("--pred-class-field", type=str, default="class")
    pa.add_argument("--iou-thr", type=float, default=0.5)
    pa.add_argument(
        "--use-per-chip",
        action="store_true",
        help="Use per-chip predictions under {preds_root}/{size}/per_chip_preds",
    )
    pa.add_argument(
        "--per-chip-subdir",
        type=str,
        default="per_chip_preds",
        help="Subdir name under each size for per-chip prediction GPKGs",
    )
    pa.add_argument("--out-csv", type=Path, default=None, help="Summary CSV (macro/weighted F1)")
    pa.add_argument(
        "--per-class-csv",
        type=Path,
        default=None,
        help="Per-class metrics CSV (size,class,precision,recall,f1,support)",
    )
    args = pa.parse_args()

    if args.sizes:
        sizes = _parse_sizes(args.sizes, args.preds_root)
    elif args.use_per_chip:
        sizes = _parse_sizes_per_chip(args.preds_root, args.per_chip_subdir)
    else:
        sizes = _parse_sizes(args.sizes, args.preds_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure val_preds.gpkg exists under preds-root.")

    gt = _load_gt(args.val_manifest, args.gt_class_field)
    if gt.empty:
        raise RuntimeError("Ground-truth crowns are empty.")
    labels = sorted(gt[args.gt_class_field].unique().tolist())
    val_pairs = json.load(open(args.val_manifest))

    summary_rows = []
    per_class_rows = []

    for sz in sizes:
        y_true_all = []
        y_pred_all = []
        n_gt = 0
        n_pred = 0
        n_matched = 0

        if args.use_per_chip:
            per_chip_dir = args.preds_root / sz / args.per_chip_subdir
            if not per_chip_dir.exists():
                print(f"[skip] missing per-chip preds: {per_chip_dir}")
                continue

            for rec in val_pairs:
                chip_name = rec.get("name", Path(rec["tif_path"]).stem)
                pred_path = per_chip_dir / f"{chip_name}.gpkg"
                if pred_path.exists():
                    preds = _load_preds(pred_path, args.pred_class_field)
                else:
                    preds = gpd.GeoDataFrame(geometry=[], crs=None)
                gt_chip = gpd.read_file(rec["gpkg_path"])
                if gt_chip.empty:
                    continue
                if args.gt_class_field not in gt_chip.columns:
                    raise RuntimeError(f"Ground-truth crowns missing '{args.gt_class_field}' field.")
                gt_chip[args.gt_class_field] = pd.to_numeric(gt_chip[args.gt_class_field], errors="coerce").astype("Int64")
                gt_chip = gt_chip.dropna(subset=[args.gt_class_field]).copy()
                gt_chip[args.gt_class_field] = gt_chip[args.gt_class_field].astype(int)

                best = _match_by_iou(preds, gt_chip, args.iou_thr)
                n_gt += len(gt_chip)
                n_pred += len(preds)
                n_matched += len(best)

                if best.empty:
                    continue
                y_true = gt_chip.loc[best["gid"], args.gt_class_field].astype(int).values
                y_pred = preds.loc[best["pid"], args.pred_class_field].astype(int).values
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
        else:
            pred_path = args.preds_root / sz / "val_preds.gpkg"
            if not pred_path.exists():
                print(f"[skip] missing preds: {pred_path}")
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
        if args.use_per_chip:
            if not y_true_all:
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
            y_true = np.concatenate(y_true_all)
            y_pred = np.concatenate(y_pred_all)
        else:
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

        rpt = _report_dict(y_true, y_pred, labels)
        for cls in labels:
            entry = rpt.get(str(cls), None)
            if entry is None:
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

    if summary_rows:
        out_csv = args.out_csv or (args.preds_root / "summary_class_metrics.csv")
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")

    if per_class_rows:
        per_csv = args.per_class_csv or (args.preds_root / "per_class_metrics.csv")
        pd.DataFrame(per_class_rows).to_csv(per_csv, index=False)
        print(f"wrote {per_csv}")


if __name__ == "__main__":
    main()
