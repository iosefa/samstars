#!/usr/bin/env python3
# STATUS: active
"""
Summarize COCO AP metrics across multiple Detectron eval runs against a fixed val set.

Reads per-size val_metrics.json written by eval_detectron_fixed_val.py and writes a
summary CSV for plotting/comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid


AP_KEYS = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
AP_THRESHES = [0.5 + 0.05 * i for i in range(10)]


def _parse_sizes_arg(sizes_arg: str | None, preds_root: Path, metrics_name: str) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(preds_root.iterdir()):
        if not p.is_dir():
            continue
        metrics_path = p / metrics_name
        if metrics_path.exists():
            sizes.append(p.name)
    return sizes


def _count_train_instances(manifest_path: Path) -> int:
    pairs = json.load(open(manifest_path))
    total = 0
    for rec in pairs:
        gpkg = Path(rec["gpkg_path"])
        gdf = gpd.read_file(gpkg)
        total += int(len(gdf))
    return total


def _load_val_stats(manifest_path: Path | None) -> tuple[int, int]:
    if manifest_path is None:
        return 0, 0
    pairs = json.load(open(manifest_path))
    n_images = len(pairs)
    total = 0
    for rec in pairs:
        gpkg = Path(rec["gpkg_path"])
        gdf = gpd.read_file(gpkg)
        total += int(len(gdf))
    return n_images, total


def _extract_metrics(metrics: dict, task: str) -> dict:
    task_metrics = metrics.get(task, {})
    if not isinstance(task_metrics, dict):
        return {}
    if not any(k in task_metrics for k in AP_KEYS):
        return {}
    out = {}
    for k in AP_KEYS:
        out[f"{task}_{k}"] = task_metrics.get(k, float("nan"))
    return out


def _valid_polys(geoms):
    polys = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        try:
            gv = make_valid(g)
        except Exception:
            try:
                gv = g.buffer(0)
            except Exception:
                continue
        if gv.is_empty:
            continue
        if gv.geom_type == "Polygon":
            polys.append(gv)
        elif gv.geom_type == "MultiPolygon":
            polys.extend([p for p in gv.geoms if p.area > 0])
    return polys


def _geom_iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def _compute_ap(gt_polys, pred_polys, pred_scores, iou_thr):
    if not gt_polys:
        return float("nan")
    if not pred_polys:
        return 0.0
    order = np.argsort(-np.array(pred_scores))
    preds = [pred_polys[i] for i in order]
    matched = [False] * len(gt_polys)
    tp = []
    fp = []
    for p in preds:
        best_iou = 0.0
        best_idx = -1
        for j, g in enumerate(gt_polys):
            if matched[j]:
                continue
            iou = _geom_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_thr and best_idx >= 0:
            matched[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / max(len(gt_polys), 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    return float(np.trapz(mpre, mrec))


def _compute_ap_from_gpkg(preds_gpkg: Path, val_manifest: Path) -> dict:
    preds = gpd.read_file(preds_gpkg)
    val_pairs = json.load(open(val_manifest))
    val_gdfs = []
    for rec in val_pairs:
        gpkg = Path(rec["gpkg_path"])
        val_gdfs.append(gpd.read_file(gpkg))
    validation = gpd.GeoDataFrame(pd.concat(val_gdfs, ignore_index=True))

    # align CRS
    if preds.crs and validation.crs and preds.crs != validation.crs:
        validation = validation.to_crs(preds.crs)
    elif preds.crs and validation.crs is None:
        validation = validation.set_crs(preds.crs)

    gt_polys = _valid_polys(validation.geometry)
    pred_polys = []
    pred_scores = []
    if "score" in preds.columns:
        for geom, score in zip(preds.geometry, preds["score"]):
            vals = _valid_polys([geom])
            for v in vals:
                pred_polys.append(v)
                pred_scores.append(float(score))
    else:
        for geom in preds.geometry:
            vals = _valid_polys([geom])
            for v in vals:
                pred_polys.append(v)
                pred_scores.append(1.0)

    ap_vals = [_compute_ap(gt_polys, pred_polys, pred_scores, thr) for thr in AP_THRESHES]
    ap = float(np.nanmean(ap_vals)) if ap_vals else float("nan")
    ap50 = _compute_ap(gt_polys, pred_polys, pred_scores, 0.5)
    ap75 = _compute_ap(gt_polys, pred_polys, pred_scores, 0.75)
    return {
        "segm_AP": ap,
        "segm_AP50": ap50,
        "segm_AP75": ap75,
        "segm_APs": float("nan"),
        "segm_APm": float("nan"),
        "segm_APl": float("nan"),
    }


def main() -> None:
    pa = argparse.ArgumentParser(description="Summarize COCO AP metrics for fixed-val Detectron runs.")
    pa.add_argument("--preds-root", type=Path, default=Path("data/detectron_fixed_val"))
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect from preds-root")
    pa.add_argument("--metrics-name", type=str, default="val_metrics.json")
    pa.add_argument("--gpkg-name", type=str, default="val_preds.gpkg")
    pa.add_argument(
        "--train-manifests-root",
        type=Path,
        default=Path("data/chips_128_100"),
        help="Root containing {size}/chips_train/detectron_manifest.json",
    )
    pa.add_argument("--val-manifest", type=Path, default=None, help="Optional; include val image/instance counts.")
    pa.add_argument("--include-bbox", action="store_true", help="Include bbox AP metrics alongside segm.")
    pa.add_argument(
        "--use-gpkg-ap",
        action="store_true",
        help="Compute AP from prediction polygons in --gpkg-name (ignore COCO metrics).",
    )
    pa.add_argument(
        "--scale-100",
        action="store_true",
        help="Scale AP metrics in [0,1] up to [0,100] (useful for StarDist).",
    )
    pa.add_argument("--out-csv", type=Path, default=None)
    args = pa.parse_args()

    sizes = _parse_sizes_arg(args.sizes, args.preds_root, args.metrics_name)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure val_metrics.json exists under preds-root.")

    n_val_images, n_val_instances = _load_val_stats(args.val_manifest)

    rows = []
    for sz in sizes:
        metrics_path = args.preds_root / sz / args.metrics_name
        if not metrics_path.exists() and args.metrics_name == "val_metrics.json":
            alt = args.preds_root / sz / "metrics.json"
            if alt.exists():
                metrics_path = alt
        metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
        row = {"size": sz}

        train_manifest = args.train_manifests_root / sz / "chips_train" / "detectron_manifest.json"
        row["n_train_instances"] = _count_train_instances(train_manifest) if train_manifest.exists() else 0
        if args.val_manifest:
            row["n_val_images"] = n_val_images
            row["n_val_instances"] = n_val_instances

        segm_metrics = {} if args.use_gpkg_ap else _extract_metrics(metrics, "segm")
        if not segm_metrics and args.val_manifest:
            preds_gpkg = args.preds_root / sz / args.gpkg_name
            if preds_gpkg.exists():
                segm_metrics = _compute_ap_from_gpkg(preds_gpkg, args.val_manifest)
        if not segm_metrics:
            print(f"[skip] missing metrics: {metrics_path}")
            continue
        row.update(segm_metrics)
        if args.include_bbox:
            row.update(_extract_metrics(metrics, "bbox"))
        if args.scale_100:
            for k, v in list(row.items()):
                if not isinstance(v, (int, float)):
                    continue
                if not k.endswith(tuple(AP_KEYS)):
                    continue
                if np.isfinite(v) and v <= 1.0:
                    row[k] = float(v) * 100.0
        rows.append(row)

    if not rows:
        raise SystemExit("No metrics collected. Check --preds-root/--sizes/--metrics-name.")

    df = pd.DataFrame(rows)
    out_csv = args.out_csv if args.out_csv else (args.preds_root / "summary_ap.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
