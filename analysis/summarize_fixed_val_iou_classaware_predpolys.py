#!/usr/bin/env python3
# STATUS: active
"""
Summarize class-aware PQ/RQ/SQ using classified prediction polygons.

Reads per-size val_preds_classified.gpkg (or override name) where each polygon
has a predicted class, and compares to ground-truth crowns from val-manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


IOU_THR = 0.5


def _parse_sizes(sizes_arg: str | None, preds_root: Path, gpkg_name: str) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(preds_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / gpkg_name).exists():
            sizes.append(p.name)
    return sizes


def _coerce_class(gdf: gpd.GeoDataFrame, field: str) -> gpd.GeoDataFrame:
    if field not in gdf.columns:
        raise RuntimeError(f"Missing class field '{field}'.")
    gdf[field] = pd.to_numeric(gdf[field], errors="coerce").astype("Int64")
    gdf = gdf.dropna(subset=[field]).copy()
    gdf[field] = gdf[field].astype(int)
    return gdf


def _load_validation(manifest_path: Path, class_field: str) -> gpd.GeoDataFrame:
    pairs = json.load(open(manifest_path))
    gdfs = []
    for rec in pairs:
        gpkg = Path(rec["gpkg_path"])
        if gpkg.exists():
            gdf = gpd.read_file(gpkg)
            gdfs.append(gdf)
    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    validation = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    validation = _coerce_class(validation, class_field)
    return validation


def _count_train_instances(manifest_path: Path) -> int:
    pairs = json.load(open(manifest_path))
    total = 0
    for rec in pairs:
        gpkg = Path(rec["gpkg_path"])
        if gpkg.exists():
            gdf = gpd.read_file(gpkg)
            total += int(len(gdf))
    return total


def _greedy_match(pairs: pd.DataFrame, iou_col: str = "iou") -> pd.DataFrame:
    out = pairs.sort_values(iou_col, ascending=False)
    used_p = set()
    used_v = set()
    rows = []
    for _, r in out.iterrows():
        pid = int(r["pid"])
        vid = int(r["vid"])
        if pid in used_p or vid in used_v:
            continue
        used_p.add(pid)
        used_v.add(vid)
        rows.append(r)
    if not rows:
        return pairs.iloc[:0]
    return pd.DataFrame(rows)


def _compute_iou_stats(preds: gpd.GeoDataFrame, validation: gpd.GeoDataFrame, iou_thr: float) -> dict:
    preds = preds.reset_index(drop=True).copy()
    preds["pid"] = preds.index
    validation = validation.reset_index(drop=True).copy()
    validation["vid"] = validation.index

    if preds.empty and validation.empty:
        return {"mean_iou_gt0": float("nan"), "mean_iou_ge05": float("nan"), "n_matched_gt0": 0,
                "n_matched_ge05": 0, "tp": 0, "fp": 0, "fn": 0}
    if preds.empty:
        return {"mean_iou_gt0": float("nan"), "mean_iou_ge05": float("nan"), "n_matched_gt0": 0,
                "n_matched_ge05": 0, "tp": 0, "fp": 0, "fn": int(len(validation))}
    if validation.empty:
        return {"mean_iou_gt0": float("nan"), "mean_iou_ge05": float("nan"), "n_matched_gt0": 0,
                "n_matched_ge05": 0, "tp": 0, "fp": int(len(preds)), "fn": 0}

    if preds.crs and validation.crs and preds.crs != validation.crs:
        validation = validation.to_crs(preds.crs)
    elif preds.crs and validation.crs is None:
        validation = validation.set_crs(preds.crs)

    inter = gpd.overlay(
        preds[["pid", "geometry"]],
        validation[["vid", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if inter.empty:
        return {"mean_iou_gt0": float("nan"), "mean_iou_ge05": float("nan"), "n_matched_gt0": 0,
                "n_matched_ge05": 0, "tp": 0, "fp": int(len(preds)), "fn": int(len(validation))}

    ap = preds.set_index("pid").geometry.area
    av = validation.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(lower=0.0, upper=1.0)

    pairs_all = _greedy_match(inter.loc[inter["iou"] > 0, ["pid", "vid", "iou"]])
    pairs_50 = pairs_all.loc[pairs_all["iou"] >= iou_thr]
    tp = int(len(pairs_50))
    fp = int(len(preds) - tp)
    fn = int(len(validation) - tp)
    return {
        "mean_iou_gt0": float(pairs_all["iou"].mean()) if not pairs_all.empty else float("nan"),
        "mean_iou_ge05": float(pairs_50["iou"].mean()) if not pairs_50.empty else float("nan"),
        "n_matched_gt0": int(len(pairs_all)),
        "n_matched_ge05": int(len(pairs_50)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    pa = argparse.ArgumentParser(description="Summarize class-aware PQ/RQ/SQ from classified prediction polygons.")
    pa.add_argument("--val-manifest", type=Path, required=True)
    pa.add_argument("--preds-root", type=Path, required=True, help="Root with <size>/val_preds_classified.gpkg")
    pa.add_argument("--preds-gpkg-name", type=str, default="val_preds_classified.gpkg")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect")
    pa.add_argument("--gt-class-field", type=str, default="class")
    pa.add_argument("--pred-class-field", type=str, default="predicted_class")
    pa.add_argument("--iou-thr", type=float, default=IOU_THR)
    pa.add_argument(
        "--train-manifests-root",
        type=Path,
        default=Path("data/chips_128_100"),
        help="Root containing {size}/chips_train/detectron_manifest.json",
    )
    pa.add_argument("--out-csv", type=Path, default=None)
    args = pa.parse_args()

    sizes = _parse_sizes(args.sizes, args.preds_root, args.preds_gpkg_name)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure gpkg exists under --preds-root.")

    validation = _load_validation(args.val_manifest, args.gt_class_field)
    if validation.empty:
        raise RuntimeError("Ground-truth crowns are empty.")
    gt_classes = set(validation[args.gt_class_field].unique().tolist())

    rows = []
    for sz in sizes:
        pred_path = args.preds_root / sz / args.preds_gpkg_name
        if not pred_path.exists():
            print(f"[skip] missing predictions: {pred_path}")
            continue
        preds = gpd.read_file(pred_path)
        preds = _coerce_class(preds, args.pred_class_field)
        pred_classes = set(preds[args.pred_class_field].unique().tolist())
        classes = sorted(gt_classes.union(pred_classes))

        if preds.crs and validation.crs and preds.crs != validation.crs:
            gt_use = validation.to_crs(preds.crs)
        elif preds.crs and validation.crs is None:
            gt_use = validation.set_crs(preds.crs)
        else:
            gt_use = validation

        total_tp = total_fp = total_fn = 0
        sum_iou_gt0 = 0.0
        sum_iou_ge05 = 0.0
        n_matched_gt0 = 0
        n_matched_ge05 = 0

        for cls in classes:
            preds_c = preds.loc[preds[args.pred_class_field] == cls]
            gt_c = gt_use.loc[gt_use[args.gt_class_field] == cls]
            stats = _compute_iou_stats(preds_c, gt_c, args.iou_thr)
            total_tp += stats["tp"]
            total_fp += stats["fp"]
            total_fn += stats["fn"]
            n_matched_gt0 += stats["n_matched_gt0"]
            n_matched_ge05 += stats["n_matched_ge05"]
            if stats["n_matched_gt0"]:
                sum_iou_gt0 += stats["mean_iou_gt0"] * stats["n_matched_gt0"]
            if stats["n_matched_ge05"]:
                sum_iou_ge05 += stats["mean_iou_ge05"] * stats["n_matched_ge05"]

        mean_iou_gt0 = (sum_iou_gt0 / n_matched_gt0) if n_matched_gt0 else float("nan")
        mean_iou_ge05 = (sum_iou_ge05 / n_matched_ge05) if n_matched_ge05 else float("nan")
        sq = (sum_iou_ge05 / total_tp) if total_tp else 0.0
        denom = total_tp + 0.5 * (total_fp + total_fn)
        rq = float(total_tp / denom) if denom > 0 else 0.0
        pq = sq * rq

        train_manifest = args.train_manifests_root / sz / "chips_train" / "detectron_manifest.json"
        n_train = _count_train_instances(train_manifest) if train_manifest.exists() else 0

        rows.append({
            "size": sz,
            "n_train_instances": n_train,
            "n_pred_instances": int(len(preds)),
            "n_gt_instances": int(len(validation)),
            "mean_iou_gt0": mean_iou_gt0,
            "mean_iou_ge05": mean_iou_ge05,
            "n_matched_gt0": n_matched_gt0,
            "n_matched_ge05": n_matched_ge05,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "sq": sq,
            "rq": rq,
            "pq": pq,
        })
        print(f"[ok] {sz}: PQ={pq:.3f}, SQ={sq:.3f}, RQ={rq:.3f}")

    df = pd.DataFrame(rows).sort_values("size")
    out_csv = args.out_csv or (args.preds_root / "summary_iou_classaware_predpolys.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
