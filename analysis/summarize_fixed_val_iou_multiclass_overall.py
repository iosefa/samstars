#!/usr/bin/env python3
# STATUS: active
"""
Summarize class-aware (overall) PQ/RQ/SQ for multiclass Detectron eval runs.

Class-aware means matches require BOTH IoU >= threshold AND class match.
This script aggregates across classes into a single PQ/RQ/SQ per size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


IOU_THR = 0.5


def _load_validation(manifest_path: Path) -> gpd.GeoDataFrame:
    pairs = json.load(open(manifest_path))
    val_gdfs = []
    for rec in pairs:
        gpkg = Path(rec["gpkg_path"])
        gdf = gpd.read_file(gpkg)
        val_gdfs.append(gdf)
    validation = gpd.GeoDataFrame(pd.concat(val_gdfs, ignore_index=True))
    return validation


def _coerce_class_col(gdf: gpd.GeoDataFrame, field: str) -> gpd.GeoDataFrame:
    if field not in gdf.columns:
        gdf[field] = 0
        return gdf
    gdf[field] = pd.to_numeric(gdf[field], errors="coerce").fillna(0).astype(int)
    return gdf


def _collect_classes_from_val(manifest_path: Path, class_field: str) -> list[int]:
    pairs = json.load(open(manifest_path))
    classes = set()
    for rec in pairs:
        gdf = gpd.read_file(rec["gpkg_path"])
        gdf = _coerce_class_col(gdf, class_field)
        classes.update(gdf[class_field].unique().tolist())
    if not classes:
        return [0]
    return sorted(classes)


def _load_class_mapping(mapping_path: Path | None) -> list[int] | None:
    if mapping_path and mapping_path.exists():
        data = json.load(open(mapping_path))
        classes = data.get("classes", None)
        if classes:
            return [int(c) for c in classes]
    return None


def _count_instances_by_class(manifest_path: Path, class_field: str) -> dict[int, int]:
    pairs = json.load(open(manifest_path))
    counts: dict[int, int] = {}
    for rec in pairs:
        gdf = gpd.read_file(rec["gpkg_path"])
        gdf = _coerce_class_col(gdf, class_field)
        vc = gdf[class_field].value_counts()
        for k, v in vc.items():
            counts[int(k)] = counts.get(int(k), 0) + int(v)
    return counts


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


def compute_iou_stats(preds: gpd.GeoDataFrame, validation: gpd.GeoDataFrame) -> dict:
    preds = preds.reset_index(drop=True).copy()
    preds["pid"] = preds.index
    validation = validation.reset_index(drop=True).copy()
    validation["vid"] = validation.index

    if preds.crs and validation.crs and preds.crs != validation.crs:
        validation = validation.to_crs(preds.crs)
    elif preds.crs and validation.crs is None:
        validation = validation.set_crs(preds.crs)

    try:
        if not preds.geometry.is_valid.all():
            preds["geometry"] = preds.geometry.buffer(0)
        if not validation.geometry.is_valid.all():
            validation["geometry"] = validation.geometry.buffer(0)
    except Exception:
        pass

    inter = gpd.overlay(
        preds[["pid", "geometry"]],
        validation[["vid", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if inter.empty:
        tp = 0
        fp = int(len(preds))
        fn = int(len(validation))
        sq = 0.0
        denom = tp + 0.5 * (fp + fn)
        rq = float(tp / denom) if denom > 0 else 0.0
        pq = sq * rq
        return {
            "mean_iou_gt0": float("nan"),
            "mean_iou_ge05": float("nan"),
            "n_matched_gt0": 0,
            "n_matched_ge05": 0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sq": sq,
            "rq": rq,
            "pq": pq,
        }

    ap = preds.set_index("pid").geometry.area
    av = validation.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(lower=0.0, upper=1.0)

    pairs_all = _greedy_match(inter.loc[inter["iou"] > 0, ["pid", "vid", "iou"]])
    pairs_50 = pairs_all.loc[pairs_all["iou"] >= IOU_THR]
    tp = int(len(pairs_50))
    fp = int(len(preds) - tp)
    fn = int(len(validation) - tp)
    sq = float(pairs_50["iou"].mean()) if tp else 0.0
    denom = tp + 0.5 * (fp + fn)
    rq = float(tp / denom) if denom > 0 else 0.0
    pq = sq * rq

    return {
        "mean_iou_gt0": float(pairs_all["iou"].mean()) if not pairs_all.empty else float("nan"),
        "mean_iou_ge05": float(pairs_50["iou"].mean()) if not pairs_50.empty else float("nan"),
        "n_matched_gt0": int(len(pairs_all)),
        "n_matched_ge05": int(len(pairs_50)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sq": sq,
        "rq": rq,
        "pq": pq,
    }


def _apply_score_thresh(preds: gpd.GeoDataFrame, score_thresh: float | None) -> gpd.GeoDataFrame:
    if score_thresh is None or "score" not in preds.columns:
        return preds
    return preds.loc[preds["score"] >= score_thresh].copy()


def _parse_sizes_arg(sizes_arg: str | None, preds_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(preds_root.iterdir()):
        if not p.is_dir():
            continue
        gpkg = p / "val_preds.gpkg"
        if gpkg.exists():
            sizes.append(p.name)
    return sizes


def main() -> None:
    pa = argparse.ArgumentParser(description="Summarize overall class-aware PQ/RQ/SQ from multiclass predictions.")
    pa.add_argument("--val-manifest", type=Path, required=True)
    pa.add_argument("--preds-root", type=Path, default=Path("data/detectron_fixed_val"))
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect from preds-root")
    pa.add_argument(
        "--train-manifests-root",
        type=Path,
        default=Path("data/chips_128_100"),
        help="Root containing {size}/chips_train/detectron_manifest.json",
    )
    pa.add_argument("--out-csv", type=Path, default=None)
    pa.add_argument("--class-field", type=str, default="class", help="Class field in GT GPKGs")
    pa.add_argument("--pred-class-field", type=str, default="class", help="Class field in prediction GPKGs")
    pa.add_argument("--class-mapping", type=Path, default=None, help="class_mapping.json to define class list")
    pa.add_argument("--classes", type=str, default=None, help="Comma-separated class list override")
    pa.add_argument("--score-thresh", type=float, default=None, help="Optional score threshold for predictions")
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
    args = pa.parse_args()

    sizes = _parse_sizes_arg(args.sizes, args.preds_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure val_preds.gpkg exists under preds-root.")

    if args.classes:
        classes = [int(s.strip()) for s in args.classes.split(",") if s.strip()]
    else:
        classes = _load_class_mapping(args.class_mapping) or _collect_classes_from_val(args.val_manifest, args.class_field)

    val_pairs = json.load(open(args.val_manifest))
    val_num_images = len(val_pairs)
    val_counts = _count_instances_by_class(args.val_manifest, args.class_field)
    val_total = sum(val_counts.values())

    rows = []
    for sz in sizes:
        train_manifest = args.train_manifests_root / sz / "chips_train" / "detectron_manifest.json"
        train_counts = _count_instances_by_class(train_manifest, args.class_field) if train_manifest.exists() else {}
        n_train_total = int(sum(train_counts.values()))

        accum = {
            "sum_iou_gt0": 0.0,
            "sum_iou_ge05": 0.0,
            "n_matched_gt0": 0,
            "n_matched_ge05": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_pred": 0,
        }

        if args.use_per_chip:
            per_chip_dir = args.preds_root / sz / args.per_chip_subdir
            if not per_chip_dir.exists():
                print(f"[skip] missing per-chip predictions: {per_chip_dir}")
                continue

            for rec in val_pairs:
                chip_name = rec.get("name", Path(rec["tif_path"]).stem)
                pred_file = per_chip_dir / f"{chip_name}.gpkg"
                if pred_file.exists():
                    preds = gpd.read_file(pred_file)
                else:
                    preds = gpd.GeoDataFrame(geometry=[], crs=None)
                preds = _coerce_class_col(preds, args.pred_class_field)
                preds = _apply_score_thresh(preds, args.score_thresh)
                gt = gpd.read_file(rec["gpkg_path"])
                gt = _coerce_class_col(gt, args.class_field)

                for cls in classes:
                    preds_c = preds.loc[preds[args.pred_class_field] == cls].copy()
                    gt_c = gt.loc[gt[args.class_field] == cls].copy()
                    if preds_c.empty:
                        preds_c = gpd.GeoDataFrame(geometry=[], crs=gt_c.crs)
                    stats = compute_iou_stats(preds_c, gt_c)
                    accum["tp"] += stats["tp"]
                    accum["fp"] += stats["fp"]
                    accum["fn"] += stats["fn"]
                    accum["n_matched_gt0"] += stats["n_matched_gt0"]
                    accum["n_matched_ge05"] += stats["n_matched_ge05"]
                    if stats["n_matched_gt0"]:
                        accum["sum_iou_gt0"] += stats["mean_iou_gt0"] * stats["n_matched_gt0"]
                    if stats["n_matched_ge05"]:
                        accum["sum_iou_ge05"] += stats["mean_iou_ge05"] * stats["n_matched_ge05"]
                    accum["n_pred"] += int(len(preds_c))
        else:
            pred_path = args.preds_root / sz / "val_preds.gpkg"
            if not pred_path.exists():
                print(f"[skip] missing predictions: {pred_path}")
                continue
            preds = gpd.read_file(pred_path)
            preds = _coerce_class_col(preds, args.pred_class_field)
            preds = _apply_score_thresh(preds, args.score_thresh)
            validation = _load_validation(args.val_manifest)
            validation = _coerce_class_col(validation, args.class_field)

            for cls in classes:
                preds_c = preds.loc[preds[args.pred_class_field] == cls].copy()
                gt_c = validation.loc[validation[args.class_field] == cls].copy()
                stats = compute_iou_stats(preds_c, gt_c)
                accum["tp"] += stats["tp"]
                accum["fp"] += stats["fp"]
                accum["fn"] += stats["fn"]
                accum["n_matched_gt0"] += stats["n_matched_gt0"]
                accum["n_matched_ge05"] += stats["n_matched_ge05"]
                if stats["n_matched_gt0"]:
                    accum["sum_iou_gt0"] += stats["mean_iou_gt0"] * stats["n_matched_gt0"]
                if stats["n_matched_ge05"]:
                    accum["sum_iou_ge05"] += stats["mean_iou_ge05"] * stats["n_matched_ge05"]
                accum["n_pred"] += int(len(preds_c))

        mean_iou_gt0 = (accum["sum_iou_gt0"] / accum["n_matched_gt0"]) if accum["n_matched_gt0"] else float("nan")
        mean_iou_ge05 = (accum["sum_iou_ge05"] / accum["n_matched_ge05"]) if accum["n_matched_ge05"] else float("nan")
        sq = (accum["sum_iou_ge05"] / accum["n_matched_ge05"]) if accum["n_matched_ge05"] else 0.0
        denom = accum["tp"] + 0.5 * (accum["fp"] + accum["fn"])
        rq = float(accum["tp"] / denom) if denom > 0 else 0.0
        pq = sq * rq

        rows.append({
            "size": sz,
            "n_train_instances": n_train_total,
            "n_val_instances": val_total,
            "n_val_images": val_num_images,
            "n_pred_instances": accum["n_pred"],
            "mean_iou_gt0": mean_iou_gt0,
            "mean_iou_ge05": mean_iou_ge05,
            "n_matched_gt0": accum["n_matched_gt0"],
            "n_matched_ge05": accum["n_matched_ge05"],
            "tp": accum["tp"],
            "fp": accum["fp"],
            "fn": accum["fn"],
            "sq": sq,
            "rq": rq,
            "pq": pq,
        })
        print(
            f"[ok] {sz}: n_train={n_train_total}, n_pred={accum['n_pred']}, "
            f"PQ={pq:.3f}, SQ={sq:.3f}, RQ={rq:.3f}"
        )

    df = pd.DataFrame(rows).sort_values("size")
    out_csv = args.out_csv or (args.preds_root / "summary_iou_classaware_overall.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
