#!/usr/bin/env python3
"""
Attach predicted classes to raw prediction polygons so class-aware PQ can be
computed on the original predicted geometry (not the object-feature GPKGs).

For each size:
  - load predictions (per-chip GPKGs or val_preds.gpkg)
  - load classified_objects.gpkg (geometry + predicted_class)
  - match by max IoU and copy predicted_class onto preds
  - write val_preds_classified.gpkg under out-root/<size>/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid


def _parse_sizes(sizes_arg: str | None, preds_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(preds_root.iterdir()):
        if p.is_dir():
            sizes.append(p.name)
    return sizes


def _valid_geom(g):
    if g is None or g.is_empty:
        return None
    try:
        gv = make_valid(g)
    except Exception:
        try:
            gv = g.buffer(0)
        except Exception:
            return None
    if gv.is_empty:
        return None
    if gv.geom_type == "Polygon":
        return gv
    if gv.geom_type == "MultiPolygon":
        parts = [p for p in gv.geoms if p.area > 0]
        if not parts:
            return None
        return max(parts, key=lambda p: p.area)
    return None


def _geom_iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def _load_preds(size_dir: Path, preds_gpkg_name: str, per_chip_subdir: str) -> gpd.GeoDataFrame:
    per_chip = size_dir / per_chip_subdir
    if per_chip.exists():
        gdfs = []
        for gpkg in sorted(per_chip.glob("*.gpkg")):
            gdf = gpd.read_file(gpkg)
            gdf["chip_name"] = gpkg.stem
            gdfs.append(gdf)
        if not gdfs:
            return gpd.GeoDataFrame(geometry=[], crs=None)
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    gpkg = size_dir / preds_gpkg_name
    if not gpkg.exists():
        return gpd.GeoDataFrame(geometry=[], crs=None)
    return gpd.read_file(gpkg)


def main() -> None:
    pa = argparse.ArgumentParser(description="Attach predicted_class to raw prediction polygons.")
    pa.add_argument("--preds-root", type=Path, required=True, help="Root with per-size predictions")
    pa.add_argument("--classified-root", type=Path, required=True, help="Root with <size>/classified_objects.gpkg")
    pa.add_argument("--out-root", type=Path, required=True)
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect")
    pa.add_argument("--preds-gpkg-name", type=str, default="val_preds.gpkg")
    pa.add_argument("--per-chip-subdir", type=str, default="per_chip_preds")
    pa.add_argument("--pred-class-field", type=str, default="predicted_class")
    pa.add_argument("--match-iou", type=float, default=0.9, help="IoU threshold for matching classes")
    args = pa.parse_args()

    sizes = _parse_sizes(args.sizes, args.preds_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or check --preds-root.")

    for sz in sizes:
        preds_dir = args.preds_root / sz
        cls_path = args.classified_root / sz / "classified_objects.gpkg"
        if not cls_path.exists():
            print(f"[skip] missing classified objects: {cls_path}")
            continue

        preds = _load_preds(preds_dir, args.preds_gpkg_name, args.per_chip_subdir)
        if preds.empty:
            print(f"[skip] missing predictions for size {sz}")
            continue

        classified = gpd.read_file(cls_path)
        if classified.empty:
            print(f"[skip] empty classified objects: {cls_path}")
            continue

        if preds.crs and classified.crs and preds.crs != classified.crs:
            classified = classified.to_crs(preds.crs)
        elif preds.crs and classified.crs is None:
            classified = classified.set_crs(preds.crs)

        preds = preds.copy()
        preds["geometry"] = preds.geometry.apply(_valid_geom)
        preds = preds[preds.geometry.notna()].reset_index(drop=True)
        if preds.empty:
            print(f"[skip] no valid predictions for size {sz}")
            continue

        classified = classified.copy()
        classified["geometry"] = classified.geometry.apply(_valid_geom)
        classified = classified[classified.geometry.notna()].reset_index(drop=True)
        if classified.empty:
            print(f"[skip] no valid classified geometry for size {sz}")
            continue

        sindex = classified.sindex
        pred_classes = []
        for geom in preds.geometry:
            cand_idx = list(sindex.intersection(geom.bounds))
            best_iou = 0.0
            best_class = None
            for j in cand_idx:
                iou = _geom_iou(geom, classified.geometry.iloc[j])
                if iou > best_iou:
                    best_iou = iou
                    best_class = classified[args.pred_class_field].iloc[j]
            if best_iou >= args.match_iou and best_class is not None:
                pred_classes.append(int(best_class))
            else:
                pred_classes.append(-1)

        preds[args.pred_class_field] = pred_classes
        out_dir = args.out_root / sz
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "val_preds_classified.gpkg"
        preds.to_file(out_path, driver="GPKG")
        print(f"✓ wrote {out_path} ({len(preds)} preds)")


if __name__ == "__main__":
    main()
