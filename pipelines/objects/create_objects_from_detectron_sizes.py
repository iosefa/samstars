#!/usr/bin/env python3
"""
Create object feature GPKGs per training size using detectron chip manifests.

For each size:
  - read chips_train/detectron_manifest.json
  - load each chip tif + clipped crowns gpkg
  - run obia.stats.create_objects
  - write detectron/<size>/objects.gpkg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
from rasterio.windows import from_bounds

from obia.stats import create_objects
from utils.utils import open_geotiff


def _parse_sizes(sizes_arg: str | None, chips_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(chips_root.iterdir()):
        if not p.is_dir():
            continue
        manifest = p / "chips_train" / "detectron_manifest.json"
        if manifest.exists():
            sizes.append(p.name)
    return sizes


def _split_name(name: str) -> tuple[str, int | None]:
    if "_" not in name:
        return name, None
    tile, chip = name.rsplit("_", 1)
    try:
        return tile, int(chip)
    except ValueError:
        return name, None


def _parse_voxel_resolution(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit("--voxel-resolution must be 'x,y,z'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _clean_polys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        return gdf
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.area > 0]
    return gdf


def _filter_zero_windows(gdf: gpd.GeoDataFrame, transform) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    keep = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            keep.append(False)
            continue
        minx, miny, maxx, maxy = geom.bounds
        if minx >= maxx or miny >= maxy:
            keep.append(False)
            continue
        win = from_bounds(minx, miny, maxx, maxy, transform=transform)
        if win.width < 1 or win.height < 1:
            keep.append(False)
            continue
        keep.append(True)
    return gdf.loc[keep].copy()

def main() -> None:
    pa = argparse.ArgumentParser(description="Create object features per detectron training size.")
    pa.add_argument("--chips-root", type=Path, default=Path("data/chips_128_100"))
    pa.add_argument("--out-root", type=Path, default=Path("data/detectron"))
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect")
    pa.add_argument("--ept", type=Path, required=True, help="Path to EPT json")
    pa.add_argument("--ept-srs", type=str, default="EPSG:32605")
    pa.add_argument(
        "--ept-geom-srs",
        type=str,
        default=None,
        help="CRS to reproject segment geometry for EPT queries (defaults to --ept-srs)",
    )
    pa.add_argument("--voxel-resolution", type=str, default="500,500,1")
    pa.add_argument("--textural-bands", type=str, default="0", help="Comma-separated 0-based band indices")
    pa.add_argument("--no-structural", action="store_true", help="Disable structural stats")
    pa.add_argument("--no-radiometric", action="store_true", help="Disable radiometric stats")
    pa.add_argument("--class-field", type=str, default="class", help="Class field to carry into objects.gpkg")
    pa.add_argument("--overwrite", action="store_true", help="Overwrite existing objects.gpkg")
    args = pa.parse_args()

    voxel_resolution = _parse_voxel_resolution(args.voxel_resolution)
    textural_bands = [int(b) for b in args.textural_bands.split(",") if b.strip()]

    sizes = _parse_sizes(args.sizes, args.chips_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure detectron_manifest.json exists under --chips-root.")

    ept_geom_srs = args.ept_geom_srs or args.ept_srs

    for sz in sizes:
        manifest = args.chips_root / sz / "chips_train" / "detectron_manifest.json"
        if not manifest.exists():
            print(f"[skip] missing manifest: {manifest}")
            continue

        out_path = args.out_root / sz / "objects.gpkg"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] exists: {out_path}")
            continue

        recs = json.load(open(manifest))
        if not recs:
            print(f"[skip] empty manifest: {manifest}")
            continue

        all_gdfs = []
        next_seg_id = 1
        for rec in recs:
            tif_path = Path(rec["tif_path"])
            gpkg_path = Path(rec["gpkg_path"])
            if not tif_path.exists() or not gpkg_path.exists():
                continue

            segments = gpd.read_file(gpkg_path)
            segments = _clean_polys(segments)
            if segments.empty:
                continue
            seg_class_map = None
            if args.class_field in segments.columns:
                seg_class_map = segments[[args.class_field]].copy()

            img = open_geotiff(str(tif_path))
            try:
                if segments.crs is None and img.crs is not None:
                    segments = segments.set_crs(img.crs)
                elif img.crs and segments.crs != img.crs:
                    segments = segments.to_crs(img.crs)

                segments = segments.copy()
                segments["segment_id"] = range(next_seg_id, next_seg_id + len(segments))
                next_seg_id += len(segments)
                if seg_class_map is not None:
                    seg_class_map = segments[["segment_id", args.class_field]].copy()
                segments = _filter_zero_windows(segments, img.transform)
                if segments.empty:
                    continue

                gdf = create_objects(
                    segments,
                    img,
                    ept=str(args.ept),
                    ept_srs=args.ept_srs,
                    ept_geom_srs=ept_geom_srs,
                    voxel_resolution=voxel_resolution,
                    textural_bands=textural_bands,
                    calculate_structural=not args.no_structural,
                    calculate_radiometric=not args.no_radiometric,
                )
                if seg_class_map is not None and args.class_field not in gdf.columns:
                    gdf = gdf.merge(seg_class_map, on="segment_id", how="left")

                name = rec.get("name", tif_path.stem)
                tile, chip_id = _split_name(name)
                gdf["chip_name"] = name
                gdf["tile"] = tile
                gdf["chip_id"] = chip_id
                gdf["size"] = sz
                all_gdfs.append(gdf)
            finally:
                try:
                    img.rasterio_obj.close()
                except Exception:
                    pass

        if not all_gdfs:
            print(f"[skip] no objects created for size {sz}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        if all_gdfs[0].crs is not None:
            out_gdf = out_gdf.set_crs(all_gdfs[0].crs)
        out_gdf.to_file(out_path, driver="GPKG")
        print(f"✓ wrote {out_path} ({len(out_gdf)} objects)")


if __name__ == "__main__":
    main()
