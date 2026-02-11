#!/usr/bin/env python3
"""
Create object feature GPKGs per size from Detectron eval outputs (per-chip predictions).

For each size:
  - read detectron_fixed_val/<size>/per_chip_preds/*.gpkg
  - map each chip name to chips_root/<val_size>/chips_val/tif/<name>.tif
  - run obia.stats.create_objects on predicted segments
  - write <out_root>/<size>/objects.gpkg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rasterio.windows import from_bounds
from shapely.validation import make_valid

from obia.stats import create_objects
from utils.utils import open_geotiff


def _parse_sizes(sizes_arg: str | None, preds_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(preds_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "per_chip_preds").exists():
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


def _chip_tif_path(chips_root: Path, val_size: str, stem: str) -> Path:
    return chips_root / val_size / "chips_val" / "tif" / f"{stem}.tif"


def main() -> None:
    pa = argparse.ArgumentParser(description="Create objects from Detectron eval outputs per size.")
    pa.add_argument("--chips-root", type=Path, default=Path("data/chips_128_100"))
    pa.add_argument("--preds-root", type=Path, default=Path("data/detectron_fixed_val"))
    pa.add_argument("--out-root", type=Path, default=Path("data/detectron_objects"))
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect")
    pa.add_argument("--val-size", type=str, default="fixed_val", help="Fixed val folder under chips-root")
    pa.add_argument("--per-chip-subdir", type=str, default="per_chip_preds")
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
    pa.add_argument("--overwrite", action="store_true", help="Overwrite existing objects.gpkg")
    args = pa.parse_args()

    voxel_resolution = _parse_voxel_resolution(args.voxel_resolution)
    textural_bands = [int(b) for b in args.textural_bands.split(",") if b.strip()]

    sizes = _parse_sizes(args.sizes, args.preds_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure per-chip preds exist under --preds-root.")

    ept_geom_srs = args.ept_geom_srs or args.ept_srs

    for sz in sizes:
        per_chip_dir = args.preds_root / sz / args.per_chip_subdir
        if not per_chip_dir.exists():
            print(f"[skip] missing per-chip predictions: {per_chip_dir}")
            continue

        out_path = args.out_root / sz / "objects.gpkg"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] exists: {out_path}")
            continue

        gpkg_files = sorted(per_chip_dir.glob("*.gpkg"))
        if not gpkg_files:
            print(f"[skip] no predictions found in {per_chip_dir}")
            continue

        all_gdfs = []
        next_seg_id = 1
        for gpkg in gpkg_files:
            stem = gpkg.stem
            tif_path = _chip_tif_path(args.chips_root, args.val_size, stem)
            if not tif_path.exists():
                print(f"[skip] missing chip tif: {tif_path}")
                continue

            preds = gpd.read_file(gpkg)
            preds = _clean_polys(preds)
            if preds.empty:
                continue

            img = open_geotiff(str(tif_path))
            try:
                if preds.crs is None and img.crs is not None:
                    preds = preds.set_crs(img.crs)
                elif img.crs and preds.crs != img.crs:
                    preds = preds.to_crs(img.crs)

                preds = _filter_zero_windows(preds, img.transform)
                if preds.empty:
                    continue

                preds = preds.copy()
                preds["segment_id"] = range(next_seg_id, next_seg_id + len(preds))
                next_seg_id += len(preds)

                gdf = create_objects(
                    preds,
                    img,
                    ept=str(args.ept),
                    ept_srs=args.ept_srs,
                    ept_geom_srs=ept_geom_srs,
                    voxel_resolution=voxel_resolution,
                    textural_bands=textural_bands,
                    calculate_structural=not args.no_structural,
                    calculate_radiometric=not args.no_radiometric,
                )

                tile, chip_id = _split_name(stem)
                gdf["chip_name"] = stem
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
