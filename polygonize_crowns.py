#!/usr/bin/env python3
"""
Polygonise every crowns.tif and merge results.

• per-tile crowns.gpkg are overwritten
• merged crowns.gpkg goes to the parent dir
• crowns closer than ½-pixel to the tile edge are dropped
• if two crowns overlap by ≥70 % of the smaller crown, keep the larger one
"""

import argparse
import pathlib
import sys
import concurrent.futures as cf
from typing import Optional, List

import rasterio
from rasterio import features
from shapely.geometry import shape, box, mapping, MultiPolygon
from shapely.strtree import STRtree
import geopandas as gpd
import pandas as pd
import fiona
from tqdm import tqdm


def process_tile(tile_dir: pathlib.Path) -> Optional[pathlib.Path]:
    tif  = tile_dir / "crowns.tif"
    gpkg = tile_dir / "crowns.gpkg"
    if not tif.exists():
        print(f"✘ {tile_dir.name}: crowns.tif missing")
        return None

    with rasterio.open(tif) as src:
        band  = src.read(1)
        trans = src.transform

        extent  = box(*src.bounds)
        border  = extent.boundary
        pix     = max(abs(trans.a), abs(trans.e))
        tol     = 0.5 * pix

        records, new_id = [], 1
        for geom_ras, val in features.shapes(
                band, mask=(band != 0), transform=trans, connectivity=8):

            if val == 0:
                continue

            shp = shape(geom_ras)
            parts = shp.geoms if isinstance(shp, MultiPolygon) else [shp]

            for part in parts:
                if part.distance(border) < tol:
                    continue

                records.append({
                    "geometry": mapping(part),
                    "properties": {"id": new_id}
                })
                new_id += 1

    if not records:
        print(f"• {tile_dir.name}: no interior crowns")
        return None

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    try:
        gpkg.unlink()
    except FileNotFoundError:
        pass

    with fiona.open(
        gpkg, "w",
        driver="GPKG",
        layer="crowns",
        schema=schema,
        crs=src.crs.to_wkt(),
        EXISTING_LAYER="OVERWRITE") as dst:
        dst.writerecords(records)

    print(f"✓ {tile_dir.name}: {len(records):,} crowns kept")
    return gpkg


def merge_crowns(tile_gpkgs: List[pathlib.Path], out_gpkg: pathlib.Path) -> None:
    gpkgs = [g for g in tile_gpkgs if g is not None]
    if not gpkgs:
        print("No per-tile GeoPackages generated – nothing to merge.")
        return

    gdfs = [gpd.read_file(g, layer="crowns") for g in gpkgs]
    crowns = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True),
                              crs=gdfs[0].crs)

    tree      = STRtree(crowns.geometry.tolist())
    visited   = [False] * len(crowns)
    keep_mask = [True]  * len(crowns)

    def is_duplicate(a, b) -> bool:
        inter = a.intersection(b).area
        if inter == 0:
            return False
        return inter / min(a.area, b.area) >= 0.70

    for i, geom in enumerate(crowns.geometry):
        if visited[i]:
            continue
        visited[i] = True
        for j in tree.query(geom):
            if i == j or visited[j]:
                continue
            other = crowns.geometry[j]
            if is_duplicate(geom, other):
                if geom.area >= other.area:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    geom = other
                visited[j] = True

    merged = crowns[keep_mask].reset_index(drop=True)
    try:
        out_gpkg.unlink()
    except FileNotFoundError:
        pass
    merged.to_file(out_gpkg, layer="crowns", driver="GPKG")
    print(f"\n⭑ merged crowns.gpkg written with {len(merged):,} polygons")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Polygonise crowns.tif in each tile_* folder and merge.")
    ap.add_argument("parent", type=pathlib.Path,
                    help="parent directory that contains tile_* folders")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="parallel workers (default: CPU count)")
    args = ap.parse_args()

    parent = args.parent.resolve()
    if not parent.is_dir():
        sys.exit(f"{parent} is not a directory")

    tile_dirs = [d for d in sorted(parent.iterdir())
                 if (d / "crowns.tif").exists()]
    if not tile_dirs:
        sys.exit("No tile folders with crowns.tif found.")

    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        gpkg_paths = list(tqdm(ex.map(process_tile, tile_dirs),
                               total=len(tile_dirs)))

    merge_crowns(gpkg_paths, parent / "crowns.gpkg")


if __name__ == "__main__":
    main()
