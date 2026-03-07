#!/usr/bin/env python3
"""
Build chips (img/msk/json/chm/cost) from tiles and write detectron_records.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.ops import transform as shp_transform
import geopandas as gpd


def clip_annotations(gdf: gpd.GeoDataFrame, bbox_geom) -> gpd.GeoDataFrame:
    """Clip annotations."""
    sub = gdf[gdf.intersects(bbox_geom)].copy()
    if sub.empty:
        return sub
    sub["geometry"] = sub.geometry.buffer(0)
    sub["geometry"] = sub.geometry.intersection(bbox_geom)
    sub = sub[~sub.geometry.is_empty]
    return sub


def poly_to_coco_segmentation(poly):
    """Poly to coco segmentation."""
    coords = np.array(poly.exterior.coords, dtype=float)
    if coords.shape[0] < 3:
        return []
    return coords.flatten().tolist()


def process_tile(tile: Path, out_dir: Path, size: int, stride: int, bands=(1, 2, 3)) -> List[dict]:
    """Process tile."""
    img_path = tile / "img.tif"
    crowns_path = tile / "crowns.gpkg"
    chm_path = tile / "chm.tif"
    cost_path = tile / "cost.tif"
    if not img_path.exists() or not crowns_path.exists() or not chm_path.exists() or not cost_path.exists():
        print(f"Skipping {tile.name}: missing img/crowns/chm/cost")
        return []

    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        tfm = src.transform
        crs = src.crs
        img_bands = src.read(bands).astype(np.float32)

    with rasterio.open(chm_path) as csrc:
        chm_all = csrc.read(1).astype(np.float32)
    with rasterio.open(cost_path) as cosrc:
        cost_all = cosrc.read(1).astype(np.float32)

    gdf = gpd.read_file(crowns_path)
    if gdf.empty:
        print(f"Skipping {tile.name}: empty crowns")
        return []
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    records: List[dict] = []
    chip_id = 0
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1, x1 = min(y0 + size, H), min(x0 + size, W)
            if y1 - y0 < 1 or x1 - x0 < 1:
                continue
            win_tfm = rasterio.windows.transform(rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0), tfm)
            win_geom = box(*rasterio.transform.array_bounds(y1 - y0, x1 - x0, win_tfm))
            clipped = clip_annotations(gdf, win_geom)
            if clipped.empty:
                continue

            chip_img = img_bands[:, y0:y1, x0:x1]
            pad_h, pad_w = size - chip_img.shape[1], size - chip_img.shape[2]
            if pad_h or pad_w:
                chip_img = np.pad(chip_img, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

            np.save(out_dir / "img" / f"{tile.name}_{chip_id}.npy", chip_img.astype(np.float32))

            chm_chip = chm_all[y0:y1, x0:x1]
            cost_chip = cost_all[y0:y1, x0:x1]
            if pad_h or pad_w:
                chm_chip = np.pad(chm_chip, ((0, pad_h), (0, pad_w)), mode="constant")
                cost_chip = np.pad(cost_chip, ((0, pad_h), (0, pad_w)), mode="constant")
            np.save(out_dir / "chm" / f"{tile.name}_{chip_id}.npy", chm_chip.astype(np.float32))
            np.save(out_dir / "cost" / f"{tile.name}_{chip_id}.npy", cost_chip.astype(np.float32))

            mask = rasterize(
                [(geom, 1) for geom in clipped.geometry],
                out_shape=(y1 - y0, x1 - x0),
                transform=win_tfm,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )
            if pad_h or pad_w:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
            np.save(out_dir / "msk" / f"{tile.name}_{chip_id}.npy", mask.astype(np.uint8))

            ys, xs = np.where(mask)
            point = [int(xs.mean()), int(ys.mean())] if xs.size else [size // 2, size // 2]
            json.dump({"point": point}, open(out_dir / "json" / f"{tile.name}_{chip_id}.json", "w"))

            annos = []
            inv = ~win_tfm
            def world_to_pixel(x, y, z=None):
                """World to pixel."""
                col, row = inv * (x, y)
                return (col, row)

            for geom in clipped.geometry:
                pix = shp_transform(world_to_pixel, geom)
                if pix.is_empty:
                    continue
                if pix.geom_type == "Polygon":
                    polys = [pix]
                elif pix.geom_type == "MultiPolygon":
                    polys = [p for p in pix.geoms if p.area > 0]
                else:
                    continue
                segs = []
                bboxes = []
                for p in polys:
                    seg = poly_to_coco_segmentation(p)
                    if len(seg) < 6:
                        continue
                    segs.append(seg)
                    x0b, y0b, x1b, y1b = p.bounds
                    bboxes.append([x0b, y0b, x1b - x0b, y1b - y0b])
                if not segs:
                    continue
                x0b = min(b[0] for b in bboxes); y0b = min(b[1] for b in bboxes)
                x1b = max(b[0] + b[2] for b in bboxes); y1b = max(b[1] + b[3] for b in bboxes)
                bbox = [x0b, y0b, x1b - x0b, y1b - y0b]
                annos.append({
                    "bbox": bbox,
                    "segmentation": segs,
                    "category_id": 0,
                    "iscrowd": 0,
                })

            if annos:
                records.append({
                    "file_name": str(out_dir / "img" / f"{tile.name}_{chip_id}.npy"),
                    "image_id": f"{tile.name}_{chip_id}",
                    "height": size,
                    "width": size,
                    "annotations": annos,
                    "meta": {
                        "source_tile": str(img_path),
                        "window": [int(x0), int(y0), int(x1), int(y1)],
                        "transform": list(win_tfm)[:6],
                        "bounds": [float(b) for b in win_geom.bounds],
                        "crs": crs.to_wkt() if crs else None,
                    },
                })
            chip_id += 1
    return records


def main():
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Build chips and annotations from tiles.")
    pa.add_argument("--tiles-dir", required=True, type=Path)
    pa.add_argument("--out", required=True, type=Path)
    pa.add_argument("--size", type=int, default=512)
    pa.add_argument("--stride", type=int, default=32, help="Chip stride")
    pa.add_argument("--skip", nargs="*", default=[], help="Tile folder names to skip")
    args = pa.parse_args()

    for sub in ("img", "msk", "json", "chm", "cost"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    skip = set(args.skip)
    for tile in sorted(p for p in args.tiles_dir.iterdir() if p.is_dir() and p.name not in skip):
        recs = process_tile(tile, args.out, args.size, args.stride)
        all_records.extend(recs)
        print(f"{tile.name}: {len(recs)} chip records")

    json.dump(all_records, open(args.out / "detectron_records.json", "w"))
    print(f"✓ wrote detectron_records.json with {len(all_records)} images")


if __name__ == "__main__":
    main()
