#!/usr/bin/env python3
"""
Enumerate chip windows (no npy writing) and produce deterministic train/val splits per size step.

For each tile directory (expects img.tif + crowns.gpkg), this script:
  - slides a window (size/stride) over the raster,
  - keeps chips that intersect crowns,
  - records chip metadata (window, bounds, transform, CRS, source paths),
  - shuffles chips with a seed,
  - for each size increment, writes train/val JSON manifests (70/30 by default).
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List

import geopandas as gpd
import rasterio
from rasterio import windows
from shapely.geometry import box


def clip_has_crowns(gdf: gpd.GeoDataFrame, geom) -> bool:
    sub = gdf[gdf.intersects(geom)]
    return not sub.empty


def count_crowns_in_geom(gdf: gpd.GeoDataFrame, geom) -> int:
    try:
        sindex = gdf.sindex
        hits = list(sindex.query(geom, predicate="intersects"))
        if not hits:
            return 0
        return int(gdf.iloc[hits].intersects(geom).sum())
    except Exception:
        return int(gdf.intersects(geom).sum())


def build_chips_for_tile(tile: Path, size: int, stride: int, count_instances: bool) -> List[dict]:
    img_path = tile / "img.tif"
    crowns_path = tile / "crowns.gpkg"
    if not img_path.exists() or not crowns_path.exists():
        print(f"Skipping {tile.name}: missing img/crowns")
        return []

    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        tfm = src.transform
        crs = src.crs
        raster_bounds = box(*src.bounds)

    gdf = gpd.read_file(crowns_path)
    if gdf.empty:
        return []
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.is_valid & gdf.intersects(raster_bounds)]
    if gdf.empty:
        return []

    recs = []
    chip_id = 0
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1, x1 = min(y0 + size, H), min(x0 + size, W)
            if y1 - y0 < 1 or x1 - x0 < 1:
                continue
            win = windows.Window(x0, y0, x1 - x0, y1 - y0)
            win_tfm = windows.transform(win, tfm)
            win_geom = box(*rasterio.transform.array_bounds(int(win.height), int(win.width), win_tfm))
            if not clip_has_crowns(gdf, win_geom):
                continue
            n_crowns = count_crowns_in_geom(gdf, win_geom) if count_instances else None
            recs.append({
                "tile": tile.name,
                "chip_id": chip_id,
                "window": [int(x0), int(y0), int(x1), int(y1)],
                "transform": list(win_tfm)[:6],
                "bounds": [float(b) for b in win_geom.bounds],
                "crs": crs.to_wkt() if crs else None,
                "img_path": str(img_path),
                "crowns_path": str(crowns_path),
                "size": size,
                "n_crowns": n_crowns,
            })
            chip_id += 1
    return recs


def main():
    pa = argparse.ArgumentParser(description="Build chip manifests and per-size train/val splits (no npy writing).")
    pa.add_argument("--tiles-dir", type=Path, required=True, help="Directory with tile subfolders (img.tif, crowns.gpkg).")
    pa.add_argument("--out-root", type=Path, required=True, help="Output root for per-size splits.")
    pa.add_argument("--size", type=int, default=512, help="Chip size (pixels)")
    pa.add_argument("--stride", type=int, default=512, help="Chip stride (pixels)")
    pa.add_argument("--holdout", type=float, default=0.3, help="Validation fraction (ignored if fixed val is used)")
    pa.add_argument("--split-seed", type=int, default=42, help="Shuffle seed for train/val split")
    pa.add_argument("--start-chips", type=int, default=10, help="First subset size")
    pa.add_argument("--step", type=int, default=10, help="Subset increment")
    pa.add_argument("--max-chips", type=int, default=None, help="Maximum subset size (defaults to all chips)")
    pa.add_argument("--skip", nargs="*", default=[], help="Tile folder names to skip")
    pa.add_argument("--count-instances", action="store_true", help="Count crowns per chip and print per-size totals")
    pa.add_argument("--fixed-val-fraction", type=float, default=None, help="Use a single fixed validation set with this fraction")
    pa.add_argument("--fixed-val-count", type=int, default=None, help="Use a single fixed validation set with this count")
    pa.add_argument("--fixed-val-crowns", type=int, default=None, help="Use a single fixed validation set with this many crowns")
    pa.add_argument("--write-per-size-val", action="store_true", help="Also write val_manifest.json per size")
    args = pa.parse_args()

    count_instances = args.count_instances or (args.fixed_val_crowns is not None)
    all_chips: List[dict] = []
    skip = set(args.skip)
    for tile in sorted(p for p in args.tiles_dir.iterdir() if p.is_dir() and p.name not in skip):
        recs = build_chips_for_tile(tile, args.size, args.stride, count_instances)
        all_chips.extend(recs)
        print(f"{tile.name}: {len(recs)} chips")

    if not all_chips:
        raise SystemExit("No chips found.")

    random.seed(args.split_seed)
    random.shuffle(all_chips)
    total = len(all_chips)

    if sum(x is not None for x in (args.fixed_val_fraction, args.fixed_val_count, args.fixed_val_crowns)) > 1:
        raise SystemExit("Only one of --fixed-val-fraction, --fixed-val-count, --fixed-val-crowns may be set.")

    fixed_val = []
    train_pool = all_chips
    if args.fixed_val_fraction is not None or args.fixed_val_count is not None or args.fixed_val_crowns is not None:
        if args.fixed_val_fraction is not None:
            n_val = int(round(total * args.fixed_val_fraction))
        elif args.fixed_val_count is not None:
            n_val = int(args.fixed_val_count or 0)
        else:
            n_val = None
        if args.fixed_val_crowns is not None:
            target = int(args.fixed_val_crowns)
            acc = 0
            fixed_val = []
            for rec in reversed(all_chips):
                n_c = int(rec.get("n_crowns") or 0)
                fixed_val.append(rec)
                acc += n_c
                if acc >= target:
                    break
            train_pool = [r for r in all_chips if r not in fixed_val]
            print(f"✓ fixed val crowns target={target}, selected={len(fixed_val)}, crowns={acc}")
        else:
            n_val = max(0, min(int(n_val or 0), total))
            fixed_val = all_chips[-n_val:] if n_val else []
            train_pool = all_chips[:-n_val] if n_val else all_chips
        if fixed_val:
            out_val = args.out_root / "fixed_val_manifest.json"
            args.out_root.mkdir(parents=True, exist_ok=True)
            json.dump(fixed_val, open(out_val, "w"))
            print(f"✓ wrote fixed val manifest ({len(fixed_val)} chips) ➜ {out_val}")

    max_chips = args.max_chips or len(train_pool)
    sizes = [s for s in range(args.start_chips, max_chips + 1, args.step) if s <= max_chips]

    for n in sizes:
        subset = train_pool[:n]
        if fixed_val:
            train = subset
            val = []
        else:
            k = int(round(n * (1.0 - args.holdout)))
            train = subset[:k]
            val = subset[k:]
        size_dir = args.out_root / f"{n:04d}"
        size_dir.mkdir(parents=True, exist_ok=True)
        json.dump(train, open(size_dir / "train_manifest.json", "w"))
        if val or args.write_per_size_val:
            json.dump(val, open(size_dir / "val_manifest.json", "w"))
        if args.count_instances:
            train_instances = sum(int(r.get("n_crowns") or 0) for r in train)
            val_instances = sum(int(r.get("n_crowns") or 0) for r in val)
            print(
                f"✓ size {n}: train={len(train)} val={len(val)} "
                f"(instances: train={train_instances} val={val_instances}) ➜ {size_dir}"
            )
        else:
            print(f"✓ size {n}: train={len(train)} val={len(val)} ➜ {size_dir}")


if __name__ == "__main__":
    main()
