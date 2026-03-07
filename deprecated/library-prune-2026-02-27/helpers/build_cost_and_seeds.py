#!/usr/bin/env python3
"""
Build cost.tif and seeds.gpkg for each tile under a tiles directory.
Uses SAM cost surface (wv3 + chm) and canonical_seeds (reusing CHM as density).
Skips tiles that already have both outputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import rasterio
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.cost import build_cost_surface, write_cost_surface
from sam.seed import canonical_seeds, SeedParams


def _align_raster(src_path: Path, ref: rasterio.DatasetReader):
    """Open a raster and reproject/resample to match ref grid if needed."""
    raw = rasterio.open(src_path)
    if (
        raw.crs != ref.crs
        or raw.transform != ref.transform
        or any(abs(a - b) > 1e-9 for a, b in zip(raw.res, ref.res))
    ):
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        return WarpedVRT(
            raw,
            crs=ref.crs,
            transform=ref.transform,
            height=ref.height,
            width=ref.width,
            resampling=Resampling.bilinear,
        ), raw
    return raw, raw


def process_tile(tile: Path, weights=(0.4, 0.3, 0.3, 0.0)) -> None:
    """Process tile."""
    img = tile / "img.tif"
    chm = tile / "chm.tif"
    cost_out = tile / "cost.tif"
    seeds_out = tile / "seeds.gpkg"

    missing = [p.name for p in (img, chm) if not p.exists()]
    if missing:
        print(f"Skipping {tile.name}: missing {', '.join(missing)}")
        return

    # Ensure CHM aligns to IMG; if not, write an aligned copy
    with rasterio.open(img) as img_ds:
        chm_ds, chm_raw = _align_raster(chm, img_ds)
        if chm_raw is chm_ds:
            chm_use = chm
        else:
            aligned_path = tile / "chm_aligned.tif"
            meta = img_ds.profile.copy()
            meta.update(count=1, dtype="float32")
            data = chm_ds.read(1).astype("float32")
            with rasterio.open(aligned_path, "w", **meta) as dst:
                dst.write(data, 1)
            chm_use = aligned_path
            chm_ds.close()
            chm_raw.close()

    # Cost surface
    if not cost_out.exists():
        cost, profile = build_cost_surface(wv3_path=img, chm_path=chm_use, weights=weights)
        write_cost_surface(cost, profile, cost_out)
        print(f"{tile.name}: wrote cost.tif")
    else:
        print(f"{tile.name}: cost.tif exists; skipping cost")

    # Seeds (reuse CHM as density proxy)
    if seeds_out.exists():
        print(f"{tile.name}: seeds.gpkg exists; skipping seeds")
        return

    sp = SeedParams(
        chm_raster=chm_use,
        den_raster=chm_use,          # reuse CHM if density raster absent
        cost_raster=cost_out,
    )
    try:
        seeds = canonical_seeds(sp)
    except SystemExit as e:
        print(f"{tile.name}: no seeds ({e})")
        return
    if seeds.empty:
        print(f"{tile.name}: no seeds generated")
        return
    seeds.to_file(seeds_out, driver="GPKG")
    print(f"{tile.name}: wrote seeds.gpkg ({len(seeds)} points)")


def main():
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Create cost.tif and seeds.gpkg for tiles.")
    pa.add_argument("--tiles-dir", required=True, type=Path, help="Parent directory with tile-* subfolders")
    pa.add_argument("--skip", nargs="*", default=[], help="Tile folder names to skip, e.g., tile-008")
    pa.add_argument("--weights", nargs=4, type=float, default=(0.4, 0.3, 0.3, 0.0),
                    metavar=("GRAD", "GAP", "TEX", "SLIC"), help="Cost surface weights (sum to 1)")
    args = pa.parse_args()

    tiles = [p for p in sorted(args.tiles_dir.iterdir()) if p.is_dir() and p.name not in set(args.skip)]
    if not tiles:
        raise SystemExit("No tiles found.")

    for tile in tiles:
        process_tile(tile, weights=tuple(args.weights))


if __name__ == "__main__":
    main()
