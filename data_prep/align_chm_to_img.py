#!/usr/bin/env python3
"""
Resample CHM rasters onto the img.tif grid for each tile directory.
Writes a new GeoTIFF (default: chm_aligned.tif) per tile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def _get_resampling(name: str) -> Resampling:
    name = name.lower()
    if name == "nearest":
        return Resampling.nearest
    if name == "bilinear":
        return Resampling.bilinear
    if name == "cubic":
        return Resampling.cubic
    if name == "average":
        return Resampling.average
    raise ValueError(f"Unsupported resampling: {name}")


def align_chm(tile_dir: Path, img_name: str, chm_name: str, out_name: str, overwrite: bool, resampling: Resampling) -> None:
    img_path = tile_dir / img_name
    chm_path = tile_dir / chm_name
    if not img_path.exists() or not chm_path.exists():
        return

    out_path = chm_path if overwrite else tile_dir / out_name
    if out_path.exists() and not overwrite:
        print(f"skip existing: {out_path}")
        return

    with rasterio.open(img_path) as img_src, rasterio.open(chm_path) as chm_src:
        dst_shape = (img_src.height, img_src.width)
        dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(chm_src, 1),
            destination=dst,
            src_transform=chm_src.transform,
            src_crs=chm_src.crs,
            dst_transform=img_src.transform,
            dst_crs=img_src.crs,
            resampling=resampling,
        )
        profile = img_src.profile.copy()
        profile.update(count=1, dtype="float32")
        if chm_src.nodata is not None:
            profile.update(nodata=chm_src.nodata)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst_out:
        dst_out.write(dst.astype(np.float32), 1)
    print(f"✓ wrote {out_path}")


def main() -> None:
    pa = argparse.ArgumentParser(description="Align CHM rasters to img.tif grid per tile.")
    pa.add_argument("--tiles-dir", type=Path, required=True)
    pa.add_argument("--img-name", type=str, default="img.tif")
    pa.add_argument("--chm-name", type=str, default="chm.tif")
    pa.add_argument("--out-name", type=str, default="chm_aligned.tif")
    pa.add_argument("--overwrite", action="store_true", help="Overwrite chm.tif in place")
    pa.add_argument("--resampling", type=str, default="bilinear", help="nearest|bilinear|cubic|average")
    args = pa.parse_args()

    resampling = _get_resampling(args.resampling)
    for tile_dir in sorted(p for p in args.tiles_dir.iterdir() if p.is_dir()):
        align_chm(tile_dir, args.img_name, args.chm_name, args.out_name, args.overwrite, resampling)


if __name__ == "__main__":
    main()
