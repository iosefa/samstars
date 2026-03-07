#!/usr/bin/env python3
"""
Tile a large raster, build cost/seeds per tile, and run SAM (fast mode) per tile.

This script:
  1) Tiles IMG/CHM/DEN into overlapping tiles (helpers/make_tiles.py)
  2) Writes a minimal SAM manifest for tiles (val_manifest.json)
  3) Runs SAM inference with --fast to produce *_sam_prob.tif, *_sam_index.tif, etc.

Run in the SAM environment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import rasterio

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.make_tiles import build_tiles


def _tile_manifest_entry(tile_dir: Path) -> dict:
    """Internal helper function."""
    img_path = tile_dir / "img.tif"
    with rasterio.open(img_path) as src:
        width = src.width
        height = src.height
        bounds = src.bounds
        crs = src.crs.to_wkt() if src.crs else None
        tfm = list(src.transform)
    return {
        "tile": tile_dir.name,
        "chip_id": 0,
        "img_path": str(img_path),
        "window": [0, 0, width, height],
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "size": width,
        "crs": crs,
        "transform": tfm,
    }


def _write_manifest(tiles_dir: Path, manifest_path: Path) -> None:
    """Internal helper function."""
    records = []
    for tile_dir in sorted(tiles_dir.iterdir()):
        if not tile_dir.is_dir():
            continue
        if not (tile_dir / "img.tif").exists():
            continue
        records.append(_tile_manifest_entry(tile_dir))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(records, f)
    print(f"✓ wrote {manifest_path} ({len(records)} records)")


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Tile raster + run SAM (fast) per tile.")
    pa.add_argument("--img", type=Path, required=True, help="Full raster (multispectral)")
    pa.add_argument("--chm", type=Path, required=True, help="Aligned CHM raster")
    pa.add_argument("--den", type=Path, required=True, help="Density raster (or CHM)")
    pa.add_argument("--slic", type=Path, default=None, help="Optional SLIC raster")
    pa.add_argument("--tiles-out", type=Path, required=True, help="Output tiles directory")
    pa.add_argument("--tile-m", type=float, default=500.0, help="Tile size in meters")
    pa.add_argument("--overlap-m", type=float, default=20.0, help="Tile overlap in meters")
    pa.add_argument(
        "--cost-weights",
        type=str,
        default="0.4,0.3,0.3,0.0",
        help="Comma-separated cost weights (grad,gap,tex,slic)",
    )
    pa.add_argument("--no-thin", action="store_true", help="Disable seed thinning")
    pa.add_argument("--write-den", action="store_true", help="Write den.tif per tile")
    pa.add_argument("--skip-tiling", action="store_true", help="Skip tile generation")
    pa.add_argument("--manifests-root", type=Path, required=True)
    pa.add_argument(
        "--sam-size",
        type=str,
        required=True,
        help="Size name used to select SAM decoder (e.g., 0250)",
    )
    pa.add_argument("--ckpt-root", type=Path, required=True, help="Root containing <size>/sam/*.pth")
    pa.add_argument("--sam-out-root", type=Path, required=True, help="SAM output root")
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--scale-stats", type=Path, default=None)
    pa.add_argument("--fast", action="store_true", help="Enable SAM fast mode")
    pa.add_argument("--fast-batch-size", type=int, default=64)
    pa.add_argument("--splits", type=str, default="val")
    args = pa.parse_args()

    if not args.skip_tiling:
        weights = tuple(float(x) for x in args.cost_weights.split(","))
        build_tiles(
            img=args.img,
            chm=args.chm,
            den=args.den,
            slic=args.slic,
            out=args.tiles_out,
            tile=args.tile_m,
            overlap=args.overlap_m,
            cost_weights=weights,
            thin_seeds=not args.no_thin,
            write_den=args.write_den,
        )

    size_dir = args.manifests_root / args.sam_size
    manifest_path = size_dir / "val_manifest.json"
    _write_manifest(args.tiles_out, manifest_path)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "pipelines" / "sam" / "predict_sam_from_manifests.py"),
        "--manifests-root",
        str(args.manifests_root),
        "--ckpt-root",
        str(args.ckpt_root),
        "--out-root",
        str(args.sam_out_root),
        "--device",
        args.device,
        "--sizes",
        args.sam_size,
        "--splits",
        args.splits,
    ]
    if args.scale_stats:
        cmd += ["--scale-stats", str(args.scale_stats)]
    if args.fast:
        cmd += ["--fast", "--fast-batch-size", str(args.fast_batch_size)]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
