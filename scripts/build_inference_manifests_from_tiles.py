#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Build manifest-style train/val JSONs from tile folders for inference-only runs."
    )
    pa.add_argument("--tiles-dir", type=Path, required=True, help="Directory containing tile_* folders")
    pa.add_argument("--out-root", type=Path, required=True, help="Output root for manifests")
    pa.add_argument("--size-id", type=str, default="0474", help="Size folder name under out-root")
    pa.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split filename to write")
    pa.add_argument("--write-both", action="store_true", help="Write both train_manifest.json and val_manifest.json")
    args = pa.parse_args()

    size_dir = args.out_root / args.size_id
    size_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    tile_dirs = sorted(p for p in args.tiles_dir.iterdir() if p.is_dir() and p.name.startswith("tile_"))
    for tile in tile_dirs:
        img = tile / "img.tif"
        chm = tile / "chm.tif"
        cost = tile / "cost.tif"
        seeds = tile / "seeds.gpkg"
        if not (img.exists() and chm.exists() and cost.exists() and seeds.exists()):
            continue

        with rasterio.open(img) as src:
            w, h = int(src.width), int(src.height)

        recs.append(
            {
                "tile": tile.name,
                "chip_id": 0,
                "window": [0, 0, w, h],
                "img_path": str(img.resolve()),
                "size": int(max(w, h)),
            }
        )

    if not recs:
        raise SystemExit(f"No valid tiles found in {args.tiles_dir}")

    split_names = ["train_manifest.json", "val_manifest.json"] if args.write_both else [f"{args.split}_manifest.json"]
    for name in split_names:
        out = size_dir / name
        json.dump(recs, open(out, "w"))
        print(f"✓ wrote {out} ({len(recs)} records)")


if __name__ == "__main__":
    main()
