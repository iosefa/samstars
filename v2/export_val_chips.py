#!/usr/bin/env python3
"""
Export validation chips (img + msk) to PNG/TIF for inspection, based on val_manifest.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import geopandas as gpd
from rasterio.features import rasterize


def export_chips(manifest: Path, out_dir: Path, size: int = 512):
    recs = json.load(open(manifest))
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        img_path = Path(rec["img_path"])
        crowns_path = Path(rec["crowns_path"])
        x0, y0, x1, y1 = rec["window"]
        win = Window(x0, y0, x1 - x0, y1 - y0)
        with rasterio.open(img_path) as src:
            bands = [7, 5, 3]  # 1-based indices for bands 6,4,2 (0-based)
            available = src.count
            bands = [b for b in bands if b <= available] or [1]
            img_win = src.read(bands, window=win, boundless=True, fill_value=0)
            tfm = rasterio.windows.transform(win, src.transform)
            crs = src.crs

        h, w = img_win.shape[1], img_win.shape[2]
        pad_h, pad_w = size - h, size - w
        if pad_h < 0 or pad_w < 0:
            img_win = img_win[:, : size, : size]
            h, w = size, size
            pad_h = pad_w = 0
        if pad_h > 0 or pad_w > 0:
            img_win = np.pad(img_win, ((0, 0), (0, pad_h), (0, pad_w)))

        crowns = gpd.read_file(crowns_path).to_crs(crs)
        mask = rasterize(
            [(geom, 1) for geom in crowns.geometry],
            out_shape=(h, w),
            transform=tfm,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        if pad_h > 0 or pad_w > 0:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)))
        # save RGB PNG and mask PNG
        denom = img_win.max() if img_win.max() > 0 else 1.0
        rgb = np.clip((img_win.transpose(1, 2, 0) / denom) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / f"{rec['tile']}_{rec['chip_id']}_img.png")
        Image.fromarray(mask * 255).save(out_dir / f"{rec['tile']}_{rec['chip_id']}_msk.png")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--val-manifest", type=Path, required=True)
    pa.add_argument("--out-dir", type=Path, required=True)
    args = pa.parse_args()
    export_chips(args.val_manifest, args.out_dir)
    print(f"✓ wrote val chip PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
