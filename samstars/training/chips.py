#!/usr/bin/env python3
"""Build proposal-decoder training chips from prepared tiles and crown annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from samstars._preprocessing import positive_int, window_starts


def _clip_annotations(gdf: gpd.GeoDataFrame, bbox_geom) -> gpd.GeoDataFrame:
    """Clip crown polygons to a chip footprint."""
    sub = gdf[gdf.intersects(bbox_geom)].copy()
    if sub.empty:
        return sub
    sub["geometry"] = sub.geometry.buffer(0)
    sub["geometry"] = sub.geometry.intersection(bbox_geom)
    sub = sub[~sub.geometry.is_empty]
    return sub


def _prompt_from_mask(mask: np.ndarray) -> dict[str, list[int]] | None:
    """Build a prompt point and box from a binary mask."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return {
        "point": [int(xs.mean()), int(ys.mean())],
        "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def _rasterize_one_geometry(geom, out_shape: tuple[int, int], transform) -> np.ndarray:
    """Rasterize one geometry into a binary chip mask."""
    mask = rasterize(
        [(geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return mask.astype(np.uint8)


def _save_prompt_json(path: Path, payload: dict[str, list[int]]) -> None:
    """Write prompt metadata for one chip."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _iter_tiles(tiles_dir: Path, skip_tiles: set[str]) -> list[Path]:
    """Return the tile directories to process."""
    return sorted(p for p in tiles_dir.iterdir() if p.is_dir() and p.name not in skip_tiles)


def build_training_chips(
    tiles_dir: Path,
    out_dir: Path,
    chip_size: int = 512,
    stride: int | None = None,
    image_bands: tuple[int, ...] = (1, 2, 3),
    skip_tiles: tuple[str, ...] = (),
) -> int:
    """Build proposal-decoder training chips from labeled tiles."""
    chip_size = positive_int(chip_size, "chip_size")
    stride = chip_size if stride is None else positive_int(stride, "stride")
    if stride > chip_size:
        raise ValueError("stride must not exceed chip_size (which would leave gaps).")
    skip_set = set(skip_tiles)

    for subdir in ("images", "masks", "chm", "prompts", "valid"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    written = 0
    for tile in _iter_tiles(tiles_dir, skip_set):
        img_path = tile / "img.tif"
        crowns_path = tile / "crowns.gpkg"
        chm_path = tile / "chm.tif"
        if not img_path.exists() or not crowns_path.exists() or not chm_path.exists():
            print(f"Skipping {tile.name}: missing img.tif, crowns.gpkg, or chm.tif")
            continue

        with rasterio.open(img_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            image_masked = src.read(image_bands, masked=True).astype(np.float32)
            image = image_masked.filled(0)
            valid = ~np.ma.getmaskarray(image_masked).any(axis=0) & np.isfinite(image).all(axis=0)

        with rasterio.open(chm_path) as chm_src:
            if (chm_src.shape != (height, width) or chm_src.transform != transform or chm_src.crs != crs):
                raise ValueError(f"{tile.name}: CHM must be aligned to img.tif before chip creation.")
            chm_masked = chm_src.read(1, masked=True).astype(np.float32)
            chm = chm_masked.filled(0)
            valid &= ~np.ma.getmaskarray(chm_masked) & np.isfinite(chm)
        image[:, ~valid] = 0
        chm[~valid] = 0

        crowns = gpd.read_file(crowns_path)
        if crowns.empty:
            print(f"Skipping {tile.name}: empty crowns")
            continue
        if crowns.crs is None:
            crowns = crowns.set_crs(crs)
        elif crowns.crs != crs:
            crowns = crowns.to_crs(crs)

        tile_count = 0
        chip_id = 0
        for y0 in window_starts(height, chip_size, stride):
            for x0 in window_starts(width, chip_size, stride):
                y1 = min(y0 + chip_size, height)
                x1 = min(x0 + chip_size, width)
                if y1 - y0 < 1 or x1 - x0 < 1:
                    continue

                window = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                win_transform = rasterio.windows.transform(window, transform)
                chip_bounds = box(*rasterio.transform.array_bounds(y1 - y0, x1 - x0, win_transform))
                clipped = _clip_annotations(crowns, chip_bounds)
                if clipped.empty:
                    continue

                image_chip = image[:, y0:y1, x0:x1]
                chm_chip = chm[y0:y1, x0:x1]
                valid_chip = valid[y0:y1, x0:x1]
                pad_h = chip_size - image_chip.shape[1]
                pad_w = chip_size - image_chip.shape[2]
                if pad_h or pad_w:
                    image_chip = np.pad(image_chip, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
                    chm_chip = np.pad(chm_chip, ((0, pad_h), (0, pad_w)), mode="constant")
                    valid_chip = np.pad(valid_chip, ((0, pad_h), (0, pad_w)), mode="constant")

                base_image_chip = image_chip.astype(np.float32)
                base_chm_chip = chm_chip.astype(np.float32)
                for geom in clipped.geometry:
                    mask_chip = _rasterize_one_geometry(geom, (y1 - y0, x1 - x0), win_transform)
                    if pad_h or pad_w:
                        mask_chip = np.pad(mask_chip, ((0, pad_h), (0, pad_w)), mode="constant")
                    mask_chip[~valid_chip] = 0

                    prompt = _prompt_from_mask(mask_chip)
                    if prompt is None:
                        continue

                    stem = f"{tile.name}_{chip_id}"
                    np.save(out_dir / "images" / f"{stem}.npy", base_image_chip)
                    np.save(out_dir / "masks" / f"{stem}.npy", mask_chip.astype(np.uint8))
                    np.save(out_dir / "chm" / f"{stem}.npy", base_chm_chip)
                    np.save(out_dir / "valid" / f"{stem}.npy", valid_chip)
                    _save_prompt_json(out_dir / "prompts" / f"{stem}.json", prompt)

                    chip_id += 1
                    tile_count += 1
                    written += 1

        print(f"{tile.name}: wrote {tile_count} training chips")

    print(f"Built {written} training chips in {out_dir}")
    return written


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Build proposal-decoder training chips from labeled tiles.")
    parser.add_argument("--tiles-dir", type=Path, required=True, help="Directory containing prepared tile folders")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write training chips")
    parser.add_argument("--chip-size", type=int, default=512, help="Output chip size in pixels")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Chip stride in pixels; defaults to the chip size when omitted",
    )
    parser.add_argument(
        "--image-bands",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="1-based raster bands to include in each training chip",
    )
    parser.add_argument(
        "--skip-tiles",
        nargs="*",
        default=(),
        help="Tile directory names to skip",
    )
    args = parser.parse_args()

    build_training_chips(
        tiles_dir=args.tiles_dir,
        out_dir=args.out_dir,
        chip_size=args.chip_size,
        stride=args.stride,
        image_bands=tuple(args.image_bands),
        skip_tiles=tuple(args.skip_tiles),
    )


if __name__ == "__main__":
    main()
