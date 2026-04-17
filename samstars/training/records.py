#!/usr/bin/env python3
"""Build StarDist training records from proposal rasters and crown labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def _proposal_stem(rec: dict) -> str:
    """Return the proposal-record stem without the raster suffix."""
    for key in ("index", "file_name", "logit"):
        value = rec.get(key, "")
        if not value:
            continue
        stem = Path(value).stem
        for suffix in ("_sam_index", "_sam_prob", "_logit"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem
    raise ValueError("Record has no usable raster path.")


def _tile_name_from_stem(stem: str) -> str:
    """Infer the tile directory name from a proposal stem."""
    tile_name, _, chip_id = stem.rpartition("_")
    if not tile_name or chip_id == "":
        raise ValueError(f"Could not infer tile name from proposal stem: {stem}")
    return tile_name


def _rasterize_crowns(crowns_path: Path, reference_raster: Path, out_path: Path) -> bool:
    """Rasterize crown polygons to the reference raster grid."""
    with rasterio.open(reference_raster) as src:
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs
        meta = src.profile | {"count": 1, "dtype": "uint16", "compress": "deflate"}

    crowns = gpd.read_file(crowns_path)
    if crowns.empty:
        return False
    if crowns.crs is None:
        crowns = crowns.set_crs(crs)
    elif crowns.crs != crs:
        crowns = crowns.to_crs(crs)

    shapes = [(geom, idx + 1) for idx, geom in enumerate(crowns.geometry) if geom is not None and not geom.is_empty]
    if not shapes:
        return False

    mask = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint16",
    )
    if np.max(mask) == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask, 1)
    return True


def build_stardist_training_records(
    *,
    proposal_records: Path,
    tiles_dir: Path,
    out_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Build train/val StarDist records from proposal rasters and labeled tiles."""
    proposal_records = Path(proposal_records)
    tiles_dir = Path(tiles_dir)
    out_dir = Path(out_dir)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    base_records = json.load(open(proposal_records))
    records: list[dict] = []
    for rec in base_records:
        stem = _proposal_stem(rec)
        tile_name = _tile_name_from_stem(stem)
        crowns_path = tiles_dir / tile_name / "crowns.gpkg"
        if not crowns_path.exists():
            continue

        reference = Path(rec.get("index") or rec.get("file_name") or rec.get("logit") or "")
        if not reference.exists():
            continue

        mask_path = masks_dir / f"{stem}_mask.tif"
        if not mask_path.exists():
            ok = _rasterize_crowns(crowns_path, reference, mask_path)
            if not ok:
                continue

        out_rec = dict(rec)
        out_rec["mask"] = str(mask_path)
        records.append(out_rec)

    rng = random.Random(seed)
    rng.shuffle(records)

    if len(records) <= 1 or val_fraction <= 0:
        train_records = records
        val_records = []
    else:
        n_val = max(1, int(round(len(records) * val_fraction)))
        n_val = min(n_val, len(records) - 1)
        val_records = records[:n_val]
        train_records = records[n_val:]

    train_path = out_dir / "train_records.json"
    val_path = out_dir / "val_records.json"
    json.dump(train_records, open(train_path, "w"))
    json.dump(val_records, open(val_path, "w"))
    print(f"✓ wrote {train_path} ({len(train_records)} records)")
    print(f"✓ wrote {val_path} ({len(val_records)} records)")
    return train_path, val_path


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Build StarDist training records from proposal rasters and crown labels.")
    parser.add_argument("--proposal-records", type=Path, required=True, help="Proposal-record JSON produced by SAM proposals")
    parser.add_argument("--tiles-dir", type=Path, required=True, help="Prepared tile directory containing crowns.gpkg labels")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write masks and train/val record JSON files")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of records to reserve for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split")
    args = parser.parse_args()

    build_stardist_training_records(
        proposal_records=args.proposal_records,
        tiles_dir=args.tiles_dir,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
