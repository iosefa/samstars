"""Run-record build and update helpers for tiled segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import rasterio


def build_tile_records(
    *,
    tiles_dir: Path,
    out_root: Path,
    filename: str = "tiles.json",
) -> tuple[list[dict], Path]:
    """Build tile-level run records from a prepared tile directory."""
    out_root.mkdir(parents=True, exist_ok=True)

    recs: list[dict] = []
    tile_dirs = sorted(p for p in tiles_dir.iterdir() if p.is_dir() and p.name.startswith("tile_"))
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
        raise RuntimeError(f"No valid tiles found in {tiles_dir}")

    out_path = out_root / filename
    json.dump(recs, open(out_path, "w"))
    return recs, out_path


def build_pending_tile_records(
    *,
    records_path: Path,
    proposal_rasters_dir: Path,
    out_path: Path,
) -> tuple[int, int, int]:
    """Write a pending-only tile-record file by skipping finished proposal outputs."""
    if not records_path.exists():
        raise FileNotFoundError(f"Missing records: {records_path}")
    if not proposal_rasters_dir.exists():
        raise FileNotFoundError(f"Missing proposal raster dir: {proposal_rasters_dir}")

    recs = json.load(open(records_path))
    if not isinstance(recs, list):
        raise RuntimeError(f"Records file is not a list: {records_path}")

    pending = []
    done = 0
    for rec in recs:
        tile = rec.get("tile")
        chip_id = rec.get("chip_id")
        if tile is None or chip_id is None:
            pending.append(rec)
            continue
        name = f"{tile}_{chip_id}"
        idx = proposal_rasters_dir / f"{name}_sam_index.tif"
        prob = proposal_rasters_dir / f"{name}_sam_prob.tif"
        logit = proposal_rasters_dir / f"{name}_logit.tif"
        if idx.exists() and prob.exists() and logit.exists():
            done += 1
            continue
        pending.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(pending, open(out_path, "w"))
    return len(recs), done, len(pending)
