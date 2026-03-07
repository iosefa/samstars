"""Manifest build, validation, and update helpers."""

from __future__ import annotations

import json
from pathlib import Path

import rasterio


def build_inference_manifest_from_tiles(
    *,
    tiles_dir: Path,
    out_root: Path,
    size_id: str = "0474",
    split: str = "train",
    write_both: bool = False,
) -> tuple[list[dict], list[Path]]:
    """Build manifest-style JSON records from tile folders."""
    size_dir = out_root / size_id
    size_dir.mkdir(parents=True, exist_ok=True)

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

    split_names = ["train_manifest.json", "val_manifest.json"] if write_both else [f"{split}_manifest.json"]
    outs: list[Path] = []
    for name in split_names:
        out = size_dir / name
        json.dump(recs, open(out, "w"))
        outs.append(out)
    return recs, outs


def build_pending_manifest(
    *,
    manifest_path: Path,
    sam_out_dir: Path,
    out_path: Path,
) -> tuple[int, int, int]:
    """Write pending-only manifest by skipping records with complete SAM outputs."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not sam_out_dir.exists():
        raise FileNotFoundError(f"Missing SAM output dir: {sam_out_dir}")

    recs = json.load(open(manifest_path))
    if not isinstance(recs, list):
        raise RuntimeError(f"Manifest is not a list: {manifest_path}")

    pending = []
    done = 0
    for rec in recs:
        tile = rec.get("tile")
        chip_id = rec.get("chip_id")
        if tile is None or chip_id is None:
            pending.append(rec)
            continue
        name = f"{tile}_{chip_id}"
        idx = sam_out_dir / f"{name}_sam_index.tif"
        prob = sam_out_dir / f"{name}_sam_prob.tif"
        logit = sam_out_dir / f"{name}_logit.tif"
        if idx.exists() and prob.exists() and logit.exists():
            done += 1
            continue
        pending.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(pending, open(out_path, "w"))
    return len(recs), done, len(pending)
