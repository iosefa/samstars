"""Unified segmentation interfaces for proposal + refinement stages."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from samstars.data import (
    build_pending_tile_records,
    build_single_tile,
    build_scale_stats_from_proposal_records,
    build_tile_records,
)


def _resolve_sam_ckpt(
    *,
    sam_ckpt: Path | None,
) -> Path:
    """Resolve the SAM decoder checkpoint path."""
    if sam_ckpt is None:
        raise ValueError("Provide sam_ckpt.")
    ckpt = Path(sam_ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {ckpt}")
    return ckpt


def _resolve_stardist_model(
    *,
    stardist_model: Path | None,
) -> tuple[Path, str]:
    """Resolve the StarDist model basedir and model name."""
    if stardist_model is None:
        raise ValueError("Provide stardist_model.")
    model_path = Path(stardist_model)
    basedir, name = model_path.parent, model_path.name
    config_path = basedir / name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing StarDist model config: {config_path}")
    return basedir, name


def _normalize_record_paths(records_json: Path) -> None:
    """Rewrite known raster paths in a records JSON file to absolute paths."""
    if not records_json.exists():
        return
    recs = json.load(open(records_json))
    if not isinstance(recs, list):
        return

    changed = False
    base = records_json.parent
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        for key in ("file_name", "chm", "cost", "logit", "index", "mask"):
            value = rec.get(key)
            if not value:
                continue
            path = Path(value)
            if path.is_absolute():
                continue
            candidate = (base / path).resolve()
            if candidate.exists():
                rec[key] = str(candidate)
                changed = True

    if changed:
        json.dump(recs, open(records_json, "w"))


def run_segmentation(
    *,
    tiles_dir: Path | None = None,
    img: Path | None = None,
    chm: Path | None = None,
    den: Path | None = None,
    rgb_bands: tuple[int, int, int] | None = None,
    out_dir: Path,
    sam_ckpt: Path,
    device: str = "cpu",
    stardist_model: Path | None = None,
    stardist_input_key: str = "index",
    stardist_cpu: bool = False,
    run_sam: bool = True,
    run_stardist: bool = True,
    skip_existing_sam: bool = True,
    fast: bool = True,
    fast_batch_size: int = 32,
    crop_smooth_sigma: float = 2.0,
    smooth_sigma: float = 4.0,
    post_smooth_sigma: float = 0.0,
) -> dict:
    """Run end-to-end segmentation from prepared tiles or one raw input scene."""
    out_dir = Path(out_dir)
    resolved_sam_ckpt: Path | None = None
    resolved_stardist_model: Path | None = None

    raw_inputs = (img, chm, den)
    if tiles_dir is not None and any(v is not None for v in raw_inputs):
        raise ValueError("Provide either tiles_dir or img/chm/den, not both.")
    if tiles_dir is None:
        if any(v is None for v in raw_inputs):
            raise ValueError("Provide either tiles_dir or all of img/chm/den.")
        source_key = json.dumps(
            {
                "img": str(Path(img).resolve()),
                "chm": str(Path(chm).resolve()),
                "den": str(Path(den).resolve()),
            },
            sort_keys=True,
        )
        prepared_root = out_dir / ".prepared_tiles" / hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
        tiles_dir = build_single_tile(
            img=Path(img),
            chm=Path(chm),
            den=Path(den),
            rgb_bands=rgb_bands,
            out=prepared_root,
            skip_existing=True,
        )
    tiles_dir = Path(tiles_dir)

    records_root = out_dir / "run_records"
    proposals_root = out_dir / "proposals"
    crowns_dir = out_dir / "crowns"
    proposal_records = proposals_root / "proposal_records.json"
    proposal_rasters_dir = proposals_root / "rasters"
    tile_records = records_root / "tiles.json"
    pending_tile_records_path = records_root / "tiles.pending.json"
    recs, _ = build_tile_records(
        tiles_dir=tiles_dir,
        out_root=records_root,
    )

    pending_count = len(recs)
    active_tile_records = tile_records
    resolved_sam_ckpt = _resolve_sam_ckpt(sam_ckpt=sam_ckpt)
    if run_sam and skip_existing_sam:
        proposal_rasters_dir.mkdir(parents=True, exist_ok=True)
        _, _, pending_count = build_pending_tile_records(
            records_path=tile_records,
            proposal_rasters_dir=proposal_rasters_dir,
            out_path=pending_tile_records_path,
        )
        if pending_count > 0:
            active_tile_records = pending_tile_records_path

    if run_sam and (not skip_existing_sam or pending_count > 0):
        from samstars.segmentation.proposals import run_tile_proposals

        run_tile_proposals(
            active_tile_records,
            proposals_root,
            resolved_sam_ckpt,
            device=device,
            rgb_lo_hi=None,
            fast=fast,
            fast_batch=fast_batch_size,
            crop_smooth_sigma=crop_smooth_sigma,
            smooth_sigma=smooth_sigma,
            post_smooth_sigma=post_smooth_sigma,
        )

    _normalize_record_paths(proposal_records)

    if not run_stardist:
        return {
            "records_root": records_root,
            "proposals_root": proposals_root,
            "crowns_dir": crowns_dir,
            "proposal_records": proposal_records,
        }

    if not proposal_records.exists() and proposal_rasters_dir.exists():
        partial = []
        for idx in sorted(proposal_rasters_dir.glob("*_sam_index.tif")):
            stem = idx.stem
            if not stem.endswith("_sam_index"):
                continue
            base = stem[:-10]
            prob = proposal_rasters_dir / f"{base}_sam_prob.tif"
            chm = proposal_rasters_dir / f"{base}_chm.tif"
            cost = proposal_rasters_dir / f"{base}_cost.tif"
            logit = proposal_rasters_dir / f"{base}_logit.tif"
            if prob.exists() and chm.exists() and cost.exists() and logit.exists():
                partial.append(
                    {
                        "file_name": str(prob.resolve()),
                        "chm": str(chm.resolve()),
                        "cost": str(cost.resolve()),
                        "logit": str(logit.resolve()),
                        "index": str(idx.resolve()),
                        "mask": "",
                        "meta": {},
                    }
                )
        if partial:
            proposal_records.parent.mkdir(parents=True, exist_ok=True)
            json.dump(partial, open(proposal_records, "w"))
            _normalize_record_paths(proposal_records)

    if not proposal_records.exists():
        raise RuntimeError(f"Missing proposal records: {proposal_records}")

    stardist_model_dir, stardist_model_name = _resolve_stardist_model(stardist_model=stardist_model)
    resolved_stardist_model = Path(stardist_model_dir) / stardist_model_name

    scale_stats = out_dir / "proposal_scale_stats.json"
    if not scale_stats.exists():
        build_scale_stats_from_proposal_records(
            proposal_records=proposal_records,
            out=scale_stats,
            percentiles=(2.0, 98.0),
            chm_key="chm",
            nodata_values=(-9999.0,),
        )

    refinement_py = Path(__file__).resolve().parent / "refinement.py"
    cmd = [
        sys.executable,
        str(refinement_py),
        "--proposal-records",
        str(proposal_records),
        "--model-dir",
        str(Path(stardist_model_dir)),
        "--model-name",
        stardist_model_name,
        "--out-dir",
        str(crowns_dir),
        "--input-key",
        stardist_input_key,
        "--use-aux",
        "--scale-stats",
        str(scale_stats),
        "--preds-name",
        "crowns.gpkg",
        "--write-labels",
    ]
    env = None
    if stardist_cpu:
        env = dict(**os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, env=env)

    return {
        "records_root": records_root,
        "proposals_root": proposals_root,
        "crowns_dir": crowns_dir,
        "proposal_records": proposal_records,
        "scale_stats": scale_stats,
        "sam_ckpt": resolved_sam_ckpt,
        "stardist_model": resolved_stardist_model,
    }
