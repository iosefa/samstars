"""Unified segmentation interfaces for proposal + refinement stages."""

from __future__ import annotations

import json
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


def _complete_proposal_records_from_rasters(proposal_rasters_dir: Path) -> list[dict]:
    """Build proposal records from every complete proposal raster set on disk."""
    recs = []
    if not proposal_rasters_dir.exists():
        return recs

    for idx in sorted(proposal_rasters_dir.glob("*_sam_index.tif")):
        stem = idx.stem
        if not stem.endswith("_sam_index"):
            continue
        base = stem[:-10]
        prob = proposal_rasters_dir / f"{base}_sam_prob.tif"
        chm = proposal_rasters_dir / f"{base}_chm.tif"
        cost = proposal_rasters_dir / f"{base}_cost.tif"
        logit = proposal_rasters_dir / f"{base}_logit.tif"
        if not (prob.exists() and chm.exists() and cost.exists() and logit.exists()):
            continue
        recs.append(
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
    return recs


def _record_count(records_json: Path) -> int | None:
    """Return the number of records in a records JSON file, if readable."""
    if not records_json.exists():
        return None
    recs = json.load(open(records_json))
    return len(recs) if isinstance(recs, list) else None


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
    stardist_use_aux: bool = True,
    stardist_prob_thr: float | None = None,
    stardist_nms_thr: float | None = None,
    stardist_window_size_m: float | None = None,
    stardist_window_overlap_m: float = 0.0,
    stardist_dedup_iou_thr: float = 0.0,
    stardist_cpu: bool = False,
    stardist_resume: bool = False,
    stardist_resume_from_labels: bool = False,
    stardist_skip_sources: str | None = None,
    stardist_skip_sources_file: Path | None = None,
    run_sam: bool = True,
    run_stardist: bool = True,
    skip_existing_sam: bool = True,
    log_refinement: bool = False,
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
        prepared_root = out_dir / "prep" / "scene"
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
    proposal_records_rebuilt = False

    if not run_stardist:
        return {
            "records_root": records_root,
            "proposals_root": proposals_root,
            "crowns_dir": crowns_dir,
            "proposal_records": proposal_records,
        }

    if proposal_rasters_dir.exists():
        partial = _complete_proposal_records_from_rasters(proposal_rasters_dir)
        existing_count = _record_count(proposal_records)
        if partial and existing_count != len(partial):
            proposal_records.parent.mkdir(parents=True, exist_ok=True)
            json.dump(partial, open(proposal_records, "w"))
            _normalize_record_paths(proposal_records)
            proposal_records_rebuilt = True

    if not proposal_records.exists():
        raise RuntimeError(f"Missing proposal records: {proposal_records}")

    stardist_model_dir, stardist_model_name = _resolve_stardist_model(stardist_model=stardist_model)
    resolved_stardist_model = Path(stardist_model_dir) / stardist_model_name

    scale_stats = out_dir / "proposal_scale_stats.json"
    if proposal_records_rebuilt and scale_stats.exists():
        scale_stats.unlink()
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
        "--preds-name",
        "crowns.gpkg",
        "--write-labels",
    ]
    if stardist_use_aux:
        cmd.extend(
            [
                "--use-aux",
                "--scale-stats",
                str(scale_stats),
            ]
        )
    if (stardist_prob_thr is None) ^ (stardist_nms_thr is None):
        raise ValueError("Provide both stardist_prob_thr and stardist_nms_thr, or neither.")
    if stardist_window_size_m is not None and stardist_window_size_m <= 0:
        raise ValueError("stardist_window_size_m must be > 0.")
    if stardist_window_overlap_m < 0:
        raise ValueError("stardist_window_overlap_m must be >= 0.")
    if (
        stardist_window_size_m is not None
        and stardist_window_overlap_m >= stardist_window_size_m
    ):
        raise ValueError("stardist_window_overlap_m must be smaller than stardist_window_size_m.")
    if not (0.0 <= stardist_dedup_iou_thr < 1.0):
        raise ValueError("stardist_dedup_iou_thr must be in [0, 1).")
    if stardist_resume_from_labels and not stardist_resume:
        raise ValueError("stardist_resume_from_labels requires stardist_resume.")
    if stardist_prob_thr is not None and stardist_nms_thr is not None:
        cmd.extend(
            [
                "--prob-thr",
                str(stardist_prob_thr),
                "--nms-thr",
                str(stardist_nms_thr),
            ]
        )
    if stardist_window_size_m is not None:
        cmd.extend(
            [
                "--window-size-m",
                str(stardist_window_size_m),
                "--window-overlap-m",
                str(stardist_window_overlap_m),
            ]
        )
    if stardist_dedup_iou_thr > 0:
        cmd.extend(
            [
                "--dedup-iou-thr",
                str(stardist_dedup_iou_thr),
            ]
        )
    if log_refinement:
        cmd.append("--log")
    if stardist_resume:
        cmd.append("--resume")
    if stardist_resume_from_labels:
        cmd.append("--resume-from-labels")
    if stardist_skip_sources:
        cmd.extend(["--skip-sources", stardist_skip_sources])
    if stardist_skip_sources_file is not None:
        cmd.extend(["--skip-sources-file", str(stardist_skip_sources_file)])
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
