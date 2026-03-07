"""Unified segmentation interfaces for proposal + refinement stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from samstars.data import (
    build_inference_manifest_from_tiles,
    build_pending_manifest,
    build_scale_stats_from_probs_manifest,
)
from samstars.segmentation.proposals import run_manifests


def run_segmentation(
    *,
    tiles_dir: Path,
    out_dir: Path,
    ckpt_root: Path,
    device: str = "cpu",
    size_id: str = "0474",
    stardist_model_dir: Path | None = None,
    stardist_model_name: str = "stardist",
    stardist_input_key: str = "index",
    stardist_cpu: bool = False,
    run_sam: bool = True,
    run_stardist: bool = True,
    skip_existing_sam: bool = True,
    fast: bool = True,
    fast_batch_size: int = 32,
) -> dict:
    """Run end-to-end segmentation from tiles to crown polygons."""
    out_dir = Path(out_dir)
    tiles_dir = Path(tiles_dir)
    ckpt_root = Path(ckpt_root)

    manifests_root = out_dir / "manifests"
    sam_out_root = out_dir / "sam_probs"
    stardist_out_dir = out_dir / "stardist_preds"
    probs_json = sam_out_root / size_id / "train_probs.json"
    train_probs_dir = sam_out_root / size_id / "train_probs"
    train_manifest = manifests_root / size_id / "train_manifest.json"
    pending_manifest = manifests_root / size_id / "train_manifest.pending.json"
    pending_manifest_root = out_dir / ".pending_manifests"

    recs, _ = build_inference_manifest_from_tiles(
        tiles_dir=tiles_dir,
        out_root=manifests_root,
        size_id=size_id,
        split="train",
        write_both=False,
    )

    pending_count = len(recs)
    sam_manifest_root = manifests_root
    if run_sam and skip_existing_sam:
        train_probs_dir.mkdir(parents=True, exist_ok=True)
        _, _, pending_count = build_pending_manifest(
            manifest_path=train_manifest,
            sam_out_dir=train_probs_dir,
            out_path=pending_manifest,
        )
        if pending_count > 0:
            dst_dir = pending_manifest_root / size_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            (dst_dir / "train_manifest.json").write_text(pending_manifest.read_text())
            sam_manifest_root = pending_manifest_root

    if run_sam and (not skip_existing_sam or pending_count > 0):
        run_manifests(
            sam_manifest_root,
            ckpt_root,
            sam_out_root,
            device=device,
            sizes=[size_id],
            splits=["train_manifest.json"],
            rgb_lo_hi=None,
            fast=fast,
            fast_batch=fast_batch_size,
        )

    if not run_stardist:
        return {
            "manifests_root": manifests_root,
            "sam_out_root": sam_out_root,
            "stardist_out_dir": stardist_out_dir,
            "probs_json": probs_json,
        }

    if not probs_json.exists() and train_probs_dir.exists():
        partial = []
        for idx in sorted(train_probs_dir.glob("*_sam_index.tif")):
            stem = idx.stem
            if not stem.endswith("_sam_index"):
                continue
            base = stem[:-10]
            prob = train_probs_dir / f"{base}_sam_prob.tif"
            chm = train_probs_dir / f"{base}_chm.tif"
            cost = train_probs_dir / f"{base}_cost.tif"
            logit = train_probs_dir / f"{base}_logit.tif"
            if prob.exists() and chm.exists() and cost.exists() and logit.exists():
                partial.append(
                    {
                        "file_name": str(prob),
                        "chm": str(chm),
                        "cost": str(cost),
                        "logit": str(logit),
                        "index": str(idx),
                        "mask": "",
                        "meta": {},
                    }
                )
        if partial:
            probs_json.parent.mkdir(parents=True, exist_ok=True)
            json.dump(partial, open(probs_json, "w"))

    if not probs_json.exists():
        raise RuntimeError(f"Missing probs manifest: {probs_json}")

    if stardist_model_dir is None:
        stardist_model_dir = Path(__file__).resolve().parents[2] / "models"
    stardist_model_dir = Path(stardist_model_dir)

    scale_stats = out_dir / "scale_2_98.json"
    if not scale_stats.exists():
        build_scale_stats_from_probs_manifest(
            probs_json=probs_json,
            out=scale_stats,
            percentiles=(2.0, 98.0),
            chm_key="chm",
            nodata_values=(-9999.0,),
        )

    refinement_py = Path(__file__).resolve().parent / "refinement.py"
    cmd = [
        sys.executable,
        str(refinement_py),
        "--probs-json",
        str(probs_json),
        "--model-dir",
        str(stardist_model_dir),
        "--model-name",
        stardist_model_name,
        "--out-dir",
        str(stardist_out_dir),
        "--input-key",
        stardist_input_key,
        "--use-aux",
        "--scale-stats",
        str(scale_stats),
        "--preds-name",
        "preds.gpkg",
        "--write-labels",
    ]
    env = None
    if stardist_cpu:
        env = dict(**os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, env=env)

    return {
        "manifests_root": manifests_root,
        "sam_out_root": sam_out_root,
        "stardist_out_dir": stardist_out_dir,
        "probs_json": probs_json,
        "scale_stats": scale_stats,
    }
