#!/usr/bin/env python3
"""
Run SAM decoder on chips (train/val) and write smoothed probability rasters + JSON manifests.
Uses per-chip point prompt, CHM/COST weighting, Gaussian smoothing similar to helpers/predict_sam.py.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.crs import CRS
from segment_anything import sam_model_registry, SamPredictor


def stretch01(arr, p=(2, 98)):
    """Stretch01."""
    lo, hi = np.percentile(arr[np.isfinite(arr)], p)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def smooth_prob(logit, core_sigma=8.0, core_gamma=1.7, logit_scale=1.4, smooth_sigma=6.0, chm=None, cost=None, w_chm=0.75, w_cost=0.25):
    """Smooth prob."""
    prob = torch.sigmoid(torch.clamp(logit_scale * logit, -12, 12))
    yy, xx = torch.meshgrid(torch.arange(prob.shape[-2], device=prob.device), torch.arange(prob.shape[-1], device=prob.device), indexing="ij")
    cy, cx = prob.shape[-2] / 2, prob.shape[-1] / 2
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core_w = torch.exp(-(d ** 2) / (2 * core_sigma ** 2)) ** core_gamma
    prob = prob * (core_w / (core_w.max() + 1e-6))
    if smooth_sigma > 0:
        prob_np = prob.cpu().numpy()
        prob_np = gaussian_filter(prob_np, smooth_sigma)
        prob = torch.from_numpy(prob_np).to(prob.device)
    def _resize(arr_np):
        """Internal helper function."""
        t = torch.from_numpy(arr_np).to(prob.device, dtype=prob.dtype)
        if t.shape != prob.shape:
            t = F.interpolate(t[None, None, ...], size=prob.shape, mode="bilinear", align_corners=False)[0, 0]
        return t

    if chm is not None:
        chm_n = _resize(stretch01(chm))
        prob = prob * (1 - w_chm + w_chm * chm_n)
    if cost is not None:
        cost_n = _resize(stretch01(cost))
        prob = prob * (1 - w_cost + w_cost * (1 - cost_n))
    return prob.cpu().numpy()


def run_split(chip_dir: Path, recs_json: Path, predictor: SamPredictor, out_dir: Path):
    """Run split."""
    recs = json.load(open(recs_json))
    out_records = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        name = Path(rec["file_name"]).stem
        base = Path(rec["file_name"]).parent.parent
        img = np.load(base / "img" / f"{name}.npy").astype(np.float32)
        msk = np.load(base / "msk" / f"{name}.npy").astype(np.float32)
        chm_path = base / "chm" / f"{name}.npy"
        cost_path = base / "cost" / f"{name}.npy"
        chm = np.load(chm_path).astype(np.float32) if chm_path.exists() else None
        cost = np.load(cost_path).astype(np.float32) if cost_path.exists() else None
        point = json.load(open(base / "json" / f"{name}.json")).get("point", [img.shape[2] // 2, img.shape[1] // 2])

        # SAM expects HWC uint8 RGB
        sam_img = np.stack([img[1], img[2], img[0]], axis=-1)
        sam_img = np.clip(sam_img, 0, 255).astype(np.uint8)
        predictor.set_image(sam_img)
        pts = torch.tensor(point, device=predictor.device, dtype=torch.float32)[None, None, :]
        labels = torch.ones((1, 1), device=predictor.device, dtype=torch.int64)
        with torch.no_grad():
            _, _, logits = predictor.predict_torch(point_coords=pts, point_labels=labels, multimask_output=False, return_logits=True)
        logit = logits[0, 0]
        logit_up = F.interpolate(logit[None, None, ...], size=(img.shape[1], img.shape[2]), mode="bilinear", align_corners=False)[0, 0]
        prob = smooth_prob(logit_up, chm=chm, cost=cost)

        np.save(out_dir / f"{name}_prob.npy", prob.astype(np.float32))
        meta = rec.get("meta", {})
        out_records.append({"file_name": str(out_dir / f"{name}_prob.npy"), "mask": str(base / "msk" / f"{name}.npy"), "meta": meta})
    return out_records


def main():
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser()
    pa.add_argument("--chip-dir", type=Path, required=True)
    pa.add_argument("--records-json", type=Path, required=True)
    pa.add_argument("--val-records-json", type=Path, required=True)
    pa.add_argument("--sam-ckpt", type=Path, required=True)
    pa.add_argument("--out-dir", type=Path, required=True)
    args = pa.parse_args()

    sam = sam_model_registry["vit_l"](checkpoint=str(args.sam_ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device).eval()
    predictor = SamPredictor(sam)

    train_out = args.out_dir / "train"
    val_out = args.out_dir / "val"
    train_recs = run_split(args.chip_dir, args.records_json, predictor, train_out)
    val_recs = run_split(args.chip_dir, args.val_records_json, predictor, val_out)

    json.dump(train_recs, open(args.out_dir / "train_probs.json", "w"))
    json.dump(val_recs, open(args.out_dir / "val_probs.json", "w"))
    print(f"✓ wrote SAM prob manifests to {args.out_dir}")


if __name__ == "__main__":
    main()
