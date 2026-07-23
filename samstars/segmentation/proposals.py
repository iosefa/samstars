#!/usr/bin/env python3
"""
Run SAM (fine-tuned decoder) on tile run records using seed prompts and the *original*
probability-building logic:

- Use SAM low-res logits upsampled to the crop size
- Optional logit Gaussian blur per crop
- CHM "boost" term
- Seed-centered core weighting in *meters*
- Mosaic max-compositing
- Large Gaussian smoothing of the mosaic with valid-mask normalization
- Build "index" by fusing SAM prob + CHM + COST + ground weighting

For each tile record, writes:
  *_logit.tif     (debug: per-chip mosaic BEFORE global smoothing; this is actually prob-mosaic)
  *_sam_prob.tif  (smoothed SAM probability mosaic, float32 in [0,1])
  *_sam_index.tif (fused index, float32 in [0,1])
  *_chm.tif
  *_cost.tif

And writes a proposal-records JSON file with file paths + spatial metadata.

Note: this script keeps your global RGB scaling (via --scale-stats) but restores the
core weighting + smoothing behavior from the original working implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from rasterio.transform import rowcol
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from samstars.data.scaling import load_scale_stats, scale_to_u8_per_channel

# ───────────── hyper-params (match original) ────────────────────────────────
CONTEXT_FCTR = 1.5
BOX_METRE = 8.0

FG_EXTRA = 80
NEG_MAX_GND = 50
NEG_MAX_CAN = 0

CHM_GROUND = 3.0
SIGMA_METRE = 0.5
CHM_BOOST = 3.0

CORE_SIGMA = BOX_METRE / 2.0
CORE_GAMMA = 1.2

LOGIT_SCALE = 1.4

# Reduced smoothing for sharper crowns in dense canopy.
CROP_SMOOTH_SIGMA = 2
SMOOTH_SIGMA = 4.0

POST_SMOOTH_SIGMA = 0

PCTL_LOW_HIGH = (5, 95)
W_CHM, W_COST = 0.75, 0.25
GROUND_THRESH, SIGMA_GROUND, W_GROUND = 3.0, 2.0, 1.0

NODATA_VAL = 0.0
# ───────────────────────────────────────────────────────────────────────────


def stretch_u8_percentile(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Robust percentile stretch to uint8 (single channel)."""
    arr = np.asarray(arr, np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], (p_lo, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - lo) * 255.0 / (hi - lo + 1e-9)
    return np.clip(out, 0, 255).astype(np.uint8)


def box_from_window(win: Window, tfm) -> box:
    # tfm is already the window transform; don't apply windows.transform twice.
    """Box from window."""
    return box(*rasterio.transform.array_bounds(int(win.height), int(win.width), tfm))


def smooth_mosaic(prob_mosaic: np.ndarray, sigma_px: float) -> np.ndarray:
    """Original: blur prob mosaic and divide by blurred valid-mask (handles sparse coverage)."""
    prob_blur = gaussian_filter(prob_mosaic, sigma_px, mode="nearest")
    val_mask = (prob_mosaic > 0).astype(np.float32)
    val_blur = gaussian_filter(val_mask, sigma_px, mode="nearest")
    prob_s = np.where(val_blur == 0, np.nan, prob_blur / val_blur)
    return np.clip(prob_s, 0, 1)


def prob_from_logit_seedcenter(logit: np.ndarray, seed_rc: tuple[int, int], res: float) -> np.ndarray:
    """Prob from logit seedcenter."""
    logit = np.asarray(logit, np.float32)
    prob = 1.0 / (1.0 + np.exp(-np.clip(LOGIT_SCALE * logit, -12, 12)))
    rr, cc = seed_rc
    yy, xx = np.indices(prob.shape)
    d_m = np.hypot(yy - rr, xx - cc) * res
    core_w = np.exp(-(d_m ** 2) / (2.0 * (CORE_SIGMA ** 2))) ** CORE_GAMMA
    prob *= core_w / (core_w.max() + 1e-6)
    return prob.astype(np.float32)


def _norm01(arr: np.ndarray) -> np.ndarray:
    """Internal helper function."""
    lo, hi = np.nanpercentile(arr, PCTL_LOW_HIGH)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def _check_sam_outputs(prob: np.ndarray, name: str) -> None:
    """Internal helper function."""
    if prob.dtype != np.float32:
        raise ValueError(f"{name}: sam_prob dtype {prob.dtype} (expected float32)")
    if not np.isfinite(prob).all():
        raise ValueError(f"{name}: sam_prob has non-finite values")
    pmin = float(prob.min())
    pmax = float(prob.max())
    if pmin < -1e-3 or pmax > 1.0 + 1e-3:
        raise ValueError(f"{name}: sam_prob out of range [{pmin:.4f}, {pmax:.4f}]")


def process_chip(
    rec: dict,
    predictor: SamPredictor,
    device: str,
    rgb_lo_hi=None,
    fast: bool = False,
    fast_batch: int = 32,
    crop_smooth_sigma: float = CROP_SMOOTH_SIGMA,
    smooth_sigma: float = SMOOTH_SIGMA,
    post_smooth_sigma: float = POST_SMOOTH_SIGMA,
):
    """Process chip."""
    img_path = Path(rec["img_path"])
    name = f"{rec['tile']}_{rec['chip_id']}"
    chm_path = img_path.parent / "chm.tif"
    cost_path = img_path.parent / "cost.tif"
    seeds_path = img_path.parent / "seeds.gpkg"

    x0, y0, x1, y1 = rec["window"]
    win = Window(x0, y0, x1 - x0, y1 - y0)

    # --- Read imagery + aux rasters for this chip window ---
    with rasterio.open(img_path) as src:
        rgb_bands = (7, 5, 3) if src.count >= 7 else (1, 2, 3)[: src.count]
        rgb = src.read(rgb_bands, window=win, boundless=True, fill_value=0).astype(np.float32)
        tfm = rasterio.windows.transform(win, src.transform)
        crs = src.crs
        res = abs(tfm.a)
        H, W = int(win.height), int(win.width)

    with rasterio.open(chm_path) as csrc:
        CHM = csrc.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
    with rasterio.open(cost_path) as cosrc:
        COST = cosrc.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)

    seeds = gpd.read_file(seeds_path).to_crs(crs)
    bounds_poly = box_from_window(win, tfm)
    seeds = seeds[seeds.intersects(bounds_poly)]
    if seeds.empty:
        return None
    print(f"seeds in chip: {len(seeds)}")

    # CHM blur for boost (within this chip window)
    sigma_px = SIGMA_METRE / max(res, 1e-9)
    chm_blur = gaussian_filter(CHM, sigma_px, mode="nearest")

    # Convert RGB to uint8 for SAM
    rgb_hwc = rgb.transpose(1, 2, 0)  # HWC float32
    if rgb_lo_hi is not None and len(rgb_lo_hi) >= rgb_hwc.shape[2]:
        RGB_u8 = scale_to_u8_per_channel(rgb_hwc, rgb_lo_hi[: rgb_hwc.shape[2]])
    else:
        RGB_u8 = np.stack([stretch_u8_percentile(rgb_hwc[..., c]) for c in range(rgb_hwc.shape[2])], axis=-1)
    if np.random.rand() < 0.05:
        sat0 = float((RGB_u8 == 0).mean())
        sat255 = float((RGB_u8 == 255).mean())
        print(f"{name}: RGB_u8 saturation 0={sat0:.3f} 255={sat255:.3f}")

    prob_mosaic = np.zeros((H, W), np.float32)

    # Group seeds by crop window (original caching strategy, per chip)
    hp_px = int(round(CONTEXT_FCTR * BOX_METRE / max(res, 1e-9)))
    hb = BOX_METRE / (2 * max(res, 1e-9))  # half box width in pixels

    if fast:
        # Fast mode: set image once per chip, batch prompts across seeds.
        predictor.set_image(RGB_u8)

        seed_infos = []
        for _, row in seeds.iterrows():
            r_arr, c_arr = rowcol(tfm, [row.geometry.x], [row.geometry.y])
            r, c = int(r_arr[0]), int(c_arr[0])
            r0, r1 = max(r - hp_px, 0), min(r + hp_px, H)
            c0, c1 = max(c - hp_px, 0), min(c + hp_px, W)
            if r1 <= r0 or c1 <= c0:
                continue

            chm_patch = CHM[r0:r1, c0:c1]
            rp, cp = (r - r0), (c - c0)

            # Foreground extra points: top CHM pixels inside the box
            mask_fg = chm_patch > CHM_GROUND
            yy, xx = np.where(mask_fg)
            in_box = ((np.abs(xx - cp) <= hb) & (np.abs(yy - rp) <= hb))
            yy, xx = yy[in_box], xx[in_box]
            if yy.size:
                keep = min(FG_EXTRA, yy.size)
                top_idx = np.argpartition(chm_patch[yy, xx], -keep)[-keep:]
                fg_pts = np.column_stack([xx[top_idx], yy[top_idx]])
                fg_pts = fg_pts + np.array([c0, r0], dtype=np.float32)
            else:
                fg_pts = np.empty((0, 2), np.float32)

            pts = np.vstack(([[c, r]], fg_pts)).astype(np.float32)
            lbl = np.hstack((np.array([1], np.int64), np.ones(len(fg_pts), np.int64)))

            # Negatives: ground
            yg, xg = np.where(chm_patch < CHM_GROUND)
            if len(yg) > NEG_MAX_GND:
                sel = np.random.choice(len(yg), NEG_MAX_GND, replace=False)
                yg, xg = yg[sel], xg[sel]
            neg_g = np.column_stack([xg, yg]).astype(np.float32)
            if neg_g.size:
                neg_g += np.array([c0, r0], dtype=np.float32)

            # Negatives: canopy outside the box
            yc, xc = np.where(chm_patch > CHM_GROUND)
            mask_out = ((np.abs(xc - cp) > hb) | (np.abs(yc - rp) > hb))
            yc, xc = yc[mask_out], xc[mask_out]
            if len(yc) > NEG_MAX_CAN:
                sel = np.random.choice(len(yc), NEG_MAX_CAN, replace=False)
                yc, xc = yc[sel], xc[sel]
            neg_c = np.column_stack([xc, yc]).astype(np.float32)
            if neg_c.size:
                neg_c += np.array([c0, r0], dtype=np.float32)

            if neg_g.size or neg_c.size:
                pts = np.vstack([pts, neg_g, neg_c])
                lbl = np.hstack([lbl, np.zeros(len(neg_g) + len(neg_c), np.int64)])

            seed_infos.append(
                {
                    "r": r,
                    "c": c,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "pts": pts,
                    "lbl": lbl,
                }
            )

        for i in range(0, len(seed_infos), max(1, fast_batch)):
            chunk = seed_infos[i : i + max(1, fast_batch)]
            if not chunk:
                continue

            pts_list = [c["pts"] for c in chunk]
            lbl_list = [c["lbl"] for c in chunk]
            boxes = [[c["c"] - hb, c["r"] - hb, c["c"] + hb, c["r"] + hb] for c in chunk]

            max_n = max(p.shape[0] for p in pts_list)
            pad_coord = np.array([[-10.0, -10.0]], np.float32)
            for j in range(len(pts_list)):
                need = max_n - pts_list[j].shape[0]
                if need:
                    pts_list[j] = np.vstack([pts_list[j], np.repeat(pad_coord, need, axis=0)])
                    lbl_list[j] = np.hstack([lbl_list[j], np.full(need, -1, np.int64)])

            pts_arr = torch.as_tensor(np.stack(pts_list), device=device)
            lbl_arr = torch.as_tensor(np.stack(lbl_list), device=device)
            boxes_t = torch.as_tensor(np.asarray(boxes, np.float32), device=device)

            orig_hw = (H, W)
            pts_in = predictor.transform.apply_coords_torch(pts_arr, orig_hw)
            boxes_in = predictor.transform.apply_boxes_torch(boxes_t, orig_hw)

            with torch.inference_mode():
                _, _, low_res_logits = predictor.predict_torch(
                    point_coords=pts_in,
                    point_labels=lbl_arr,
                    boxes=boxes_in,
                    multimask_output=False,
                    return_logits=True,
                )

            for info, low_res in zip(chunk, low_res_logits):
                r0, r1, c0, c1 = info["r0"], info["r1"], info["c0"], info["c1"]
                h_p, w_p = r1 - r0, c1 - c0

                if low_res.dim() == 2:
                    low_res = low_res.unsqueeze(0).unsqueeze(0)
                elif low_res.dim() == 3:
                    low_res = low_res.unsqueeze(0)
                logit_full = F.interpolate(low_res, (H, W), mode="bilinear", align_corners=False)[0, 0]
                logit = logit_full[r0:r1, c0:c1].detach().cpu().numpy().astype(np.float32)

                if crop_smooth_sigma > 0:
                    logit = gaussian_filter(logit, crop_smooth_sigma, mode="nearest")

                chm_patch = CHM[r0:r1, c0:c1]
                chm_blur_p = chm_blur[r0:r1, c0:c1]
                rr, cc = (info["r"] - r0), (info["c"] - c0)
                seed_h = chm_patch[rr, cc]
                dh = np.clip(chm_blur_p - seed_h, 0, None)
                dh_max = float(np.nanmax(dh)) if np.isfinite(dh).any() else 0.0
                if dh_max > 0:
                    logit = logit + (CHM_BOOST * dh / (dh_max + 1e-6)).astype(np.float32)

                seed_rc = (rr, cc)
                prob = prob_from_logit_seedcenter(logit, seed_rc=seed_rc, res=res)
                if np.random.rand() < 0.01:
                    print("seed prob pcts:", np.percentile(prob, [0, 50, 90, 99, 100]))

                prob_mosaic[r0:r1, c0:c1] = np.maximum(prob_mosaic[r0:r1, c0:c1], prob.astype(np.float32))
    else:
        buckets: dict[tuple[int, int, int, int], list[tuple[int, int]]] = {}
        for _, row in seeds.iterrows():
            r_arr, c_arr = rowcol(tfm, [row.geometry.x], [row.geometry.y])
            r, c = int(r_arr[0]), int(c_arr[0])
            r0, r1 = max(r - hp_px, 0), min(r + hp_px, H)
            c0, c1 = max(c - hp_px, 0), min(c + hp_px, W)
            buckets.setdefault((r0, r1, c0, c1), []).append((r, c))

        for (r0, r1, c0, c1), pts_seed in buckets.items():
            h_p, w_p = r1 - r0, c1 - c0
            if h_p <= 0 or w_p <= 0:
                continue

            # Set SAM image for this crop
            predictor.set_image(RGB_u8[r0:r1, c0:c1])

            chm_patch = CHM[r0:r1, c0:c1]
            chm_blur_p = chm_blur[r0:r1, c0:c1]

            # Build prompts for each seed in this crop (batched)
            boxes, pts_list, lbl_list = [], [], []
            hb = BOX_METRE / (2 * max(res, 1e-9))  # half box width in pixels (crop coordinates)

            for (r, c) in pts_seed:
                rp, cp = (r - r0), (c - c0)
                boxes.append([cp - hb, rp - hb, cp + hb, rp + hb])

                # Foreground extra points: top CHM pixels inside the box
                mask_fg = chm_patch > CHM_GROUND
                yy, xx = np.where(mask_fg)
                in_box = ((np.abs(xx - (c - c0)) <= hb) & (np.abs(yy - (r - r0)) <= hb))
                yy, xx = yy[in_box], xx[in_box]
                if yy.size:
                    keep = min(FG_EXTRA, yy.size)
                    top_idx = np.argpartition(chm_patch[yy, xx], -keep)[-keep:]
                    fg_pts = np.column_stack([xx[top_idx], yy[top_idx]])
                else:
                    fg_pts = np.empty((0, 2), np.float32)

                pts = np.vstack(([[cp, rp]], fg_pts)).astype(np.float32)
                lbl = np.hstack((np.array([1], np.int64), np.ones(len(fg_pts), np.int64)))

                # Negatives: ground
                yg, xg = np.where(chm_patch < CHM_GROUND)
                if len(yg) > NEG_MAX_GND:
                    sel = np.random.choice(len(yg), NEG_MAX_GND, replace=False)
                    yg, xg = yg[sel], xg[sel]
                neg_g = np.column_stack([xg, yg]).astype(np.float32)

                # Negatives: canopy outside the box
                yc, xc = np.where(chm_patch > CHM_GROUND)
                mask_out = ((np.abs(xc - (c - c0)) > hb) | (np.abs(yc - (r - r0)) > hb))
                yc, xc = yc[mask_out], xc[mask_out]
                if len(yc) > NEG_MAX_CAN:
                    sel = np.random.choice(len(yc), NEG_MAX_CAN, replace=False)
                    yc, xc = yc[sel], xc[sel]
                neg_c = np.column_stack([xc, yc]).astype(np.float32)

                if neg_g.size or neg_c.size:
                    pts = np.vstack([pts, neg_g, neg_c])
                    lbl = np.hstack([lbl, np.zeros(len(neg_g) + len(neg_c), np.int64)])

                pts_list.append(pts)
                lbl_list.append(lbl)

            # Pad variable-length prompt lists to one tensor
            max_n = max(p.shape[0] for p in pts_list)
            pad_coord = np.array([[-10.0, -10.0]], np.float32)
            for i in range(len(pts_list)):
                need = max_n - pts_list[i].shape[0]
                if need:
                    pts_list[i] = np.vstack([pts_list[i], np.repeat(pad_coord, need, axis=0)])
                    lbl_list[i] = np.hstack([lbl_list[i], np.full(need, -1, np.int64)])

            pts_arr = torch.as_tensor(np.stack(pts_list), device=device)
            lbl_arr = torch.as_tensor(np.stack(lbl_list), device=device)
            boxes_t = torch.as_tensor(np.asarray(boxes, np.float32), device=device)

            # IMPORTANT: predict_torch expects coords in the *transformed* (1024-long-side) space.
            orig_hw = (h_p, w_p)
            pts_in = predictor.transform.apply_coords_torch(pts_arr, orig_hw)
            boxes_in = predictor.transform.apply_boxes_torch(boxes_t, orig_hw)

            with torch.inference_mode():
                # returns: (masks_full, iou_preds, low_res_logits)
                _, _, low_res_logits = predictor.predict_torch(
                    point_coords=pts_in,
                    point_labels=lbl_arr,
                    boxes=boxes_in,
                    multimask_output=False,
                    return_logits=True,
                )

            for (r, c), low_res in zip(pts_seed, low_res_logits):
                # Upsample low-res logits to crop size (original behavior)
                if low_res.dim() == 2:
                    low_res = low_res.unsqueeze(0).unsqueeze(0)
                elif low_res.dim() == 3:
                    low_res = low_res.unsqueeze(0)
                logit = F.interpolate(low_res, (h_p, w_p), mode="bilinear", align_corners=False)[0, 0]
                logit = logit.detach().cpu().numpy().astype(np.float32)

                # Light blur in logit space (original used cv2.GaussianBlur)
                if crop_smooth_sigma > 0:
                    logit = gaussian_filter(logit, crop_smooth_sigma, mode="nearest")

                # CHM boost: compare blurred CHM to seed CHM
                rr, cc = (r - r0), (c - c0)
                seed_h = chm_patch[rr, cc]
                dh = np.clip(chm_blur_p - seed_h, 0, None)  # (h_p, w_p)
                dh_max = float(np.nanmax(dh)) if np.isfinite(dh).any() else 0.0
                if dh_max > 0:
                    logit = logit + (CHM_BOOST * dh / (dh_max + 1e-6)).astype(np.float32)

                seed_rc = (rr, cc)
                prob = prob_from_logit_seedcenter(logit, seed_rc=seed_rc, res=res)
                if np.random.rand() < 0.01:
                    print("seed prob pcts:", np.percentile(prob, [0, 50, 90, 99, 100]))

                # Mosaic max compositing
                prob_mosaic[r0:r1, c0:c1] = np.maximum(prob_mosaic[r0:r1, c0:c1], prob.astype(np.float32))

    # --- Global smoothing and index fusion (original) ---
    prob_mosaic_out = prob_mosaic.copy()  # debug output (pre-smooth mosaic)

    prob_s = smooth_mosaic(prob_mosaic, smooth_sigma)
    prob_s = np.nan_to_num(prob_s, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    print("prob_s pcts:", np.nanpercentile(prob_s, [0, 50, 90, 99, 100]))

    base = prob_s * (1 - W_CHM + W_CHM * _norm01(CHM)) * (1 - W_COST + W_COST * (1 - _norm01(COST)))
    z = np.clip(-(CHM - GROUND_THRESH) / SIGMA_GROUND, -60, 60)
    ground_w = (1 / (1 + np.exp(z))) ** W_GROUND

    index = base * ground_w

    # Original: compute min/max BEFORE post smoothing, then normalize AFTER smoothing.
    idx_min = float(np.nanmin(index)) if np.isfinite(index).any() else 0.0
    idx_max = float(np.nanmax(index)) if np.isfinite(index).any() else 1.0

    if post_smooth_sigma > 0:
        index = gaussian_filter(index, post_smooth_sigma, mode="nearest")

    denom = (idx_max - idx_min)
    if not np.isfinite(denom) or denom <= 0:
        index = np.zeros_like(index, dtype=np.float32)
    else:
        index = np.where(
            np.isnan(index),
            NODATA_VAL,
            ((index - idx_min) / (denom + 1e-6)).astype(np.float32),
        )

    if CHM.shape != prob_s.shape or COST.shape != prob_s.shape:
        raise ValueError("CHM/COST shapes do not match SAM prob shape")

    return prob_mosaic_out, prob_s, index, CHM, COST, crs, tfm


def run_tile_proposals(
    records_json: Path,
    out_root: Path,
    ckpt_path: Path,
    *,
    device: str = "cpu",
    rgb_lo_hi=None,
    fast: bool = False,
    fast_batch: int = 32,
    crop_smooth_sigma: float = CROP_SMOOTH_SIGMA,
    smooth_sigma: float = SMOOTH_SIGMA,
    post_smooth_sigma: float = POST_SMOOTH_SIGMA,
):
    """Run SAM proposals from one tile-record file."""
    records_path = Path(records_json)
    if not records_path.exists():
        raise FileNotFoundError(f"Missing records file: {records_path}")

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {ckpt}")

    sam = sam_model_registry["vit_l"](checkpoint=str(ckpt))
    sam.to(device).eval()
    predictor = SamPredictor(sam)

    out_dir = out_root / "rasters"
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = json.load(open(records_path))
    proposal_records = []

    for rec in recs:
        res = process_chip(
            rec,
            predictor,
            device,
            rgb_lo_hi=rgb_lo_hi,
            fast=fast,
            fast_batch=fast_batch,
            crop_smooth_sigma=crop_smooth_sigma,
            smooth_sigma=smooth_sigma,
            post_smooth_sigma=post_smooth_sigma,
        )
        if res is None:
            continue

        mosaic_prob, prob_s, index, chm_out, cost_out, crs, tfm = res
        name = f"{rec['tile']}_{rec['chip_id']}"

        _check_sam_outputs(prob_s, name)

        meta = {
            "driver": "GTiff",
            "height": index.shape[0],
            "width": index.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": tfm,
            "compress": "deflate",
        }

        with rasterio.open(out_dir / f"{name}_logit.tif", "w", **meta) as dst:
            dst.write(mosaic_prob.astype(np.float32), 1)
        with rasterio.open(out_dir / f"{name}_sam_prob.tif", "w", **meta) as dst:
            dst.write(prob_s.astype(np.float32), 1)
        with rasterio.open(out_dir / f"{name}_sam_index.tif", "w", **meta) as dst:
            dst.write(index.astype(np.float32), 1)
        with rasterio.open(out_dir / f"{name}_chm.tif", "w", **meta) as dst:
            dst.write(chm_out.astype(np.float32), 1)
        with rasterio.open(out_dir / f"{name}_cost.tif", "w", **meta) as dst:
            dst.write(cost_out.astype(np.float32), 1)

        proposal_records.append(
            {
                "file_name": str((out_dir / f"{name}_sam_prob.tif").resolve()),
                "chm": str((out_dir / f"{name}_chm.tif").resolve()),
                "cost": str((out_dir / f"{name}_cost.tif").resolve()),
                "logit": str((out_dir / f"{name}_logit.tif").resolve()),
                "index": str((out_dir / f"{name}_sam_index.tif").resolve()),
                "mask": "",
                "meta": {
                    "crs": crs.to_wkt() if crs else None,
                    "transform": list(tfm) if tfm is not None else None,
                },
            }
        )

    out_json = out_root / "proposal_records.json"
    with open(out_json, "w") as f:
        json.dump(proposal_records, f)
    print(f"✓ wrote proposal outputs ({len(proposal_records)} records)")


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Run SAM proposals on tile records and write proposal rasters.")
    pa.add_argument("--tile-records", type=Path, required=True, help="Path to a tile-record JSON file")
    pa.add_argument("--ckpt-path", type=Path, required=True, help="Direct path to a SAM decoder checkpoint")
    pa.add_argument("--out-root", type=Path, required=True)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--scale-stats", type=Path, default=None, help="Global scaling stats JSON")
    pa.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: reuse a single SAM embedding per chip and batch prompts.",
    )
    pa.add_argument("--fast-batch-size", type=int, default=32, help="Batch size for fast mode prompts")
    pa.add_argument(
        "--crop-smooth-sigma",
        type=float,
        default=CROP_SMOOTH_SIGMA,
        help="Per-crop Gaussian blur sigma applied before seed-centered probability weighting.",
    )
    pa.add_argument(
        "--smooth-sigma",
        type=float,
        default=SMOOTH_SIGMA,
        help="Global Gaussian blur sigma for the SAM probability mosaic.",
    )
    pa.add_argument(
        "--post-smooth-sigma",
        type=float,
        default=POST_SMOOTH_SIGMA,
        help="Optional post-fusion Gaussian blur sigma for the index raster.",
    )
    args = pa.parse_args()

    rgb_lo_hi = None
    if args.scale_stats:
        stats = load_scale_stats(args.scale_stats)
        per_channel = stats.get("sam_rgb_u8", {}).get("per_channel", [])
        rgb_lo_hi = [(float(c["lo"]), float(c["hi"])) for c in per_channel]

    run_tile_proposals(
        args.tile_records,
        args.out_root,
        args.ckpt_path,
        device=args.device,
        rgb_lo_hi=rgb_lo_hi,
        fast=args.fast,
        fast_batch=args.fast_batch_size,
        crop_smooth_sigma=args.crop_smooth_sigma,
        smooth_sigma=args.smooth_sigma,
        post_smooth_sigma=args.post_smooth_sigma,
    )


if __name__ == "__main__":
    main()
