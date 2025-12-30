#!/usr/bin/env python3
"""
Run SAM (fine-tuned decoder) on chip manifests using seed prompts and original smoothing logic.
For each chip window (train/val manifests), uses seeds.gpkg to build fg/bg prompts, applies CHM boost,
core weighting, Gaussian smoothing, cost weighting, and writes logit/prob GeoTIFFs per chip.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from rasterio.crs import CRS
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from rasterio.transform import rowcol
from shapely.geometry import box

# Constants from helpers/predict_sam.py
BOX_METRE = 8.0
FG_EXTRA = 80
NEG_MAX_GND = 150
NEG_MAX_CAN = 300
CHM_GROUND = 3.0
CORE_GAMMA = 1.7
LOGIT_SCALE = 1.4
SMOOTH_SIGMA = 12.0
CORE_SIGMA = BOX_METRE / 3
CROP_SMOOTH_SIGMA = 4
BLEND_W = 0.3  # weight for blurred prob vs raw in per-seed smoothing
POST_SMOOTH_SIGMA = 6
PCTL_LOW_HIGH, W_CHM, W_COST = (5, 95), 0.75, 0.25
GROUND_THRESH, SIGMA_GROUND, W_GROUND = 3.0, 2.0, 1.0
SIGMA_METRE = 0.5
CHM_BOOST = 3.0


def stretch01(arr, p_low=2, p_high=98):
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo, hi = np.percentile(arr[finite], (p_low, p_high))
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def smooth_mosaic(prob_mosaic, res):
    sigma_px = SMOOTH_SIGMA
    prob_blur = gaussian_filter(prob_mosaic, sigma_px, mode="nearest")
    val_mask = (prob_mosaic > 0).astype(np.float32)
    val_blur = gaussian_filter(val_mask, sigma_px, mode="nearest")
    prob_s = np.where(val_blur == 0, np.nan, prob_blur / val_blur).clip(0, 1)
    return prob_s


def smooth_prob(logit):
    """Apply sigmoid + core weighting, plus a light blur blended with raw."""
    prob = torch.sigmoid(torch.clamp(LOGIT_SCALE * torch.as_tensor(logit), -12, 12))
    yy, xx = torch.meshgrid(torch.arange(prob.shape[-2], device=prob.device), torch.arange(prob.shape[-1], device=prob.device), indexing="ij")
    cy, cx = prob.shape[-2] / 2, prob.shape[-1] / 2
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core_w = torch.exp(-(d ** 2) / (2 * CORE_SIGMA ** 2)) ** CORE_GAMMA
    prob = prob * (core_w / (core_w.max() + 1e-6))
    if CROP_SMOOTH_SIGMA > 0:
        prob_np = prob.cpu().numpy()
        prob_blur = gaussian_filter(prob_np, CROP_SMOOTH_SIGMA)
        prob = torch.from_numpy((1 - BLEND_W) * prob_np + BLEND_W * prob_blur).to(prob.device)
    return prob.cpu().numpy()


def scale_u8(arr, lo, hi):
    arr = np.asarray(arr, np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_scale_stats(path: Path):
    data = json.load(open(path))
    low = data.get("low")
    high = data.get("high")
    if low is None or high is None:
        raise ValueError("scale stats must include 'low' and 'high'")
    return np.asarray(low, np.float32), np.asarray(high, np.float32)


def process_chip(rec, sam: SamPredictor, device, scale_stats=None):
    img_path = Path(rec["img_path"])
    chm_path = img_path.parent / "chm.tif"
    cost_path = img_path.parent / "cost.tif"
    seeds_path = img_path.parent / "seeds.gpkg"

    x0, y0, x1, y1 = rec["window"]
    chip_size = int(rec.get("size") or (x1 - x0))
    win = Window(x0, y0, x1 - x0, y1 - y0)
    with rasterio.open(img_path) as src:
        rgb_bands = (7, 5, 3) if src.count >= 7 else (1, 2, 3)[: src.count]
        RGB16 = src.read(rgb_bands, window=win, boundless=True, fill_value=0)
        tfm = rasterio.windows.transform(win, src.transform)
        crs = src.crs
        res = abs(tfm.a)
        H, W = int(win.height), int(win.width)
    with rasterio.open(chm_path) as csrc:
        CHM = csrc.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
    with rasterio.open(cost_path) as cosrc:
        COST = cosrc.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)

    seeds = gpd.read_file(seeds_path).to_crs(crs)
    mask_bounds = box_from_window(win, tfm)
    seeds = seeds[seeds.intersects(mask_bounds)]
    if seeds.empty:
        return None

    # CHM blur for boost
    sigma_px = SIGMA_METRE / res
    chm_blur = gaussian_filter(CHM, sigma_px, mode="nearest")

    def stretch(arr):
        return np.clip((arr - arr.min()) * 255 / (np.ptp(arr) + 1e-9), 0, 255).astype("uint8")

    if scale_stats is not None:
        low, high = scale_stats
        if len(low) >= RGB16.shape[0] and len(high) >= RGB16.shape[0]:
            RGB_u8 = np.stack([scale_u8(RGB16[i], low[i], high[i]) for i in range(RGB16.shape[0])], axis=-1)
        else:
            RGB_u8 = np.stack([stretch(RGB16[i]) for i in range(RGB16.shape[0])], axis=-1)
    else:
        RGB_u8 = np.stack([stretch(RGB16[i]) for i in range(RGB16.shape[0])], axis=-1)
    sam_img = RGB_u8
    prob_mosaic = np.zeros((H, W), np.float32)

    hp_px = int(round(BOX_METRE / res * 1.5))
    buckets = {}
    for idx, row in seeds.iterrows():
        # tfm is the window transform, so row/col are already in chip coords
        r_arr, c_arr = rowcol(tfm, [row.geometry.x], [row.geometry.y])
        r, c = int(r_arr[0]), int(c_arr[0])
        r0, r1 = max(r - hp_px, 0), min(r + hp_px, H)
        c0, c1 = max(c - hp_px, 0), min(c + hp_px, W)
        buckets.setdefault((r0, r1, c0, c1), []).append((r, c))

    for (r0, r1, c0, c1), pts_seed in buckets.items():
        h_p, w_p = r1 - r0, c1 - c0
        if h_p <= 0 or w_p <= 0:
            continue
        sam.set_image(sam_img[r0:r1, c0:c1])
        chm_patch = CHM[r0:r1, c0:c1]
        chm_blur_p = chm_blur[r0:r1, c0:c1]
        cost_patch = COST[r0:r1, c0:c1]
        half_pix = BOX_METRE / (2 * res)

        boxes, pts_list, lbl_list = [], [], []
        for r, c in pts_seed:
            rp, cp = (r - r0), (c - c0)
            hb = BOX_METRE / (2 * res)
            boxes.append([cp - hb, rp - hb, cp + hb, rp + hb])

            mask_fg = chm_patch > CHM_GROUND
            yy, xx = np.where(mask_fg)
            in_box = ((abs(xx - (c - c0)) <= hb) & (abs(yy - (r - r0)) <= hb))
            yy, xx = yy[in_box], xx[in_box]
            if yy.size:
                keep = min(FG_EXTRA, yy.size)
                top_idx = np.argpartition(chm_patch[yy, xx], -keep)[-keep:]
                fg_pts = np.column_stack([xx[top_idx], yy[top_idx]])
            else:
                fg_pts = np.empty((0, 2), np.float32)

            pts = np.vstack(([ [cp, rp] ], fg_pts))
            lbl = np.hstack(([1], np.ones(len(fg_pts), np.int64)))

            yg, xg = np.where(chm_patch < CHM_GROUND)
            if len(yg) > NEG_MAX_GND:
                sel = np.random.choice(len(yg), NEG_MAX_GND, replace=False)
                yg, xg = yg[sel], xg[sel]
            neg_g = np.column_stack([xg, yg])

            yc, xc = np.where(chm_patch > CHM_GROUND)
            mask_out = ((abs(xc - (c - c0)) > hb) | (abs(yc - (r - r0)) > hb))
            yc, xc = yc[mask_out], xc[mask_out]
            if len(yc) > NEG_MAX_CAN:
                sel = np.random.choice(len(yc), NEG_MAX_CAN, replace=False)
                yc, xc = yc[sel], xc[sel]
            neg_c = np.column_stack([xc, yc])

            if neg_g.size or neg_c.size:
                pts = np.vstack([pts, neg_g, neg_c])
                lbl = np.hstack([lbl, np.zeros(len(neg_g) + len(neg_c), np.int64)])

            pts_list.append(pts.astype(np.float32))
            lbl_list.append(lbl.astype(np.int64))

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

        orig_hw = (h_p, w_p)

        pts_arr_in = sam.transform.apply_coords_torch(pts_arr, orig_hw)
        boxes_in = sam.transform.apply_boxes_torch(boxes_t, orig_hw)

        with torch.inference_mode():
            logits_full, _, low_res_logits = sam.predict_torch(
                point_coords=pts_arr_in,
                point_labels=lbl_arr,
                boxes=boxes_in,
                multimask_output=False,
                return_logits=True,
            )

        for (r, c), logit_full in zip(pts_seed, logits_full):
            logit = logit_full[0].cpu().numpy()
            if CROP_SMOOTH_SIGMA > 0:
                logit = gaussian_filter(logit, CROP_SMOOTH_SIGMA)
            dh = np.clip(chm_blur_p - chm_patch[(r - r0), (c - c0)], 0, None)
            logit += CHM_BOOST * dh / (dh.max() + 1e-6) if np.isfinite(dh).any() else 0
            prob = smooth_prob(logit)
            r_abs, c_abs = r0, c0
            prob_mosaic[r_abs:r_abs + h_p, c_abs:c_abs + w_p] = np.maximum(prob_mosaic[r_abs:r_abs + h_p, c_abs:c_abs + w_p], prob)

    logit_out = prob_mosaic.copy()
    prob_s = smooth_mosaic(prob_mosaic, res)
    def _norm(arr):
        lo, hi = np.nanpercentile(arr, PCTL_LOW_HIGH)
        return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

    base = prob_s * (1 - W_CHM + W_CHM * _norm(CHM)) * (1 - W_COST + W_COST * (1 - _norm(COST)))
    ground_w = (1 / (1 + np.exp(-(CHM - GROUND_THRESH) / SIGMA_GROUND))) ** W_GROUND
    index = base * ground_w
    if POST_SMOOTH_SIGMA > 0:
        index = gaussian_filter(index, POST_SMOOTH_SIGMA, mode="nearest")
    idx_min, idx_max = np.nanmin(index), np.nanmax(index)
    index = np.where(np.isnan(index), 0.0, ((index - idx_min) / (idx_max - idx_min + 1e-6)).astype("float32"))
    # Pad outputs to full chip size to match masks.
    if chip_size and (H < chip_size or W < chip_size):
        pad_h = max(0, chip_size - H)
        pad_w = max(0, chip_size - W)
        logit_out = np.pad(logit_out, ((0, pad_h), (0, pad_w)), constant_values=0)
        index = np.pad(index, ((0, pad_h), (0, pad_w)), constant_values=0)
        prob_s = np.pad(prob_s, ((0, pad_h), (0, pad_w)), constant_values=0)
        H, W = logit_out.shape
    meta = {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "float32", "crs": crs, "transform": tfm, "compress": "deflate"}
    return logit_out, prob_s, index, crs, tfm


def box_from_window(win, tfm):
    # tfm is already the window transform; don't apply windows.transform twice.
    return box(*rasterio.transform.array_bounds(int(win.height), int(win.width), tfm))


def run_manifests(manifests_root: Path, ckpt_root: Path, out_root: Path, device="cpu", sizes=None, splits=None, scale_stats=None):
    if sizes:
        size_dirs = [manifests_root / s for s in sorted(sizes)]
    else:
        size_dirs = sorted(p for p in manifests_root.iterdir() if p.is_dir())
    split_files = splits or ["train_manifest.json", "val_manifest.json"]
    for size_dir in size_dirs:
        ckpt = ckpt_root / size_dir.name / "sam" / "sam_crowns_decoder.pth"
        if not ckpt.exists():
            continue
        for split in split_files:
            manifest = size_dir / split
            if not manifest.exists():
                continue
            out_dir = out_root / size_dir.name / ("train_probs" if "train" in split else "val_probs")
            out_dir.mkdir(parents=True, exist_ok=True)
            sam = sam_model_registry["vit_l"](checkpoint=str(ckpt))
            sam.to(device).eval()
            predictor = SamPredictor(sam)
            recs = json.load(open(manifest))
            manifest_out = []
            for rec in recs:
                res = process_chip(rec, predictor, device, scale_stats=scale_stats)
                if res is None:
                    continue
                logit_out, prob_s, index, crs, tfm = res
                name = f"{rec['tile']}_{rec['chip_id']}"
                meta = {"driver": "GTiff", "height": index.shape[0], "width": index.shape[1], "count": 1, "dtype": "float32", "crs": crs, "transform": tfm, "compress": "deflate"}
                with rasterio.open(out_dir / f"{name}_logit.tif", "w", **meta) as dst:
                    dst.write(logit_out.astype(np.float32), 1)
                with rasterio.open(out_dir / f"{name}_prob.tif", "w", **meta) as dst:
                    dst.write(index.astype(np.float32), 1)
                manifest_out.append({"file_name": str(out_dir / f"{name}_prob.tif"), "logit": str(out_dir / f"{name}_logit.tif"), "mask": ""})
            with open(out_root / size_dir.name / ("train_probs.json" if "train" in split else "val_probs.json"), "w") as f:
                json.dump(manifest_out, f)
            print(f"✓ wrote probs for {size_dir.name}/{split} ({len(manifest_out)} records)")


def main():
    pa = argparse.ArgumentParser(description="Run SAM on manifests with seeds and write smoothed logits/probs.")
    pa.add_argument("--manifests-root", type=Path, required=True)
    pa.add_argument("--ckpt-root", type=Path, required=True)
    pa.add_argument("--out-root", type=Path, required=True)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes to process")
    pa.add_argument("--scale-stats", type=Path, default=None, help="Global scaling stats JSON")
    pa.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated split list (train,val). Default runs both.",
    )
    args = pa.parse_args()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()] if args.sizes else None
    splits = None
    if args.splits:
        splits = []
        for s in args.splits.split(","):
            s = s.strip().lower()
            if not s:
                continue
            splits.append("train_manifest.json" if s.startswith("train") else "val_manifest.json")
    scale_stats = load_scale_stats(args.scale_stats) if args.scale_stats else None
    run_manifests(args.manifests_root, args.ckpt_root, args.out_root, device=args.device, sizes=sizes, splits=splits, scale_stats=scale_stats)


if __name__ == "__main__":
    main()
