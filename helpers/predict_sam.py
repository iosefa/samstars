#!/usr/bin/env python3
"""
make_canopy_index.py – faster fp32 version
──────────────────────────────────────────────────────────────────────────────
* cache one 1024-px SAM embedding **per crop**, not per seed
* batch all seeds that share the crop – single decoder call
* GPU separable Gaussian blur (fallback to SciPy on CPU)
"""
from __future__ import annotations
import argparse, numpy as np, cv2, torch, rasterio, geopandas as gpd
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

# ───────────── hyper-params ────────────────────────────────────────────────
CONTEXT_FCTR   = 1.5
BOX_METRE      = 8.0
FG_EXTRA       = 80
NEG_MAX_GND    = 150
NEG_MAX_CAN    = 300
CHM_GROUND     = 3.0
CORE_GAMMA     = 1.7
LOGIT_SCALE    = 1.4
SIGMA_METRE    = 0.5
CHM_BOOST      = 3.0
CORE_SIGMA, SMOOTH_SIGMA = BOX_METRE / 3, 36
CROP_SMOOTH_SIGMA = 8
POST_SMOOTH_SIGMA = 6
PCTL_LOW_HIGH, W_CHM, W_COST = (5, 95), 0.75, 0.25
GROUND_THRESH, SIGMA_GROUND, W_GROUND = 3.0, 2.0, 1.0
NODATA_VAL = 0.0
# ───────────────────────────────────────────────────────────────────────────


def stretch(arr, p2=2, p98=98):
    lo, hi = np.percentile(arr, (p2, p98))
    return np.clip((arr - lo) * 255 / (hi - lo + 1e-9), 0, 255).astype("uint8")


def resize_long(img: np.ndarray, long=1024):
    h, w = img.shape[:2]
    s = long / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_LINEAR), s


def gauss_blur_gpu(arr: np.ndarray, sigma_px: float, device) -> np.ndarray:
    if sigma_px < 0.01:
        return arr

    k  = int(6 * sigma_px + 1) | 1
    x  = torch.arange(k, device=device) - k // 2
    kern = torch.exp(-(x**2) / (2 * sigma_px**2))
    kern /= kern.sum()

    t = torch.as_tensor(arr.astype("float32"), device=device)
    kern = kern.to(t.dtype)[None, None, :]     # (1,1,k)

    t = t.unsqueeze(0).unsqueeze(0)
    t = F.pad(t, (k//2, k//2, 0, 0), mode='replicate')
    t = F.conv2d(t, kern[:, :, None, :])
    t = F.pad(t, (0, 0, k//2, k//2), mode='replicate')
    t = F.conv2d(t, kern[:, :, :, None])
    return t.squeeze().cpu().numpy()


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--img",   default="img.tif")
    pa.add_argument("--chm",   default="chm.tif")
    pa.add_argument("--cost",  default="cost.tif")
    pa.add_argument("--seeds", default="seeds.gpkg")
    pa.add_argument("--sam-meta", default="sam_vit_l_0b3195.pth")
    pa.add_argument("--sam-dec",  default="sam_crowns_decoder.pth")
    # Legacy: --out is treated as sam_index output path.
    pa.add_argument("--out",   default="canopy_index.tif")
    pa.add_argument("--out-index", default=None,
                    help="Output path for fused SAM index raster (default: --out).")
    pa.add_argument("--out-prob", default=None,
                    help="Optional output path for smoothed SAM probability raster.")
    pa.add_argument("--out-logit", default=None,
                    help="Optional output path for pre-smoothing probability mosaic (legacy logit debug).")
    args = pa.parse_args()

    with rasterio.open(args.img) as img_ds, \
         rasterio.open(args.chm) as chm_ds, \
         rasterio.open(args.cost) as cost_ds:

        tfm, crs = img_ds.transform, img_ds.crs
        res      = abs(tfm.a)
        H, W     = img_ds.height, img_ds.width
        rgb_bands = (7, 5, 3) if img_ds.count >= 7 else (1, 2, 3)
        RGB16 = np.stack([img_ds.read(b) for b in rgb_bands])
        CHM   = chm_ds.read(1).astype("float32")
        COST  = cost_ds.read(1).astype("float32")

    seeds = gpd.read_file(args.seeds).to_crs(crs).set_index("id")
    if seeds.empty:
        raise RuntimeError("seeds.gpkg is empty")

    sigma_px = SIGMA_METRE / res
    CHM_BLUR = gaussian_filter1d(CHM, sigma_px, axis=0, mode="nearest")
    CHM_BLUR = gaussian_filter1d(CHM_BLUR, sigma_px, axis=1, mode="nearest")

    torch.set_float32_matmul_precision("high")
    device = "mps" if torch.backends.mps.is_available() else "cuda:0"
    sam = sam_model_registry["vit_l"](checkpoint=args.sam_meta).to(device).eval()
    sam.load_state_dict(torch.load(args.sam_dec, map_location=device), strict=False)
    sam.mask_decoder.num_multimask_outputs = 1
    sam.mask_decoder.mask_tokens = torch.nn.Embedding(
        1, sam.mask_decoder.mask_tokens.embedding_dim,
        device=device)
    predictor = SamPredictor(sam)

    hp_px = int(round(CONTEXT_FCTR * BOX_METRE / res))
    buckets: dict[tuple[int,int,int,int], list[tuple[int,int,int]]] = {}
    for cid, row in seeds.iterrows():
        r_arr, c_arr = rowcol(tfm, [row.geometry.x], [row.geometry.y])
        r, c = int(r_arr[0]), int(c_arr[0])
        r0, r1 = max(r - hp_px, 0), min(r + hp_px, H)
        c0, c1 = max(c - hp_px, 0), min(c + hp_px, W)
        buckets.setdefault((r0, r1, c0, c1), []).append((cid, r, c))

    rgb_cache: dict[tuple[int,int,int,int], tuple[np.ndarray,float]] = {}
    prob_mosaic = np.zeros((H, W), np.float32)

    for (r0, r1, c0, c1), lst in tqdm(buckets.items(), desc="SAM", unit="crop"):
        if (r0, r1, c0, c1) not in rgb_cache:
            rgb_patch = np.moveaxis(RGB16[:, r0:r1, c0:c1], 0, -1)
            rgb_u8 = np.stack([stretch(rgb_patch[..., i]) for i in range(3)], -1)
            rgb_1024, s = resize_long(rgb_u8)
            rgb_cache[(r0, r1, c0, c1)] = (rgb_1024, s)
        else:
            rgb_1024, s = rgb_cache[(r0, r1, c0, c1)]

        predictor.set_image(rgb_1024)
        h_p, w_p = r1 - r0, c1 - c0
        chm_patch  = CHM[r0:r1, c0:c1]
        chm_blur_p = CHM_BLUR[r0:r1, c0:c1]
        half_pix   = BOX_METRE / (2 * res)

        boxes, pts_list, lbl_list = [], [], []

        for cid, r, c in lst:
            rp, cp = (r - r0) * s, (c - c0) * s
            hb_s = BOX_METRE / (2 * res) * s
            boxes.append([cp - hb_s, rp - hb_s, cp + hb_s, rp + hb_s])

            mask_fg = chm_patch > CHM_GROUND
            yy, xx = np.where(mask_fg)
            in_box = ((abs(xx - (c - c0)) <= half_pix) &
                      (abs(yy - (r - r0)) <= half_pix))
            yy, xx = yy[in_box], xx[in_box]
            if yy.size:
                keep = min(FG_EXTRA, yy.size)
                top_idx = np.argpartition(chm_patch[yy, xx], -keep)[-keep:]
                fg_pts = np.column_stack([xx[top_idx] * s, yy[top_idx] * s])
            else:
                fg_pts = np.empty((0, 2), np.float32)

            pts = np.vstack(([ [cp, rp] ], fg_pts))
            lbl = np.hstack(([1], np.ones(len(fg_pts), np.int32)))

            yg, xg = np.where(chm_patch < CHM_GROUND)
            if len(yg) > NEG_MAX_GND:
                sel = np.random.choice(len(yg), NEG_MAX_GND, replace=False)
                yg, xg = yg[sel], xg[sel]
            neg_g = np.column_stack([xg * s, yg * s])

            yc, xc = np.where(chm_patch > CHM_GROUND)
            mask_out = ((abs(xc - (c - c0)) > half_pix) |
                        (abs(yc - (r - r0)) > half_pix))
            yc, xc = yc[mask_out], xc[mask_out]
            if len(yc) > NEG_MAX_CAN:
                sel = np.random.choice(len(yc), NEG_MAX_CAN, replace=False)
                yc, xc = yc[sel], xc[sel]
            neg_c = np.column_stack([xc * s, yc * s])

            if neg_g.size or neg_c.size:
                pts = np.vstack([pts, neg_g, neg_c])
                lbl = np.hstack([lbl, np.zeros(len(neg_g) + len(neg_c), np.int32)])

            pts_list.append(pts.astype(np.float32))
            lbl_list.append(lbl.astype(np.int64))

        max_n = max(p.shape[0] for p in pts_list)
        pad_coord = np.array([[-10.0, -10.0]], np.float32)
        for i in range(len(pts_list)):
            need = max_n - pts_list[i].shape[0]
            if need:
                pts_list[i] = np.vstack([pts_list[i],
                                         np.repeat(pad_coord, need, axis=0)])
                lbl_list[i] = np.hstack([lbl_list[i],
                                          np.full(need, -1, np.int64)])

        pts_arr = torch.as_tensor(np.stack(pts_list), device=device)   # fp32
        lbl_arr = torch.as_tensor(np.stack(lbl_list), device=device)
        boxes_t = torch.as_tensor(np.asarray(boxes, np.float32), device=device)

        with torch.inference_mode():
            _, _, logits_rs = predictor.predict_torch(
                point_coords     = pts_arr,
                point_labels     = lbl_arr,
                boxes            = boxes_t,
                multimask_output = False,
                return_logits    = True,
            )

        for (cid, r, c), logit_rs in zip(lst, logits_rs):
            logit = torch.nn.functional.interpolate(
                logit_rs.unsqueeze(0), (h_p, w_p),
                mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()

            if CROP_SMOOTH_SIGMA > 0:
                logit = cv2.GaussianBlur(
                    logit, (0, 0), CROP_SMOOTH_SIGMA,
                    borderType = cv2.BORDER_REPLICATE
                )

            dh = np.clip(chm_blur_p - chm_patch[r - r0, c - c0], 0, None)
            logit += CHM_BOOST * dh / (dh.max() + 1e-6)

            prob = 1 / (1 + np.exp(-np.clip(LOGIT_SCALE * logit, -12, 12)))
            yy, xx = np.indices(prob.shape)
            d_m = np.hypot(yy - (r - r0), xx - (c - c0)) * res
            core_w = np.exp(-(d_m ** 2) / (2 * CORE_SIGMA ** 2))
            core_w  = core_w ** CORE_GAMMA
            prob *= core_w / core_w.max()
            prob_mosaic[r0:r1, c0:c1] = np.maximum(prob_mosaic[r0:r1, c0:c1], prob)

    if device != "cpu":
        prob_blur = gauss_blur_gpu(prob_mosaic, SMOOTH_SIGMA, device)
        val_mask  = (prob_mosaic > 0).astype(np.float32)
        val_blur  = gauss_blur_gpu(val_mask,  SMOOTH_SIGMA, device)
    else:
        prob_blur = gaussian_filter1d(prob_mosaic, SMOOTH_SIGMA, axis=0, mode="nearest")
        prob_blur = gaussian_filter1d(prob_blur,   SMOOTH_SIGMA, axis=1, mode="nearest")
        val_mask  = (prob_mosaic > 0).astype(np.float32)
        val_blur  = gaussian_filter1d(val_mask, SMOOTH_SIGMA, axis=0, mode="nearest")
        val_blur  = gaussian_filter1d(val_blur, SMOOTH_SIGMA, axis=1, mode="nearest")

    prob_s = np.where(val_blur == 0, np.nan, prob_blur / val_blur).clip(0, 1)

    def _norm(arr):
        lo, hi = np.nanpercentile(arr, PCTL_LOW_HIGH)
        return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

    base = prob_s * (1 - W_CHM + W_CHM * _norm(CHM)) \
                  * (1 - W_COST + W_COST * (1 - _norm(COST)))
    ground_w = (1 / (1 + np.exp(-(CHM - GROUND_THRESH) / SIGMA_GROUND))) ** W_GROUND
    index = base * ground_w
    idx_min, idx_max = np.nanmin(index), np.nanmax(index)

    if POST_SMOOTH_SIGMA > 0:
        if device != "cpu":
                index = gauss_blur_gpu(index, POST_SMOOTH_SIGMA, device)
        else:
            index = gaussian_filter1d(index, POST_SMOOTH_SIGMA, axis=0, mode = "nearest")
            index = gaussian_filter1d(index, POST_SMOOTH_SIGMA, axis=1, mode = "nearest")

    index = np.where(np.isnan(index), NODATA_VAL,
                     ((index - idx_min) / (idx_max - idx_min + 1e-6)).astype("float32"))

    out_index = args.out_index if args.out_index else args.out

    meta = {"driver": "GTiff", "height": H, "width": W, "count": 1,
            "dtype": "float32", "crs": crs, "transform": tfm,
            "compress": "deflate", "predictor": 2, "zlevel": 6,
            "nodata": NODATA_VAL}

    with rasterio.open(out_index, "w", **meta) as dst:
        dst.write(index, 1)

    if args.out_prob:
        with rasterio.open(args.out_prob, "w", **meta) as dst:
            dst.write(prob_s.astype("float32"), 1)

    if args.out_logit:
        with rasterio.open(args.out_logit, "w", **meta) as dst:
            dst.write(prob_mosaic.astype("float32"), 1)


if __name__ == "__main__":
    main()
