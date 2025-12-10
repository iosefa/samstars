#!/usr/bin/env python3
"""
Train logit-aware StarDist head on chips with SAM conditioning + CHM/COST weighting.
Supports train/val splits (via holdout or provided val records) and writes IoU/Dice + COCO segm AP metrics.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
import shapely.geometry as geom
import geopandas as gpd
import rasterio
from rasterio.crs import CRS

# Ensure repo root is on path so `v2.*` imports work when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from v2.logit_aware_stardist.model import LogitAwareStarDist
from v2.logit_aware_stardist.targets import compute_ray_targets
from v2.logit_aware_stardist.postproc import decode_rays


class ChipDataset(Dataset):
    def __init__(self, chip_dir: Path, records_json: Path):
        self.chip_dir = chip_dir
        self.records = json.load(open(records_json))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = Path(rec["file_name"])
        name = img_path.stem
        base = img_path.parent.parent
        msk_path = base / "msk" / f"{name}.npy"
        json_path = base / "json" / f"{name}.json"
        chm_path = base / "chm" / f"{name}.npy"
        cost_path = base / "cost" / f"{name}.npy"

        img = np.load(img_path).astype(np.float32)
        msk = np.load(msk_path).astype(np.float32)
        if json_path.exists():
            point = np.array(json.load(open(json_path)).get("point", [img.shape[2] // 2, img.shape[1] // 2]), dtype=np.float32)
        else:
            point = np.array([img.shape[2] // 2, img.shape[1] // 2], dtype=np.float32)
        chm = np.load(chm_path).astype(np.float32) if chm_path.exists() else None
        cost = np.load(cost_path).astype(np.float32) if cost_path.exists() else None
        meta = rec.get("meta", {})
        return img, msk, point, name, chm, cost, meta


def gaussian_blur_torch(t: torch.Tensor, sigma: float = 2.0):
    if sigma <= 0:
        return t
    size = int(6 * sigma + 1) | 1
    coords = torch.arange(size, dtype=t.dtype, device=t.device) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    k1 = kernel.view(1, 1, 1, size)
    k2 = kernel.view(1, 1, size, 1)
    t = F.pad(t, (size // 2, size // 2, size // 2, size // 2), mode="reflect")
    t = F.conv2d(t, k1)
    t = F.conv2d(t, k2)
    return t


def normalize_band(band: np.ndarray):
    finite = np.isfinite(band)
    if finite.any():
        lo, hi = np.percentile(band[finite], [2, 98])
        if hi <= lo:
            lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
        band = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        band = np.zeros_like(band)
    return band


def preprocess_img(img: np.ndarray):
    bands = [normalize_band(img[c]) for c in range(img.shape[0])]
    return np.stack(bands, axis=0)


def blend_prior(prob: torch.Tensor, chm: Optional[torch.Tensor], cost: Optional[torch.Tensor], w_chm: float, w_cost: float):
    # prob: (H, W). Resize ancillary rasters to match prob spatial size.
    def _match_shape(t: torch.Tensor):
        if t.shape == prob.shape:
            return t
        t4 = t[None, None, ...]
        t4 = F.interpolate(t4, size=prob.shape, mode="bilinear", align_corners=False)
        return t4[0, 0]

    base = prob
    if chm is not None:
        chm = _match_shape(chm)
        chm_n = (chm - torch.quantile(chm, 0.05)) / (torch.quantile(chm, 0.95) - torch.quantile(chm, 0.05) + 1e-6)
        chm_n = torch.clamp(chm_n, 0, 1)
        base = base * (1 - w_chm + w_chm * chm_n)
    if cost is not None:
        cost = _match_shape(cost)
        cost_n = (cost - torch.quantile(cost, 0.05)) / (torch.quantile(cost, 0.95) - torch.quantile(cost, 0.05) + 1e-6)
        cost_n = torch.clamp(cost_n, 0, 1)
        base = base * (1 - w_cost + w_cost * (1 - cost_n))
    return torch.clamp(base, 0, 1)


def predict_prior(prior_inputs, predictor: SamPredictor, device, args):
    img_np, point_np, chm_np, cost_np = prior_inputs
    img_norm = preprocess_img(img_np)
    sam_img = (img_norm[[1, 2, 0]] * 255).astype(np.uint8).transpose(1, 2, 0)
    predictor.set_image(sam_img)
    pts = torch.tensor(point_np[None, None, :], device=device, dtype=torch.float32)
    labels = torch.ones((1, 1), device=device, dtype=torch.int64)
    with torch.no_grad():
        _, _, logits = predictor.predict_torch(point_coords=pts, point_labels=labels, multimask_output=False, return_logits=True)
    logit = logits[0, 0]
    if args.blur_sigma > 0:
        logit = gaussian_blur_torch(logit[None, None, ...], sigma=args.blur_sigma)[0, 0]
    yy, xx = torch.meshgrid(torch.arange(logit.shape[0], device=device), torch.arange(logit.shape[1], device=device), indexing="ij")
    d = torch.sqrt((yy - point_np[1]) ** 2 + (xx - point_np[0]) ** 2)
    core_w = torch.exp(-(d ** 2) / (2 * args.core_sigma ** 2)) ** args.core_gamma
    logit = logit * (core_w / (core_w.max() + 1e-6))
    logit_up = F.interpolate(logit[None, None, ...], size=img_norm.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    prob = torch.sigmoid(torch.clamp(args.logit_scale * logit_up, -12, 12))

    chm_t = torch.tensor(chm_np, device=device, dtype=torch.float32) if chm_np is not None else None
    cost_t = torch.tensor(cost_np, device=device, dtype=torch.float32) if cost_np is not None else None
    prior = blend_prior(prob, chm_t, cost_t, args.chm_weight, args.cost_weight)
    return img_norm, prior, logit_up


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred_bin = pred_mask > 0.5
    gt_bin = gt_mask > 0.5
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) > 0 else 0.0
    return iou, dice


def mask_to_polygons(mask: np.ndarray):
    from skimage import measure
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        coords = np.flip(c, axis=1)
        polys.append(coords.flatten().tolist())
    return polys


def build_coco_gt(val_items):
    images = []
    annotations = []
    ann_id = 1
    for img_id, (name, msk) in enumerate(val_items, start=1):
        if isinstance(msk, torch.Tensor):
            msk = msk.cpu().numpy()
        images.append({"id": img_id, "file_name": name, "height": msk.shape[0], "width": msk.shape[1]})
        labeled = (msk > 0).astype(np.uint8)
        polys = mask_to_polygons(labeled)
        if not polys:
            continue
        rle = mask_utils.encode(np.asfortranarray(labeled))
        area = float(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": polys,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
        })
        ann_id += 1
    coco_dict = {
        "info": {"description": "logit-aware stardist val set"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "crown"}],
    }
    return coco_dict


def coco_eval(preds, coco_gt_dict):
    # If no preds, return zeros to avoid pycocotools errors.
    if not preds:
        return {"coco_ap": 0.0, "coco_ap50": 0.0}
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    if not coco_gt_dict.get("annotations"):
        return {"coco_ap": 0.0, "coco_ap50": 0.0}
    try:
        coco_dt = coco_gt.loadRes(preds)
        if len(coco_dt.getImgIds()) == 0:
            return {"coco_ap": 0.0, "coco_ap50": 0.0}
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        return {"coco_ap": float(stats[0] * 100), "coco_ap50": float(stats[1] * 100)}
    except Exception:
        return {"coco_ap": 0.0, "coco_ap50": 0.0}


def eval_on_val(model, predictor, dl, device, args):
    model.eval()
    ious, dices = [], []
    geoms_world = []
    crs_out = None
    with torch.no_grad():
        for img_np, msk_np, point_np, name, chm_np, cost_np, meta in dl:
            img_np = img_np[0]; msk_np = msk_np[0]; point_np = point_np[0]
            name = name[0] if isinstance(name, (list, tuple)) else name
            chm_np = chm_np[0] if chm_np[0] is not None else None
            cost_np = cost_np[0] if cost_np[0] is not None else None
            img_norm, prior, logit_up = predict_prior((img_np, point_np, chm_np, cost_np), predictor, device, args)
            img_t = torch.tensor(img_norm, device=device, dtype=torch.float32)
            prior_resized = F.interpolate(prior.unsqueeze(0).unsqueeze(0), size=img_t.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            prior_t = prior_resized.unsqueeze(0)
            x = torch.cat([img_t, prior_t], dim=0).unsqueeze(0)
            pred_prob, pred_rays = model(x)
            pred_prob_sig = torch.sigmoid(pred_prob)[0, 0].cpu().numpy()

            iou, dice = compute_mask_metrics(pred_prob_sig, msk_np)
            ious.append(iou); dices.append(dice)

            # write logits/prob rasters if georef is available
            tfm = meta.get("transform")
            crs_meta = meta.get("crs")
            if isinstance(crs_meta, (list, tuple)):
                crs_meta = crs_meta[0] if len(crs_meta) > 0 else None
            if tfm and crs_meta:
                try:
                    crs_obj = CRS.from_user_input(crs_meta)
                    aff = rasterio.Affine(*tfm)
                    prob_dir = args.out.parent / "val_probs"
                    prob_dir.mkdir(parents=True, exist_ok=True)
                    for arr, suffix in [(logit_up.cpu().numpy(), "logit"), (prior_resized.cpu().numpy(), "prior")]:
                        path = prob_dir / f"{name}_{suffix}.tif"
                        with rasterio.open(
                            path,
                            "w",
                            driver="GTiff",
                            height=arr.shape[0],
                            width=arr.shape[1],
                            count=1,
                            dtype="float32",
                            crs=crs_obj,
                            transform=aff,
                        ) as dst:
                            dst.write(arr.astype(np.float32), 1)
                except Exception:
                    pass

            pred_rays_np = pred_rays[0].cpu().numpy().transpose(1, 2, 0)
            dec = decode_rays(pred_prob_sig, pred_rays_np, prob_thresh=0.5, nms_thresh=0.3)
            polys_pixel = [p for _, p in dec] if dec else []
            if not polys_pixel:
                # fallback: polygonize the binary mask
                from skimage import measure
                mask_bin = pred_prob_sig > 0.5
                contours = measure.find_contours(mask_bin.astype(np.uint8), 0.5)
                polys_pixel = [geom.Polygon(cnt[:, ::-1]) for cnt in contours if len(cnt) >= 3]

            tfm = meta.get("transform")
            crs_meta = meta.get("crs")
            if isinstance(crs_meta, (list, tuple)):
                crs_meta = crs_meta[0] if len(crs_meta) > 0 else None
            if polys_pixel and tfm:
                from affine import Affine
                aff = Affine(*tfm)
                if crs_out is None:
                    crs_out = crs_meta
                xs_list, ys_list = [], []
                for poly in polys_pixel:
                    coords = np.array(poly.exterior.coords)
                    xs_list.append(coords[:, 0])
                    ys_list.append(coords[:, 1])
                xs_cat = np.concatenate(xs_list)
                ys_cat = np.concatenate(ys_list)
                xw_cat = aff.c + aff.a * xs_cat + aff.b * ys_cat
                yw_cat = aff.f + aff.d * xs_cat + aff.e * ys_cat
                offset = 0
                for poly, xs, ys in zip(polys_pixel, xs_list, ys_list):
                    n = len(xs)
                    xw = xw_cat[offset:offset + n]
                    yw = yw_cat[offset:offset + n]
                    offset += n
                    poly_world = geom.Polygon(zip(xw, yw))
                    if poly_world.is_valid and poly_world.area > 0:
                        geoms_world.append((0.0, poly_world))
    model.train()
    return {"iou": float(np.mean(ious)) if ious else 0.0,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "geoms_world": geoms_world,
            "crs": crs_out}


def main():
    pa = argparse.ArgumentParser(description="Train logit-aware StarDist head using chips and SAM logits + CHM/COST weighting.")
    pa.add_argument("--chip-dir", required=True, type=Path)
    pa.add_argument("--records-json", required=True, type=Path, help="train records JSON")
    pa.add_argument("--val-records-json", type=Path, default=None, help="val records JSON; if omitted, uses holdout")
    pa.add_argument("--holdout", type=float, default=0.0, help="Holdout fraction if no val JSON provided")
    pa.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic holdout split")
    pa.add_argument("--sam-ckpt", required=True, type=Path)
    pa.add_argument("--epochs", type=int, default=2)
    pa.add_argument("--batch", type=int, default=1)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--n-rays", type=int, default=32)
    pa.add_argument("--blur-sigma", type=float, default=2.0)
    pa.add_argument("--core-sigma", type=float, default=8.0, help="core weighting sigma (pixels)")
    pa.add_argument("--core-gamma", type=float, default=1.7, help="core weighting exponent")
    pa.add_argument("--logit-scale", type=float, default=1.4)
    pa.add_argument("--chm-weight", type=float, default=0.75, help="weight for CHM in prior blend")
    pa.add_argument("--cost-weight", type=float, default=0.25, help="weight for COST in prior blend")
    pa.add_argument("--out", type=Path, default=Path("v2/models/logit_stardist_head.pth"))
    pa.add_argument("--metrics-out", type=Path, default=None)
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_json = args.records_json
    if args.val_records_json and args.val_records_json.exists():
        val_json = args.val_records_json
    elif args.holdout > 0:
        recs = json.load(open(args.records_json))
        random.seed(args.split_seed)
        random.shuffle(recs)
        k = int(round(len(recs) * (1.0 - args.holdout)))
        train_recs = recs[:k]
        val_recs = recs[k:]
        val_json = args.records_json.parent / "val_records_autosplit.json"
        train_json = args.records_json.parent / "train_records_autosplit.json"
        json.dump(train_recs, open(train_json, "w"))
        json.dump(val_recs, open(val_json, "w"))
    else:
        val_json = None

    ds_train = ChipDataset(args.chip_dir, train_json)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
    dl_val = None
    if val_json and Path(val_json).exists():
        ds_val = ChipDataset(args.chip_dir, val_json)
        dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_l"](checkpoint=str(args.sam_ckpt)).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    model = LogitAwareStarDist(in_ch=4, n_rays=args.n_rays).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for batch in dl_train:
            img_np, msk_np, point_np, name, chm_np, cost_np, meta = batch
            img_np = img_np[0]; msk_np = msk_np[0]; point_np = point_np[0]
            chm_np = chm_np[0] if chm_np[0] is not None else None
            cost_np = cost_np[0] if cost_np[0] is not None else None

            img_norm, prior = predict_prior((img_np, point_np, chm_np, cost_np), predictor, device, args)
            img_t = torch.tensor(img_norm, device=device, dtype=torch.float32)
            prior_resized = F.interpolate(prior.unsqueeze(0).unsqueeze(0), size=img_t.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            prior_t = prior_resized.unsqueeze(0)
            x = torch.cat([img_t, prior_t], dim=0).unsqueeze(0)

            rays_np, obj_np = compute_ray_targets(msk_np, n_rays=args.n_rays, erosion=1)
            rays_t = torch.tensor(rays_np.transpose(2, 0, 1), device=device, dtype=torch.float32).unsqueeze(0)
            obj_t = torch.tensor(obj_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            pred_prob, pred_rays = model(x)
            loss_prob = F.binary_cross_entropy_with_logits(pred_prob, obj_t)
            fg_mask = (obj_t > 0.5).float()
            if fg_mask.sum() > 0:
                loss_rays = (torch.abs(pred_rays - rays_t) * fg_mask).sum() / (fg_mask.sum() * args.n_rays)
            else:
                loss_rays = torch.tensor(0.0, device=device)

            loss = loss_prob + loss_rays
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        print(f"epoch {ep}/{args.epochs} loss={tot_loss/len(dl_train):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": vars(args)}, args.out)
    print(f"✓ saved fused head to {args.out}")

    if dl_val:
        train_count = sum(len(r.get("annotations", [])) for r in ds_train.records)
        val_count = sum(len(r.get("annotations", [])) for r in ds_val.records) if ds_val else 0
        metrics = eval_on_val(model, predictor, dl_val, device, args)
        geoms_world = metrics.pop("geoms_world", [])
        crs_out = metrics.pop("crs", None)
        metrics = {"train_instances": train_count, "val_instances": val_count, **metrics}
        print(f"Val split: train_instances={train_count}, val_instances={val_count}")
        metrics_path = args.metrics_out or args.out.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        csv_path = metrics_path.with_suffix(".csv")
        with open(csv_path, "w") as f:
            f.write("metric,value\n")
            for k, v in metrics.items():
                f.write(f"{k},{v}\n")
        if geoms_world:
            crs_write = crs_out[0] if isinstance(crs_out, (list, tuple)) else crs_out
            gdf = gpd.GeoDataFrame(
                {"score": [g[0] for g in geoms_world]},
                geometry=[g[1] for g in geoms_world],
                crs=crs_write,
            )
            gpkg_path = args.out.parent / "val_preds.gpkg"
            gdf.to_file(gpkg_path, layer="crowns", driver="GPKG")
            print(f"✓ wrote val predictions to {gpkg_path}")
        print(f"✓ val metrics written to {metrics_path} and {csv_path}")


if __name__ == "__main__":
    main()
