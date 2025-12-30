#!/usr/bin/env python3
"""
Run trained StarDist on validation probability rasters, write polygons (GPKG) and IoU/Dice.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import geopandas as gpd
import shapely.geometry as geom
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist import random_label_cmap
from stardist.matching import matching
from rasterio.crs import CRS
import rasterio
import rasterio


def compute_iou_dice(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    return iou, dice


def compute_pq(pred_mask, gt_mask, iou_thr=0.5):
    m = matching(gt_mask, pred_mask, thresh=iou_thr, criterion="iou")
    tp = int(m.tp)
    fp = int(m.fp)
    fn = int(m.fn)
    sq = float(m.mean_matched_score) if tp else 0.0
    denom = tp + 0.5 * (fp + fn)
    rq = float(tp / denom) if denom > 0 else 0.0
    pq = sq * rq
    return pq, sq, rq


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--val-probs", type=Path, required=True, help="val_probs.json from run_sam_probs_on_chips")
    pa.add_argument("--model-dir", type=Path, required=True)
    pa.add_argument("--out-dir", type=Path, required=True)
    pa.add_argument("--prob-thr", type=float, default=0.2)
    pa.add_argument("--nms-thr", type=float, default=0.2)
    args = pa.parse_args()

    val_recs = json.load(open(args.val_probs))
    axis_norm = (0, 1)
    model = StarDist2D(None, name="stardist", basedir=str(args.model_dir))

    ious, dices = [], []
    pqs, sqs, rqs = [], [], []
    polys_world = []
    crs_out = None
    for rec in val_recs:
        fname = rec.get("file_name", "")
        mpath = rec.get("mask", "")
        if not fname or not mpath or not Path(mpath).exists():
            continue
        if fname.endswith(".tif"):
            with rasterio.open(fname) as src:
                prob = src.read(1).astype(np.float32)
                tfm = src.transform
                crs_meta = src.crs
        else:
            prob = np.load(fname).astype(np.float32)
            tfm = rec.get("meta", {}).get("transform")
            crs_meta = rec.get("meta", {}).get("crs")
        if mpath.endswith(".tif"):
            with rasterio.open(mpath) as src:
                msk = src.read(1).astype(np.uint16)
        else:
            msk = np.load(mpath).astype(np.uint16)
        # Ensure prob matches mask shape (pad/crop as needed).
        if prob.shape != msk.shape:
            ph, pw = prob.shape
            mh, mw = msk.shape
            # Crop if prob is larger.
            prob = prob[:mh, :mw]
            ph, pw = prob.shape
            # Pad if prob is smaller.
            if ph < mh or pw < mw:
                pad_h = max(0, mh - ph)
                pad_w = max(0, mw - pw)
                prob = np.pad(prob, ((0, pad_h), (0, pad_w)), constant_values=0)
        prob_n = normalize(prob, 1, 99.8, axis=axis_norm)[..., None]
        labels, _ = model.predict_instances(
            prob_n,
            n_tiles=model._guess_n_tiles(prob_n),
            prob_thresh=args.prob_thr,
            nms_thresh=args.nms_thr,
            show_tile_progress=False,
        )
        iou, dice = compute_iou_dice(labels, msk)
        pq, sq, rq = compute_pq(labels, msk, iou_thr=0.5)
        ious.append(iou); dices.append(dice)
        pqs.append(pq); sqs.append(sq); rqs.append(rq)

        if tfm and crs_meta:
            from affine import Affine
            aff = Affine(*tfm) if not isinstance(tfm, Affine) else tfm
            crs_out = crs_meta if crs_out is None else crs_out
            from skimage import measure
            for lbl_id in np.unique(labels):
                if lbl_id == 0:
                    continue
                mask_lbl = labels == lbl_id
                contours = measure.find_contours(mask_lbl.astype(np.uint8), 0.5)
                for cnt in contours:
                    xs, ys = cnt[:, 1], cnt[:, 0]
                    xw = aff.c + aff.a * xs + aff.b * ys
                    yw = aff.f + aff.d * xs + aff.e * ys
                    poly = geom.Polygon(zip(xw, yw))
                    if poly.is_valid and poly.area > 0:
                        polys_world.append(poly)

    metrics = {"iou": float(np.mean(ious)) if ious else 0.0,
               "dice": float(np.mean(dices)) if dices else 0.0,
               "pq": float(np.mean(pqs)) if pqs else 0.0,
               "sq": float(np.mean(sqs)) if sqs else 0.0,
               "rq": float(np.mean(rqs)) if rqs else 0.0,
               "val_instances": int(sum(len(np.unique(np.load(r['mask']))) - 1 for r in val_recs if r.get('mask') and Path(r['mask']).exists())),
               "train_instances": None}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.out_dir / "metrics.csv", "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v if v is not None else ''}\n")

    if polys_world and crs_out:
        gdf = gpd.GeoDataFrame({"id": range(1, len(polys_world)+1)}, geometry=polys_world, crs=crs_out)
        gdf.to_file(args.out_dir / "val_preds.gpkg", layer="crowns", driver="GPKG")
        print(f"✓ wrote val polygons to {args.out_dir / 'val_preds.gpkg'}")
    print(f"✓ IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f} PQ={metrics['pq']:.3f} SQ={metrics['sq']:.3f} RQ={metrics['rq']:.3f}")


if __name__ == "__main__":
    main()
