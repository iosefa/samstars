#!/usr/bin/env python3
"""
Run trained StarDist on validation rasters, write polygons (GPKG) and IoU/Dice/PQ.

This version matches the original preprocessing:
- Default input is "index" (canopy index) from the manifest
- csbdeep percentile normalization (1..99.8) is applied (per-channel if multi-channel)
- Optional aux channels (CHM/COST) can be stacked

If you want to evaluate on raw SAM prob instead of index, pass --input-key file_name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import shapely.geometry as geom
import rasterio
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.matching import matching

from scale_stats import load_scale_stats, scale_to_01


def compute_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    pred = pred_mask > 0
    gt = gt_mask > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    return float(iou), float(dice)


def compute_pq(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_thr: float = 0.5) -> tuple[float, float, float]:
    m = matching(gt_mask, pred_mask, thresh=iou_thr, criterion="iou")
    tp = int(m.tp)
    fp = int(m.fp)
    fn = int(m.fn)
    sq = float(m.mean_matched_score) if tp else 0.0
    denom = tp + 0.5 * (fp + fn)
    rq = float(tp / denom) if denom > 0 else 0.0
    pq = sq * rq
    return float(pq), float(sq), float(rq)


def _align_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    th, tw = target_shape
    arr = arr[:th, :tw]
    ah, aw = arr.shape
    if ah < th or aw < tw:
        pad_h = max(0, th - ah)
        pad_w = max(0, tw - aw)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=0)
    return arr


def _scale_aux(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return scale_to_01(arr, lo, hi)


def _extract_thresholds(thr) -> tuple[float | None, float | None]:
    if thr is None:
        return None, None
    if isinstance(thr, dict):
        prob = thr.get("prob")
        nms = thr.get("nms")
        if prob is None:
            prob = thr.get("prob_thresh")
        if nms is None:
            nms = thr.get("nms_thresh")
        return prob, nms
    prob = getattr(thr, "prob", None)
    nms = getattr(thr, "nms", None)
    return prob, nms


def _load_thresholds_from_json(model_dir: Path) -> tuple[float | None, float | None]:
    thr_path = model_dir / "stardist" / "thresholds.json"
    if not thr_path.exists():
        return None, None
    data = json.load(open(thr_path))
    return _extract_thresholds(data)


def _read_float_raster(path: str | Path) -> np.ndarray:
    path = str(path)
    if path.endswith(".tif") or path.endswith(".tiff"):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    return np.load(path).astype(np.float32)


def _read_u16_mask(path: str | Path) -> np.ndarray:
    path = str(path)
    if path.endswith(".tif") or path.endswith(".tiff"):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.uint16)
    return np.load(path).astype(np.uint16)


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--val-probs", type=Path, required=True, help="val_probs.json from build_prob_manifests")
    pa.add_argument("--model-dir", type=Path, required=True)
    pa.add_argument("--out-dir", type=Path, required=True)

    pa.add_argument("--prob-thr", type=float, default=None)
    pa.add_argument("--nms-thr", type=float, default=None)

    # Original-matching knobs
    pa.add_argument(
        "--input-key",
        type=str,
        default="index",
        help="Manifest key to use as StarDist input (default: index). Use file_name for raw SAM prob.",
    )
    pa.add_argument("--normalize", action="store_true", help="Enable csbdeep percentile normalization")
    pa.add_argument("--no-normalize", action="store_true", help="Disable csbdeep percentile normalization (original had normalization ON).")

    # Aux
    pa.add_argument("--scale-stats", type=Path, default=None, help="Global scaling stats JSON for CHM/COST")
    pa.add_argument("--use-aux", action="store_true", help="Use CHM aux channel")
    pa.add_argument("--chm-key", type=str, default="chm", help="Manifest key for CHM path")
    args = pa.parse_args()

    val_recs = json.load(open(args.val_probs))

    scale_stats = load_scale_stats(args.scale_stats) if args.scale_stats else None
    if args.use_aux and scale_stats is None:
        raise SystemExit("--scale-stats is required when --use-aux is set")

    chm_lo = chm_hi = cost_lo = cost_hi = None
    if scale_stats:
        chm_lo = float(scale_stats.get("chm_01", {}).get("lo", 0.0))
        chm_hi = float(scale_stats.get("chm_01", {}).get("hi", 1.0))

    model = StarDist2D(None, name="stardist", basedir=str(args.model_dir))

    prob_thr = args.prob_thr
    nms_thr = args.nms_thr
    if prob_thr is None and nms_thr is None:
        p1, n1 = _extract_thresholds(getattr(model, "thresholds", None))
        if p1 is None or n1 is None:
            p1, n1 = _load_thresholds_from_json(args.model_dir)
        prob_thr, nms_thr = p1, n1
    if (prob_thr is None) ^ (nms_thr is None):
        raise SystemExit("Provide both --prob-thr and --nms-thr, or neither to use optimized thresholds.")

    if args.normalize and args.no_normalize:
        raise SystemExit("Use only one of --normalize or --no-normalize.")
    if args.normalize:
        do_normalize = True
    elif args.no_normalize:
        do_normalize = False
    else:
        do_normalize = True
    axis_norm = (0, 1)

    ious, dices = [], []
    pqs, sqs, rqs = [], [], []
    polys_world = []
    crs_out = None
    val_instances = 0

    for rec in val_recs:
        x_path = rec.get(args.input_key, "")
        mpath = rec.get("mask", "")
        if not x_path or not mpath or not Path(mpath).exists():
            continue
        if not Path(x_path).exists():
            continue

        if str(x_path).endswith(".tif"):
            with rasterio.open(x_path) as src:
                x = src.read(1).astype(np.float32)
                tfm = src.transform
                crs_meta = src.crs
        else:
            x = _read_float_raster(x_path)
            tfm = rec.get("meta", {}).get("transform")
            crs_meta = rec.get("meta", {}).get("crs")

        msk = _read_u16_mask(mpath)
        val_instances += int(np.unique(msk).size - 1)

        x = _align_to_shape(x, msk.shape)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if args.use_aux:
            chm_path = rec.get(args.chm_key, "")
            if not chm_path:
                continue
            if not Path(chm_path).exists():
                continue
            chm = _read_float_raster(chm_path)
            chm = _align_to_shape(chm, msk.shape)
            if scale_stats is not None:
                chm = _scale_aux(chm, chm_lo, chm_hi)
            x_in = np.stack([x, chm.astype(np.float32)], axis=-1).astype(np.float32)
        else:
            x_in = x[..., None].astype(np.float32)

        if do_normalize:
            x_in = normalize(x_in, 1, 99.8, axis=axis_norm).astype(np.float32)

        predict_kwargs = {
            "n_tiles": model._guess_n_tiles(x_in),
            "show_tile_progress": False,
        }
        if prob_thr is not None and nms_thr is not None:
            predict_kwargs["prob_thresh"] = prob_thr
            predict_kwargs["nms_thresh"] = nms_thr

        labels, _ = model.predict_instances(x_in, **predict_kwargs)

        iou, dice = compute_iou_dice(labels, msk)
        pq, sq, rq = compute_pq(labels, msk, iou_thr=0.5)
        ious.append(iou)
        dices.append(dice)
        pqs.append(pq)
        sqs.append(sq)
        rqs.append(rq)

        # polygons
        if tfm and crs_meta:
            from affine import Affine
            from skimage import measure

            aff = Affine(*tfm) if not isinstance(tfm, Affine) else tfm
            crs_out = crs_meta if crs_out is None else crs_out

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

    metrics = {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "pq": float(np.mean(pqs)) if pqs else 0.0,
        "sq": float(np.mean(sqs)) if sqs else 0.0,
        "rq": float(np.mean(rqs)) if rqs else 0.0,
        "val_instances": int(val_instances),
        "train_instances": None,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.out_dir / "metrics.csv", "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v if v is not None else ''}\n")

    if polys_world and crs_out:
        gdf = gpd.GeoDataFrame({"id": range(1, len(polys_world) + 1)}, geometry=polys_world, crs=crs_out)
        gdf.to_file(args.out_dir / "val_preds.gpkg", layer="crowns", driver="GPKG")
        print(f"✓ wrote val polygons to {args.out_dir / 'val_preds.gpkg'}")

    print(
        f"✓ IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f} "
        f"PQ={metrics['pq']:.3f} SQ={metrics['sq']:.3f} RQ={metrics['rq']:.3f}"
    )


if __name__ == "__main__":
    main()
