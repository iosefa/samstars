#!/usr/bin/env python3
"""
Run Detectron2 inference on chip GeoTIFFs and optionally write predictions to a GPKG.
Also supports COCO evaluation if ground-truth GPKGs are provided via detectron_manifest.json.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union, transform as shp_transform
from shapely.validation import make_valid

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def robust_norm255(band):
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(band[finite], [2, 98])
    if hi <= lo:
        lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(band, dtype=np.uint8)
    band = np.clip(band, lo, hi)
    band = (band - lo) / max(hi - lo, 1e-6) * 255.0
    return band.astype(np.uint8)


def read_rgb_from_tif(path: Path, bands_1idx=(7, 5, 3)):
    with rasterio.open(path) as src:
        available = src.count
        sel = [b for b in bands_1idx if b <= available] or [1]
        arr = src.read(sel).astype(np.float32)
    if arr.shape[0] == 1:
        nir = red = grn = arr[0]
    elif arr.shape[0] == 2:
        nir, red = arr[0], arr[1]
        grn = arr[1]
    else:
        nir, red, grn = arr[0], arr[1], arr[2]
    nir8, red8, grn8 = robust_norm255(nir), robust_norm255(red), robust_norm255(grn)
    rgb = np.dstack([red8, grn8, nir8]).astype(np.uint8)
    return rgb


def mask_to_polygons(mask_bool, transform, min_mask_px=50):
    if mask_bool.sum() < min_mask_px:
        return []
    polys = []
    for geom, val in features.shapes(mask_bool.astype(np.uint8), mask=mask_bool, transform=transform):
        if val == 0:
            continue
        poly = make_valid(shape(geom))
        if poly.is_empty:
            continue
        polys.append(poly)
    if not polys:
        return []
    return [unary_union(polys)]


def _geom_to_img_coords(geom, src: rasterio.io.DatasetReader):
    inv_affine = ~src.transform

    def _affine_xy(x, y, z=None):
        col, row = inv_affine * (x, y)
        return (col, row)

    g = make_valid(geom)
    if g.is_empty:
        return None
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        g = g.buffer(0)
    if g.is_empty:
        return None
    return shp_transform(_affine_xy, g)


def _poly_to_coco_segmentation(poly: Polygon):
    if not isinstance(poly, Polygon):
        return []
    exterior = np.array(poly.exterior.coords, dtype=np.float32)
    if exterior.shape[0] < 3:
        return []
    return exterior.flatten().tolist()


def build_records(manifest_path: Path):
    data = json.load(open(manifest_path))
    records = []
    for rec in data:
        tif_path = Path(rec["tif_path"])
        gpkg_path = Path(rec["gpkg_path"])
        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
            img_crs = src.crs
            gdf = gpd.read_file(gpkg_path)
            if gdf.crs is None:
                gdf = gdf.set_crs(img_crs)
            elif img_crs and gdf.crs != img_crs:
                gdf = gdf.to_crs(img_crs)

            annos = []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                g_img = _geom_to_img_coords(geom, src)
                if g_img is None or g_img.is_empty:
                    continue
                if isinstance(g_img, Polygon):
                    polys = [g_img]
                elif isinstance(g_img, MultiPolygon):
                    polys = [p for p in g_img.geoms if p.area > 0]
                else:
                    polys = []
                segs = []
                bboxes = []
                for p in polys:
                    seg = _poly_to_coco_segmentation(p)
                    if len(seg) < 6:
                        continue
                    segs.append(seg)
                    x, y, maxx, maxy = p.bounds
                    bboxes.append([x, y, maxx - x, maxy - y])
                if not segs:
                    continue
                x0 = min(b[0] for b in bboxes)
                y0 = min(b[1] for b in bboxes)
                x1 = max(b[0] + b[2] for b in bboxes)
                y1 = max(b[1] + b[3] for b in bboxes)
                annos.append({
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segs,
                    "category_id": 0,
                    "iscrowd": 0,
                })

            records.append({
                "file_name": str(tif_path),
                "image_id": rec.get("name", tif_path.stem),
                "height": H,
                "width": W,
                "annotations": annos,
            })
    return records


def mapper_test(dataset_dict, bands_1idx):
    d = dataset_dict.copy()
    rgb = read_rgb_from_tif(Path(d["file_name"]), bands_1idx=bands_1idx)
    d["image"] = torch.as_tensor(rgb.transpose(2, 0, 1))
    annos = [a for a in d.get("annotations", []) if a.get("iscrowd", 0) == 0]
    d["instances"] = utils.annotations_to_instances(annos, image_size=rgb.shape[:2], mask_format="polygon")
    d.pop("annotations", None)
    return d


def write_metrics_csv(metrics, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def _valid_polys(geoms):
    polys = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        gv = make_valid(g)
        if gv.is_empty:
            continue
        if gv.geom_type == "Polygon":
            polys.append(gv)
        elif gv.geom_type == "MultiPolygon":
            polys.extend([p for p in gv.geoms if p.area > 0])
    return polys


def _geom_iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def compute_union_iou(gt_polys, pred_polys):
    if not gt_polys and not pred_polys:
        return 1.0
    if not gt_polys or not pred_polys:
        return 0.0
    gt_union = unary_union(gt_polys)
    pr_union = unary_union(pred_polys)
    return _geom_iou(gt_union, pr_union)


def compute_ap50(gt_polys, pred_polys, pred_scores, iou_thr=0.5):
    if not gt_polys:
        return float("nan")
    if not pred_polys:
        return 0.0
    # sort by score desc
    order = np.argsort(-np.array(pred_scores))
    preds = [pred_polys[i] for i in order]
    scores = [pred_scores[i] for i in order]
    matched = [False] * len(gt_polys)
    tp = []
    fp = []
    for p in preds:
        best_iou = 0.0
        best_idx = -1
        for j, g in enumerate(gt_polys):
            if matched[j]:
                continue
            iou = _geom_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_thr and best_idx >= 0:
            matched[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / max(len(gt_polys), 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    # precision envelope
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = np.trapezoid(mpre, mrec)
    return float(ap)


def main():
    pa = argparse.ArgumentParser(description="Evaluate Detectron2 on chip GeoTIFFs.")
    pa.add_argument("--val-manifest", type=Path, required=True, help="detectron_manifest.json for val chips")
    pa.add_argument("--weights", type=Path, required=True, help="Trained weights .pth")
    pa.add_argument("--output-gpkg", type=Path, default=None, help="Optional merged predictions GPKG")
    pa.add_argument("--per-chip-dir", type=Path, default=None, help="Optional directory for per-chip prediction GPKGs")
    pa.add_argument("--metrics-out", type=Path, default=None, help="Optional metrics JSON output")
    pa.add_argument("--bands", type=str, default="7,5,3", help="1-indexed band list, e.g. 7,5,3")
    pa.add_argument("--score-thresh", type=float, default=0.5)
    pa.add_argument("--min-mask-px", type=int, default=50)
    pa.add_argument("--max-dets", type=int, default=100, help="Max detections per image at test time")
    pa.add_argument("--rpn-post-nms-topk", type=int, default=1000, help="RPN post-NMS topk at test time")
    pa.add_argument("--device", type=str, default="cpu")
    args = pa.parse_args()

    bands_1idx = tuple(int(b) for b in args.bands.split(",") if b.strip())

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # Match training mapper which feeds RGB-ordered tensors directly.
    # DefaultPredictor will flip channels if INPUT.FORMAT == "RGB",
    # so use BGR here to avoid unintended channel reversal.
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.TEST.DETECTIONS_PER_IMAGE = args.max_dets
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = args.rpn_post_nms_topk
    cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.DATALOADER.NUM_WORKERS = 0

    predictor = DefaultPredictor(cfg)

    # COCO eval (if GT exists)
    records = build_records(args.val_manifest)
    DatasetCatalog.register("chips_val_eval", lambda: records)
    MetadataCatalog.get("chips_val_eval").set(thing_classes=["crown"])
    val_loader = build_detection_test_loader(cfg, "chips_val_eval", mapper=lambda d: mapper_test(d, bands_1idx))
    metrics_dir = (args.metrics_out or Path("metrics.json")).parent
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator = COCOEvaluator("chips_val_eval", output_dir=str(metrics_dir))
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    if args.metrics_out:
        json.dump(metrics, open(args.metrics_out, "w"))
        write_metrics_csv(metrics, args.metrics_out.with_suffix(".csv"))

    # Inference → merged GPKG
    if args.output_gpkg or args.per_chip_dir:
        geoms = []
        scores = []
        chip_ids = []
        per_chip_rows = []
        if args.per_chip_dir:
            args.per_chip_dir.mkdir(parents=True, exist_ok=True)
        with open(args.val_manifest) as f:
            pairs = json.load(f)
        for rec in pairs:
            tif_path = Path(rec["tif_path"])
            per_chip_geoms = []
            per_chip_scores = []
            with rasterio.open(tif_path) as src:
                rgb = read_rgb_from_tif(tif_path, bands_1idx=bands_1idx)
                outputs = predictor(rgb)
                instances = outputs["instances"].to("cpu")
                if len(instances) == 0:
                    preds_polys = []
                    preds_scores = []
                else:
                    masks = instances.pred_masks.numpy()
                    sc = instances.scores.numpy()
                    for k in range(masks.shape[0]):
                        if sc[k] < args.score_thresh:
                            continue
                        polys = mask_to_polygons(masks[k].astype(bool), src.transform, min_mask_px=args.min_mask_px)
                        for poly in polys:
                            geoms.append(poly)
                            scores.append(float(sc[k]))
                            chip_ids.append(rec.get("name", tif_path.stem))
                            per_chip_geoms.append(poly)
                            per_chip_scores.append(float(sc[k]))
                    preds_polys = per_chip_geoms[:]
                    preds_scores = per_chip_scores[:]

            gt_gpkg = Path(rec["gpkg_path"])
            gdf_gt = gpd.read_file(gt_gpkg)
            if gdf_gt.crs is None:
                with rasterio.open(tif_path) as src:
                    gdf_gt = gdf_gt.set_crs(src.crs)
            gt_polys = _valid_polys(gdf_gt.geometry)
            preds_polys = _valid_polys(preds_polys)
            iou_union = compute_union_iou(gt_polys, preds_polys)
            ap50 = compute_ap50(gt_polys, preds_polys, preds_scores, iou_thr=0.5)
            per_chip_rows.append({
                "chip_id": rec.get("name", tif_path.stem),
                "n_gt": len(gt_polys),
                "n_pred": len(preds_polys),
                "iou_union": iou_union,
                "ap50": ap50,
            })

            if args.per_chip_dir and per_chip_geoms:
                gdf_chip = gpd.GeoDataFrame(
                    {"score": per_chip_scores},
                    geometry=per_chip_geoms,
                    crs=src.crs,
                )
                chip_name = rec.get("name", tif_path.stem)
                out_path = args.per_chip_dir / f"{chip_name}.gpkg"
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                gdf_chip.to_file(out_path, layer="crowns_pred", driver="GPKG")

        if args.output_gpkg and geoms:
            with rasterio.open(Path(pairs[0]["tif_path"])) as src:
                crs = src.crs
            gdf = gpd.GeoDataFrame({"score": scores, "chip_id": chip_ids}, geometry=geoms, crs=crs)
            out_path = Path(args.output_gpkg)
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            gdf.to_file(out_path, layer="crowns_pred", driver="GPKG")
            print(f"✓ wrote predictions to {out_path}")

        if per_chip_rows:
            base = args.metrics_out if args.metrics_out else (metrics_dir / "metrics.json")
            per_chip_csv = base.with_name(f"{base.stem}_per_chip.csv")
            per_chip_json = base.with_name(f"{base.stem}_per_chip.json")
            with open(per_chip_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["chip_id", "n_gt", "n_pred", "iou_union", "ap50"])
                writer.writeheader()
                for row in per_chip_rows:
                    writer.writerow(row)
            json.dump(per_chip_rows, open(per_chip_json, "w"))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
