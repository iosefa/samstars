#!/usr/bin/env python3
"""
Evaluate multiclass Mask R-CNN on chip GeoTIFFs using all raster bands + CHM.
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


def _chm_path_for_tif(tif_path: Path, chm_suffix: str) -> Path:
    return tif_path.with_name(f"{tif_path.stem}{chm_suffix}")


def read_multiband_with_chm(path: Path, chm_suffix: str):
    with rasterio.open(path) as src:
        arr = src.read(list(range(1, src.count + 1))).astype(np.float32)
    bands_u8 = [robust_norm255(arr[i]) for i in range(arr.shape[0])]
    chm_path = _chm_path_for_tif(path, chm_suffix)
    if not chm_path.exists():
        raise FileNotFoundError(f"Missing CHM chip: {chm_path}")
    with rasterio.open(chm_path) as csrc:
        chm = csrc.read(1).astype(np.float32)
    bands_u8.append(robust_norm255(chm))
    return np.stack(bands_u8, axis=-1).astype(np.uint8)


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


def _collect_classes(manifests: list[Path], class_field: str) -> list[int]:
    classes = set()
    for manifest in manifests:
        data = json.load(open(manifest))
        for rec in data:
            gpkg_path = Path(rec["gpkg_path"])
            if not gpkg_path.exists():
                continue
            gdf = gpd.read_file(gpkg_path)
            if class_field not in gdf.columns:
                continue
            vals = gdf[class_field].dropna().unique().tolist()
            for v in vals:
                try:
                    classes.add(int(v))
                except Exception:
                    pass
    if not classes:
        return [0]
    return sorted(classes)


def load_class_mapping(mapping_path: Path, manifest_path: Path, class_field: str):
    if mapping_path and mapping_path.exists():
        data = json.load(open(mapping_path))
        raw_map = data.get("class_to_id", {})
        class_to_id = {int(k): int(v) for k, v in raw_map.items()}
        id_to_class = {int(v): int(k) for k, v in class_to_id.items()}
        if id_to_class:
            max_id = max(id_to_class)
            classes = [id_to_class.get(i, i) for i in range(max_id + 1)]
            return classes, class_to_id, id_to_class
    classes = _collect_classes([manifest_path], class_field)
    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {i: c for c, i in class_to_id.items()}
    return classes, class_to_id, id_to_class


def build_records(manifest_path: Path, class_to_id: dict[int, int], class_field: str):
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
            for _, row in gdf.iterrows():
                geom = row.geometry
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
                cls_val = row.get(class_field, None)
                try:
                    cls_val = int(cls_val)
                except Exception:
                    cls_val = 0
                class_id = class_to_id.get(cls_val, 0)
                annos.append({
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segs,
                    "category_id": int(class_id),
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


def mapper_test(dataset_dict, chm_suffix: str):
    d = dataset_dict.copy()
    img = read_multiband_with_chm(Path(d["file_name"]), chm_suffix=chm_suffix)
    img = np.ascontiguousarray(img)
    d["image"] = torch.as_tensor(img.transpose(2, 0, 1))
    annos = [a for a in d.get("annotations", []) if a.get("iscrowd", 0) == 0]
    d["instances"] = utils.annotations_to_instances(annos, image_size=img.shape[:2], mask_format="polygon")
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
    order = np.argsort(-np.array(pred_scores))
    preds = [pred_polys[i] for i in order]
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
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = np.trapezoid(mpre, mrec)
    return float(ap)


def main():
    pa = argparse.ArgumentParser(description="Evaluate multiclass Detectron2 on chip GeoTIFFs.")
    pa.add_argument("--val-manifest", type=Path, required=True, help="detectron_manifest.json for val chips")
    pa.add_argument("--weights", type=Path, required=True, help="Trained weights .pth")
    pa.add_argument("--class-mapping", type=Path, default=None, help="class_mapping.json from training output")
    pa.add_argument("--class-field", type=str, default="class")
    pa.add_argument("--chm-suffix", type=str, default="_chm.tif", help="Suffix for CHM chip tif files")
    pa.add_argument("--output-gpkg", type=Path, default=None, help="Optional merged predictions GPKG")
    pa.add_argument("--per-chip-dir", type=Path, default=None, help="Optional directory for per-chip prediction GPKGs")
    pa.add_argument("--metrics-out", type=Path, default=None, help="Optional metrics JSON output")
    pa.add_argument("--score-thresh", type=float, default=0.5)
    pa.add_argument("--min-mask-px", type=int, default=50)
    pa.add_argument("--max-dets", type=int, default=100, help="Max detections per image at test time")
    pa.add_argument("--rpn-post-nms-topk", type=int, default=1000, help="RPN post-NMS topk at test time")
    pa.add_argument("--device", type=str, default="cpu")
    args = pa.parse_args()

    mapping_path = args.class_mapping
    if mapping_path is None:
        candidate = args.weights.parent / "class_mapping.json"
        if candidate.exists():
            mapping_path = candidate
    classes, class_to_id, id_to_class = load_class_mapping(mapping_path, args.val_manifest, args.class_field)

    records = build_records(args.val_manifest, class_to_id, args.class_field)
    if not records:
        raise RuntimeError("No validation records found.")

    sample_tif = Path(records[0]["file_name"])
    with rasterio.open(sample_tif) as src:
        n_bands = src.count
    n_channels = n_bands + 1

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.TEST.DETECTIONS_PER_IMAGE = args.max_dets
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = args.rpn_post_nms_topk
    cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.RESNETS.STEM_IN_CHANNELS = n_channels
    cfg.MODEL.PIXEL_MEAN = [0.0] * n_channels
    cfg.MODEL.PIXEL_STD = [1.0] * n_channels

    predictor = DefaultPredictor(cfg)
    model = predictor.model
    model.eval()
    model_device = next(model.parameters()).device

    DatasetCatalog.register("chips_val_eval_mc", lambda: records)
    MetadataCatalog.get("chips_val_eval_mc").set(thing_classes=[str(c) for c in classes])
    val_loader = build_detection_test_loader(
        cfg,
        "chips_val_eval_mc",
        mapper=lambda d: mapper_test(d, args.chm_suffix),
    )
    metrics_dir = (args.metrics_out or Path("metrics.json")).parent
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator = COCOEvaluator("chips_val_eval_mc", output_dir=str(metrics_dir))
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    if args.metrics_out:
        json.dump(metrics, open(args.metrics_out, "w"))
        write_metrics_csv(metrics, args.metrics_out.with_suffix(".csv"))

    if args.output_gpkg or args.per_chip_dir:
        geoms = []
        scores = []
        class_ids = []
        class_labels = []
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
            per_chip_class_ids = []
            per_chip_class_labels = []
            with rasterio.open(tif_path) as src:
                img = read_multiband_with_chm(tif_path, chm_suffix=args.chm_suffix)
                img = np.ascontiguousarray(img)
                img_tensor = torch.as_tensor(img.transpose(2, 0, 1)).to(model_device)
                inputs = [{
                    "image": img_tensor,
                    "height": img.shape[0],
                    "width": img.shape[1],
                }]
                with torch.no_grad():
                    outputs = model(inputs)[0]
                instances = outputs["instances"].to("cpu")
                if len(instances) == 0:
                    preds_polys = []
                    preds_scores = []
                else:
                    masks = instances.pred_masks.numpy()
                    sc = instances.scores.numpy()
                    cl = instances.pred_classes.numpy()
                    for k in range(masks.shape[0]):
                        if sc[k] < args.score_thresh:
                            continue
                        polys = mask_to_polygons(masks[k].astype(bool), src.transform, min_mask_px=args.min_mask_px)
                        if not polys:
                            continue
                        cls_id = int(cl[k])
                        cls_label = int(id_to_class.get(cls_id, cls_id))
                        for poly in polys:
                            geoms.append(poly)
                            scores.append(float(sc[k]))
                            class_ids.append(cls_id)
                            class_labels.append(cls_label)
                            chip_ids.append(rec.get("name", tif_path.stem))
                            per_chip_geoms.append(poly)
                            per_chip_scores.append(float(sc[k]))
                            per_chip_class_ids.append(cls_id)
                            per_chip_class_labels.append(cls_label)
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
                    {
                        "score": per_chip_scores,
                        "class_id": per_chip_class_ids,
                        "class": per_chip_class_labels,
                    },
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
            gdf = gpd.GeoDataFrame(
                {
                    "score": scores,
                    "class_id": class_ids,
                    "class": class_labels,
                    "chip_id": chip_ids,
                },
                geometry=geoms,
                crs=crs,
            )
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
