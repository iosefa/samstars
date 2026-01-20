#!/usr/bin/env python3
"""
Train Mask R-CNN on chip GeoTIFFs + GPKG crowns with class labels.
Uses all raster bands plus CHM as an extra channel.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import math
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# ---- defaults ----
SEED = 42
IMS_PER_BATCH = 2
BASE_LR = 0.00025
MAX_ITERS = 200
WARMUP_ITERS = 20
BATCH_SIZE_PER_IMAGE = 128
NUM_WORKERS = 2
SCORE_THRESH_TEST = 0.5


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    chm8 = robust_norm255(chm)
    bands_u8.append(chm8)
    return np.stack(bands_u8, axis=-1).astype(np.uint8)


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


def count_instances_from_manifest(manifest_path: Path) -> int:
    data = json.load(open(manifest_path))
    n_inst = 0
    for rec in data:
        gpkg = Path(rec["gpkg_path"])
        if gpkg.exists():
            gdf = gpd.read_file(gpkg)
            n_inst += len(gdf)
    return n_inst


def mapper_train(dataset_dict, chm_suffix: str):
    d = copy.deepcopy(dataset_dict)
    img = read_multiband_with_chm(Path(d["file_name"]), chm_suffix=chm_suffix)
    aug_input = T.AugInput(img)
    transforms = [
        T.RandomFlip(horizontal=True, vertical=False, prob=0.5),
        T.RandomFlip(horizontal=False, vertical=True, prob=0.5),
    ]
    applied = [aug(aug_input) for aug in transforms]
    aug_img = np.ascontiguousarray(aug_input.image)
    annos = [a for a in d.get("annotations", []) if a.get("iscrowd", 0) == 0]
    annos = [utils.transform_instance_annotations(a, applied, aug_img.shape[:2]) for a in annos]
    d["image"] = torch.as_tensor(aug_img.transpose(2, 0, 1))
    d["instances"] = utils.annotations_to_instances(annos, image_size=aug_img.shape[:2], mask_format="polygon")
    d.pop("annotations", None)
    return d


def mapper_test(dataset_dict, chm_suffix: str):
    d = copy.deepcopy(dataset_dict)
    img = read_multiband_with_chm(Path(d["file_name"]), chm_suffix=chm_suffix)
    img = np.ascontiguousarray(img)
    d["image"] = torch.as_tensor(img.transpose(2, 0, 1))
    annos = [a for a in d.get("annotations", []) if a.get("iscrowd", 0) == 0]
    d["instances"] = utils.annotations_to_instances(annos, image_size=img.shape[:2], mask_format="polygon")
    d.pop("annotations", None)
    return d


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        chm_suffix = str(getattr(cfg.INPUT, "CHM_SUFFIX", "_chm.tif"))
        return build_detection_train_loader(cfg, mapper=lambda d: mapper_train(d, chm_suffix))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        chm_suffix = str(getattr(cfg.INPUT, "CHM_SUFFIX", "_chm.tif"))
        return build_detection_test_loader(cfg, dataset_name, mapper=lambda d: mapper_test(d, chm_suffix))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = str(Path(cfg.OUTPUT_DIR) / "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def write_metrics_csv(metrics, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def main():
    pa = argparse.ArgumentParser(description="Train Detectron2 (multiclass, all bands + CHM).")
    pa.add_argument("--train-manifest", type=Path, required=True, help="detectron_manifest.json for train chips")
    pa.add_argument("--val-manifest", type=Path, default=None, help="detectron_manifest.json for val chips")
    pa.add_argument("--output-dir", type=Path, required=True)
    pa.add_argument("--holdout", type=float, default=0.0, help="Holdout fraction if --val-manifest is not set")
    pa.add_argument("--split-seed", type=int, default=SEED)
    pa.add_argument("--max-iters", type=int, default=MAX_ITERS)
    pa.add_argument("--epochs", type=float, default=None, help="If set, derive max-iters from epochs and train size")
    pa.add_argument("--min-iters", type=int, default=0, help="Minimum training iterations when using --epochs")
    pa.add_argument("--ims-per-batch", type=int, default=IMS_PER_BATCH)
    pa.add_argument("--base-lr", type=float, default=BASE_LR)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--skip-val-eval", action="store_true", help="Skip validation split and COCO eval")
    pa.add_argument("--quick", action="store_true", help="Override settings for a fast smoke test")
    pa.add_argument("--metrics-out", type=Path, default=None)
    pa.add_argument("--class-field", type=str, default="class")
    pa.add_argument("--chm-suffix", type=str, default="_chm.tif")
    args = pa.parse_args()

    set_seed(SEED)

    train_records = None
    if args.val_manifest:
        class_manifests = [args.train_manifest, args.val_manifest]
    else:
        class_manifests = [args.train_manifest]

    classes = _collect_classes(class_manifests, args.class_field)
    class_to_id = {c: i for i, c in enumerate(classes)}
    class_mapping = {
        "classes": classes,
        "class_to_id": {str(k): int(v) for k, v in class_to_id.items()},
    }

    train_records = build_records(args.train_manifest, class_to_id, args.class_field)
    if not train_records:
        raise RuntimeError("No train records found.")

    if args.skip_val_eval:
        val_records = []
    elif args.val_manifest:
        val_records = build_records(args.val_manifest, class_to_id, args.class_field)
    elif args.holdout > 0.0:
        rng = args.split_seed
        random.seed(rng)
        random.shuffle(train_records)
        k = int(round(len(train_records) * (1.0 - args.holdout)))
        val_records = train_records[k:]
        train_records = train_records[:k]
    else:
        val_records = []

    DatasetCatalog.register("chips_train_mc", lambda: train_records)
    MetadataCatalog.get("chips_train_mc").set(thing_classes=[str(c) for c in classes])
    DatasetCatalog.register("chips_val_mc", lambda: val_records or train_records)
    MetadataCatalog.get("chips_val_mc").set(thing_classes=[str(c) for c in classes])

    # Determine channel count (all bands + CHM)
    sample_tif = Path(train_records[0]["file_name"])
    with rasterio.open(sample_tif) as src:
        n_bands = src.count
    n_channels = n_bands + 1

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.INPUT.CHM_SUFFIX = str(args.chm_suffix)
    cfg.DATASETS.TRAIN = ("chips_train_mc",)
    cfg.DATASETS.TEST = ("chips_val_mc",)
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    if args.quick:
        args.max_iters = min(args.max_iters, 200)
        args.ims_per_batch = min(args.ims_per_batch, 1)
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    if args.device == "cpu":
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.DATALOADER.NUM_WORKERS = 0

    if args.epochs is not None:
        iters_per_epoch = math.ceil(len(train_records) / max(cfg.SOLVER.IMS_PER_BATCH, 1))
        derived = int(math.ceil(iters_per_epoch * args.epochs))
        if args.min_iters > 0:
            cfg.SOLVER.MAX_ITER = max(args.min_iters, derived)
        else:
            cfg.SOLVER.MAX_ITER = max(1, derived)
        print(f"✓ epochs={args.epochs} -> max_iters={cfg.SOLVER.MAX_ITER} (iters/epoch={iters_per_epoch})")
    else:
        cfg.SOLVER.MAX_ITER = args.max_iters
    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    cfg.TEST.EVAL_PERIOD = 500 if (val_records or args.holdout > 0.0) else 0
    cfg.OUTPUT_DIR = str(args.output_dir)
    cfg.MODEL.DEVICE = args.device

    cfg.MODEL.RESNETS.STEM_IN_CHANNELS = n_channels
    cfg.MODEL.PIXEL_MEAN = [0.0] * n_channels
    cfg.MODEL.PIXEL_STD = [1.0] * n_channels

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    json.dump(class_mapping, open(Path(cfg.OUTPUT_DIR) / "class_mapping.json", "w"))

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    src_final = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    final_weights = Path(cfg.OUTPUT_DIR) / "maskrcnn_crowns_model_final.pth"
    if src_final.exists():
        src_final.replace(final_weights)

    if not args.skip_val_eval and (val_records or args.holdout > 0.0):
        evaluator = COCOEvaluator("chips_val_mc", output_dir=str(Path(cfg.OUTPUT_DIR) / "inference"))
        val_loader = build_detection_test_loader(cfg, "chips_val_mc", mapper=lambda d: mapper_test(d, args.chm_suffix))
        metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
        metrics_json = args.metrics_out or Path(cfg.OUTPUT_DIR) / "metrics.json"
        json.dump(metrics, open(metrics_json, "w"))
        write_metrics_csv(metrics, metrics_json.with_suffix(".csv"))

    print(f"✓ saved weights to {final_weights}")


if __name__ == "__main__":
    main()
