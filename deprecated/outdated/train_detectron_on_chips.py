#!/usr/bin/env python3
"""
Train Mask R-CNN on chip subsets produced by v2/build_chips.py.
Supports pre-split train/val JSONs and deterministic holdout.
Writes metrics (train/val instance counts + COCO AP) to JSON/CSV.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.structures import BoxMode

ROOT = Path(__file__).resolve().parent

# Hyperparams
SEED = 42
IMS_PER_BATCH = 2
BASE_LR = 0.00025
MAX_ITERS = 10000
WARMUP_ITERS = 500
BATCH_SIZE_PER_IMAGE = 128
NUM_WORKERS = 2
SCORE_THRESH_TEST = 0.5
HOLDOUT_FRACTION = 0.0
DEFAULT_RECORDS = ROOT / "chips" / "detectron_records.json"
DEFAULT_IMG_DIR = ROOT / "chips" / "img"

AUGS = [
    T.RandomFlip(horizontal=True, vertical=False, prob=0.5),
    T.RandomFlip(horizontal=False, vertical=True, prob=0.5),
    T.RandomBrightness(0.9, 1.1),
    T.RandomContrast(0.9, 1.1),
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed_all(seed)


def load_records(records_path: Path):
    data = json.load(open(records_path))
    for rec in data:
        for ann in rec["annotations"]:
            ann["bbox_mode"] = BoxMode.XYWH_ABS
            ann["category_id"] = 0
            ann["iscrowd"] = 0
    return data


def norm255(band):
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


def mapper(dataset_dict):
    d = dataset_dict.copy()
    img = np.load(d["file_name"]).astype(np.float32)
    if img.shape[0] >= 3:
        nir, red, grn = img[0], img[1], img[2]
    else:
        nir = red = grn = img[0]
    nir8, red8, grn8 = norm255(nir), norm255(red), norm255(grn)
    rgb = np.dstack([red8, grn8, nir8])
    aug_input = T.AugInput(rgb)
    transforms = [aug(aug_input) for aug in AUGS]
    aug_rgb = aug_input.image

    annos = [a for a in d.get("annotations", []) if a.get("iscrowd", 0) == 0]
    annos = [utils.transform_instance_annotations(a, transforms, aug_rgb.shape[:2]) for a in annos]

    d["image"] = torch.as_tensor(aug_rgb.transpose(2, 0, 1))
    d["instances"] = utils.annotations_to_instances(annos, image_size=aug_rgb.shape[:2], mask_format="polygon")
    d.pop("annotations", None)
    return d


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.evaluation import COCOEvaluator
        import os
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def count_instances(records):
    return sum(len(r.get("annotations", [])) for r in records)


def main():
    set_seed(SEED)
    import argparse
    pa = argparse.ArgumentParser(description="Train Detectron2 on chip subsets")
    pa.add_argument("--records", type=Path, default=DEFAULT_RECORDS, help="Path to train records JSON")
    pa.add_argument("--val-records", type=Path, default=None, help="Optional path to val records; if set, skips holdout split")
    pa.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR, help="Directory containing training chip npy images")
    pa.add_argument("--val-img-dir", type=Path, default=None, help="Directory containing validation chip npy images (defaults to --img-dir)")
    pa.add_argument("--output-dir", type=Path, default=ROOT / "detectron_chips_train_logs")
    pa.add_argument("--holdout", type=float, default=HOLDOUT_FRACTION, help="Holdout fraction for validation (chip-level)")
    pa.add_argument("--metrics-out", type=Path, default=None, help="Optional path to write metrics CSV/JSON (defaults under output-dir)")
    pa.add_argument("--split-seed", type=int, default=None, help="Seed for deterministic train/val split")
    args = pa.parse_args()

    train_records = load_records(args.records)
    if args.img_dir:
        for rec in train_records:
            rec["file_name"] = str(args.img_dir / Path(rec["file_name"]).name)

    if args.val_records:
        val_records = load_records(args.val_records)
        val_dir = args.val_img_dir if args.val_img_dir else args.img_dir
        if val_dir:
            for rec in val_records:
                rec["file_name"] = str(val_dir / Path(rec["file_name"]).name)
    else:
        val_records = None

    if not train_records:
        raise RuntimeError("No chip records found; run v2/build_chips.py first.")

    if val_records is None and args.holdout > 0.0:
        rng = args.split_seed if args.split_seed is not None else SEED
        random.seed(rng)
        random.shuffle(train_records)
        k = int(round(len(train_records) * (1.0 - args.holdout)))
        val_records = train_records[k:]
        train_records = train_records[:k]
    elif val_records is None:
        val_records = []

    DatasetCatalog.register("chips_train", lambda: train_records)
    MetadataCatalog.get("chips_train").set(thing_classes=["crown"])
    DatasetCatalog.register("chips_val", lambda: val_records or train_records)
    MetadataCatalog.get("chips_val").set(thing_classes=["crown"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.FORMAT = "BGR"
    cfg.DATASETS.TRAIN = ("chips_train",)
    cfg.DATASETS.TEST = ("chips_val",)
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = MAX_ITERS
    cfg.SOLVER.STEPS = [7000, 9000]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    cfg.TEST.EVAL_PERIOD = 500 if (val_records or args.holdout > 0.0) else 0
    cfg.OUTPUT_DIR = str(args.output_dir)
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    src_final = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    final_weights = Path(cfg.OUTPUT_DIR) / "maskrcnn_crowns_model_final.pth"
    if src_final.exists():
        src_final.replace(final_weights)
    print(f"✓ saved weights to {final_weights}")

    if val_records or args.holdout > 0.0:
        results = trainer.test(cfg, trainer.model)
        metrics_json = args.metrics_out or Path(cfg.OUTPUT_DIR) / "metrics.json"
        metrics_csv = metrics_json.with_suffix(".csv")
        train_count = count_instances(train_records)
        val_count = count_instances(val_records)
        # crude pixel-union IoU/Dice between union of predicted and GT masks on val set
        def compute_union_iou_dice(model, records):
            if not records:
                return {"iou": 0.0, "dice": 0.0}
            model.eval()
            ious, dices = [], []
            with torch.no_grad():
                for rec in records:
                    name = Path(rec["file_name"]).name
                    img = np.load(rec["file_name"]).astype(np.float32)
                    if img.shape[0] >= 3:
                        nir, red, grn = img[0], img[1], img[2]
                    else:
                        nir = red = grn = img[0]
                    nir8, red8, grn8 = norm255(nir), norm255(red), norm255(grn)
                    rgb = np.dstack([red8, grn8, nir8])
                    inputs = {
                        "image": torch.as_tensor(rgb.transpose(2, 0, 1)),
                        "height": rgb.shape[0],
                        "width": rgb.shape[1],
                    }
                    outputs = model([inputs])[0]
                    pred_mask = torch.zeros((rgb.shape[0], rgb.shape[1]), dtype=torch.float32)
                    if "instances" in outputs and outputs["instances"].has("pred_masks"):
                        pm = outputs["instances"].pred_masks
                        if pm.numel() > 0:
                            pred_mask = torch.clamp(pm.float().sum(0), 0, 1)
                    gt_mask = torch.zeros_like(pred_mask)
                    for ann in rec.get("annotations", []):
                        if "segmentation" in ann and len(ann["segmentation"]) > 0:
                            r = utils.annotations_to_instances([ann], image_size=rgb.shape[:2], mask_format="polygon")
                            if r.has("gt_masks"):
                                gm = None
                                # PolygonMasks has to_bitmasks(H,W); BitMasks has .tensor
                                if hasattr(r.gt_masks, "to_bitmasks"):
                                    bm = r.gt_masks.to_bitmasks(rgb.shape[0], rgb.shape[1])
                                    gm = bm.tensor if hasattr(bm, "tensor") else None
                                elif hasattr(r.gt_masks, "tensor"):
                                    gm = r.gt_masks.tensor
                                if gm is not None and gm.numel() > 0:
                                    gt_mask = torch.clamp(gt_mask + gm.float().sum(0), 0, 1)
                    inter = (pred_mask * gt_mask).sum().item()
                    union = (pred_mask + gt_mask - pred_mask * gt_mask).sum().item()
                    iou = inter / union if union > 0 else 0.0
                    dice = (2 * inter) / (pred_mask.sum().item() + gt_mask.sum().item()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
                    ious.append(iou); dices.append(dice)
            return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices))}

        pix_stats = compute_union_iou_dice(trainer.model, val_records)
        with open(metrics_json, "w") as f:
            json.dump({"train_instances": train_count, "val_instances": val_count, "metrics": results, **pix_stats}, f, indent=2)
        with open(metrics_csv, "w") as f:
            f.write("metric,value\n")
            f.write(f"train_instances,{train_count}\n")
            f.write(f"val_instances,{val_count}\n")
            f.write(f"iou,{pix_stats['iou']}\n")
            f.write(f"dice,{pix_stats['dice']}\n")
            for k, v in results.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        f.write(f"{k}.{kk},{vv}\n")
                else:
                    f.write(f"{k},{v}\n")
        print(f"✓ eval metrics written to {metrics_json} and {metrics_csv}")


if __name__ == "__main__":
    main()
