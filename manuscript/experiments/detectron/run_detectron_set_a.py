#!/usr/bin/env python3
"""
Batch Mask R-CNN inference over manuscript/data/set_a/*.tif and write a single
GeoPackage with all detected crown polygons.

This reuses the same tiling + polygonization logic as run_detectron.py, but
loops over all Set A tiles and merges results into one layer.

Edit the CONFIG section below if paths/parameters differ.
"""
import os
from pathlib import Path
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.engine import DefaultPredictor


# -----------------------------
# CONFIG (adjust if needed)
# -----------------------------
WEIGHTS_PTH   = "manuscript/experiments/cnn/01_mask_rcnn/maskrcnn_crowns_model_final.pth"
INPUT_DIR     = Path("manuscript/data/set_a")
INPUT_GLOB    = "setA_plot[1-6]_ms.tif"
OUTPUT_GPKG   = INPUT_DIR / "detectron_crowns.gpkg"
OUTPUT_LAYER  = "crowns_pred"

# Bands to use (1-indexed WV3): NIR,R,G = 7,5,3
BANDS_753     = (7, 5, 3)

# Sliding window + thresholds
TILE_SIZE     = 1024        # pixels per side
TILE_OVERLAP  = 128         # pixels overlap on each side
SCORE_THRESH  = 0.5         # drop low-confidence instances
MIN_MASK_PX   = 50          # drop tiny masks (in pixels)
NMS_IOU       = 0.5         # IoU threshold for cross-tile dedup (on boxes)

# Device: "cpu" recommended on Mac unless CUDA available
DEVICE        = "cpu"


# -----------------------------
# HELPERS (mirroring run_detectron.py)
# -----------------------------
def robust_norm255(band: np.ndarray) -> np.ndarray:
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


def read_tile_rgb(src, row_off: int, col_off: int, size: int, band_idx=BANDS_753):
    h = min(size, src.height - row_off)
    w = min(size, src.width  - col_off)
    if h <= 0 or w <= 0:
        return None, None
    win = Window(col_off, row_off, w, h)
    nir = src.read(band_idx[0], window=win).astype(np.float32)
    red = src.read(band_idx[1], window=win).astype(np.float32)
    grn = src.read(band_idx[2], window=win).astype(np.float32)
    nir8, red8, grn8 = robust_norm255(nir), robust_norm255(red), robust_norm255(grn)
    rgb = np.dstack([red8, grn8, nir8]).astype(np.uint8)  # (H,W,3) RGB order
    return rgb, win


def tile_indices(width: int, height: int, size: int, overlap: int):
    step = size - overlap
    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            yield int(row_off), int(col_off)


def boxes_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    union = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter/union, 0.0)


def nms_global(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5):
    idxs = np.argsort(-scores)
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = boxes_iou_xyxy(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thr]
    return keep


def mask_to_polygons(mask_bool: np.ndarray, transform) -> list:
    if mask_bool.sum() < MIN_MASK_PX:
        return []
    out = []
    for geom, val in features.shapes(mask_bool.astype(np.uint8), mask=mask_bool, transform=transform):
        if val == 0:
            continue
        poly = shape(geom)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        out.append(poly)
    return out


def build_predictor(weights_path: str, score_thresh: float = SCORE_THRESH, device: str = DEVICE):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    # Increase test-time cap; default is 100 per call (tile). We tile the image
    # and apply a global NMS later, so allow more per tile to avoid early truncation.
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = 0
    return DefaultPredictor(cfg)


def infer_one_image(tif_path: Path, predictor: DefaultPredictor):
    geoms = []
    scores_all = []
    areas_m2 = []
    tile_ids = []
    img_id = tif_path.stem

    with rasterio.open(tif_path) as src:
        crs = src.crs
        H, W = src.height, src.width

        all_boxes = []
        all_scores = []
        all_polys = []

        for t, (row_off, col_off) in enumerate(tile_indices(W, H, TILE_SIZE, TILE_OVERLAP), start=1):
            rgb, win = read_tile_rgb(src, row_off, col_off, TILE_SIZE, BANDS_753)
            if rgb is None:
                continue
            outputs = predictor(rgb)
            instances = outputs["instances"].to("cpu")
            if len(instances) == 0:
                continue

            # tile-local → global pixel coords for boxes
            boxes = instances.pred_boxes.tensor.numpy()  # (N,4)
            boxes[:, [0,2]] += col_off
            boxes[:, [1,3]] += row_off
            scores = instances.scores.numpy()
            masks  = instances.pred_masks.numpy()        # (N,h,w)

            win_transform = src.window_transform(win)

            for k in range(masks.shape[0]):
                if scores[k] < SCORE_THRESH:
                    continue
                polys = mask_to_polygons(masks[k].astype(bool), win_transform)
                if not polys:
                    continue
                poly = unary_union(polys)
                if poly.is_empty:
                    continue
                all_polys.append(poly)
                all_boxes.append(boxes[k].copy())
                all_scores.append(float(scores[k]))
                tile_ids.append(t)

        if not all_polys:
            return gpd.GeoDataFrame(columns=["score","area_m2","tile_id","image"], geometry=[], crs=crs)

        all_boxes = np.array(all_boxes, dtype=np.float32)
        all_scores = np.array(all_scores, dtype=np.float32)
        keep_idx = nms_global(all_boxes, all_scores, iou_thr=NMS_IOU)

        kept_geoms = []
        kept_scores= []
        kept_area  = []
        kept_tile  = []

        for i in keep_idx:
            g = all_polys[i]
            kept_geoms.append(g)
            kept_scores.append(all_scores[i])
            kept_area.append(float(g.area))
            kept_tile.append(tile_ids[i])

        gdf = gpd.GeoDataFrame({
            "score": kept_scores,
            "area_m2": kept_area,
            "tile_id": kept_tile,
            "image": img_id,
        }, geometry=kept_geoms, crs=crs)
        return gdf


def main():
    tif_paths = sorted(INPUT_DIR.glob(INPUT_GLOB))
    if not tif_paths:
        raise SystemExit(f"No TIFs found at {INPUT_DIR}/{INPUT_GLOB}")

    predictor = build_predictor(WEIGHTS_PTH, SCORE_THRESH, DEVICE)

    merged_gdf = None
    base_crs = None
    total = 0
    for p in tif_paths:
        print(f"[RUN] {p.name}")
        gdf = infer_one_image(p, predictor)
        if gdf.empty:
            print(f"  -> no detections")
            continue
        if merged_gdf is None:
            merged_gdf = gdf
            base_crs = gdf.crs
        else:
            # reproject if needed
            if gdf.crs and base_crs and gdf.crs != base_crs:
                gdf = gdf.to_crs(base_crs)
            merged_gdf = gpd.pd.concat([merged_gdf, gdf], ignore_index=True)
        total += len(gdf)
        print(f"  -> kept {len(gdf)} crowns")

    if merged_gdf is None or merged_gdf.empty:
        print("No crowns detected in any image; nothing to write.")
        return

    out_path = Path(OUTPUT_GPKG)
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    merged_gdf.to_file(out_path, layer=OUTPUT_LAYER, driver="GPKG")
    print(f"[OK] Wrote {out_path} | crowns={len(merged_gdf)} | images={len(tif_paths)}")


if __name__ == "__main__":
    main()
