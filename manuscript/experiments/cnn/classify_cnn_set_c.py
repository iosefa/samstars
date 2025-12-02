#!/usr/bin/env python3
import os, sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import laspy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Add repo root for local imports if needed
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyforestscan.handlers import read_lidar


# ------------------
# Config (edit or pass via CLI later if needed)
# ------------------
DEFAULT_CROWNS_GPKG = "manuscript/data/set_c/set_c_cnn_predicted.gpkg"
DEFAULT_IMAGE_TIF   = "manuscript/data/set_c/set_c.tif.tif"  # orthomosaic covering Set C
DEFAULT_EPT_JSON    = "manuscript/data/set_c/ept/ept.json"  # EPT for lidar
DEFAULT_EPT_SRS     = "EPSG:32605"

# Model files (dual-encoder CNN+point MLP)
MODEL_PTH  = "/Users/iosefa/repos/sam/manuscript/experiments/cnn/02_dual_encoder_cnn/cnn_crowns_resnet34_mlp_pruned_ft1.pth"
MODEL_META = "/Users/iosefa/repos/sam/manuscript/results/cnn_crowns_resnet34_mlp_pruned_ft1.meta.json"

OUT_GPKG   = "manuscript/data/set_c/set_c_cnn_predicted_classified.gpkg"
OUT_LAYER  = "cnn_predicted"


class CombinedNet(nn.Module):
    def __init__(self, num_classes, freeze_resnet=False, resnet="resnet34", dual_pool=True):
        super().__init__()
        if resnet == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_resnet:
            for p in base.parameters():
                p.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.dual_pool = dual_pool
        pts_in = 512 if dual_pool else 256
        self.fc4 = nn.Linear(pts_in, 128)
        self.fc5 = nn.Linear(128, 64)
        self.classifier = nn.Linear(512 + 64, num_classes)

    def forward(self, img, pts):
        x_img = self.backbone(img).view(img.size(0), -1)
        h = F.relu(self.fc1(pts))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        if self.dual_pool:
            h_max, _ = torch.max(h, dim=1)
            h_mean = h.mean(dim=1)
            h = torch.cat([h_max, h_mean], dim=1)
        else:
            h, _ = torch.max(h, dim=1)
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        x = torch.cat([x_img, h], dim=1)
        return self.classifier(x)


def read_image_patch_masked(img_path: str, geom, band_sel_9, band_sel_8, img_size: int):
    # Mask image by polygon and crop
    with rasterio.open(img_path) as src:
        # read window and mask to polygon, preserve masked array
        out_ma, _ = rio_mask(src, [geom.__geo_interface__], crop=True, filled=False)
        arr = out_ma.astype("float32")  # (C,H,W)

    # Select 3 bands
    if arr.shape[0] == 9 and band_sel_9:
        arr = arr[band_sel_9, :, :]
    elif arr.shape[0] == 8 and band_sel_8:
        arr = arr[band_sel_8, :, :]
    elif arr.shape[0] >= 3:
        arr = arr[:3, :, :]
    else:
        raise ValueError(f"Unexpected band count {arr.shape[0]} in masked chip")

    # Robust per-band normalization (2–98 percentile), treat masked as NaN
    img2 = []
    for c in range(arr.shape[0]):
        band = arr[c]
        if isinstance(band, np.ma.MaskedArray):
            band = band.filled(np.nan)
        finite = np.isfinite(band)
        if not np.any(finite):
            band = np.zeros_like(band, dtype=np.float32)
        else:
            lo, hi = np.nanpercentile(band[finite], [2, 98])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0
            band = np.clip(band, lo, hi)
            band = (band - lo) / max(hi - lo, 1e-6)
        img2.append(band.astype(np.float32))
    img = np.stack(img2, axis=0)

    # Resize to model input
    t = torch.from_numpy(img).unsqueeze(0)  # (1,3,H,W)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = (t - mean) / std
    return t.float()


def read_points_from_ept(poly_geom, ept_json: str, ept_srs: str, max_points: int):
    # Read LAS points from EPT within polygon bounds; returns fixed-length tensor
    minx, miny, maxx, maxy = poly_geom.bounds
    bounds = ([minx, maxx], [miny, maxy])
    pc = read_lidar(ept_json, ept_srs, bounds, crop_poly=True, poly=poly_geom.wkt)
    if pc is None or (hasattr(pc, "__len__") and len(pc) == 0):
        pts = np.zeros((0, 3), dtype=np.float32)
    else:
        # pc may be numpy structured or list; extract X,Y,Z
        try:
            X = pc['X'].astype(float); Y = pc['Y'].astype(float); Z = pc['Z'].astype(float)
            pts = np.vstack([X, Y, Z]).T.astype(np.float32)
        except Exception:
            arr = np.asarray(pc)
            if arr.size == 0:
                pts = np.zeros((0, 3), dtype=np.float32)
            else:
                pts = arr.astype(np.float32)

    # Normalize points (center XY; scale; standardize Z)
    if pts.shape[0] == 0:
        ptsn = np.zeros((max_points, 3), dtype=np.float32)
    else:
        xy = pts[:, :2]; z = pts[:, 2:3]
        xy_c = xy - xy.mean(axis=0, keepdims=True)
        std = xy_c.std(axis=0, keepdims=True) + 1e-6
        xy_n = xy_c / std
        z_n = (z - z.mean()) / (z.std() + 1e-6)
        ptsn = np.hstack([xy_n, z_n]).astype(np.float32)
        # pad or sample to max_points
        n = ptsn.shape[0]
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            ptsn = ptsn[idx, :]
        elif n < max_points:
            add = np.random.choice(n, max_points - n, replace=True) if n > 0 else []
            ptsn = np.vstack([ptsn, ptsn[add, :]]) if n > 0 else np.zeros((max_points, 3), dtype=np.float32)
    return torch.from_numpy(ptsn)


def main():
    # Load model meta
    with open(MODEL_META, "r") as f:
        meta = json.load(f)
    num_classes = int(meta["num_classes"])
    band_sel_9 = meta.get("band_selection_9")
    band_sel_8 = meta.get("band_selection_8")
    img_size   = int(meta.get("img_size", 224))
    max_points = int(meta.get("max_points", 2048))

    # Build model; infer backbone from filename, allow meta override; handle dual_pool
    fname = Path(MODEL_PTH).name.lower()
    backbone = "resnet34" if "resnet34" in fname else ("resnet18" if "resnet18" in fname else meta.get("resnet", "resnet18"))
    dual = bool(meta.get("dual_pool", False))
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CombinedNet(num_classes=num_classes, freeze_resnet=meta.get("freeze_resnet", False), resnet=backbone, dual_pool=dual).to(device)
    state = torch.load(MODEL_PTH, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        if "fc4.weight" in str(e) and "[128, 512]" in str(e):
            model = CombinedNet(num_classes=num_classes, freeze_resnet=meta.get("freeze_resnet", False), resnet=backbone, dual_pool=True).to(device)
            model.load_state_dict(state)
        else:
            raise
    model.eval()

    # Load crowns and ensure CRS for both image and EPT contexts
    crowns = gpd.read_file(DEFAULT_CROWNS_GPKG)
    if crowns.crs is None:
        raise RuntimeError("Crowns GPKG lacks CRS.")
    # Prepare versions in image CRS and EPT CRS
    with rasterio.open(DEFAULT_IMAGE_TIF) as src:
        img_crs = src.crs
    crowns_img = crowns.to_crs(img_crs)
    crowns_ept = crowns.to_crs(DEFAULT_EPT_SRS)

    pred_ids, prob_rows = [], []
    # Iterate crowns and classify
    with torch.no_grad():
        for (geom_img, geom_ept) in zip(crowns_img.geometry, crowns_ept.geometry):
            if geom_img is None or geom_img.is_empty or geom_ept is None or geom_ept.is_empty:
                pred_ids.append(None); prob_rows.append([np.nan]*num_classes); continue
            try:
                img_t = read_image_patch_masked(DEFAULT_IMAGE_TIF, geom_img, band_sel_9, band_sel_8, img_size).to(device)
            except Exception:
                pred_ids.append(None); prob_rows.append([np.nan]*num_classes); continue
            pts_t = read_points_from_ept(geom_ept, DEFAULT_EPT_JSON, DEFAULT_EPT_SRS, max_points).to(device)
            img_t = img_t.unsqueeze(0)
            pts_t = pts_t.unsqueeze(0)
            logits = model(img_t, pts_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            pred_ids.append(pred)
            prob_rows.append(probs.tolist())

    # Map model id back to raw label using meta.class_map inverse
    raw_map = meta.get("class_map", {})
    class_map = {int(k): int(v) for k, v in raw_map.items()}  # raw->model id
    inv_map = {v: k for k, v in class_map.items()}            # model id->raw label
    predicted_class_raw = [int(inv_map.get(i, i)) if i is not None else None for i in pred_ids]

    out = crowns.copy()
    out["pred_id"] = pred_ids
    out["predicted_class"] = predicted_class_raw
    # attach probabilities
    for k in range(num_classes):
        out[f"prob_{k}"] = [row[k] if row and len(row) == num_classes else np.nan for row in prob_rows]

    # Write output
    out_path = Path(OUT_GPKG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try: out_path.unlink()
        except Exception: pass
    out.to_file(out_path, layer=OUT_LAYER, driver="GPKG")
    print(f"[OK] Wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()

