#!/usr/bin/env python3
"""
CNN dual-encoder training (v2): validation split, class balancing, dataset norms, no class pruning.

Outputs
- Weights:   <OUT_DIR>/<RUN_NAME>.best.pth (best val PQ proxy / accuracy)
- Metadata:  manuscript/results/<RUN_NAME>.meta.json

Notes
- Keeps all observed raw labels; builds a contiguous class_map but records the inverse map.
- Uses dataset-specific mean/std for 3 bands selected via BAND_SELECTION_8/9 (or first 3).
- Uses focal loss or class-weighted CE; basic image augmentations; dual-pool option for point MLP.
"""

import os, glob, json, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import laspy
from shapely.geometry import box

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# ------------- Config -------------
TRAINING_GPKG = Path("/Users/iosefa/repos/sam/trainings_filtered.gpkg")
IMG_DIR       = Path("/Users/iosefa/repos/sam/manuscript/data/set_b/crowns_img")
LAS_DIR       = Path("/Users/iosefa/repos/sam/manuscript/data/set_b/crowns_las")

OUT_DIR   = Path("manuscript/experiments/cnn/02_dual_encoder_cnn")
RESULTS_DIR = Path("manuscript/results")
RUN_NAME  = "cnn_crowns_resnet34_dual_v2"

BACKBONE = "resnet34"  # resnet18 or resnet34
DUAL_POOL = True        # if True, use max+mean pooling for point MLP
FREEZE_RESNET = False

IMG_SIZE = 224
MAX_POINTS = 2048
BATCH_SIZE = 32
EPOCHS = 40
LR_BACKBONE = 5e-5
LR_HEAD = 1.5e-4
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Loss
USE_FOCAL = True
FOCAL_GAMMA = 1.5
LABEL_SMOOTH = 0.02

# Bands
BAND_SELECTION_9 = [4, 2, 1]
BAND_SELECTION_8 = [6, 4, 2]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def read_image_as_3band_tensor(tif_path: Path, img_size: int, band_sel_9: List[int], band_sel_8: List[int],
                               mean=None, std=None, augment=False):
    with rasterio.open(tif_path) as src:
        img = src.read()  # (C,H,W)
    img = img.astype(np.float32)
    if img.shape[0] == 9 and band_sel_9:
        img = img[band_sel_9, :, :]
    elif img.shape[0] == 8 and band_sel_8:
        img = img[band_sel_8, :, :]
    elif img.shape[0] >= 3:
        img = img[:3, :, :]
    else:
        raise ValueError(f"Unexpected band count {img.shape[0]} in {tif_path}")

    # 2–98% per-channel normalization then standardize to dataset mean/std
    img2 = []
    for c in range(img.shape[0]):
        band = img[c]
        finite = np.isfinite(band)
        if not np.any(finite):
            band = np.zeros_like(band, dtype=np.float32)
        else:
            lo, hi = np.percentile(band[finite], [2, 98])
            if hi <= lo:
                lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0
            band = np.clip(band, lo, hi)
            band = (band - lo) / max(hi - lo, 1e-6)
        img2.append(band.astype(np.float32))
    img = np.stack(img2, axis=0)

    # Simple augmentations: horizontal/vertical flip
    if augment:
        if random.random() < 0.5:
            img = img[:, :, ::-1]
        if random.random() < 0.5:
            img = img[:, ::-1, :]

    # Ensure contiguous array (flips can create negative strides)
    img = np.ascontiguousarray(img)
    t = torch.from_numpy(img).unsqueeze(0)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)

    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    else:
        mean = torch.tensor(mean).view(3,1,1)
        std  = torch.tensor(std).view(3,1,1)
    t = (t - mean) / (std + 1e-6)
    return t.float()


def read_las_points(las_path: Path):
    with laspy.open(las_path) as f:
        las = f.read()
        pts = np.vstack([las.x, las.y, las.z]).T
    return pts


def normalize_points_xy_center(points: np.ndarray):
    xy = points[:, :2]
    z  = points[:, 2:3]
    xy_c = xy - xy.mean(axis=0, keepdims=True)
    std = xy_c.std(axis=0, keepdims=True) + 1e-6
    xy_n = xy_c / std
    z_n = (z - z.mean()) / (z.std() + 1e-6)
    return np.hstack([xy_n, z_n])


def pad_or_sample_points(points: np.ndarray, n=MAX_POINTS):
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if n_pts > n:
        idx = np.random.choice(n_pts, n, replace=False)
        return points[idx, :].astype(np.float32)
    if n_pts < n:
        add = np.random.choice(n_pts, n - n_pts, replace=True)
        return np.vstack([points, points[add, :]]).astype(np.float32)
    return points.astype(np.float32)


class CrownDataset(Dataset):
    def __init__(self, recs, class_map, mean=None, std=None, augment=False):
        self.recs = recs
        self.class_map = class_map
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        tif_path, las_path, y_raw = self.recs[idx]
        y = self.class_map[y_raw]
        img = read_image_as_3band_tensor(tif_path, IMG_SIZE, BAND_SELECTION_9, BAND_SELECTION_8,
                                         mean=self.mean, std=self.std, augment=self.augment)
        pts = read_las_points(las_path)
        pts = normalize_points_xy_center(pts)
        pts = pad_or_sample_points(pts, MAX_POINTS)
        pts = torch.from_numpy(pts)
        return img, pts, y


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', weight=self.weight, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def stratified_split(records: List[Tuple[str,str,int]], val_ratio=0.1, seed=42):
    by_label = {}
    for r in records:
        by_label.setdefault(r[2], []).append(r)
    rng = random.Random(seed)
    train, val = [], []
    for lbl, recs in by_label.items():
        rng.shuffle(recs)
        k = max(1, int(len(recs) * val_ratio))
        val.extend(recs[:k])
        train.extend(recs[k:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def compute_dataset_mean_std(tif_paths: List[Path], sample_max=300):
    rng = random.Random(RANDOM_SEED)
    sample = tif_paths if len(tif_paths) <= sample_max else rng.sample(tif_paths, sample_max)
    means, vars_ = [], []
    for p in sample:
        with rasterio.open(p) as src:
            img = src.read()
        if img.shape[0] == 9 and BAND_SELECTION_9:
            img = img[BAND_SELECTION_9, :, :]
        elif img.shape[0] == 8 and BAND_SELECTION_8:
            img = img[BAND_SELECTION_8, :, :]
        elif img.shape[0] >= 3:
            img = img[:3, :, :]
        img = img.astype(np.float32)
        finite = np.isfinite(img)
        m = np.nanmean(np.where(finite, img, np.nan), axis=(1,2))
        v = np.nanvar(np.where(finite, img, np.nan), axis=(1,2))
        means.append(m)
        vars_.append(v)
    mean = np.mean(np.stack(means, 0), 0).tolist()
    std  = np.sqrt(np.mean(np.stack(vars_, 0), 0)).tolist()
    return mean, std


def main():
    # labels gpkg
    gdf = gpd.read_file(TRAINING_GPKG)
    gdf["feature_class"] = pd.to_numeric(gdf["class"], errors='coerce').astype('Int64')
    gdf = gdf.dropna(subset=["feature_class"]).copy()

    # collect recs from IMG_DIR/LAS_DIR
    tifs = sorted(IMG_DIR.glob("*.tif"))
    lass = sorted(list(LAS_DIR.glob("*.las")) + list(LAS_DIR.glob("*.laz")))
    tmap = {p.stem: p for p in tifs}
    lmap = {p.stem: p for p in lass}
    keys = sorted(set(tmap.keys()).intersection(lmap.keys()))
    recs = []
    for k in keys:
        tifp = tmap[k]
        lasp = lmap[k]
        # match label by geometry intersection with chip bounds
        with rasterio.open(tifp) as src:
            left, bottom, right, top = src.bounds
            crs = src.crs
        chip = gpd.GeoDataFrame({"key":[k]}, geometry=[box(left,bottom,right,top)], crs=crs)
        gg = gdf if gdf.crs == chip.crs else gdf.to_crs(chip.crs)
        hits = gg[gg.geometry.intersects(chip.geometry.iloc[0])]
        if hits.empty:
            continue
        inter_areas = hits.geometry.intersection(chip.geometry.iloc[0]).area
        idx = inter_areas.idxmax()
        y = int(hits.loc[idx, "feature_class"])  # keep as raw label
        recs.append((str(tifp), str(lasp), y))

    if not recs:
        raise RuntimeError("No training records matched.")

    raw_labels = sorted({y for _,_,y in recs})
    class_map = {lbl:i for i,lbl in enumerate(raw_labels)}
    inv_map = {i:lbl for lbl,i in class_map.items()}

    # stratified split
    train_recs, val_recs = stratified_split(recs, val_ratio=VAL_SPLIT, seed=RANDOM_SEED)

    # dataset mean/std from training images
    mean, std = compute_dataset_mean_std([Path(r[0]) for r in train_recs], sample_max=300)

    # datasets/loaders
    ds_tr = CrownDataset(train_recs, class_map, mean=mean, std=std, augment=True)
    ds_va = CrownDataset(val_recs,   class_map, mean=mean, std=std, augment=False)

    def get_device():
        if torch.backends.mps.is_available(): return torch.device('mps')
        if torch.cuda.is_available(): return torch.device('cuda')
        return torch.device('cpu')

    device = get_device()
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device.type=='cuda'))
    loader_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device.type=='cuda'))

    # model
    model = CombinedNet(num_classes=len(class_map), freeze_resnet=FREEZE_RESNET, resnet=BACKBONE, dual_pool=DUAL_POOL).to(device)
    # optim with different LRs
    params = [
        {"params": [p for n,p in model.named_parameters() if n.startswith('backbone') and p.requires_grad], "lr": LR_BACKBONE},
        {"params": [p for n,p in model.named_parameters() if (not n.startswith('backbone')) and p.requires_grad], "lr": LR_HEAD},
    ]
    optim = torch.optim.Adam(params)

    # class weights
    counts = np.bincount([class_map[y] for _,_,y in train_recs], minlength=len(class_map))
    weights = counts.astype(np.float32)
    weights = weights.max() / np.clip(weights, 1, None)
    w = torch.tensor(weights, dtype=torch.float32, device=device)

    if USE_FOCAL:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=w, label_smoothing=LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)

    best_acc = -1.0
    best_path = OUT_DIR / f"{RUN_NAME}.best.pth"

    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for img, pts, y in loader_tr:
            img, pts, y = img.to(device), pts.to(device), y.to(device)
            optim.zero_grad()
            logits = model(img, pts)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * y.size(0)
            tr_corr += (logits.argmax(1) == y).sum().item()
            tr_tot  += y.size(0)

        # val
        model.eval()
        va_loss, va_corr, va_tot = 0.0, 0, 0
        with torch.no_grad():
            for img, pts, y in loader_va:
                img, pts, y = img.to(device), pts.to(device), y.to(device)
                logits = model(img, pts)
                loss = criterion(logits, y)
                va_loss += loss.item() * y.size(0)
                va_corr += (logits.argmax(1) == y).sum().item()
                va_tot  += y.size(0)

        tr_loss /= max(tr_tot,1)
        tr_acc  = tr_corr / max(tr_tot,1)
        va_loss /= max(va_tot,1)
        va_acc  = va_corr / max(va_tot,1)
        print(f"Epoch {epoch:02d}/{EPOCHS} | tr_loss {tr_loss:.4f} acc {tr_acc:.3f} | va_loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # write meta
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "class_map": {int(k): int(v) for k,v in class_map.items()},
        "inverse_class_map": {int(k): int(v) for k,v in inv_map.items()},
        "raw_labels_kept": [int(x) for x in raw_labels],
        "dropped_raw_labels": [],
        "num_classes": len(class_map),
        "img_size": IMG_SIZE,
        "max_points": MAX_POINTS,
        "band_selection_9": BAND_SELECTION_9,
        "band_selection_8": BAND_SELECTION_8,
        "norm_mean": [float(x) for x in mean],
        "norm_std": [float(x) for x in std],
        "freeze_resnet": FREEZE_RESNET,
        "architecture": f"{BACKBONE}+pointMLP({'dual' if DUAL_POOL else 'max'})",
        "weights": "IMAGENET1K_V1",
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "label_smoothing": LABEL_SMOOTH,
        "use_focal": USE_FOCAL,
        "focal_gamma": FOCAL_GAMMA,
        "expected_band_idx": BAND_SELECTION_9 if BAND_SELECTION_9 else BAND_SELECTION_8,
        "note": "v2 training: no class pruning; stratified val; dataset norms; focal/weights; basic aug"
    }
    with open(RESULTS_DIR / f"{RUN_NAME}.meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved best weights → {best_path}")
    print(f"[OK] Saved meta → {RESULTS_DIR / f'{RUN_NAME}.meta.json'}")


if __name__ == "__main__":
    main()
