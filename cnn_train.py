import os, glob, warnings, random, json
from pathlib import Path

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

# -------------------------------
# CONFIG
# -------------------------------
TRAINING_GPKG = "/Users/iosefa/repos/sam/trainings_filtered.gpkg"
INDIVIDUAL_CROWNS_DIR = "/Users/iosefa/repos/sam/crowns_img"
INDIVIDUAL_CROWNS_POINTS_DIR = "/Users/iosefa/repos/sam/crowns_las"

# where to save the trained model + metadata
MODEL_PTH  = "/Users/iosefa/repos/sam/cnn_crowns_resnet18_mlp.pth"
MODEL_META = "/Users/iosefa/repos/sam/cnn_crowns_resnet18_mlp.meta.json"

MAX_POINTS = 2048
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
RANDOM_SEED = 42
FREEZE_RESNET = False    # set True to freeze image backbone if dataset is small

# Normalize bands: pick 3 bands (e.g., WV3 7,5,3)
# If your chips are already 3-band composites, set these to [0,1,2]
BAND_SELECTION_9 = [4, 2, 1]   # example if 9-band input
BAND_SELECTION_8 = [6, 4, 2]   # adjust for your stack (e.g., NIR, R, G). Change if needed.

# -------------------------------
# Reproducibility
# -------------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------------------
# 1) READ & MATCH TIF / LAS + attach crown geometry + 0-indexed label
# -------------------------------
gdf_labels = gpd.read_file(TRAINING_GPKG)

# Ensure integer type, then convert to 0-indexed
gdf_labels["feature_class"] = (
    pd.to_numeric(gdf_labels["class"], errors="coerce").astype("Int64") - 1
)
gdf_labels = gdf_labels.dropna(subset=["feature_class"])

# Glob both .las and .laz
tif_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_DIR, "*.tif")))
las_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_POINTS_DIR, "*.las")))

tif_dict = {Path(p).stem: p for p in tif_files}
las_dict = {Path(p).stem: p for p in las_files}
common_keys = sorted(set(tif_dict.keys()).intersection(las_dict.keys()))
if not common_keys:
    raise RuntimeError("No common *.tif/*.las keys found.")

records = []
geoms = []
labels = []
keys_ok = []

# Ensure CRS
if gdf_labels.crs is None or gdf_labels.crs.is_geographic:
    # default to EPSG:6635 if missing/geo; adjust if needed
    gdf_labels = gdf_labels.to_crs(6635)

for key in common_keys:
    tif_path = tif_dict[key]
    las_path = las_dict[key]

    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
        crs_tif = src.crs
    if crs_tif is None:
        warnings.warn(f"{tif_path} has no CRS; skipping.")
        continue

    chip_geom = box(left, bottom, right, top)
    chip_gdf = gpd.GeoDataFrame({"key":[key]}, geometry=[chip_geom], crs=crs_tif)
    if chip_gdf.crs != gdf_labels.crs:
        chip_gdf = chip_gdf.to_crs(gdf_labels.crs)
        chip_geom = chip_gdf.geometry.iloc[0]

    # find labelled crown intersecting this chip
    hits = gdf_labels[gdf_labels.geometry.intersects(chip_geom)]
    if hits.empty:
        continue

    # if multiple, take the one with max intersection area
    inter_areas = hits.geometry.intersection(chip_geom).area
    idx = inter_areas.idxmax()
    raw_label = hits.loc[idx, "feature_class"]  # already 0-indexed

    if pd.isna(raw_label):
        continue

    records.append((tif_path, las_path, int(raw_label)))
    geoms.append(hits.loc[idx, "geometry"])  # crown polygon geometry
    labels.append(int(raw_label))
    keys_ok.append(key)

print(f"Matched crowns with label & geometry: {len(records)}")

gdf_records = gpd.GeoDataFrame({
    "key": keys_ok,
    "raw_label": labels
}, geometry=geoms, crs=gdf_labels.crs)

# Build CLASS_MAP dynamically from observed labels (handles 0-indexed cleanly)
UNIQUE_LABELS = sorted(set(labels))
CLASS_MAP = {lbl: i for i, lbl in enumerate(UNIQUE_LABELS)}
NUM_CLASSES = len(CLASS_MAP)
print(f"Classes observed (0-indexed): {UNIQUE_LABELS}")

# -------------------------------
# 2) DATASET
# -------------------------------
def read_image_as_3band_tensor(tif_path):
    with rasterio.open(tif_path) as src:
        img = src.read()  # (C,H,W)
    img = img.astype(np.float32)

    # pick bands
    if img.shape[0] == 9 and BAND_SELECTION_9:
        img = img[BAND_SELECTION_9, :, :]
    elif img.shape[0] == 8 and BAND_SELECTION_8:
        img = img[BAND_SELECTION_8, :, :]
    elif img.shape[0] >= 3:
        img = img[:3, :, :]
    else:
        raise ValueError(f"Unexpected band count {img.shape[0]} in {tif_path}")

    # robust per-band normalization (2–98 percentile)
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
    img = np.stack(img2, axis=0)  # (3,H,W)

    # Resize to 224x224 with bilinear
    t = torch.from_numpy(img).unsqueeze(0)  # (1,3,H,W)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    t = t.squeeze(0)  # (3,224,224)

    # Standardize with ImageNet stats (optional)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = (t - mean) / std
    return t.float()

def read_las_points(las_path):
    with laspy.open(las_path) as f:
        las = f.read()
        pts = np.vstack([las.x, las.y, las.z]).T  # (N,3)
    return pts

def normalize_points_xy_center(points):
    # center XY; scale to unit std; standardize Z
    xy = points[:, :2]
    z  = points[:, 2:3]
    xy_c = xy - xy.mean(axis=0, keepdims=True)
    std = xy_c.std(axis=0, keepdims=True) + 1e-6
    xy_n = xy_c / std
    z_n = (z - z.mean()) / (z.std() + 1e-6)
    return np.hstack([xy_n, z_n])

def pad_or_sample_points(points, n=MAX_POINTS):
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
    def __init__(self, recs, class_map):
        self.recs = recs
        self.class_map = class_map

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        tif_path, las_path, y_raw = self.recs[idx]
        y = self.class_map[y_raw]

        img = read_image_as_3band_tensor(tif_path)         # torch (3,224,224)
        pts = read_las_points(las_path)                    # np (N,3)
        pts = normalize_points_xy_center(pts)
        pts = pad_or_sample_points(pts, MAX_POINTS)
        pts = torch.from_numpy(pts)                        # (MAX_POINTS,3)

        return img, pts, y, Path(tif_path).stem

full_dataset = CrownDataset(records, CLASS_MAP)

# -------------------------------
# 3) MODEL
# -------------------------------
class CombinedNet(nn.Module):
    def __init__(self, num_classes=len(CLASS_MAP), freeze_resnet=FREEZE_RESNET):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_resnet:
            for p in base.parameters():
                p.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)

        # point MLP (very lightweight)
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)

        self.classifier = nn.Linear(512 + 64, num_classes)

    def forward(self, img, pts):
        x_img = self.backbone(img).view(img.size(0), -1)     # (B,512)
        h = F.relu(self.fc1(pts))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h, _ = torch.max(h, dim=1)                           # (B,256) global pooling
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))                              # (B,64)
        x = torch.cat([x_img, h], dim=1)                     # (B,576)
        logits = self.classifier(x)
        return logits

# -------------------------------
# Helpers
# -------------------------------
def get_device_and_loader_opts():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, 0, False  # num_workers, pin_memory
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, 2, True
    return torch.device("cpu"), 0, False

def run_epoch(model, loader, criterion, optimizer=None, device=torch.device("cpu")):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for img, pts, y, _ in loader:
            img, pts, y = img.to(device), pts.to(device), torch.as_tensor(y, dtype=torch.long, device=device)
            if train: optimizer.zero_grad()
            logits = model(img, pts)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total,1), correct / max(total,1)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    device, NUM_WORKERS, PIN_MEMORY = get_device_and_loader_opts()
    print(f"Using device: {device}. workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")

    # Train on the FULL dataset (no split)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = CombinedNet(num_classes=len(CLASS_MAP), freeze_resnet=FREEZE_RESNET).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")

    # ---- Save trained weights + minimal metadata ----
    torch.save(model.state_dict(), MODEL_PTH)
    meta = {
        "class_map": CLASS_MAP,          # maps raw label (0-indexed) -> contiguous class id
        "num_classes": len(CLASS_MAP),
        "img_size": IMG_SIZE,
        "max_points": MAX_POINTS,
        "band_selection_9": BAND_SELECTION_9,
        "band_selection_8": BAND_SELECTION_8,
        "freeze_resnet": FREEZE_RESNET,
        "architecture": "resnet18+pointMLP",
        "weights": "IMAGENET1K_V1"
    }
    with open(MODEL_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {MODEL_PTH}")
    print(f"Saved metadata to {MODEL_META}")