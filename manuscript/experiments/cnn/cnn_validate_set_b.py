import os, glob, json, warnings, sys
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Ensure repo root on sys.path (for utils)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils.utils import label_segments


# ========= PATHS (EDIT THESE) =========
IMG_DIR = "manuscript/data/set_b/crowns_img"
LAS_DIR = "manuscript/data/set_b/crowns_las"

# Optional: a pre-made labeled crowns GPKG with 'feature_class'. If None, derive
# labels from objects + training points via label_segments.
LABELS_GPKG = None  # e.g., "manuscript/data/set_b/labeled_crowns.gpkg"
OBJECTS_GPKG = "objects.gpkg"
TRAINING_POINTS_GPKG = "trainings_filtered.gpkg"

# Model (dual-encoder CNN+point MLP) — must match training
MODEL_PTH  = "/Users/iosefa/repos/sam/manuscript/experiments/cnn/02_dual_encoder_cnn/cnn_crowns_resnet34_dualpool.pth"
MODEL_META = "/Users/iosefa/repos/sam/manuscript/results/cnn_crowns_resnet34_dualpool.meta.json"

# Outputs
OUT_DIR = Path("manuscript/data/set_b")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = OUT_DIR / "cnn_predictions_set_b.csv"
REPORT_CSV = OUT_DIR / "cnn_report_set_b.csv"
CONFUSION_CSV = OUT_DIR / "cnn_confusion_set_b.csv"


# ========= RUNTIME/DEVICE =========
def get_device_and_loader_opts():
    if torch.backends.mps.is_available():
        return torch.device("mps"), 0, False
    if torch.cuda.is_available():
        return torch.device("cuda"), 2, True
    return torch.device("cpu"), 0, False


# ========= MODEL (must mirror training) =========
class CombinedNet(nn.Module):
    def __init__(self, num_classes, freeze_resnet=False, resnet="resnet18", dual_pool=False):
        super().__init__()
        if resnet == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_resnet:
            for p in base.parameters():
                p.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.dual_pool = dual_pool
        pts_in = 512 if dual_pool else 256
        self.fc4 = nn.Linear(pts_in, 128)
        self.fc5 = nn.Linear(128, 64)
        self.classifier = nn.Linear(512 + 64, num_classes)

    def forward(self, img, pts):
        x_img = self.backbone(img).view(img.size(0), -1)  # (B,512)
        h = F.relu(self.fc1(pts))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        if self.dual_pool:
            h_max, _ = torch.max(h, dim=1)
            h_mean = h.mean(dim=1)
            h = torch.cat([h_max, h_mean], dim=1)         # (B,512)
        else:
            h, _ = torch.max(h, dim=1)                    # (B,256)
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))                           # (B,64)
        x = torch.cat([x_img, h], dim=1)                  # (B,576)
        return self.classifier(x)


# ========= IO / PREPROC =========
def read_image_as_3band_tensor(tif_path, band_sel_9, band_sel_8, img_size):
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

    t = torch.from_numpy(img).unsqueeze(0)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = (t - mean) / std
    return t.float()


def read_las_points(las_path):
    with laspy.open(las_path) as f:
        las = f.read()
        pts = np.vstack([las.x, las.y, las.z]).T
    return pts


def normalize_points_xy_center(points):
    xy = points[:, :2]
    z  = points[:, 2:3]
    xy_c = xy - xy.mean(axis=0, keepdims=True)
    std = xy_c.std(axis=0, keepdims=True) + 1e-6
    xy_n = xy_c / std
    z_n = (z - z.mean()) / (z.std() + 1e-6)
    return np.hstack([xy_n, z_n])


def pad_or_sample_points(points, n):
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
    def __init__(self, records, meta):
        self.records = records
        self.meta = meta

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tif_path, las_path, label = self.records[idx]
        img = read_image_as_3band_tensor(tif_path,
                                         self.meta["band_selection_9"],
                                         self.meta["band_selection_8"],
                                         self.meta["img_size"])
        pts = read_las_points(las_path)
        pts = normalize_points_xy_center(pts)
        pts = pad_or_sample_points(pts, self.meta["max_points"])
        pts = torch.from_numpy(pts)
        return img, pts, int(label), Path(tif_path).stem


def build_records_and_geoms(img_dir, las_dir):
    tif_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    las_files = sorted(glob.glob(os.path.join(las_dir, "*.las")) +
                       glob.glob(os.path.join(las_dir, "*.laz")))
    tif_dict = {Path(p).stem: p for p in tif_files}
    las_dict = {Path(p).stem: p for p in las_files}
    keys = sorted(set(tif_dict.keys()).intersection(las_dict.keys()))
    if not keys:
        raise RuntimeError("No matching *.tif with *.las(laz) basenames found.")

    recs, geoms, keys_ok, crs_out = [], [], [], None
    for k in keys:
        tif_path = tif_dict[k]
        las_path = las_dict[k]
        with rasterio.open(tif_path) as src:
            left, bottom, right, top = src.bounds
            crs_tif = src.crs
        if crs_tif is None:
            warnings.warn(f"{tif_path} has no CRS; skipping.")
            continue
        if crs_out is None:
            crs_out = crs_tif
        geom = box(left, bottom, right, top)
        recs.append((tif_path, las_path, None))  # label to fill later
        geoms.append(geom)
        keys_ok.append(k)
    gdf = gpd.GeoDataFrame({"key": keys_ok}, geometry=geoms, crs=crs_out)
    return recs, gdf


def derive_labels_for_crowns(crowns_gdf: gpd.GeoDataFrame) -> pd.Series:
    if LABELS_GPKG and Path(LABELS_GPKG).exists():
        lab = gpd.read_file(LABELS_GPKG)
    else:
        # Build labeled crowns from objects + training points
        objects = gpd.read_file(OBJECTS_GPKG)
        pts = gpd.read_file(TRAINING_POINTS_GPKG)
        pts["class"] = pd.to_numeric(pts["class"], errors="coerce").astype("Int64")
        pts = pts.dropna(subset=["class"])[["class", "geometry"]]
        pts["feature_class"] = pts["class"]
        labelled, _ = label_segments(objects, pts)
        if not isinstance(labelled, gpd.GeoDataFrame):
            labelled = gpd.GeoDataFrame(labelled, geometry="geometry", crs=objects.crs)
        lab = labelled[["feature_class","geometry"]].copy()

    # Normalize label dtype and indexing (expect 0-indexed raw labels)
    if "feature_class" not in lab.columns:
        raise RuntimeError("Label GeoDataFrame lacks 'feature_class' column.")
    lab["feature_class"] = pd.to_numeric(lab["feature_class"], errors="coerce").astype("Int64")
    vals = lab["feature_class"].dropna().unique()
    if len(vals) > 0 and 0 not in set(int(v) for v in vals):
        # Likely 1-indexed; shift down to 0-index to match model's raw labels
        lab["feature_class"] = (lab["feature_class"] - 1).astype("Int64")

    if lab.crs != crowns_gdf.crs:
        lab = lab.to_crs(crowns_gdf.crs)

    # Spatial join and reduce to one label per crown (mode of overlaps)
    left = crowns_gdf.reset_index().rename(columns={"index": "left_index"})
    joined = gpd.sjoin(left, lab, how="left", predicate="intersects")
    # pick mode label per crown
    lbl_mode = (joined.dropna(subset=["feature_class"])  # rows with a label
                      .groupby("left_index")["feature_class"]
                      .agg(lambda s: int(pd.Series(s).mode().iloc[0]) if len(s) else np.nan))
    y = lbl_mode.reindex(range(len(crowns_gdf)))
    return y.astype("Int64")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    device, NUM_WORKERS, PIN_MEMORY = get_device_and_loader_opts()
    print(f"Device: {device} | workers={NUM_WORKERS} pin_memory={PIN_MEMORY}")

    # Load metadata + model
    with open(MODEL_META, "r") as f:
        meta = json.load(f)
    num_classes = int(meta["num_classes"])
    # Ensure class_map keys/values are ints (JSON may turn keys into strings)
    raw_map = meta.get("class_map", {})
    class_map = {int(k): int(v) for k, v in raw_map.items()}
    inv_class_map = {v: k for k, v in class_map.items()}  # model id -> raw int label

    # Instantiate by inferring backbone from filename; allow meta override
    state = torch.load(MODEL_PTH, map_location=device)
    fname = Path(MODEL_PTH).name.lower()
    if "resnet34" in fname:
        backbone = "resnet34"
    elif "resnet18" in fname:
        backbone = "resnet18"
    else:
        backbone = meta.get("resnet", "resnet18")
    dual = bool(meta.get("dual_pool", False))
    model = CombinedNet(num_classes=num_classes,
                        freeze_resnet=meta.get("freeze_resnet", False),
                        resnet=backbone,
                        dual_pool=dual).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        msg = str(e)
        # If fc4 expects 512 input (dual_pool training), rebuild with dual_pool=True
        if "fc4.weight" in msg and "[128, 512]" in msg:
            del model
            model = CombinedNet(num_classes=num_classes,
                                freeze_resnet=meta.get("freeze_resnet", False),
                                resnet=backbone,
                                dual_pool=True).to(device)
            model.load_state_dict(state)
        else:
            raise
    model.eval()

    # Build records + crowns geoms from the TIF footprints
    records, crowns_gdf = build_records_and_geoms(IMG_DIR, LAS_DIR)

    # Derive labels via sjoin (from LABELS_GPKG if provided, else label_segments)
    y_all = derive_labels_for_crowns(crowns_gdf)
    mask_lab = y_all.notna()
    crowns_gdf = crowns_gdf.loc[mask_lab].copy()
    y_all = y_all.loc[mask_lab].astype(int)

    # Attach tif/las paths and labels to records
    key_to_label = dict(zip(crowns_gdf["key"].values, y_all.values))
    key_to_tif = {Path(p).stem: p for p in glob.glob(os.path.join(IMG_DIR, "*.tif"))}
    key_to_las = {Path(p).stem: p for p in glob.glob(os.path.join(LAS_DIR, "*.las"))}

    samples = []
    for k in crowns_gdf["key"].values:
        if k in key_to_tif and k in key_to_las and k in key_to_label:
            samples.append((key_to_tif[k], key_to_las[k], int(key_to_label[k]), k))

    if not samples:
        raise RuntimeError("No labeled samples found in crowns_img/crowns_las.")

    # Train/test split (70/30 stratified)
    keys = [s[3] for s in samples]
    y = np.array([s[2] for s in samples])
    train_idx, test_idx = train_test_split(np.arange(len(samples)), test_size=0.30, random_state=42, stratify=y)

    train_records = [(samples[i][0], samples[i][1], samples[i][2]) for i in train_idx]
    test_records  = [(samples[i][0], samples[i][1], samples[i][2]) for i in test_idx]
    test_keys     = [samples[i][3] for i in test_idx]
    y_test_true   = [samples[i][2] for i in test_idx]

    # DataLoaders
    ds_test = CrownDataset(test_records, meta)
    loader_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Inference on test split
    pred_ids, prob_rows = [], []
    with torch.no_grad():
        for img, pts, y_lab, key in loader_test:
            img, pts = img.to(device), pts.to(device)
            logits = model(img, pts)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            pred_ids.extend(pred.tolist())
            prob_rows.extend(probs.tolist())

    # Map model id to raw label
    y_pred_raw = [int(inv_class_map.get(int(i), int(i))) for i in pred_ids]
    y_true_raw = [int(v) for v in y_test_true]

    # Harmonize label dtypes (all ints). If any cast fails, fall back to strings
    try:
        y_true_lab = [int(x) for x in y_true_raw]
        y_pred_lab = [int(x) for x in y_pred_raw]
    except Exception:
        y_true_lab = [str(x) for x in y_true_raw]
        y_pred_lab = [str(x) for x in y_pred_raw]

    # Metrics
    report = classification_report(y_true_lab, y_pred_lab, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true_lab, y_pred_lab)
    cm = confusion_matrix(y_true_lab, y_pred_lab)

    # Debug: show unique label sets
    try:
        print(f"Unique y_true: {sorted(set(y_true_lab))}")
        print(f"Unique y_pred: {sorted(set(y_pred_lab))}")
    except Exception:
        pass

    # Save report CSV (per-class precision/recall/f1/support and averages)
    rep_df = pd.DataFrame(report).transpose()
    rep_df.insert(0, "label", rep_df.index)
    rep_df.to_csv(REPORT_CSV, index=False)

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(CONFUSION_CSV, index=False)

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "key": test_keys,
        "y_true": y_true_lab,
        "y_pred": y_pred_lab,
    })
    # add probabilities per model class id (columns prob_0..prob_{num_classes-1})
    for k in range(num_classes):
        pred_df[f"prob_{k}"] = np.array(prob_rows)[:, k]
    pred_df.to_csv(PRED_CSV, index=False)

    print(f"[OK] Test size={len(test_keys)} | accuracy={acc:.4f}")
    print(f"Saved: {REPORT_CSV}, {CONFUSION_CSV}, {PRED_CSV}")
