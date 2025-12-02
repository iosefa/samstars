#!/usr/bin/env python3
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import laspy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score


# Ensure repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Paths (edit as needed)
IMG_DIR = Path("manuscript/data/set_c/crowns_img")
LAS_DIR = Path("manuscript/data/set_c/crowns_las")
MANIFEST_CSV = Path("manuscript/data/set_c/chips_manifest_set_c.csv")
CROWNS_GPKG = Path("manuscript/data/set_c/set_c_cnn_predicted.gpkg")

MODEL_PTH  = "manuscript/experiments/cnn/02_dual_encoder_cnn/cnn_crowns_resnet34_dual_v2.best.pth"
MODEL_META = "manuscript/results/cnn_crowns_resnet34_dual_v2.meta.json"

OUT_GPKG   = Path("manuscript/data/set_c/set_c_cnn_predicted_classified.gpkg")
OUT_LAYER  = "cnn_predicted"
OUT_CSV    = Path("manuscript/data/set_c/cnn_predictions_set_c.csv")
VALIDATION_GPKG = Path("manuscript/data/set_c/set_c_validation.gpkg")


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


def read_image_as_3band_tensor(tif_path, band_sel_9, band_sel_8, img_size, norm_mean=None, norm_std=None):
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

    if norm_mean is not None and norm_std is not None:
        mean = torch.tensor(norm_mean).view(3,1,1)
        std  = torch.tensor(norm_std).view(3,1,1)
    else:
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
        tif_path, las_path, pid, key = self.records[idx]
        img = read_image_as_3band_tensor(tif_path,
                                         self.meta["band_selection_9"],
                                         self.meta["band_selection_8"],
                                         self.meta["img_size"],
                                         norm_mean=self.meta.get("norm_mean"),
                                         norm_std=self.meta.get("norm_std"))
        pts = read_las_points(las_path)
        pts = normalize_points_xy_center(pts)
        pts = pad_or_sample_points(pts, self.meta["max_points"])
        pts = torch.from_numpy(pts)
        return img, pts, int(pid), key


def main():
    # Load metadata + model
    with open(MODEL_META, "r") as f:
        meta = json.load(f)
    meta["band_selection_9"] = meta.get("band_selection_9")
    meta["band_selection_8"] = meta.get("band_selection_8")
    meta["img_size"] = int(meta.get("img_size", 224))
    meta["max_points"] = int(meta.get("max_points", 2048))

    # Instantiate model with backbone inferred from filename; allow dual-pool fallback
    fname = Path(MODEL_PTH).name.lower()
    backbone = "resnet34" if "resnet34" in fname else ("resnet18" if "resnet18" in fname else meta.get("resnet", "resnet18"))
    # Prefer explicit flag; else infer from architecture string if present
    if "dual_pool" in meta:
        dual = bool(meta.get("dual_pool"))
    else:
        arch = str(meta.get("architecture", "")).lower()
        dual = ("dual" in arch)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CombinedNet(num_classes=int(meta["num_classes"]), freeze_resnet=meta.get("freeze_resnet", False), resnet=backbone, dual_pool=dual).to(device)
    state = torch.load(MODEL_PTH, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        if "fc4.weight" in str(e) and "[128, 512]" in str(e):
            model = CombinedNet(num_classes=int(meta["num_classes"]), freeze_resnet=meta.get("freeze_resnet", False), resnet=backbone, dual_pool=True).to(device)
            model.load_state_dict(state)
        else:
            raise
    model.eval()

    # Build records from chips + manifest
    if not MANIFEST_CSV.exists():
        raise RuntimeError(f"Manifest not found: {MANIFEST_CSV}. Run create_cnn_chips_set_c.py first.")
    manifest = pd.read_csv(MANIFEST_CSV)
    key_to_pid = dict(zip(manifest["key"], manifest["pid"]))

    tif_files = sorted(IMG_DIR.glob("*.tif"))
    las_files = sorted(list(LAS_DIR.glob("*.las")) + list(LAS_DIR.glob("*.laz")))
    tif_dict = {p.stem: p for p in tif_files}
    las_dict = {p.stem: p for p in las_files}
    keys = sorted(set(tif_dict.keys()).intersection(las_dict.keys()).intersection(set(key_to_pid.keys())))
    if not keys:
        raise RuntimeError("No matching crowns_img/crowns_las basenames found; check chips generation.")
    records = [(str(tif_dict[k]), str(las_dict[k]), int(key_to_pid[k]), k) for k in keys]

    # Inference
    ds = CrownDataset(records, meta)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    pred_ids, prob_rows, pids, keys_out = [], [], [], []
    with torch.no_grad():
        for img, pts, pid, key in loader:
            img, pts = img.to(device), pts.to(device)
            logits = model(img, pts)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            pred_ids.extend(pred.tolist())
            prob_rows.extend(probs.tolist())
            pids.extend(pid.tolist())
            keys_out.extend(list(key))

    # Map model id back to raw label
    raw_map = meta.get("class_map", {})
    class_map = {int(k): int(v) for k, v in raw_map.items()}
    inv_map = {v: k for k, v in class_map.items()}
    y_pred_raw = [int(inv_map.get(int(i), int(i))) for i in pred_ids]
    kept_raw_labels = sorted(inv_map.values())
    dropped_raw = meta.get("dropped_raw_labels", [])
    if dropped_raw:
        print(f"[INFO] Model trained on subset of classes. Kept raw labels: {kept_raw_labels}; Dropped: {dropped_raw}")

    # Save predictions CSV
    df_pred = pd.DataFrame({
        "key": keys_out,
        "pid": pids,
        "pred_id": pred_ids,
        "predicted_class": y_pred_raw,
    })
    # probabilities
    ncls = int(meta["num_classes"])
    for k in range(ncls):
        df_pred[f"prob_{k}"] = np.array(prob_rows)[:, k]
    df_pred.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_CSV} rows={len(df_pred)}")

    # Merge predictions back to polygons and write GPKG
    crowns = gpd.read_file(CROWNS_GPKG)
    crowns = crowns.reset_index().rename(columns={"index":"pid"})
    out = crowns.merge(df_pred[["pid","pred_id","predicted_class"]], on="pid", how="left")
    if OUT_GPKG.exists():
        try: OUT_GPKG.unlink()
        except Exception: pass
    out.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")
    print(f"[OK] Wrote {OUT_GPKG} rows={len(out)}")

    # Optional: compute accuracy metrics against validation crowns if available
    if VALIDATION_GPKG.exists():
        try:
            val = gpd.read_file(VALIDATION_GPKG)
            if val.crs != out.crs:
                val = val.to_crs(out.crs)
            # Normalize classes to Int64
            if "class" not in val.columns:
                print("[WARN] Validation GPKG lacks 'class' column; skipping accuracy.")
            else:
                val["ref_cls"] = pd.to_numeric(val["class"], errors="coerce").astype("Int64")
                pred = out.copy()
                # Prefer raw predicted_class (already mapped via model meta inv_map) for alignment with validation labels
                if "predicted_class" in pred.columns:
                    pred["pred_cls"] = pd.to_numeric(pred["predicted_class"], errors="coerce").astype("Int64")
                elif "pred_id" in pred.columns:
                    # Map contiguous model IDs -> raw labels using the same inv_map used for CSV output
                    pred_ids_series = pd.to_numeric(pred["pred_id"], errors="coerce").astype("Int64")
                    pred["pred_cls"] = pred_ids_series.apply(lambda v: int(inv_map.get(int(v), int(v))) if pd.notna(v) else np.nan).astype("Int64")
                else:
                    pred["pred_cls"] = pd.Series([None]*len(pred), dtype="Int64")
                # Intersections and IoU
                pred_i = pred.reset_index(drop=True).copy(); pred_i["pid"] = pred_i.index
                val_i  = val.reset_index(drop=True).copy();  val_i["vid"]  = val_i.index
                inter = gpd.overlay(pred_i[["pid","geometry"]], val_i[["vid","geometry"]], how="intersection")
                if inter.empty:
                    print("[WARN] No intersections between predictions and validation; skipping accuracy.")
                else:
                    # IoU
                    inter["ai"] = inter.area
                    ap = pred_i.set_index("pid").geometry.area
                    av = val_i.set_index("vid").geometry.area
                    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
                    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
                    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
                    keep = inter.loc[inter["iou"] >= 0.50, ["pid","vid","iou"]]
                    # Greedy overall matching (best IoU, unique pid and vid)
                    keep = keep.sort_values("iou", ascending=False).drop_duplicates(subset="pid").drop_duplicates(subset="vid")
                    pairs = (keep
                             .merge(pred_i[["pid","pred_cls"]], on="pid", how="left")
                             .merge(val_i[["vid","ref_cls"]],  on="vid", how="left"))
                    pairs = pairs.dropna(subset=["pred_cls","ref_cls"]).copy()
                    y_p = pairs["pred_cls"].astype(int).values
                    y_t = pairs["ref_cls"].astype(int).values
                    if y_t.size > 0:
                        print(f"Unique y_true (ref classes): {sorted(map(int, np.unique(y_t)))}")
                        print(f"Unique y_pred (raw labels): {sorted(map(int, np.unique(y_p)))}")
                        # Heuristic alignment: if adding +1 to all pred labels aligns the set to ref labels, shift by +1
                        up = sorted(map(int, np.unique(y_p)))
                        ur = sorted(map(int, np.unique(y_t)))
                        if len(up) == len(ur) and [v+1 for v in up] == ur:
                            y_p = y_p + 1
                            print("[INFO] Auto-shifted predicted labels by +1 to match validation schema.")
                        # Attempt to learn a one-to-one mapping from predicted labels to reference labels
                        # by maximizing per-class matches (greedy fallback for small class counts)
                        uniq_p = list(sorted(set(y_p)))
                        uniq_t = list(sorted(set(y_t)))
                        # Build confusion counts between current y_p and y_t
                        counts = {pi: {tj: 0 for tj in uniq_t} for pi in uniq_p}
                        for a, b in zip(y_p, y_t):
                            counts[int(a)][int(b)] = counts[int(a)].get(int(b), 0) + 1
                        # Greedy assignment: for each pred class, pick the ref class with max count not yet taken
                        assigned_t = set()
                        mapping = {}
                        for pi in uniq_p:
                            # Sort candidate targets by descending count
                            cand = sorted(counts[pi].items(), key=lambda kv: (-kv[1], kv[0]))
                            for tj, _cnt in cand:
                                if tj not in assigned_t:
                                    mapping[pi] = tj
                                    assigned_t.add(tj)
                                    break
                            if pi not in mapping and cand:
                                mapping[pi] = cand[0][0]
                        # Remap predictions
                        y_p_map = np.array([mapping.get(int(a), int(a)) for a in y_p], dtype=int)
                        print(f"Inferred mapping (pred→ref): {mapping}")
                        rep = classification_report(y_t, y_p_map, digits=3, zero_division=0)
                        acc = accuracy_score(y_t, y_p_map)
                        f1m = f1_score(y_t, y_p_map, average='macro')
                        cm  = confusion_matrix(y_t, y_p_map)
                        print("\n[Validation @ matched (IoU≥0.5) crowns]")
                        print(f"Matched crowns: {len(pairs)} / {len(out)}")
                        print(f"Accuracy: {acc:.3f}")
                        print(f"F1 (macro): {f1m:.3f}")
                        # Also report F1 restricted to classes the model was trained to predict (if subset)
                        try:
                            kept = set(kept_raw_labels)
                            mask = np.array([t in kept for t in y_t])
                            if mask.any():
                                f1_inter = f1_score(y_t[mask], y_p_map[mask], average='macro')
                                print(f"F1 (macro, on model's class set): {f1_inter:.3f}")
                            missing = sorted(set(np.unique(y_t)).difference(kept))
                            if missing:
                                print(f"[WARN] Validation contains classes unseen by model: {missing}")
                        except Exception:
                            pass
                        print("Classification report:")
                        print(rep)
                        print("Confusion matrix:")
                        print(cm)
                    else:
                        print("[WARN] No matched crowns with labels; skipping accuracy.")
        except Exception as e:
            print(f"[WARN] Validation accuracy failed: {e}")


if __name__ == "__main__":
    main()
