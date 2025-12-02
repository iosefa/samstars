#!/usr/bin/env python3
"""
Build per-crown feature tensors using StarDist encoder features + raster descriptors.
Outputs X (features), y (class indices), and metadata needed for KAN training.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rasterio
import geopandas as gpd
import tensorflow as tf
from rasterio.features import rasterize
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def pick_feature_layer(kmodel: tf.keras.Model, layer_name: Optional[str] = None) -> tf.keras.layers.Layer:
    """Return a convolutional feature layer; default to last 4D layer before output."""
    if layer_name:
        return kmodel.get_layer(layer_name)
    for layer in reversed(kmodel.layers):
        try:
            shp = layer.output_shape
        except Exception:
            continue
        if isinstance(shp, (list, tuple)) and len(shp) == 4:
            return layer
    raise RuntimeError("No suitable 4D feature layer found in StarDist model.")


def build_feature_model(model: StarDist2D, layer_name: Optional[str]) -> tf.keras.Model:
    """Wrap StarDist keras_model to also output chosen feature map."""
    km = model.keras_model
    feat_layer = pick_feature_layer(km, layer_name)
    outputs = [km.outputs, feat_layer.output] if isinstance(km.outputs, list) else [km.outputs, feat_layer.output]
    return tf.keras.Model(inputs=km.inputs, outputs=outputs)


def rasterize_mask(crowns: gpd.GeoDataFrame, transform, shape, id_field: str) -> np.ndarray:
    """Rasterize crowns to integer mask using id_field (assumed unique per row)."""
    shapes = [(geom, int(idx)) for geom, idx in zip(crowns.geometry, crowns[id_field])]
    mask = rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="int32")
    return mask


def resample_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize mask to target shape (nearest)."""
    import cv2
    return cv2.resize(mask.astype("float32"), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def pool_features(feat_map: np.ndarray, mask_small: np.ndarray, crown_id: int) -> np.ndarray:
    """Average pool feature map over a given crown id in resized mask."""
    m = (mask_small == crown_id)
    if not m.any():
        return np.zeros((feat_map.shape[-1],), dtype=np.float32)
    vals = feat_map[m]
    return vals.mean(axis=0).astype(np.float32)


def geometry_features(geom) -> List[float]:
    """Basic shape metrics."""
    area = geom.area
    length = geom.length
    circ = 4 * np.pi * area / (length ** 2 + 1e-6)
    minrect = geom.minimum_rotated_rectangle
    if minrect and not minrect.is_empty:
        coords = np.array(minrect.exterior.coords)
        edges = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        edges = edges[edges > 0]
        elong = float(edges.max() / edges.min()) if edges.size else 1.0
    else:
        elong = 1.0
    return [area, length, circ, elong]


def raster_stats(arr: np.ndarray, mask: np.ndarray) -> List[float]:
    m = arr[mask]
    if m.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(m.mean()), float(m.std()), float(np.percentile(m, 10)), float(np.percentile(m, 90))]


def spectral_features(img_arr: np.ndarray, mask: np.ndarray) -> List[float]:
    feats = []
    for b in range(img_arr.shape[0]):
        feats.extend(raster_stats(img_arr[b], mask))
    return feats


def build_tile_features(tile_dir: Path, feature_model: tf.keras.Model,
                        crowns_field: str = "class") -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for one tile."""
    prob_path = tile_dir / "crown_probability.tif"
    crowns_path = tile_dir / "crowns.gpkg"
    img_path = tile_dir / "img.tif"
    chm_path = tile_dir / "chm.tif"
    if not (prob_path.exists() and crowns_path.exists() and img_path.exists() and chm_path.exists()):
        raise FileNotFoundError(f"Missing required files in {tile_dir}")

    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype("float32")
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
    prob_norm = normalize(prob, 1, 99.8, axis=(0, 1))[..., None]

    crowns = gpd.read_file(crowns_path).to_crs(crs)
    if crowns_field not in crowns.columns:
        # use first non-geometry column
        fallback_cols = [c for c in crowns.columns if c != "geometry"]
        if not fallback_cols:
            raise ValueError("No class column found in crowns.gpkg")
        crowns_field = fallback_cols[0]
    crowns = crowns.reset_index().rename(columns={"index": "tmp_id"})
    crowns["tmp_id"] = crowns.index + 1  # 0 reserved for background

    mask_full = rasterize_mask(crowns, transform, shape, "tmp_id")

    outputs, feat_map = feature_model.predict(prob_norm[None], verbose=0)
    feat_map = np.array(feat_map[0])
    mask_small = resample_mask(mask_full, feat_map.shape[:2])

    with rasterio.open(img_path) as src:
        img = src.read().astype("float32")
    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype("float32")

    X_list, y_list = [], []
    for _, row in crowns.iterrows():
        cid = int(row["tmp_id"])
        label_val = row[crowns_field]
        crow_mask = mask_full == cid
        prob_stats = raster_stats(prob, crow_mask)
        chm_stats = raster_stats(chm, crow_mask)
        spec_stats = spectral_features(img, crow_mask)
        geom_stats = geometry_features(row.geometry)
        pooled = pool_features(feat_map, mask_small, cid).tolist()
        feats = pooled + prob_stats + chm_stats + spec_stats + geom_stats
        X_list.append(feats)
        y_list.append(label_val)

    return np.array(X_list, dtype=np.float32), np.array(y_list)


def main():
    pa = argparse.ArgumentParser(description="Build KAN training features from StarDist encoder + descriptors.")
    pa.add_argument("--tiles-dir", required=True, type=Path)
    pa.add_argument("--stardist-model-dir", default="models", type=Path)
    pa.add_argument("--stardist-model-name", default="stardist")
    pa.add_argument("--feature-layer", default=None, help="Optional Keras layer name to tap for encoder features")
    pa.add_argument("--class-field", default="class", help="Column in crowns.gpkg with class labels")
    pa.add_argument("--out", default="kan_dataset.npz", type=Path)
    args = pa.parse_args()

    sd_model = StarDist2D(None, name=args.stardist_model_name, basedir=str(args.stardist_model_dir))
    feat_model = build_feature_model(sd_model, args.feature_layer)

    X_all, y_all = [], []
    for tile in sorted(Path(args.tiles_dir).iterdir()):
        if not tile.is_dir():
            continue
        try:
            X_tile, y_tile = build_tile_features(tile, feat_model, args.class_field)
            X_all.append(X_tile)
            y_all.append(y_tile)
            print(f"{tile.name}: {len(X_tile)} crowns")
        except Exception as exc:
            print(f"Skipping {tile.name}: {exc}")

    if not X_all:
        raise SystemExit("No features built.")
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    meta = {"class_field": args.class_field, "feature_layer": args.feature_layer}
    np.savez_compressed(args.out, X=X, y=y, meta=json.dumps(meta))
    print(f"Saved {X.shape[0]} samples to {args.out}")


if __name__ == "__main__":
    main()
