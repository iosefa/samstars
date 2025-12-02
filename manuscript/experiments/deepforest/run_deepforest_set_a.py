#!/usr/bin/env python3
"""
DeepForest inference-only script over Set A multispectral tiles.

Runs DeepForest's pretrained release over each
  manuscript/data/set_a/setA_plot[1-6]_ms.tif
and writes a single merged GeoPackage of crown polygons (DeepForest bboxes).

Notes:
- If input has >3 bands, builds a temporary 3-band RGB (uint8) with 2–98%% stretch.
- Attempts multiple predict_tile signatures for compatibility across DeepForest versions.
- Adds an 'image' column with the source tile stem.
"""
from pathlib import Path
import tempfile
import numpy as np
import geopandas as gpd
import rasterio


# --------------- Config ---------------
INPUT_DIR     = Path("manuscript/data/set_a")
INPUT_GLOB    = "setA_plot[1-6]_ms.tif"
OUTPUT_GPKG   = INPUT_DIR / "deepforest_crowns.gpkg"
OUTPUT_LAYER  = "crowns_pred"

# RGB band selection for multispectral inputs (1-based indexes). If None, use first 3 bands.
RGB_BANDS_1IDX = None  # e.g., (5, 3, 2) for WV3 R,G,B; None -> first three

# Tiling and thresholds
PATCH_SIZE   = 800
OVERLAP      = 0.20
SCORE_THR    = 0.20


def build_rgb_uint8(image_path: Path, bands_1idx=None) -> Path:
    """Return a temporary 3-band uint8 RGB GeoTIFF path for DeepForest.
    If the source already has 3 bands, just return the original path.
    """
    with rasterio.open(image_path) as src:
        if src.count == 3:
            return image_path
        # choose bands
        if bands_1idx is None:
            bands = [1, 2, 3]
        else:
            bands = list(bands_1idx)
        arr = src.read(bands).astype("float32")  # (3,H,W)
        # per-band robust stretch to 0–255
        for i in range(arr.shape[0]):
            band = arr[i]
            finite = np.isfinite(band)
            if finite.any():
                lo, hi = np.percentile(band[finite], [2, 98])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    band = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
                else:
                    band = np.zeros_like(band, dtype="float32")
            else:
                band = np.zeros_like(band, dtype="float32")
            arr[i] = (band * 255.0).astype("uint8")
        arr = arr.astype("uint8")
        profile = src.profile.copy()
        profile.update(count=3, dtype="uint8", nodata=None)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_df_rgb.tif")
    tmp_path = Path(tmp.name)
    tmp.close()
    with rasterio.open(tmp_path, 'w', **profile) as dst:
        dst.write(arr)
    return tmp_path


def predict_one(image_path: Path):
    try:
        from deepforest import main as df_main
    except Exception as e:
        raise RuntimeError("DeepForest not available in this environment. Install deepforest to run inference.") from e

    rgb_path = build_rgb_uint8(image_path, bands_1idx=RGB_BANDS_1IDX)

    # Prepare model (pretrained release)
    model = df_main.deepforest()
    try:
        model.use_release()
    except Exception:
        pass

    # Try multiple predict_tile signatures (version differences)
    attempts = [
        dict(raster_path=str(rgb_path), return_plot=False, patch_size=PATCH_SIZE,
             patch_overlap=OVERLAP, iou_threshold=0.1, thresh=SCORE_THR),
        dict(raster_path=str(rgb_path), return_plot=False, patch_size=PATCH_SIZE,
             patch_overlap=OVERLAP, iou_threshold=0.1, score_threshold=SCORE_THR),
        dict(raster_path=str(rgb_path), return_plot=False, patch_size=PATCH_SIZE,
             patch_overlap=OVERLAP, iou_threshold=0.1),
    ]

    gdf = None
    last_err = None
    for kw in attempts:
        try:
            gdf = model.predict_tile(**kw)
            break
        except TypeError as e:
            last_err = e
            continue
    if gdf is None or gdf.empty:
        raise RuntimeError(f"DeepForest produced no detections on {image_path.name} (last_err={last_err})")

    # Ensure CRS/geometries are in map coords (some versions return pixel coords)
    try:
        with rasterio.open(rgb_path) as src:
            img_crs = src.crs
            img_transform = src.transform
            img_w, img_h = src.width, src.height
    except Exception:
        img_crs = None
        img_transform = None
        img_w = img_h = None

    if getattr(gdf, 'crs', None) is None and img_crs is not None:
        gdf.set_crs(img_crs, inplace=True)

    try:
        bx = gdf.total_bounds
        if img_transform is not None and img_w is not None and img_h is not None:
            within_pixels = (0.0 <= bx[0] <= img_w) and (0.0 <= bx[2] <= img_w) and \
                            (0.0 <= bx[1] <= img_h) and (0.0 <= bx[3] <= img_h)
            if within_pixels:
                from shapely.ops import transform as shp_transform
                a = img_transform.a; b = img_transform.b; c = img_transform.c
                d = img_transform.d; e = img_transform.e; f = img_transform.f
                def px_to_map(x, y, z=None):
                    x = np.asarray(x); y = np.asarray(y)
                    X = a * x + b * y + c
                    Y = d * x + e * y + f
                    return (X, Y) if z is None else (X, Y, z)
                gdf = gdf.copy()
                gdf["geometry"] = gdf.geometry.apply(lambda g: shp_transform(px_to_map, g))
                if img_crs is not None:
                    gdf.set_crs(img_crs, inplace=True)
    except Exception:
        pass

    # cleanup temp RGB if we created one
    if rgb_path != image_path:
        try:
            Path(rgb_path).unlink(missing_ok=True)
        except Exception:
            pass

    return gdf


def main():
    tif_paths = sorted(INPUT_DIR.glob(INPUT_GLOB))
    if not tif_paths:
        raise SystemExit(f"No TIFs found at {INPUT_DIR}/{INPUT_GLOB}")

    merged = None
    crs0 = None
    for p in tif_paths:
        print(f"[DF] {p.name}")
        gdf = predict_one(p)
        gdf = gdf.copy()
        gdf["image"] = p.stem
        if merged is None:
            merged = gdf
            crs0 = gdf.crs
        else:
            if crs0 and gdf.crs and gdf.crs != crs0:
                gdf = gdf.to_crs(crs0)
            merged = gpd.pd.concat([merged, gdf], ignore_index=True)
        print(f"  -> detections: {len(gdf)}")

    if merged is None or merged.empty:
        print("No DeepForest detections; nothing to write.")
        return

    out_path = Path(OUTPUT_GPKG)
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    merged.to_file(out_path, layer=OUTPUT_LAYER, driver="GPKG")
    print(f"[OK] Wrote {out_path} | crowns={len(merged)} | images={len(tif_paths)}")


if __name__ == "__main__":
    main()

