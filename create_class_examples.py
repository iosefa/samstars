#!/usr/bin/env python3
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.errors import RasterioIOError
from shapely.geometry import mapping, box

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded inputs/outputs
# ─────────────────────────────────────────────────────────────────────────────
WV3_PATH   = "/Users/iosefa/Dropbox/MergedImage2.tif"   # multispectral stack
CHM_PATH   = "/Users/iosefa/Dropbox/chm_full.tif"       # canopy height model
OBJECTS_GPKG = Path("objects.gpkg")                     # must have 'segment_id'
OUT_DIR    = Path("six_class_chips")

# One segment per class (edit as needed)
SEGMENT_BY_CLASS = {
    1: 154789,
    2: 133906,
    3: 111325,
    4: 84270,
    5: 33578,
    6: 112807,
}

# WV3 RGB bands (1-based). If WV3 has <5 bands, we’ll fall back to (1,2,3).
PREF_WV3_BANDS_532 = (5, 3, 2)

# Square padding in metres (applied to crop mask; outlines are unpadded)
PAD_M = 5.0
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def write_geopackage(out_path: Path, gdf: gpd.GeoDataFrame, layer: str = "polygons"):
    if out_path.exists():
        out_path.unlink()
    gdf.to_file(out_path, driver="GPKG", layer=layer)

def square_pad_envelope(geom, pad=PAD_M):
    """Strict square pad around geom bounds."""
    minx, miny, maxx, maxy = geom.bounds
    return box(minx - pad, miny - pad, maxx + pad, maxy + pad)

def _predictor_for_dtype(dtype: np.dtype):
    """Choose TIFF predictor compatible with dtype."""
    if np.issubdtype(dtype, np.integer):
        return 2  # horizontal differencing (good for DEFLATE/LZW on ints)
    if np.issubdtype(dtype, np.floating):
        return 3  # floating-point predictor
    return None

def write_gtiff(path: Path, src, arr, transform, count, force_dtype=None, prefer_deflate=True):
    """
    Robust GTiff writer:
      - Normalizes shape (bands, H, W)
      - Picks a safe predictor based on dtype
      - Retries without predictor if libtiff complains
      - Final fallback: LZW, no predictor
    """
    meta = src.meta.copy()
    if arr.ndim == 2:
        arr = arr[None, :, :]

    write_dtype = force_dtype if force_dtype is not None else arr.dtype
    arr = arr.astype(write_dtype, copy=False)

    meta.update({
        "driver": "GTiff",
        "height": arr.shape[1],
        "width":  arr.shape[2],
        "count":  count,
        "transform": transform,
        "dtype": str(write_dtype),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE" if prefer_deflate else "LZW",
    })
    if src.nodata is not None:
        meta["nodata"] = src.nodata

    predictor = _predictor_for_dtype(np.dtype(write_dtype))
    if predictor is not None:
        meta["predictor"] = predictor

    def _do_write(m):
        with rasterio.open(path, "w", **m) as dst:
            for i in range(count):
                dst.write(arr[i], i + 1)

    try:
        _do_write(meta)
    except RasterioIOError:
        # Retry without predictor
        meta.pop("predictor", None)
        try:
            _do_write(meta)
        except RasterioIOError:
            # Final fallback
            meta["compress"] = "LZW"
            meta.pop("predictor", None)
            _do_write(meta)

def pick_rgb_bands(src):
    """Prefer (5,3,2) if present; else (1,2,3)."""
    if src.count >= max(PREF_WV3_BANDS_532):
        return PREF_WV3_BANDS_532
    return (1, 2, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not OBJECTS_GPKG.exists():
        raise FileNotFoundError(f"Missing {OBJECTS_GPKG}")

    objects = gpd.read_file(OBJECTS_GPKG)
    if "segment_id" not in objects.columns:
        raise RuntimeError(f"{OBJECTS_GPKG} must contain a 'segment_id' column.")

    with rasterio.open(WV3_PATH) as wv3_src, rasterio.open(CHM_PATH) as chm_src:
        if wv3_src.crs is None:
            raise RuntimeError(f"{WV3_PATH} has no CRS.")
        if chm_src.crs is None:
            raise RuntimeError(f"{CHM_PATH} has no CRS.")

        rgb_bands = pick_rgb_bands(wv3_src)

        manifest = {
            "wv3_raster": str(WV3_PATH),
            "chm_raster": str(CHM_PATH),
            "objects": str(OBJECTS_GPKG),
            "wv3_bands": rgb_bands,
            "pad_m": PAD_M,
            "outputs": []
        }

        for cls in range(1, 7):
            seg_id = SEGMENT_BY_CLASS.get(cls)
            sel = objects[objects["segment_id"] == seg_id]
            if sel.empty:
                print(f"[WARN] class {cls}: segment_id={seg_id} not found; skipping.")
                continue

            row = sel.iloc[0]
            geom_obj = row.geometry

            # Reproject polygon to WV3 CRS for outline and WV3 crop
            poly_in_wv3 = gpd.GeoDataFrame(geometry=[geom_obj], crs=objects.crs)
            if poly_in_wv3.crs != wv3_src.crs:
                poly_in_wv3 = poly_in_wv3.to_crs(wv3_src.crs)

            # 1) Outline GPKG: ORIGINAL polygon (no padding), written in WV3 CRS
            gpkg_out = OUT_DIR / f"class_{cls}_outline.gpkg"
            write_geopackage(gpkg_out, poly_in_wv3, layer="class_outline")
            print(f"[OK] class {cls}: wrote {gpkg_out.name}")

            # 2) WV3 crop using SQUARE padded envelope in WV3 CRS
            padded_sq_wv3 = square_pad_envelope(poly_in_wv3.geometry.iloc[0], PAD_M)
            out_img_wv3, out_transform_wv3 = rio_mask(
                wv3_src, [mapping(padded_sq_wv3)], crop=True, filled=True, indexes=list(rgb_bands)
            )
            tif_out_wv3 = OUT_DIR / f"class_{cls}.tif"
            write_gtiff(tif_out_wv3, wv3_src, out_img_wv3, out_transform_wv3, count=len(rgb_bands))
            print(f"[OK] class {cls}: wrote {tif_out_wv3.name} (bands {rgb_bands})")

            # 3) CHM crop (reproject the padded square to CHM CRS first if needed)
            padded_sq_chm = gpd.GeoSeries([padded_sq_wv3], crs=wv3_src.crs)
            if padded_sq_chm.crs != chm_src.crs:
                padded_sq_chm = padded_sq_chm.to_crs(chm_src.crs)

            out_img_chm, out_transform_chm = rio_mask(
                chm_src, [mapping(padded_sq_chm.iloc[0])], crop=True, filled=True, indexes=1
            )
            if out_img_chm.ndim == 2:
                out_img_chm = out_img_chm[None, :, :]
            out_img_chm = out_img_chm.astype("float32", copy=False)

            tif_out_chm = OUT_DIR / f"class_{cls}_chm.tif"
            write_gtiff(tif_out_chm, chm_src, out_img_chm, out_transform_chm, count=1, force_dtype="float32")
            print(f"[OK] class {cls}: wrote {tif_out_chm.name} (CHM)")

            manifest["outputs"].append({
                "class": cls,
                "segment_id": int(seg_id),
                "outline_gpkg": str(gpkg_out),
                "chip_tif_wv3": str(tif_out_wv3),
                "chip_tif_chm": str(tif_out_chm),
            })

        (OUT_DIR / "chips_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[OK] Wrote manifest → {OUT_DIR/'chips_manifest.json'}")

if __name__ == "__main__":
    main()