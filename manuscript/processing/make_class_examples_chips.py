#!/usr/bin/env python3
"""
Create 30 m x 30 m class example chips (RGB 7/5/3 and CHM) centered on points.

Inputs
- Centers: manuscript/assets/six_class_chips/class_chip_centers.gpkg (must have field 'class')
- Image:   /Users/iosefa/Dropbox/MergedImage2.tif (reads bands 7,5,3 as RGB)
- CHM:     /Users/iosefa/Dropbox/chm_full.tif

Outputs
- Writes chips under manuscript/data/ahupuaa/class_examples/
  - RGB: class_<cls>_<idx>.tif (3 bands: 7/5/3 from source)
  - CHM: class_<cls>_<idx>_chm.tif (1 band)

Notes
- Chip side length is 30 meters (square), centered on each point.
- Points are reprojected to each raster's CRS for spatial indexing.
"""

from pathlib import Path
import sys
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling


# Config
CENTERS_GPKG = Path("manuscript/assets/six_class_chips/class_chip_centers.gpkg")
IMG_PATH     = Path("/Users/iosefa/Dropbox/MergedImage2.tif")
CHM_PATH     = Path("/Users/iosefa/Dropbox/chm_full.tif")
OUT_DIR      = Path("manuscript/data/ahupuaa/class_examples")
CHIP_SIDE_M  = 30.0
DST_CRS_EPSG = 6635


def ensure_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf[gdf.geometry.notna()].copy()
    g = g[g.geometry.type == 'Point']
    return g


def meters_to_pixels(side_m: float, transform) -> tuple[int, int]:
    px_w = max(1, int(round(side_m / abs(transform.a))))
    px_h = max(1, int(round(side_m / abs(transform.e))))
    return px_w, px_h


def clamp_window(win: Window, width: int, height: int) -> Window:
    c0 = max(0, int(np.floor(win.col_off)))
    r0 = max(0, int(np.floor(win.row_off)))
    c1 = min(width,  int(np.ceil(win.col_off + win.width)))
    r1 = min(height, int(np.ceil(win.row_off + win.height)))
    w = max(0, c1 - c0)
    h = max(0, r1 - r0)
    return Window(c0, r0, w, h)


def extract_centered_window(src: rasterio.io.DatasetReader, x: float, y: float, side_m: float) -> tuple[Window, rasterio.Affine]:
    cols, rows = src.index(x, y)
    pw, ph = meters_to_pixels(side_m, src.transform)
    c0 = int(cols - pw // 2)
    r0 = int(rows - ph // 2)
    win = Window(c0, r0, pw, ph)
    win = clamp_window(win, src.width, src.height)
    if win.width <= 0 or win.height <= 0:
        return None, None
    w_transform = window_transform(win, src.transform)
    return win, w_transform


def write_rgb_chip(img_src: rasterio.io.DatasetReader, bbox_dst: tuple[float,float,float,float], out_path: Path, bands=(5,3,2)):
    # Ensure source is in EPSG:6635 (per user note); otherwise project centers earlier
    if not (img_src.crs and img_src.crs.to_epsg() == DST_CRS_EPSG):
        raise RuntimeError(f"Image CRS must be EPSG:{DST_CRS_EPSG}, found {img_src.crs}")
    # Compute window from bounds
    minx, miny, maxx, maxy = bbox_dst
    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, img_src.transform)
    win = clamp_window(win, img_src.width, img_src.height)
    if win.width <= 0 or win.height <= 0:
        raise RuntimeError("Computed RGB window out of bounds")
    # Read bands 7/5/3
    bidx = [b for b in bands if 1 <= b <= img_src.count]
    if len(bidx) != 3:
        raise ValueError(f"Bands {bands} not available in {img_src.name}")
    arr = img_src.read(indexes=bidx, window=win)
    meta = img_src.meta.copy()
    meta.update({
        'driver': 'GTiff',
        'width': int(win.width),
        'height': int(win.height),
        'count': 3,
        'crs': img_src.crs,
        'transform': window_transform(win, img_src.transform),
        'dtype': arr.dtype,
        'compress': 'deflate'
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(arr)


def write_chm_chip(chm_src: rasterio.io.DatasetReader, bbox_dst: tuple[float,float,float,float], out_path: Path):
    if not (chm_src.crs and chm_src.crs.to_epsg() == DST_CRS_EPSG):
        raise RuntimeError(f"CHM CRS must be EPSG:{DST_CRS_EPSG}, found {chm_src.crs}")
    minx, miny, maxx, maxy = bbox_dst
    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, chm_src.transform)
    win = clamp_window(win, chm_src.width, chm_src.height)
    if win.width <= 0 or win.height <= 0:
        raise RuntimeError("Computed CHM window out of bounds")
    arr = chm_src.read(1, window=win)
    meta = chm_src.meta.copy()
    meta.update({
        'driver': 'GTiff',
        'width': int(win.width),
        'height': int(win.height),
        'count': 1,
        'crs': chm_src.crs,
        'transform': window_transform(win, chm_src.transform),
        'dtype': arr.dtype,
        'compress': 'deflate'
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(arr, 1)


def main():
    if not CENTERS_GPKG.exists():
        print(f"Centers not found: {CENTERS_GPKG}", file=sys.stderr); sys.exit(1)
    gdf = gpd.read_file(CENTERS_GPKG)
    if 'class' not in gdf.columns:
        print("'class' field missing in centers GPKG", file=sys.stderr); sys.exit(1)
    gdf = ensure_points(gdf)
    if gdf.empty:
        print("No points found in centers GPKG."); return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(IMG_PATH) as img_src, rasterio.open(CHM_PATH) as chm_src:
        # For each point, reproject to each raster CRS and extract chips
        for idx, row in gdf.reset_index(drop=True).iterrows():
            cls = int(row['class'])
            geom = row.geometry
            # Center point in EPSG:6635 (all inputs should be EPSG:6635)
            if not (gdf.crs and gdf.crs.to_epsg() == DST_CRS_EPSG):
                pt = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(DST_CRS_EPSG).iloc[0]
                cx, cy = float(pt.x), float(pt.y)
            else:
                cx, cy = float(geom.x), float(geom.y)
            half = CHIP_SIDE_M / 2.0
            bbox = (cx - half, cy - half, cx + half, cy + half)

            # RGB chip (bands 7/5/3)
            out_rgb = OUT_DIR / f"rgb_{cls}.tif"
            write_rgb_chip(img_src, bbox, out_rgb, bands=(5,3,2))
            print(f"[OK] RGB → {out_rgb}")

            # CHM chip (band 1)
            out_chm = OUT_DIR / f"chm_{cls}.tif"
            write_chm_chip(chm_src, bbox, out_chm)
            print(f"[OK] CHM → {out_chm}")


if __name__ == "__main__":
    main()
