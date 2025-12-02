import os
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask

# Add repo root to import path so 'utils' resolves when running via relative path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyforestscan.handlers import read_lidar, write_las
from utils.utils import label_segments

# --- Inputs (edit as needed) ---
ept = "/Users/iosefa/Downloads/ept6635/ept.json"
ept_srs = "EPSG:6635"
img_path = "/Users/iosefa/Dropbox/MergedImage2.tif"

# Crown sources (same behavior as the Set A script)
training_path = "trainings_filtered.gpkg"
objects_path = "objects.gpkg"

# --- Outputs (Set B) ---
base_out = Path("manuscript/data/set_b")
las_out_dir = base_out / "crowns_las"
img_out_dir = base_out / "crowns_img"
las_out_dir.mkdir(parents=True, exist_ok=True)
img_out_dir.mkdir(parents=True, exist_ok=True)

# --- Load crowns via intersecting training & objects ---
objects = gpd.read_file(objects_path)
training_points = gpd.read_file(training_path)

training_points["class"] = (
    pd.to_numeric(training_points["class"], errors="coerce").astype("Int64")
)
training_points = training_points.dropna(subset=["class"])[["class", "geometry"]]
training_points["feature_class"] = training_points["class"]

# label_segments should return a GeoDataFrame for 'training'
training, mixed = label_segments(objects, training_points)
if not isinstance(training, gpd.GeoDataFrame):
    training = gpd.GeoDataFrame(training, geometry="geometry", crs=objects.crs)

# Keep a copy with attributes; ensure valid geometries
crowns = training.copy()
crowns = crowns[~crowns.geometry.is_empty & crowns.geometry.notna()].copy()
crowns["geometry"] = crowns.geometry.buffer(0)  # clean minor topology issues

if crowns.crs is None:
    raise ValueError("Crowns have no CRS; please assign one before running.")

# Prepare geometry in both CRS: EPT for LAS, Raster CRS for imagery
crowns_las = crowns.to_crs(ept_srs)

with rasterio.open(img_path) as src:
    img_crs = src.crs

crowns_img = crowns.to_crs(img_crs)

# --- Iterate and export ---
for i, (geom_las, geom_img) in enumerate(zip(crowns_las.geometry, crowns_img.geometry)):
    if geom_las is None or geom_las.is_empty or geom_img is None or geom_img.is_empty:
        print(f"Skipping crown {i}: empty/invalid geometry.")
        continue

    # ---------- LAS per crown ----------
    try:
        minx, miny, maxx, maxy = geom_las.bounds
        bounds = ([minx, maxx], [miny, maxy])
        pc = read_lidar(ept, ept_srs, bounds, crop_poly=True, poly=geom_las.wkt)

        if pc is None or (hasattr(pc, "__len__") and len(pc) == 0):
            print(f"No lidar points for crown {i}; skipping LAS.")
        else:
            las_path = las_out_dir / f"crown_{i}.las"
            write_las(pc, str(las_path), compress=False)
            print(f"Wrote {las_path}")
    except Exception as exc:
        print(f"Failed LAS on crown {i}: {exc}")

    # ---------- Raster clip+mask per crown ----------
    try:
        with rasterio.open(img_path) as src:
            # Get masked array (avoid filling with NaN while dtype is int)
            out_ma, out_transform = mask(
                src,
                [geom_img.__geo_interface__],
                crop=True,
                filled=False  # important: keep as masked array
            )

            # Upcast to float32 and fill mask with NaN
            # (masked elements include outside-crown and source nodata)
            out_ma = out_ma.astype("float32")
            out_arr = out_ma.filled(np.nan)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_arr.shape[1],
                "width":  out_arr.shape[2],
                "transform": out_transform,
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "LZW",
                "tiled": True,
            })

        img_path_out = img_out_dir / f"crown_{i}.tif"
        with rasterio.open(img_path_out, "w", **out_meta) as dst:
            dst.write(out_arr)
        print(f"Wrote {img_path_out}")

    except ValueError as ve:
        # Often raised when no pixels intersect the geometry
        print(f"No raster pixels for crown {i}; skipping image. ({ve})")
    except Exception as exc:
        print(f"Failed raster on crown {i}: {exc}")
