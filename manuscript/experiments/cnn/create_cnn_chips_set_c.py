#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask

# Ensure repo root for local libs
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyforestscan.handlers import read_lidar, write_las


# --- Inputs (edit as needed) ---
CROWNS_GPKG = Path("manuscript/data/set_c/set_c_cnn_predicted.gpkg")
IMAGE_TIF   = Path("manuscript/data/set_c/set_c.tif")  # Set C orthomosaic
EPT_JSON    = Path("manuscript/data/set_c/ept/ept.json")
EPT_SRS     = "EPSG:32605"

# --- Outputs ---
OUT_DIR     = Path("manuscript/data/set_c")
LAS_DIR     = OUT_DIR / "crowns_las"
IMG_DIR     = OUT_DIR / "crowns_img"
MANIFEST_CSV= OUT_DIR / "chips_manifest_set_c.csv"

LAS_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Read predicted crowns polygons
    gdf = gpd.read_file(CROWNS_GPKG)
    if gdf.crs is None:
        raise RuntimeError("Crowns GPKG lacks CRS.")

    # Prepare crown geometries in EPT and image CRSs
    with rasterio.open(IMAGE_TIF) as src:
        img_crs = src.crs

    crowns_las = gdf.to_crs(EPT_SRS)
    crowns_img = gdf.to_crs(img_crs)

    records = []  # (key, pid)

    for i, (geom_las, geom_img) in enumerate(zip(crowns_las.geometry, crowns_img.geometry)):
        key = f"crown_{i}"
        records.append((key, i))

        if geom_las is None or geom_las.is_empty or geom_img is None or geom_img.is_empty:
            print(f"Skipping crown {i}: empty/invalid geometry.")
            continue

        # ---------- LAS per crown ----------
        try:
            minx, miny, maxx, maxy = geom_las.bounds
            bounds = ([minx, maxx], [miny, maxy])
            pc = read_lidar(str(EPT_JSON), EPT_SRS, bounds, crop_poly=True, poly=geom_las.wkt)
            if pc is None or (hasattr(pc, "__len__") and len(pc) == 0):
                print(f"No lidar points for crown {i}; skipping LAS.")
            else:
                las_path = LAS_DIR / f"{key}.las"
                write_las(pc, str(las_path), compress=False)
                print(f"Wrote {las_path}")
        except Exception as exc:
            print(f"Failed LAS on crown {i}: {exc}")

        # ---------- Raster clip+mask per crown ----------
        try:
            with rasterio.open(IMAGE_TIF) as src:
                out_ma, out_transform = mask(
                    src,
                    [geom_img.__geo_interface__],
                    crop=True,
                    filled=False
                )
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

            img_path_out = IMG_DIR / f"{key}.tif"
            with rasterio.open(img_path_out, "w", **out_meta) as dst:
                dst.write(out_arr)
            print(f"Wrote {img_path_out}")
        except ValueError as ve:
            print(f"No raster pixels for crown {i}; skipping image. ({ve})")
        except Exception as exc:
            print(f"Failed raster on crown {i}: {exc}")

    # Save manifest
    pd.DataFrame(records, columns=["key","pid"]).to_csv(MANIFEST_CSV, index=False)
    print(f"[OK] Wrote manifest → {MANIFEST_CSV}")


if __name__ == "__main__":
    main()

