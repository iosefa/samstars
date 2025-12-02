#!/usr/bin/env python3
import numpy as np
import geopandas as gpd
from shapely.prepared import prep
from pyforestscan.handlers import read_lidar

# -------------------
# CONFIG / PATHS
# -------------------
ept = "/Users/iosefa/repos/obia/docs/example_data/full/ept/ept.json"
ept_srs = "EPSG:32605"
final_srs = "EPSG:6635"  # reproject outputs back to this CRS

# Inputs
cnn_in = "manuscript/figures/set_c/set_c_cnn_predicted.gpkg"
df_in  = "manuscript/figures/set_c/set_c_df_predicted.gpkg"
val_in = "manuscript/figures/set_c/set_c_validation.gpkg"

# Outputs
cnn_out = "manuscript/figures/set_c/set_c_cnn_predicted_with_ch.gpkg"
df_out  = "manuscript/figures/set_c/set_c_df_predicted_with_ch.gpkg"
val_out = "manuscript/figures/set_c/set_c_validation_with_ch.gpkg"

# -------------------
# UTILITIES
# -------------------

def load_polys_in_ept_crs(gpkg_path: str, target_srs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        raise ValueError(f"{gpkg_path} has no CRS set.")
    return gdf.to_crs(target_srs)

def build_bounds_str(gdf: gpd.GeoDataFrame) -> str:
    minx, miny, maxx, maxy = gdf.total_bounds
    return f"([{minx},{maxx}], [{miny},{maxy}])"

def read_points_for_gdf(gdf: gpd.GeoDataFrame):
    """Robust EPT read: try bounds+poly, fallback to poly-only to dodge PDAL range quirks."""
    bounds_str = build_bounds_str(gdf)
    poly_union = gdf.union_all()
    try:
        return read_lidar(
            ept,
            ept_srs,
            bounds_str,
            crop_poly=True,
            poly=poly_union.wkt,
        )[0]
    except RuntimeError as e:
        if "No valid minimum value for range" in str(e) or "filters.range" in str(e):
            return read_lidar(
                ept,
                ept_srs,
                None,
                crop_poly=True,
                poly=poly_union.wkt,
            )[0]
        raise

# Try fast vectorized contains; fall back to safe loop if unavailable
try:
    from shapely import vectorized as shv
    def points_in_poly_mask(geom, xi, yi):
        return shv.contains(geom, xi, yi)
except Exception:
    def points_in_poly_mask(geom, xi, yi):
        gprep = prep(geom)
        from shapely.geometry import Point
        return np.fromiter(
            (gprep.contains(Point(xx, yy)) for xx, yy in zip(xi, yi)),
            count=len(xi),
            dtype=bool,
        )

def choose_hag_key(pts) -> str:
    names = pts.dtype.names
    hag_key = next((k for k in ("HeightAboveGround", "HAG") if k in names), None)
    if hag_key is None:
        raise ValueError(f"No HeightAboveGround/HAG in point fields: {names}")
    return hag_key

def compute_stat_per_polygon(gdf: gpd.GeoDataFrame, pts, stat: str) -> np.ndarray:
    hag_key = choose_hag_key(pts)
    X = pts["X"].astype(float)
    Y = pts["Y"].astype(float)
    H = pts[hag_key].astype(float)

    out = np.full(len(gdf), np.nan, dtype=float)

    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        gxmin, gymin, gxmax, gymax = geom.bounds
        bbox_mask = (X >= gxmin) & (X <= gxmax) & (Y >= gymin) & (Y <= gymax)
        if not np.any(bbox_mask):
            continue
        idxs = np.nonzero(bbox_mask)[0]
        inside = points_in_poly_mask(geom, X[idxs], Y[idxs])
        if not np.any(inside):
            continue
        vals = H[idxs][inside]
        if stat == "max":
            out[i] = np.nanmax(vals)
        elif stat == "median":
            out[i] = np.nanmedian(vals)
        elif stat == "mean":
            out[i] = np.nanmean(vals)
        else:
            raise ValueError("stat must be 'max' or 'median' or 'mean'")
    return out

def process_one(in_path: str, out_path: str, stat: str, drop_below: float | None = None):
    gdf = load_polys_in_ept_crs(in_path, ept_srs)
    n_start = len(gdf)

    pts = read_points_for_gdf(gdf)
    ch_vals = compute_stat_per_polygon(gdf, pts, stat=stat)
    gdf["ch"] = ch_vals

    if drop_below is not None:
        mask = np.isfinite(gdf["ch"]) & (gdf["ch"] >= float(drop_below))
        removed = n_start - mask.sum()
        gdf = gdf[mask].copy()
    else:
        removed = 0

    gdf = gdf.to_crs(final_srs)
    gdf.to_file(out_path, driver="GPKG")

    print(f"Wrote {out_path} with 'ch' = {stat} HAG per polygon. "
          f"Removed {removed} of {n_start} polygons.")

# -------------------
# RUN ALL THREE
# -------------------

# 1) CNN predicted → max HAG (drop < 3)
process_one(cnn_in, cnn_out, stat="max", drop_below=3)

# 2) DeepForest predicted → max HAG (drop < 3)
process_one(df_in, df_out, stat="max", drop_below=3)

# 3) Validation → median HAG (drop < 3)
process_one(val_in, val_out, stat="max", drop_below=3.0)