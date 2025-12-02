#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.prepared import prep
from shapely.ops import unary_union as shp_union
from pyforestscan.handlers import read_lidar

# -------------------
# CONFIG / PATHS (Set A)
# -------------------
# EPT source for HAG
ept = "/Users/iosefa/repos/obia/docs/example_data/full/ept/ept.json"
ept_srs = "EPSG:32605"
final_srs = "EPSG:6635"    # reproject outputs back to this CRS

# Inputs (Set A predictions and validation)
base = "manuscript/data/set_a/"
det_in = base + "detectron_crowns.gpkg"
df_in  = base + "deepforest_crowns.gpkg"
sam_in = base + "sam_crowns.gpkg"
val_in = base + "validation_crowns.gpkg"

# Outputs with 'ch' field added
det_out = base + "detectron_crowns_with_ch.gpkg"
df_out  = base + "deepforest_crowns_with_ch.gpkg"
sam_out = base + "sam_crowns_with_ch.gpkg"
val_out = base + "validation_crowns_with_ch.gpkg"


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
    poly_union = gdf.union_all() if hasattr(gdf, "union_all") else gdf.geometry.unary_union
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


def get_plot_extents_in_ept() -> list:
    """Return list of (plot_id, extent_geom) in ept_srs CRS.
    Prefers setA_plot{i}_extent.gpkg; falls back to setA_plot{i}.gpkg.
    """
    out = []
    for i in range(1, 7):
        p1 = Path(base) / f"setA_plot{i}_extent.gpkg"
        p2 = Path(base) / f"setA_plot{i}.gpkg"
        p = p1 if p1.exists() else p2
        if not Path(p).exists():
            continue
        g = gpd.read_file(p)
        if g.empty:
            continue
        if g.crs is None:
            raise ValueError(f"Plot extent {p} has no CRS.")
        g = g.to_crs(ept_srs)
        # Use unary union in case of multi features
        geom = g.geometry.unary_union
        out.append((i, geom))
    return out


def process_by_plot():
    # Load all method layers in EPT CRS once
    det = load_polys_in_ept_crs(det_in, ept_srs)
    df  = load_polys_in_ept_crs(df_in,  ept_srs)
    sam = load_polys_in_ept_crs(sam_in, ept_srs)
    val = load_polys_in_ept_crs(val_in, ept_srs)

    acc = {"det": [], "df": [], "sam": [], "val": []}

    plots = get_plot_extents_in_ept()
    if not plots:
        raise SystemExit("No plot extents found in manuscript/data/set_a/")

    for pid, geom in plots:
        # subset polygons that intersect this plot (keep original geometry)
        det_s = det.loc[det.geometry.intersects(geom)].copy()
        df_s  = df.loc[df.geometry.intersects(geom)].copy()
        sam_s = sam.loc[sam.geometry.intersects(geom)].copy()
        val_s = val.loc[val.geometry.intersects(geom)].copy()

        # Skip if nothing in this plot
        if all(g.empty for g in (det_s, df_s, sam_s, val_s)):
            continue

        # Read points once for all subsets in this plot
        geoms_all = []
        for g in (det_s, df_s, sam_s, val_s):
            if not g.empty:
                geoms_all.extend(list(g.geometry))
        if not geoms_all:
            continue
        union_geom = shp_union(geoms_all)
        pts = read_points_for_gdf(gpd.GeoDataFrame(geometry=[union_geom], crs=ept_srs))

        # Compute ch per subset and append
        if not det_s.empty:
            det_s = det_s.copy()
            det_s["ch"] = compute_stat_per_polygon(det_s, pts, stat="max")
            acc["det"].append(det_s)
        if not df_s.empty:
            df_s = df_s.copy()
            df_s["ch"] = compute_stat_per_polygon(df_s, pts, stat="max")
            acc["df"].append(df_s)
        if not sam_s.empty:
            sam_s = sam_s.copy()
            sam_s["ch"] = compute_stat_per_polygon(sam_s, pts, stat="max")
            acc["sam"].append(sam_s)
        if not val_s.empty:
            val_s = val_s.copy()
            val_s["ch"] = compute_stat_per_polygon(val_s, pts, stat="median")
            acc["val"].append(val_s)

    # Concatenate and post-process per method
    def finalize(parts, drop_below):
        if not parts:
            return gpd.GeoDataFrame(geometry=[], crs=ept_srs)
        g = pd.concat(parts, ignore_index=True)
        if drop_below is not None and "ch" in g.columns:
            mask = np.isfinite(g["ch"]) & (g["ch"] >= float(drop_below))
            g = g.loc[mask].copy()
        return g.to_crs(final_srs)

    det_f = finalize(acc["det"], drop_below=3)
    df_f  = finalize(acc["df"],  drop_below=3)
    sam_f = finalize(acc["sam"], drop_below=3)
    val_f = finalize(acc["val"], drop_below=3)

    det_f.to_file(det_out, driver="GPKG")
    df_f.to_file(df_out, driver="GPKG")
    sam_f.to_file(sam_out, driver="GPKG")
    val_f.to_file(val_out, driver="GPKG")

    print(f"[OK] Wrote {det_out} | n={len(det_f)}")
    print(f"[OK] Wrote {df_out} | n={len(df_f)}")
    print(f"[OK] Wrote {sam_out} | n={len(sam_f)}")
    print(f"[OK] Wrote {val_out} | n={len(val_f)}")


# -------------------
# RUN ALL (Set A)
# -------------------

process_by_plot()
