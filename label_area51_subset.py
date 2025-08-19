#!/usr/bin/env python3
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# --- inputs ---
validated_fp = "/Users/iosefa/repos/sam/area51_validated_subset1.gpkg"
labels_fp    = "/Users/iosefa/repos/sam/area51_labels_subset1.gpkg"
out_fp       = "/Users/iosefa/repos/sam/area51_labelled_validated_subset1.gpkg"

# --- 1) read ---
validated = gpd.read_file(validated_fp)
labels    = gpd.read_file(labels_fp)

# sanity: CRS
if validated.crs != labels.crs:
    raise ValueError(f"CRS mismatch: validated={validated.crs}, labels={labels.crs}")

# --- 2) map code → class on labels ---
code2class = {
    "METPOL": 1,
    "GREROB": 2,
    "DIOSAN": 4,
}
labels["class"] = labels.get("code").map(code2class).fillna(6).astype(int)

# keep only what we need for the transfer
labels_min = labels[["class", "geometry"]].copy()

# --- 3) assign class to validated by largest-area overlap ---
# Compute intersections so we can pick the label class with the maximum intersected area per validated polygon
# Keep original row id to map back
validated = validated.reset_index().rename(columns={"index": "_val_id"})
labels_min = labels_min.reset_index().rename(columns={"index": "_lab_id"})

# Use overlay intersection (handles polygons robustly)
inter = gpd.overlay(validated[["_val_id", "geometry"]],
                    labels_min[["_lab_id", "class", "geometry"]],
                    how="intersection")

# If nothing intersects, we’ll fall back to sjoin (rare but just in case)
if inter.empty:
    sj = gpd.sjoin(validated[["_val_id", "geometry"]],
                   labels_min[["class", "geometry"]],
                   how="left",
                   predicate="intersects")
    class_by_val = sj.groupby("_val_id")["class"].agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else 6)
else:
    inter["__area"] = inter.geometry.area
    # for each validated polygon, keep the class of the intersecting label with the largest area
    inter_sorted = inter.sort_values(["_val_id", "__area"], ascending=[True, False])
    class_by_val = inter_sorted.drop_duplicates("_val_id").set_index("_val_id")["class"]

# attach class back to validated; default to 6 if no match
validated["class"] = validated["_val_id"].map(class_by_val).fillna(6).astype(int)

# --- 4) write output ---
# If the layer exists, overwrite by using a new filename or driver-specific option.
validated.drop(columns=["_val_id"], errors="ignore").to_file(out_fp, driver="GPKG")

print(f"Done. Wrote: {out_fp}")