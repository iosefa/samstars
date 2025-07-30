#!/usr/bin/env python3
"""
Filter the cleaned training-point layer so that:

1. Every crown polygon keeps *only* the points whose class is
   consistent across all its points (crowns exhibiting > 1 class are
   removed entirely from the point layer).

2. Points whose *code* is in
      {"UNKNOWN", "UNKOWN", "UNPRED", "UNSURE"}  (case-insensitive)
   are removed.

3. Points that fall outside every crown are removed.

Finally, the script prints two frequency tables (and bar-chart
histograms) for the remaining points:
    • distribution of `code`
    • distribution of `class`
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────── paths ──────────────────────────────────────────
TRAIN_GPKG  = Path("./train_classifier.gpkg").expanduser()
CROWNS_GPKG = Path("~/Dropbox/crowns.gpkg").expanduser()

assert TRAIN_GPKG.exists(),  f"File not found: {TRAIN_GPKG}"
assert CROWNS_GPKG.exists(), f"File not found: {CROWNS_GPKG}"

# ──────────────────────── read layers ───────────────────────────────────────
pts    = gpd.read_file(TRAIN_GPKG)
crowns = gpd.read_file(CROWNS_GPKG)

# Common CRS
if pts.crs != crowns.crs:
    pts = pts.to_crs(crowns.crs)

# ──────────────────── annotate points with crown_id ─────────────────────────
crowns = crowns.reset_index(drop=True).copy()
crowns["crown_id"] = crowns.index

pts_crown = gpd.sjoin(
    pts,
    crowns[["crown_id", "geometry"]],
    how="left",
    predicate="within"
).drop(columns="index_right")

# ─────────────────────────── 1. drop bad codes ─────────────────────────────
BAD_CODES = {"UNKNOWN", "UNKOWN", "UNPRED", "UNSURE"}

mask_bad_code = pts_crown["code"].str.upper().isin(BAD_CODES)
n_bad_code = int(mask_bad_code.sum())
pts_crown = pts_crown.loc[~mask_bad_code]
print(f"Removed points with bad codes           : {n_bad_code:,}")

# ─────────────────────────── 2. drop no-crown ──────────────────────────────
mask_no_crown = pts_crown["crown_id"].isna()
n_no_crown = int(mask_no_crown.sum())
pts_crown = pts_crown.loc[~mask_no_crown]
print(f"Removed points outside any crown        : {n_no_crown:,}")

# ──────────────────── 3. drop crowns with >1 class ─────────────────────────
n_unique_per_crown = (
    pts_crown.groupby("crown_id", observed=True)["class"]
    .nunique()
)

conflicting_crowns = n_unique_per_crown[n_unique_per_crown > 1].index
mask_conflict = pts_crown["crown_id"].isin(conflicting_crowns)
n_conflict_pts = int(mask_conflict.sum())
n_conflict_crowns = len(conflicting_crowns)

pts_filtered = pts_crown.loc[~mask_conflict].copy()

print(f"Removed crowns with >1 class            : {n_conflict_crowns:,}")
print(f"…which contained points                 : {n_conflict_pts:,}")
print("────────────────────────────────────────────────────────────")
print(f"Remaining training points               : {len(pts_filtered):,}\n")

# ─────────────────────── summary plots / tables ────────────────────────────
def _print_and_plot(series: pd.Series, title: str) -> None:
    counts = series.value_counts().sort_index()
    print(f"{title}\n{counts.to_string()}\n")

    counts.plot.bar(rot=45)
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


_print_and_plot(pts_filtered["code"],  "Frequency of codes (filtered)")
_print_and_plot(pts_filtered["class"], "Frequency of classes (filtered)")

pts_filtered.to_file("trainings_filtered.gpkg", driver="GPKG")
