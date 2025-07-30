#!/usr/bin/env python3
"""
Produce basic summary statistics for the cleaned training points layer
(`train_classifier.gpkg`) against an existing crowns layer
(`~/Dropbox/crowns.gpkg`).

The script reports:
• ────────────────────────────────────────────────────────────────────────────
  1.   # points that fall inside at least one crown polygon
       # points that do *not* intersect any crown
       percentage outside

  2.   Frequency tables (and bar-chart histograms) for the
       a) `code` column
       b) `class` column

  3.   # crowns that contain points belonging to *more than one* class
• ────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# ──────────────────────── file locations ────────────────────────────────────
TRAIN_GPKG  = Path("./train_classifier.gpkg").expanduser()
CROWNS_GPKG = Path("~/Dropbox/crowns.gpkg").expanduser()

assert TRAIN_GPKG.exists(),  f"File not found: {TRAIN_GPKG}"
assert CROWNS_GPKG.exists(), f"File not found: {CROWNS_GPKG}"

# ────────────────────────── read layers ─────────────────────────────────────
points  = gpd.read_file(TRAIN_GPKG)
crowns  = gpd.read_file(CROWNS_GPKG)

# Ensure a common CRS
if points.crs != crowns.crs:
    points = points.to_crs(crowns.crs)

# ────────────────────────── point-in-polygon test ──────────────────────────
# Spatial join: keep the crowns' index so we can refer back later
crowns["crown_id"] = crowns.index  # guarantee an identifier
points_in_crowns = gpd.sjoin(
    points,
    crowns[["crown_id", "geometry"]],
    how="inner",
    predicate="within"
)

n_points_total = len(points)
n_points_inside = len(points_in_crowns)
n_points_outside = n_points_total - n_points_inside
pct_outside = 100 * n_points_outside / n_points_total if n_points_total else 0

print("─ Point coverage ────────────────────────────────────────────────")
print(f"Total points                : {n_points_total:,}")
print(f"Points INSIDE a crown       : {n_points_inside:,}")
print(f"Points OUTSIDE any crown    : {n_points_outside:,}"
      f"  ({pct_outside:.1f} %)")
print()

# ─────────────────────────── code / class stats ─────────────────────────────
def _print_and_plot_counts(series: pd.Series, title: str):
    counts = series.value_counts().sort_index()
    print(f"─ {title} ─")
    print(counts.to_string())
    print()

    # Plot histogram
    counts.plot.bar(rot=45)
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


_print_and_plot_counts(points["code"],  "Frequency of codes")
_print_and_plot_counts(points["class"], "Frequency of classes")

# ───────────────────── crowns with conflicting classes ──────────────────────
# For each crown, how many *distinct* class labels are present?
classes_per_crown = (
    points_in_crowns.groupby("crown_id")["class"]
    .nunique()
)

n_conflicting_crowns = (classes_per_crown > 1).sum()
print("─ Crowns with mixed classes ────────────────────────────────────")
print(f"Crowns that contain >1 class label: {n_conflicting_crowns:,}")