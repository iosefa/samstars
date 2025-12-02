#!/usr/bin/env python3
"""
Figure 11 magnified panels only (detail view for canopy cover comparison).

This script reuses the Figure 11 data preparation but only renders the
three detail panels: CAH canopy fraction, SAM–StarDist canopy fraction,
and their difference over the magnified extent.

Output
- manuscript/figures/output/tmp_subplot11.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[2]
CROWNS_PATH = ROOT.parent / "classified_xgb.gpkg"
AOI_PATH = ROOT / "data" / "ahupuaa" / "extent.gpkg"
JACOBI_PATH = ROOT.parent / "CAH_LandCover.gpkg"
DETAIL_PATH = ROOT / "data" / "landcover_detail.gpkg"

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "tmp_subplot11.png"

CROWN_CLASS_FIELD = "predicted_class"
JACOBI_CODE_FIELD = "DN"
GRID_RES_M = 30.0

DN_TO_CLASS = {
    1: "Closed ohia wet forest",
    2: "Closed koa-ohia wet forest",
    3: "Low-stature ohia wet forest",
    4: "Open ohia wet forest",
    5: "Open koa-ohia wet forest",
    6: "Closed ohia mesic forest",
    7: "Closed hala forest",
    8: "Closed koa-ohia mesic forest",
    9: "Open ohia mesic forest",
    10: "Open koa-ohia mesic forest",
    11: "Closed ohia dry forest",
    12: "Mixed mamane-naio-native trees dry woodland",
    13: "Open ohia dry forest",
    14: "Open koa-mamane dry forest",
    15: "Native mesic to dry forest and shrubland",
    16: "Mixed native-alien wet forest",
    17: "Mixed native-alien mesic forest",
    18: "Mixed native-alien dry forest",
    19: "Alien wet forest",
    20: "Alien mesic forest",
    21: "Alien dry forest",
    22: "Kiawe dry forest and shrubland",
    23: "Plantation forest",
}
FOREST_DN = list(DN_TO_CLASS.keys())


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    """Project GeoDataFrame to target EPSG if needed."""
    if gdf.crs is None or (gdf.crs.to_epsg() or -1) != epsg:
        return gdf.to_crs(epsg)
    return gdf


def clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf[gdf.geometry.notna()].copy()
    out = out[~out.geometry.is_empty]
    try:
        out = out[out.geometry.is_valid]
    except Exception:
        pass
    return out


def make_grid(aoi: gpd.GeoDataFrame, res: float) -> gpd.GeoDataFrame:
    union = aoi.geometry.unary_union
    minx, miny, maxx, maxy = union.bounds
    nx = int(np.ceil((maxx - minx) / res))
    ny = int(np.ceil((maxy - miny) / res))
    cells = [
        box(minx + i * res, miny + j * res, minx + (i + 1) * res, miny + (j + 1) * res)
        for i in range(nx)
        for j in range(ny)
    ]
    grid = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=aoi.crs)
    grid = gpd.overlay(grid, aoi, how="intersection")
    grid["cell_area"] = grid.geometry.area
    return grid.reset_index(drop=True)


def canopy_fraction_by_cell(grid: gpd.GeoDataFrame, crowns: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    inter = gpd.overlay(crowns[["geometry"]], grid[["cell_id", "geometry", "cell_area"]], how="intersection")
    if inter.empty:
        grid["canopy_frac"] = 0.0
        return grid
    inter["part_area"] = inter.geometry.area
    add = inter.groupby("cell_id")["part_area"].sum().rename("canopy_area")
    grid = grid.merge(add, on="cell_id", how="left")
    grid["canopy_area"] = grid["canopy_area"].fillna(0.0)
    grid["canopy_frac"] = grid["canopy_area"] / grid["cell_area"]
    return grid


def jacobi_fraction_by_cell(grid: gpd.GeoDataFrame, jacobi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    inter = gpd.overlay(grid[["cell_id", "geometry", "cell_area"]], jacobi[["geometry"]], how="intersection")
    if inter.empty:
        grid["jacobi_frac"] = 0.0
        return grid
    inter["part_area"] = inter.geometry.area
    add = inter.groupby("cell_id")["part_area"].sum().rename("jacobi_area")
    grid = grid.merge(add, on="cell_id", how="left")
    grid["jacobi_area"] = grid["jacobi_area"].fillna(0.0)
    grid["jacobi_frac"] = grid["jacobi_area"] / grid["cell_area"]
    return grid


def add_scale_bar(ax, bounds, length_m=2000.0):
    """Draw a scale bar with background box inside the subplot."""
    import matplotlib.patches as mpatches

    minx, miny, maxx, maxy = bounds
    width = float(maxx - minx)
    height = float(maxy - miny)
    pad_x = width * 0.03
    pad_y = height * 0.03
    bar_h = max(height * 0.006, length_m * 0.01)

    inner_x0 = minx + pad_x
    inner_y0 = miny + pad_y
    label_gap = pad_y * 0.45
    outer_px = width * 0.02
    outer_py = height * 0.02

    bg_x = inner_x0 - outer_px
    bg_y = inner_y0 - outer_py
    bg_w = length_m + 2 * outer_px
    bg_h = bar_h + label_gap + 2 * outer_py

    bg = mpatches.Rectangle(
        (bg_x, bg_y), bg_w, bg_h, facecolor="white", edgecolor="black", linewidth=0.6, zorder=3.8
    )
    ax.add_patch(bg)
    ax.add_patch(
        mpatches.Rectangle(
            (inner_x0, inner_y0),
            length_m,
            bar_h,
            facecolor="black",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    )
    ax.text(
        inner_x0 + length_m / 2,
        inner_y0 + bar_h + (label_gap * 0.8),
        f"{int(length_m/1000)} km",
        ha="center",
        va="center",
        color="black",
        fontsize=11,
        zorder=4,
    )


def main():
    crowns = ensure_crs(clean_geoms(gpd.read_file(CROWNS_PATH)), 6635)
    aoi = ensure_crs(clean_geoms(gpd.read_file(AOI_PATH)), 6635)
    jacobi = ensure_crs(gpd.read_file(JACOBI_PATH), 26904).to_crs(6635)
    if JACOBI_CODE_FIELD not in jacobi.columns:
        raise RuntimeError("Jacobi layer missing DN field")
    jacobi = gpd.overlay(jacobi[[JACOBI_CODE_FIELD, "geometry"]], aoi, how="intersection")
    jacobi = jacobi[jacobi[JACOBI_CODE_FIELD].isin(FOREST_DN)].copy()
    jacobi = clean_geoms(jacobi)

    detail = gpd.read_file(DETAIL_PATH)
    detail = ensure_crs(clean_geoms(detail), 6635)
    if detail.empty:
        raise RuntimeError("Detail extent is empty; cannot render magnified panels.")

    grid = make_grid(aoi, GRID_RES_M)
    grid = canopy_fraction_by_cell(grid, crowns)
    grid = jacobi_fraction_by_cell(grid, jacobi)
    grid["canopy_frac"] = pd.to_numeric(grid["canopy_frac"], errors="coerce").fillna(0.0).clip(0, 1)
    grid["jacobi_frac"] = pd.to_numeric(grid["jacobi_frac"], errors="coerce").fillna(0.0).clip(0, 1)
    grid["diff_frac"] = grid["canopy_frac"] - grid["jacobi_frac"]

    grid_detail = gpd.overlay(grid, detail[["geometry"]], how="intersection")
    if grid_detail.empty:
        raise RuntimeError("No grid cells intersect the detail extent.")

    bounds = detail.total_bounds
    dabs = float(np.nanmax(np.abs(grid_detail["diff_frac"].to_numpy()))) if len(grid_detail) else 0.0
    dmax = max(dabs, 0.5)

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 12,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=False)
    fig.subplots_adjust(wspace=0.022)
    titles = [
        "(a) Canopy cover\n(CAH, detail)",
        "(b) Canopy cover\n(SAM–StarDist, detail)",
        "(c) Difference\n(SAM–StarDist − CAH, detail)",
    ]
    columns = ["jacobi_frac", "canopy_frac", "diff_frac"]
    cmaps = ["viridis", "viridis", "RdBu"]
    norms = [
        (0.0, 1.0),
        (0.0, 1.0),
        (-dmax, dmax),
    ]

    for ax, title, col, cmap, (vmin, vmax) in zip(axes, titles, columns, cmaps, norms):
        grid_detail.plot(column=col, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0, legend=False)
        detail.boundary.plot(ax=ax, color="#ff00ff", linewidth=1.6, alpha=1.0, zorder=4)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_axis_off()
        ax.text(
            0.5,
            1.04,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            linespacing=1.1,
        )
        add_scale_bar(ax, bounds)

        # Add colorbar for each subplot
        mappable = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        mappable.set_array([])
        cb = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.012)
        cb.ax.tick_params(labelsize=10)
        if col != "diff_frac":
            cb.set_label("Fraction")
        else:
            cb.set_label("Difference")

    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
