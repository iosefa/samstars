#!/usr/bin/env python3
"""
Figure 11 — 30 m canopy cover: crowns vs CAH (Jacobi) and difference

Panels:
  A) Crown-derived canopy fraction (30 m)
  B) CAH/Jacobi canopy fraction (forest classes only; 30 m)
  C) Difference (crowns − Jacobi)

Inputs
- crowns:  classified_xgb.gpkg with polygons (predicted_class) [EPSG:6635]
- AOI:     extent.gpkg                                          [EPSG:6635]
- Jacobi:  CAH_LandCover.gpkg (polygonized)                    [EPSG:26904]

Output
- manuscript/figures/output/figure11.png
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CROWNS_PATH = ROOT.parent / "classified_xgb.gpkg"
# AOI extent moved under manuscript/data/ahupuaa
AOI_PATH    = ROOT / "data" / "ahupuaa" / "extent.gpkg"
JACOBI_PATH = ROOT / "CAH_LandCover.gpkg"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure11.png"

CROWN_CLASS_FIELD = "predicted_class"
JACOBI_CODE_FIELD = "DN"
GRID_RES_M = 30.0

DN_TO_CLASS = {  # subset; used to flag forest DNs
    1:"Closed ohia wet forest", 2:"Closed koa-ohia wet forest", 3:"Low-stature ohia wet forest",
    4:"Open ohia wet forest", 5:"Open koa-ohia wet forest", 6:"Closed ohia mesic forest",
    7:"Closed hala forest", 8:"Closed koa-ohia mesic forest", 9:"Open ohia mesic forest",
    10:"Open koa-ohia mesic forest", 11:"Closed ohia dry forest",
    12:"Mixed mamane-naio-native trees dry woodland", 13:"Open ohia dry forest",
    14:"Open koa-mamane dry forest", 15:"Native mesic to dry forest and shrubland",
    16:"Mixed native-alien wet forest", 17:"Mixed native-alien mesic forest",
    18:"Mixed native-alien dry forest", 19:"Alien wet forest", 20:"Alien mesic forest",
    21:"Alien dry forest", 22:"Kiawe dry forest and shrubland", 23:"Plantation forest",
}
FOREST_DN = list(DN_TO_CLASS.keys())

plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Figure 13 (detail) inputs reused for bottom row
DETAIL_PATH = ROOT / "data" / "landcover_detail.gpkg"  # EPSG:6635


def ensure_crs(gdf, epsg):
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


def make_grid(aoi: gpd.GeoDataFrame, res: float):
    aoi_union = aoi.geometry.unary_union
    minx, miny, maxx, maxy = aoi_union.bounds
    nx = math.ceil((maxx - minx) / res)
    ny = math.ceil((maxy - miny) / res)
    cells = [box(minx + i*res, miny + j*res,
                 minx + (i+1)*res, miny + (j+1)*res)
             for i in range(nx) for j in range(ny)]
    grid = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=aoi.crs)
    grid = gpd.overlay(grid, aoi, how="intersection")
    grid["cell_area"] = grid.geometry.area
    return grid.reset_index(drop=True)


def canopy_fraction_by_cell(grid, crowns):
    inter = gpd.overlay(crowns[["geometry"]], grid[["cell_id","geometry","cell_area"]], how="intersection")
    if inter.empty:
        grid["canopy_frac"] = 0.0
        return grid
    inter["part_area"] = inter.geometry.area
    add = inter.groupby("cell_id")["part_area"].sum().rename("canopy_area")
    grid = grid.merge(add, on="cell_id", how="left")
    grid["canopy_area"] = grid["canopy_area"].fillna(0.0)
    grid["canopy_frac"] = grid["canopy_area"] / grid["cell_area"]
    return grid


def jacobi_fraction_by_cell(grid, jacobi):
    inter = gpd.overlay(grid[["cell_id","geometry","cell_area"]], jacobi[["geometry"]], how="intersection")
    if inter.empty:
        grid["jacobi_frac"] = 0.0
        return grid
    inter["part_area"] = inter.geometry.area
    add = inter.groupby("cell_id")["part_area"].sum().rename("jacobi_area")
    grid = grid.merge(add, on="cell_id", how="left")
    grid["jacobi_area"] = grid["jacobi_area"].fillna(0.0)
    grid["jacobi_frac"] = grid["jacobi_area"] / grid["cell_area"]
    return grid


def main():
    crowns = ensure_crs(clean_geoms(gpd.read_file(CROWNS_PATH)), 6635)
    aoi    = ensure_crs(clean_geoms(gpd.read_file(AOI_PATH)), 6635)
    jacobi = ensure_crs(gpd.read_file(JACOBI_PATH), 26904).to_crs(6635)
    if JACOBI_CODE_FIELD not in jacobi.columns:
        raise RuntimeError("Jacobi layer missing DN field")
    jacobi = gpd.overlay(jacobi[[JACOBI_CODE_FIELD, "geometry"]], aoi, how="intersection")
    jacobi = jacobi[jacobi[JACOBI_CODE_FIELD].isin(FOREST_DN)].copy()
    jacobi = clean_geoms(jacobi)

    grid = make_grid(aoi, GRID_RES_M)
    grid = canopy_fraction_by_cell(grid, crowns)
    grid = jacobi_fraction_by_cell(grid, jacobi)
    grid["canopy_frac"] = pd.to_numeric(grid["canopy_frac"], errors="coerce").fillna(0.0).clip(0, 1)
    grid["jacobi_frac"] = pd.to_numeric(grid["jacobi_frac"], errors="coerce").fillna(0.0).clip(0, 1)
    grid["diff_frac"] = grid["canopy_frac"] - grid["jacobi_frac"]

    # Load detailed extent (if available) for overlay in panels A and B
    detail = None
    try:
        detail = gpd.read_file(DETAIL_PATH)
        if detail.crs is None or (detail.crs.to_epsg() or -1) != 6635:
            detail = detail.to_crs(6635)
        detail = clean_geoms(detail)
    except Exception:
        detail = None

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16.5, 11.0))
    gs_main = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.08)

    # Prepare detailed grid once
    if detail is None:
        try:
            detail = gpd.read_file(DETAIL_PATH)
            if detail.crs is None or (detail.crs.to_epsg() or -1) != 6635:
                detail = detail.to_crs(6635)
            detail = clean_geoms(detail)
        except Exception:
            detail = None
    grid_d = gpd.overlay(grid, detail, how='intersection') if detail is not None and not detail.empty else grid

    # Column 0: CAH (A) over AOI, detail (D) below
    sub0 = gs_main[0, 0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.02)
    axA = fig.add_subplot(sub0[0, 0])
    axD = fig.add_subplot(sub0[1, 0])
    # A
    imA = grid.plot(column="jacobi_frac", ax=axA, cmap="viridis", vmin=0.0, vmax=1.0, linewidth=0, legend=False)
    aoi.boundary.plot(ax=axA, color="k", linewidth=0.4)
    try:
        if detail is not None and not detail.empty:
            detail.boundary.plot(ax=axA, color="#ff00ff", linewidth=2.0, alpha=1.0, zorder=4)
    except Exception:
        pass
    axA.set_axis_off(); axA.set_title("(a) Canopy cover (CAH, 30 m)", pad=10, fontsize=16)
    # D
    grid_d.plot(column="jacobi_frac", ax=axD, cmap="viridis", vmin=0.0, vmax=1.0, linewidth=0, legend=False)
    try:
        minx, miny, maxx, maxy = detail.total_bounds
        axD.set_xlim(minx, maxx); axD.set_ylim(miny, maxy)
    except Exception:
        pass
    try:
        if detail is not None and not detail.empty:
            detail.boundary.plot(ax=axD, color="#ff00ff", linewidth=2.0, alpha=1.0, zorder=4)
    except Exception:
        pass
    axD.set_axis_off()

    # Column 1: Crowns (B) over AOI, detail (E) below
    sub1 = gs_main[0, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.02)
    axB = fig.add_subplot(sub1[0, 0])
    axE = fig.add_subplot(sub1[1, 0])
    imB = grid.plot(column="canopy_frac", ax=axB, cmap="viridis", vmin=0.0, vmax=1.0, linewidth=0, legend=False)
    aoi.boundary.plot(ax=axB, color="k", linewidth=0.4)
    try:
        if detail is not None and not detail.empty:
            detail.boundary.plot(ax=axB, color="#ff00ff", linewidth=2.0, alpha=1.0, zorder=4)
    except Exception:
        pass
    axB.set_axis_off(); axB.set_title("(b) Canopy cover (SAM–StarDist, 30 m)", pad=10, fontsize=16)
    grid_d.plot(column="canopy_frac", ax=axE, cmap="viridis", vmin=0.0, vmax=1.0, linewidth=0, legend=False)
    try:
        minx, miny, maxx, maxy = detail.total_bounds
        axE.set_xlim(minx, maxx); axE.set_ylim(miny, maxy)
    except Exception:
        pass
    try:
        if detail is not None and not detail.empty:
            detail.boundary.plot(ax=axE, color="#ff00ff", linewidth=2.0, alpha=1.0, zorder=4)
    except Exception:
        pass
    axE.set_axis_off()

    # Column 2: Difference (C) over AOI, detail (F) below
    sub2 = gs_main[0, 2].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.02)
    axC = fig.add_subplot(sub2[0, 0])
    axF = fig.add_subplot(sub2[1, 0])
    dabs = float(np.nanmax(np.abs(grid["diff_frac"].to_numpy()))) if len(grid) else 0.0
    dmax = max(dabs, 0.5)
    imC = grid.plot(column="diff_frac", ax=axC, cmap="RdBu", vmin=-dmax, vmax=dmax, linewidth=0, legend=False)
    aoi.boundary.plot(ax=axC, color="k", linewidth=0.4)
    try:
        if detail is not None and not detail.empty:
            detail.boundary.plot(ax=axC, color="#ff00ff", linewidth=2.0, alpha=1.0, zorder=4)
    except Exception:
        pass
    axC.set_axis_off(); axC.set_title("(c) Difference (SAM–StarDist − CAH)", pad=10, fontsize=16)
    try:
        grid_d.plot(column="diff_frac", ax=axF, cmap="RdBu", vmin=-dmax, vmax=dmax, linewidth=0, legend=False)
    except Exception:
        try:
            dabs_d = float(np.nanmax(np.abs(grid_d["diff_frac"].to_numpy()))) if len(grid_d) else 0.0
            dmax_d = max(dabs_d, 0.5)
            grid_d.plot(column="diff_frac", ax=axF, cmap="RdBu", vmin=-dmax_d, vmax=dmax_d, linewidth=0, legend=False)
        except Exception:
            pass
    try:
        minx, miny, maxx, maxy = detail.total_bounds
        axF.set_xlim(minx, maxx); axF.set_ylim(miny, maxy)
    except Exception:
        pass
    try:
        if detail is not None and not detail.empty:
            detail.boundary.plot(ax=axF, color="#ff00ff", linewidth=2.0, alpha=1.0, zorder=4)
    except Exception:
        pass
    axF.set_axis_off()

    # Add tall colorbars per column
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    # Single merged viridis colorbar for columns A/B (0..1), aligned to column 2 (B) for clean spacing
    p_top, p_bot = axB.get_position(), axE.get_position()
    cax_v = fig.add_axes([p_top.x1 + 0.01, p_bot.y0, 0.018, p_top.y1 - p_bot.y0])
    sm_v = ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap='viridis')
    sm_v.set_array([])
    cb_v = fig.colorbar(sm_v, cax=cax_v)
    cb_v.set_label('Fraction')
    cb_v.ax.tick_params(labelsize=11)
    # Column 2 colorbar (RdBu -dmax..dmax)
    p_top, p_bot = axC.get_position(), axF.get_position()
    cax2 = fig.add_axes([p_top.x1 + 0.01, p_bot.y0, 0.018, p_top.y1 - p_bot.y0])
    sm2 = ScalarMappable(norm=mcolors.Normalize(vmin=-dmax, vmax=dmax), cmap='RdBu')
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cax2)
    cb2.set_label('Difference')
    cb2.ax.tick_params(labelsize=11)

    # Scale bars in each detail panel (D, E, F) — fixed 2 km, with outer padding around bar+label
    def _add_detail_scalebar(ax, bounds):
        try:
            minx, miny, maxx, maxy = bounds
            width = float(maxx - minx); height = float(maxy - miny)
            bar_len = 2000.0  # fixed 2 km
            pad_x = width * 0.03; pad_y = height * 0.03
            bar_h = max(height * 0.006, bar_len * 0.01)
            # Content (bar + label) layout
            inner_x0 = minx + pad_x
            inner_y0 = miny + pad_y
            # Increase spacing between bar and label
            label_gap = pad_y * 0.40
            # More outer padding around entire content (bar + label)
            outer_px = width * 0.020
            outer_py = height * 0.020
            # Background rectangle around entire content
            bg_x = inner_x0 - outer_px
            bg_y = inner_y0 - outer_py
            bg_w = bar_len + 2 * outer_px
            bg_h = bar_h + label_gap + 2 * outer_py
            import matplotlib.patches as mpatches
            bg = mpatches.Rectangle((bg_x, bg_y), bg_w, bg_h, facecolor='white', edgecolor='black', linewidth=0.6, zorder=3.8)
            ax.add_patch(bg)
            # Bar centered within content horizontally (left aligned), at inner origin vertically
            ax.add_patch(mpatches.Rectangle((inner_x0, inner_y0), bar_len, bar_h, facecolor='black', edgecolor='white', linewidth=0.8, zorder=4))
            # Label inside bg above bar
            label_val = "2 km"
            # Place label centered in the label band inside the background box
            # Nudge label even higher within the background box
            ax.text(inner_x0 + bar_len/2, inner_y0 + bar_h + (label_gap * 0.85), label_val,
                    ha='center', va='center', color='black', fontsize=11, zorder=4)
        except Exception:
            pass

    try:
        if detail is not None and not detail.empty:
            bounds = detail.total_bounds
            _add_detail_scalebar(axD, bounds)
            _add_detail_scalebar(axE, bounds)
            _add_detail_scalebar(axF, bounds)
    except Exception:
        pass

    # Connector lines per column (corners -> corners)
    try:
        if detail is not None and not detail.empty:
            minx, miny, maxx, maxy = detail.total_bounds
            corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
            from matplotlib.patches import ConnectionPatch
            for top_ax, bot_ax in [(axA, axD), (axB, axE), (axC, axF)]:
                for (cx, cy) in corners:
                    conn = ConnectionPatch(xyA=(cx, cy), xyB=(cx, cy), coordsA='data', coordsB='data',
                                            axesA=top_ax, axesB=bot_ax, color="#ff00ff", linewidth=1.2, alpha=0.9)
                    fig.add_artist(conn)
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
