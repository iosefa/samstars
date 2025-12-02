#!/usr/bin/env python3
# section34_landscape_stats_and_jacobi.py
# ============================================================
# Landscape-scale crown stats + per-species summaries,
# comparison to Jacobi land cover (forested classes only),
# clipped to AOI (extent.gpkg). Writes CSVs + two figures:
#   - Figure 12 (AOI map + species composition bars)
#   - Figure 13 (30 m canopy cover: crowns vs Jacobi + difference)
# Optional: scatter plot of canopy fractions (30 m).
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.lines import Line2D

# ------------------------
# USER CONFIG
# ------------------------
CROWNS_PATH   = "classified_xgb.gpkg"   # EPSG:6635, polygons with predicted_class
JACOBI_PATH   = "CAH_LandCover.gpkg"    # EPSG:26904 (statewide), polygonized land cover
AOI_PATH      = "extent.gpkg"           # EPSG:6635, study AOI
GRID_RES_M    = 30.0                    # grid size for cell stats and agreement
OUT_PREFIX    = "pw_section34"

# Field names
CROWN_CLASS_FIELD = "predicted_class"   # species class code on crowns
JACOBI_CODE_FIELD = "DN"                # integer code of Jacobi land cover

# Class labels and plotting order
SPECIES_LABELS = {1:"METPOL", 2:"GREROB", 3:"ACAKOA", 4:"DIOSAN", 5:"PROPAL", 6:"OTHER"}
SPECIES_ORDER  = ["METPOL","ACAKOA","GREROB","DIOSAN","PROPAL","OTHER"]

# Plot palette (color-blind friendly-ish)
PALETTE = {
    "METPOL": "#1f78b4",  # blue
    "ACAKOA": "#6a3d9a",  # purple
    "GREROB": "#33a02c",  # green
    "DIOSAN": "#e31a1c",  # red
    "PROPAL": "#ff7f00",  # orange
    "OTHER":  "#b2b2b2",  # grey
}

# ------------------------
# Jacobi DN → class name mapping
# ------------------------
DN_TO_CLASS = {
    1:"Closed ohia wet forest", 2:"Closed koa-ohia wet forest", 3:"Low-stature ohia wet forest",
    4:"Open ohia wet forest", 5:"Open koa-ohia wet forest", 6:"Closed ohia mesic forest",
    7:"Closed hala forest", 8:"Closed koa-ohia mesic forest", 9:"Open ohia mesic forest",
    10:"Open koa-ohia mesic forest", 11:"Closed ohia dry forest",
    12:"Mixed mamane-naio-native trees dry woodland", 13:"Open ohia dry forest",
    14:"Open koa-mamane dry forest", 15:"Native mesic to dry forest and shrubland",
    16:"Mixed native-alien wet forest", 17:"Mixed native-alien mesic forest",
    18:"Mixed native-alien dry forest", 19:"Alien wet forest", 20:"Alien mesic forest",
    21:"Alien dry forest", 22:"Kiawe dry forest and shrubland", 23:"Plantation forest",
    24:"Native wet cliff community", 25:"Native wet shrubland", 26:"Native mesic shrubland",
    27:"Mixed native-alien dry cliff community", 28:"Native dry shrubland",
    29:"Coastal strand vegetation", 30:"Uluhe ferns and native shrubs",
    31:"Mixed native-alien wet shrubs and grass/sedges",
    32:"Mixed native-alien mesic shrubs and grass", 33:"Alien wet shrubland",
    34:"Alien mesic shrubland", 35:"Alien dry shrubland", 36:"Native Deschampsia grassland",
    37:"Alien wet grassland", 38:"Alien mesic grassland", 39:"Alien dry grassland",
    40:"Native bog community", 41:"Wetland vegetation", 42:"Cultivated agriculture",
    43:"Developed open space", 44:"High intensity developed", 45:"Low intensity developed",
    46:"Medium intensity developed", 47:"Very sparse vegetation to unvegetated", 48:"Water"
}
FOREST_DN = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

# ------------------------
# Matplotlib style
# ------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

# ------------------------
# Helpers
# ------------------------
def ensure_crs(gdf, epsg):
    if gdf.crs is None or int(gdf.crs.to_epsg() or -1) != epsg:
        return gdf.to_crs(epsg)
    return gdf

def require_field(gdf, col, name_for_error):
    if col not in gdf.columns:
        raise ValueError(f"Expected field '{col}' not found on {name_for_error}.")
    return gdf

def make_grid(aoi, res):
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

def canopy_fraction_by_cell(grid, crowns, class_field):
    inter = gpd.overlay(crowns[["geometry", class_field]], grid, how="intersection")
    inter["part_area"] = inter.geometry.area
    per_cell_area  = inter.groupby("cell_id")["part_area"].sum().rename("canopy_area")
    per_cell_count = inter.groupby("cell_id").size().rename("crown_count")
    grid = grid.merge(per_cell_area, on="cell_id", how="left").merge(per_cell_count, on="cell_id", how="left")
    grid["canopy_area"] = grid["canopy_area"].fillna(0.0)
    grid["crown_count"] = grid["crown_count"].fillna(0).astype(int)
    grid["canopy_frac"] = grid["canopy_area"] / grid["cell_area"]
    return grid

def jacobi_fraction_by_cell(grid, jacobi_forest):
    if jacobi_forest.empty:
        grid["jacobi_frac"] = 0.0
        return grid
    inter = gpd.overlay(grid[["cell_id","geometry","cell_area"]],
                        jacobi_forest[["geometry"]], how="intersection")
    if inter.empty:
        grid["jacobi_frac"] = 0.0
        return grid
    inter["part_area"] = inter.area
    add = inter.groupby("cell_id")["part_area"].sum().rename("jacobi_area")
    grid = grid.merge(add, on="cell_id", how="left")
    grid["jacobi_area"] = grid["jacobi_area"].fillna(0.0)
    grid["jacobi_frac"] = grid["jacobi_area"] / grid["cell_area"]
    return grid

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

def _to_int_codes(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.round().astype("Int64")

# ------------------------
# Load data
# ------------------------
crowns = gpd.read_file(CROWNS_PATH)
crowns = ensure_crs(crowns, 6635)
require_field(crowns, CROWN_CLASS_FIELD, "crowns")

# normalize class codes (critical for mapping)
crowns[CROWN_CLASS_FIELD] = _to_int_codes(crowns[CROWN_CLASS_FIELD])

aoi = ensure_crs(gpd.read_file(AOI_PATH), 6635)

jacobi = ensure_crs(gpd.read_file(JACOBI_PATH), 26904).to_crs(6635)
require_field(jacobi, JACOBI_CODE_FIELD, "Jacobi land-cover")

# Clip Jacobi to AOI and keep forest classes
jacobi = gpd.overlay(jacobi[[JACOBI_CODE_FIELD, "geometry"]], aoi, how="intersection")
jacobi["jacobi_class"] = jacobi[JACOBI_CODE_FIELD].map(DN_TO_CLASS).fillna("Unknown")
jacobi["is_forest"]    = jacobi[JACOBI_CODE_FIELD].isin(FOREST_DN)
jacobi_forest = jacobi[jacobi["is_forest"]].copy()

crowns = clean_geoms(crowns)
aoi    = clean_geoms(aoi)
jacobi_forest = clean_geoms(jacobi_forest)

# ------------------------
# AOI-wide stats + per-species summary (CSVs)
# ------------------------
aoi_area_m2 = float(aoi.geometry.area.sum())
aoi_area_ha = aoi_area_m2 / 1e4

crowns["area_m2"] = crowns.geometry.area
sp = (crowns
      .groupby(CROWN_CLASS_FIELD, dropna=False)["area_m2"]
      .agg(count="size", area_m2="sum")
      .reset_index())

sp["species"] = sp[CROWN_CLASS_FIELD].map(SPECIES_LABELS).fillna(sp[CROWN_CLASS_FIELD].astype(str))

total_crowns        = int(sp["count"].sum())
total_crown_area_m2 = float(sp["area_m2"].sum())
total_crown_area_ha = total_crown_area_m2 / 1e4
crown_cover_pct     = 100.0 * total_crown_area_m2 / aoi_area_m2

sp["area_ha"] = sp["area_m2"] / 1e4
sp["share_of_crowns_pct"] = 100.0 * sp["count"] / total_crowns if total_crowns > 0 else 0.0
sp["share_of_canopy_pct"] = 100.0 * sp["area_m2"] / total_crown_area_m2 if total_crown_area_m2 > 0 else 0.0
sp = sp.sort_values("area_m2", ascending=False)

pd.DataFrame([{
    "aoi_area_ha": aoi_area_ha,
    "total_crowns": total_crowns,
    "total_crown_area_ha": total_crown_area_ha,
    "crown_cover_pct": crown_cover_pct
}]).to_csv(f"{OUT_PREFIX}_overall_crown_stats.csv", index=False)
sp.to_csv(f"{OUT_PREFIX}_species_summary.csv", index=False)

print("\n=== AOI-wide Crown Statistics ===")
print(f"AOI area:                {aoi_area_ha:,.1f} ha")
print(f"Total crowns:            {total_crowns:,}")
print(f"Total crown area:        {total_crown_area_ha:,.1f} ha")
print(f"Canopy cover (crowns):   {crown_cover_pct:,.1f} %")

# ------------------------
# 30 m grid + canopy fraction (crowns & Jacobi)
# ------------------------
grid = make_grid(aoi, GRID_RES_M)
grid = canopy_fraction_by_cell(grid, crowns, CROWN_CLASS_FIELD)  # canopy_frac
grid = jacobi_fraction_by_cell(grid, jacobi_forest)              # jacobi_frac
grid["canopy_frac"] = pd.to_numeric(grid["canopy_frac"], errors="coerce").fillna(0.0).clip(0, 1)
grid["jacobi_frac"] = pd.to_numeric(grid["jacobi_frac"], errors="coerce").fillna(0.0).clip(0, 1)

grid.to_file(f"{OUT_PREFIX}_grid_30m.gpkg", driver="GPKG")

# ------------------------
# Figure 12 – AOI map + species composition bars
# ------------------------
def make_fig12(crowns_gdf: gpd.GeoDataFrame, species_df: pd.DataFrame,
               aoi_gdf: gpd.GeoDataFrame, outpath: str):

    crowns2 = clean_geoms(crowns_gdf).copy()
    aoi2    = clean_geoms(aoi_gdf).copy()
    if crowns2.empty or aoi2.empty:
        raise RuntimeError("[Fig12] Nothing to plot (empty crowns or AOI).")

    crowns2[CROWN_CLASS_FIELD] = _to_int_codes(crowns2[CROWN_CLASS_FIELD])
    crowns2["species"] = crowns2[CROWN_CLASS_FIELD].map(SPECIES_LABELS).fillna(
        crowns2[CROWN_CLASS_FIELD].astype(str)
    )
    crowns2["species"] = pd.Categorical(crowns2["species"], categories=SPECIES_ORDER, ordered=True)

    # For speed/readability at AOI scale: plot representative points
    pts = crowns2.copy()
    pts["geometry"] = pts.geometry.representative_point()
    pts = clean_geoms(pts)

    # Composition table aligned to order
    comp = (species_df[["species","share_of_crowns_pct","share_of_canopy_pct"]]
            .copy()
            .set_index("species")
            .reindex(SPECIES_ORDER)
            .fillna(0.0)
            .reset_index())

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 6.0), gridspec_kw={"width_ratios":[1.3, 1]})

    # Panel A – AOI map
    aoi2.boundary.plot(ax=ax0, color="black", linewidth=0.6)
    for sp_name, sub in pts.groupby("species", observed=True):
        if sub.empty:
            continue
        color = PALETTE.get(sp_name, "#999999")
        sub.plot(ax=ax0, markersize=1.0, color=color, alpha=0.85)
    ax0.set_axis_off()
    ax0.set_title("A) AOI crown map colored by species", pad=8)

    # legend
    handles = [Line2D([0],[0], marker="o", linestyle="none",
                      markerfacecolor=PALETTE[s], markeredgecolor="none",
                      markersize=8, label=s) for s in SPECIES_ORDER]
    leg = ax0.legend(handles=handles, loc="upper left", title="Species")
    leg._legend_box.align = "left"

    # Panel B – composition bars
    x = np.arange(len(comp))
    w = 0.38
    ax1.bar(x - w/2, comp["share_of_crowns_pct"].values, width=w, label="% crowns",
            color=[PALETTE.get(s,"#999999") for s in comp["species"]])
    ax1.bar(x + w/2, comp["share_of_canopy_pct"].values, width=w, label="% canopy area",
            color=[PALETTE.get(s,"#bbbbbb") for s in comp["species"]], alpha=0.95, edgecolor="none")

    ax1.set_xticks(x)
    ax1.set_xticklabels(comp["species"], rotation=15, ha="right")
    ax1.set_ylabel("Share (%)")
    ymax = float(comp[["share_of_crowns_pct","share_of_canopy_pct"]].to_numpy().max())
    ax1.set_ylim(0, max(ymax*1.15, 5))
    ax1.set_title("B) Species composition (crowns vs canopy area)", pad=8)
    ax1.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {outpath}")

# ------------------------
# Figure 13 – canopy cover maps + difference
# ------------------------
def make_fig13(grid_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, outprefix: str):
    grid2 = grid_gdf.copy()
    aoi2  = clean_geoms(aoi_gdf)
    if grid2.empty or aoi2.empty:
        raise RuntimeError("[Fig13] Nothing to plot (empty grid or AOI).")

    grid2["diff_frac"] = grid2["canopy_frac"] - grid2["jacobi_frac"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.0), gridspec_kw={"width_ratios":[1,1,1.05]})

    # A) crowns cover
    ax = axes[0]
    grid2.plot(column="canopy_frac", ax=ax, cmap="viridis", vmin=0.0, vmax=1.0, linewidth=0, legend=True)
    aoi2.boundary.plot(ax=ax, color="k", linewidth=0.4)
    ax.set_axis_off()
    ax.set_title("A) Canopy cover (crowns, 30 m)", pad=8)

    # B) Jacobi cover
    ax = axes[1]
    grid2.plot(column="jacobi_frac", ax=ax, cmap="viridis", vmin=0.0, vmax=1.0, linewidth=0, legend=True)
    aoi2.boundary.plot(ax=ax, color="k", linewidth=0.4)
    ax.set_axis_off()
    ax.set_title("B) Canopy cover (Jacobi, 30 m)", pad=8)

    # C) Difference (crowns − Jacobi)
    ax = axes[2]
    dabs = float(np.nanmax(np.abs(grid2["diff_frac"].to_numpy()))) if len(grid2) else 0.0
    dmax = max(dabs, 0.5)  # at least ±0.5
    grid2.plot(column="diff_frac", ax=ax, cmap="RdBu_r", vmin=-dmax, vmax=dmax, linewidth=0, legend=True)
    aoi2.boundary.plot(ax=ax, color="k", linewidth=0.4)
    ax.set_axis_off()
    ax.set_title("C) Difference (crowns − Jacobi)", pad=8)

    plt.tight_layout()
    fig.savefig(f"{outprefix}_fig13_canopy_cover_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {outprefix}_fig13_canopy_cover_maps.png")

# ------------------------
# Run the figures
# ------------------------
make_fig12(crowns, sp, aoi, f"{OUT_PREFIX}_fig12_aoi_and_species.png")
make_fig13(grid, aoi, OUT_PREFIX)

# Optional scatter
MAKE_SCATTER = True
if MAKE_SCATTER:
    sfig, sax = plt.subplots(figsize=(5, 5))
    x = grid["jacobi_frac"].astype(float).to_numpy()
    y = grid["canopy_frac"].astype(float).to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.clip(x[m], 0, 1), np.clip(y[m], 0, 1)
    sax.scatter(x, y, s=6, alpha=0.25, edgecolor="none")
    sax.plot([0,1],[0,1], ls="--", lw=1, color="k")
    sax.set_xlim(0,1); sax.set_ylim(0,1)
    sax.set_xlabel("Jacobi canopy fraction (30 m)")
    sax.set_ylabel("Crown-derived canopy fraction (30 m)")
    sax.set_title("Jacobi vs crown canopy cover (30 m)")
    sfig.tight_layout()
    sfig.savefig(f"{OUT_PREFIX}_fig13_cover_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(sfig)
    print(f"[OK] Saved scatter → {OUT_PREFIX}_fig13_cover_scatter.png")


# # section34_landscape_stats_and_jacobi.py
# # ============================================================
# # Landscape-scale crown stats + per-species summaries + plots,
# # plus comparison to Jacobi land-cover (forested classes only),
# # clipped to AOI (extent.gpkg). Writes CSVs and prints results.
# # ============================================================
#
# import warnings
# warnings.filterwarnings("ignore")
#
# import math
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from shapely.geometry import box
#
# # ------------------------
# # USER CONFIG
# # ------------------------
# CROWNS_PATH   = "classified_xgb.gpkg"   # EPSG:6635, polygons with predicted_class
# JACOBI_PATH   = "CAH_LandCover.gpkg"    # EPSG:26904 (statewide), polygonized land cover
# AOI_PATH      = "extent.gpkg"           # EPSG:6635, study AOI
# GRID_RES_M    = 30.0                    # grid size for cell stats and agreement
# OUT_PREFIX    = "pw_section34"
#
# # Field names
# CROWN_CLASS_FIELD = "predicted_class"   # species class code on crowns
# JACOBI_CODE_FIELD = "DN"                # integer code of Jacobi land cover
#
# # Optional: map your class codes to readable short names (adjust as needed)
# SPECIES_LABELS = {
#     1: "METPOL",
#     2: "GREROB",
#     3: "ACAKOA",
#     4: "DIOSAN",
#     5: "PROPAL",
#     6: "OTHER",
# }
#
# # Agreement thresholds for % of cells with crown canopy present
# PRESENT_THRESHOLDS = [0.10, 0.20]  # i.e., ≥10% and ≥20% canopy in a 30 m cell
#
# # ------------------------
# # Jacobi DN → class name mapping (from your enum)
# # ------------------------
# DN_TO_CLASS = {
#     1: "Closed ohia wet forest",
#     2: "Closed koa-ohia wet forest",
#     3: "Low-stature ohia wet forest",
#     4: "Open ohia wet forest",
#     5: "Open koa-ohia wet forest",
#     6: "Closed ohia mesic forest",
#     7: "Closed hala forest",
#     8: "Closed koa-ohia mesic forest",
#     9: "Open ohia mesic forest",
#     10: "Open koa-ohia mesic forest",
#     11: "Closed ohia dry forest",
#     12: "Mixed mamane-naio-native trees dry woodland",
#     13: "Open ohia dry forest",
#     14: "Open koa-mamane dry forest",
#     15: "Native mesic to dry forest and shrubland",
#     16: "Mixed native-alien wet forest",
#     17: "Mixed native-alien mesic forest",
#     18: "Mixed native-alien dry forest",
#     19: "Alien wet forest",
#     20: "Alien mesic forest",
#     21: "Alien dry forest",
#     22: "Kiawe dry forest and shrubland",
#     23: "Plantation forest",
#     24: "Native wet cliff community",
#     25: "Native wet shrubland",
#     26: "Native mesic shrubland",
#     27: "Mixed native-alien dry cliff community",
#     28: "Native dry shrubland",
#     29: "Coastal strand vegetation",
#     30: "Uluhe ferns and native shrubs",
#     31: "Mixed native-alien wet shrubs and grass/sedges",
#     32: "Mixed native-alien mesic shrubs and grass",
#     33: "Alien wet shrubland",
#     34: "Alien mesic shrubland",
#     35: "Alien dry shrubland",
#     36: "Native Deschampsia grassland",
#     37: "Alien wet grassland",
#     38: "Alien mesic grassland",
#     39: "Alien dry grassland",
#     40: "Native bog community",
#     41: "Wetland vegetation",
#     42: "Cultivated agriculture",
#     43: "Developed open space",
#     44: "High intensity developed",
#     45: "Low intensity developed",
#     46: "Medium intensity developed",
#     47: "Very sparse vegetation to unvegetated",
#     48: "Water"
# }
#
# # Which DN codes are "forest/woody" for Jacobi summaries
# FOREST_DN = [
#     1,2,3,4,5,6,7,8,9,10,11,
#     12,13,14,15,16,17,18,19,20,21,22,23
# ]
# FOREST_CLASSES = {DN_TO_CLASS[k] for k in FOREST_DN}
#
# # ------------------------
# # Helpers
# # ------------------------
# def ensure_crs(gdf, epsg):
#     if gdf.crs is None or int(gdf.crs.to_epsg() or -1) != epsg:
#         return gdf.to_crs(epsg)
#     return gdf
#
# def require_field(gdf, col, name_for_error):
#     if col not in gdf.columns:
#         raise ValueError(f"Expected field '{col}' not found on {name_for_error}.")
#     return gdf
#
# def make_grid(aoi, res):
#     aoi_union = aoi.geometry.unary_union
#     minx, miny, maxx, maxy = aoi_union.bounds
#     nx = math.ceil((maxx - minx) / res)
#     ny = math.ceil((maxy - miny) / res)
#     cells = [box(minx + i*res, miny + j*res,
#                  minx + (i+1)*res, miny + (j+1)*res)
#              for i in range(nx) for j in range(ny)]
#     grid = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=aoi.crs)
#     grid = gpd.overlay(grid, aoi, how="intersection")
#     grid["cell_area"] = grid.geometry.area
#     return grid.reset_index(drop=True)
#
# def canopy_fraction_by_cell(grid, crowns, class_field):
#     inter = gpd.overlay(crowns[["geometry", class_field]], grid, how="intersection")
#     inter["part_area"] = inter.geometry.area
#     per_cell_area = inter.groupby("cell_id")["part_area"].sum().rename("canopy_area")
#     per_cell_count = inter.groupby("cell_id").size().rename("crown_count")
#     grid = grid.merge(per_cell_area, on="cell_id", how="left").merge(per_cell_count, on="cell_id", how="left")
#     grid["canopy_area"] = grid["canopy_area"].fillna(0.0)
#     grid["crown_count"] = grid["crown_count"].fillna(0).astype(int)
#     grid["canopy_frac"] = grid["canopy_area"] / grid["cell_area"]
#     return grid
#
# def overlay_majority_class(grid, jacobi_forest, class_field):
#     if jacobi_forest.empty:
#         grid["jacobi_class"] = "Non-forest"
#         grid["jacobi_woody"] = 0
#         return grid
#     inter = gpd.overlay(grid, jacobi_forest, how="intersection")
#     inter["part_area"] = inter.geometry.area
#     inter_sorted = inter.sort_values(["cell_id","part_area"], ascending=[True, False])
#     winners = inter_sorted.drop_duplicates("cell_id").set_index("cell_id")
#     grid = grid.set_index("cell_id")
#     grid["jacobi_class"] = winners[class_field]
#     grid["jacobi_woody"] = 1
#     grid = grid.reset_index()
#     grid["jacobi_class"] = grid["jacobi_class"].fillna("Non-forest")
#     grid["jacobi_woody"] = grid["jacobi_woody"].fillna(0).astype(int)
#     return grid
#
# # ------------------------
# # Load data
# # ------------------------
# crowns = gpd.read_file(CROWNS_PATH)
# crowns = ensure_crs(crowns, 6635)
# require_field(crowns, CROWN_CLASS_FIELD, "crowns")
#
# aoi = ensure_crs(gpd.read_file(AOI_PATH), 6635)
#
# jacobi = ensure_crs(gpd.read_file(JACOBI_PATH), 26904).to_crs(6635)
# require_field(jacobi, JACOBI_CODE_FIELD, "Jacobi land-cover")
#
# # Clip Jacobi to AOI and map classes
# jacobi = gpd.overlay(jacobi[[JACOBI_CODE_FIELD, "geometry"]], aoi, how="intersection")
# jacobi["jacobi_class"] = jacobi[JACOBI_CODE_FIELD].map(DN_TO_CLASS).fillna("Unknown")
# jacobi["is_forest"] = jacobi[JACOBI_CODE_FIELD].isin(FOREST_DN)
# jacobi_forest = jacobi[jacobi["is_forest"]].copy()
#
# # ------------------------
# # AOI-wide crown statistics
# # ------------------------
# aoi_area_m2 = aoi.geometry.area.sum()
# aoi_area_ha = aoi_area_m2 / 1e4
#
# crowns["area_m2"] = crowns.geometry.area
# total_crowns = len(crowns)
# total_crown_area_m2 = crowns["area_m2"].sum()
# total_crown_area_ha = total_crown_area_m2 / 1e4
# crown_cover_pct = 100.0 * total_crown_area_m2 / aoi_area_m2
#
# # Per-species summary
# sp = (crowns
#       .groupby(CROWN_CLASS_FIELD, dropna=False)["area_m2"]
#       .agg(count="size",
#            area_m2="sum",
#            mean_area_m2="mean",
#            median_area_m2="median")
#       .reset_index())
#
# sp["species"] = sp[CROWN_CLASS_FIELD].map(SPECIES_LABELS).fillna(sp[CROWN_CLASS_FIELD].astype(str))
# sp["area_ha"] = sp["area_m2"] / 1e4
# sp["share_of_crowns_pct"] = 100.0 * sp["count"] / total_crowns
# sp["share_of_canopy_pct"] = 100.0 * sp["area_m2"] / total_crown_area_m2
# sp = sp.sort_values("area_m2", ascending=False)
#
# # Save and print AOI-wide summaries
# overall_df = pd.DataFrame([{
#     "aoi_area_ha": aoi_area_ha,
#     "total_crowns": total_crowns,
#     "total_crown_area_ha": total_crown_area_ha,
#     "crown_cover_pct": crown_cover_pct
# }])
# overall_df.to_csv(f"{OUT_PREFIX}_overall_crown_stats.csv", index=False)
# sp.to_csv(f"{OUT_PREFIX}_species_summary.csv", index=False)
#
# print("\n=== AOI-wide Crown Statistics ===")
# print(f"AOI area:                {aoi_area_ha:,.1f} ha")
# print(f"Total crowns:            {total_crowns:,}")
# print(f"Total crown area:        {total_crown_area_ha:,.1f} ha")
# print(f"Canopy cover (crowns):   {crown_cover_pct:,.1f} %")
#
# print("\n=== Species Summary (top rows) ===")
# print(sp[["species","count","area_ha","share_of_crowns_pct","share_of_canopy_pct",
#           "mean_area_m2","median_area_m2"]].head(10).to_string(index=False))
#
# # ------------------------
# # Species crown-area box plot
# # ------------------------
# # Only keep crowns with a mapped species label
# crowns["species"] = crowns[CROWN_CLASS_FIELD].map(SPECIES_LABELS).fillna(crowns[CROWN_CLASS_FIELD].astype(str))
# # Optional: drop polygons with zero/negative area just in case
# crowns = crowns[crowns["area_m2"] > 0].copy()
#
# plt.figure(figsize=(9, 6))
# # Sort species by median area for nicer ordering
# order = (crowns.groupby("species")["area_m2"].median().sort_values(ascending=False).index.tolist())
# data = [crowns.loc[crowns["species"] == s, "area_m2"].values for s in order]
# plt.boxplot(data, labels=order, showfliers=False)
# plt.ylabel("Crown area (m²)")
# plt.title("Distribution of crown area by species")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig(f"{OUT_PREFIX}_species_crown_area_boxplot.png", dpi=300)
# plt.close()
# print(f"\nSaved box plot: {OUT_PREFIX}_species_crown_area_boxplot.png")
#
# # ------------------------
# # 30 m grid + canopy fraction and Jacobi overlay
# # ------------------------
# grid = make_grid(aoi, GRID_RES_M)
# grid = canopy_fraction_by_cell(grid, crowns, CROWN_CLASS_FIELD)
# grid = overlay_majority_class(grid, jacobi_forest, "jacobi_class")
#
# # Jacobi woody percentage at 30 m
# jacobi_woody_pct_30m = 100.0 * grid["jacobi_woody"].mean()
#
# # Agreement: % of cells with canopy ≥ threshold inside/outside Jacobi woody
# rows = []
# for thr in PRESENT_THRESHOLDS:
#     in_woody = grid["jacobi_woody"] == 1
#     pct_in_woody = 100.0 * (grid.loc[in_woody, "canopy_frac"] >= thr).mean() if in_woody.any() else np.nan
#     pct_in_nonwoody = 100.0 * (grid.loc[~in_woody, "canopy_frac"] >= thr).mean() if (~in_woody).any() else np.nan
#     rows.append({
#         "threshold_canopy_frac": thr,
#         "pct_cells_meeting_thr_in_jacobi_woody": pct_in_woody,
#         "pct_cells_meeting_thr_in_jacobi_nonwoody": pct_in_nonwoody
#     })
# agreement_df = pd.DataFrame(rows)
# agreement_df.to_csv(f"{OUT_PREFIX}_agreement_30m.csv", index=False)
#
# # Per Jacobi class: mean±sd canopy fraction at 30 m
# by_class = (grid
#             .groupby("jacobi_class")["canopy_frac"]
#             .agg(mean="mean", sd="std", n="count")
#             .reset_index()
#             .sort_values("mean", ascending=False))
# by_class.to_csv(f"{OUT_PREFIX}_by_jacobi_class_30m.csv", index=False)
#
# # Save the annotated grid for inspection
# grid.to_file(f"{OUT_PREFIX}_grid_30m.gpkg", driver="GPKG")
#
# print("\n=== 30 m Cell Summaries ===")
# print(f"Jacobi 'woody' cells (majority-forest classes): {jacobi_woody_pct_30m:,.1f} % of grid cells")
# print("\nAgreement (% cells with crown canopy ≥ threshold):")
# print(agreement_df.to_string(index=False))
# print("\nMean crown canopy fraction by Jacobi class (30 m cells) — top rows:")
# print(by_class.head(12).to_string(index=False))
#
# # ------------------------
# # Optional: species composition by Jacobi class (area share), AOI-wide
# # ------------------------
# # Intersect crowns with Jacobi forest to attribute each crown piece to a Jacobi class
# try:
#     inter = gpd.overlay(crowns[["geometry", "species", "area_m2"]], jacobi_forest[["geometry","jacobi_class"]], how="intersection")
#     inter["part_area_m2"] = inter.geometry.area
#     # area by (Jacobi class × species)
#     comp = (inter.groupby(["jacobi_class","species"])["part_area_m2"]
#                  .sum()
#                  .reset_index())
#     # within each Jacobi class, compute species share of area
#     class_tot = comp.groupby("jacobi_class")["part_area_m2"].sum().rename("class_area_m2")
#     comp = comp.merge(class_tot, on="jacobi_class", how="left")
#     comp["species_share_pct"] = 100.0 * comp["part_area_m2"] / comp["class_area_m2"].replace(0, np.nan)
#     comp = comp.sort_values(["jacobi_class","species_share_pct"], ascending=[True, False])
#     comp.to_csv(f"{OUT_PREFIX}_species_composition_by_jacobi_class.csv", index=False)
#
#     print("\n=== Species composition within Jacobi forest classes (area share, %) — sample ===")
#     sample = comp.groupby("jacobi_class").head(5)  # top 5 species per class, if available
#     print(sample.head(20).to_string(index=False))
# except Exception as e:
#     print("\n[Note] Skipped species-by-Jacobi composition (intersection issue):", e)
