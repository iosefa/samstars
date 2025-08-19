#!/usr/bin/env python3
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from shapely.ops import unary_union
from affine import Affine
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from rasterio.enums import Resampling

# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────
RGB_TIF          = "rgb.tif"
EXTENT_GPKG      = "extent.gpkg"
IOU_DIR          = Path("IoU")
BIG_ISLAND_GPKG  = "hawaii.gpkg"

# Set C
SETC_TIF          = "/Users/iosefa/repos/sam/area51_subset1.tif"
SETC_LABEL        = "Set C"
SETC_EXTENT_GPKG  = "area51__ss1_extent.gpkg"

TARGET_MAX_DIM    = 4000
OUT_FIG           = "figure1.png"

# ─────────────────────────────────────────────────────────────────────────────
# Typography (publication-ready – adjust here)
# ─────────────────────────────────────────────────────────────────────────────
BASE_FONT               = 14
MAP_OUTLINE_LW          = 2.0     # Pu‘u Wa‘awa‘a outline
PLOT_BOX_LW             = 1.8
SETC_BOX_LW             = 2.2

LEGEND_FONT             = 14
LEGEND_HANDLE_LW        = 3.2

SCALEBAR_FONT           = 14
INSET_LABEL_FONT        = 14
INSET_ISLAND_TITLE_FONT = 16
PLOT_TAG_FONT           = 15      # “P1…P6” on main map
SUBPLOT_TITLE_FONT      = 14      # titles over the small chips
SETC_TITLE_FONT         = 14

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": BASE_FONT,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def union_all_safe(gdf: gpd.GeoDataFrame):
    try:
        return gdf.geometry.union_all()
    except Exception:
        return unary_union(gdf.geometry.tolist())

def percentile_normalize(img3_chw, lo=2, hi=98):
    out = np.zeros_like(img3_chw, dtype=np.float32)
    for i in range(img3_chw.shape[0]):
        band = img3_chw[i].astype(np.float32)
        finite = np.isfinite(band)
        if finite.any():
            p_lo, p_hi = np.percentile(band[finite], [lo, hi])
            if np.isfinite(p_lo) and np.isfinite(p_hi) and p_hi > p_lo:
                band = np.clip(band, p_lo, p_hi)
                band = (band - p_lo) / (p_hi - p_lo)
            else:
                band = np.zeros_like(band, dtype=np.float32)
        else:
            band = np.zeros_like(band, dtype=np.float32)
        out[i] = band
    return out

def array_extent(transform: Affine, width: int, height: int):
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return (left, right, bottom, top)

def nice_round_scale(meters):
    if meters <= 0: return 0
    exp = int(np.floor(np.log10(meters)))
    base = meters / (10 ** exp)
    if base < 1.5:   nice = 1
    elif base < 3.5: nice = 2
    elif base < 7.5: nice = 5
    else:            nice = 10
    return nice * (10 ** exp)

def add_scalebar(ax, where="center-bottom", length_px_frac=0.22, height_frac=0.010,
                 pad_frac=0.030, label_pad=0.010, face="white", edge="black",
                 text_color="black", fontsize=SCALEBAR_FONT):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width_m = abs(x1 - x0)
    height_m = abs(y1 - y0)

    target_m = width_m * length_px_frac
    bar_m = nice_round_scale(target_m)
    if bar_m <= 0:
        return

    pad_x = width_m * pad_frac
    pad_y = height_m * pad_frac
    bar_h = height_m * height_frac

    if where == "center-bottom":
        bl_x = (x0 + x1) / 2.0 - bar_m / 2.0
    else:
        bl_x = x1 - pad_x - bar_m
    bl_y = y0 + pad_y

    rect = Rectangle((bl_x, bl_y), bar_m, bar_h,
                     facecolor=face, alpha=0.9,
                     edgecolor=edge, linewidth=1.2)
    ax.add_patch(rect)
    km = bar_m / 1000.0
    label = f"{km:.0f} km" if km >= 1 else f"{km:.1f} km"
    ax.text(bl_x + bar_m / 2.0, bl_y + bar_h + height_m * label_pad,
            label, ha="center", va="bottom", color=text_color, fontsize=fontsize,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

def read_rgb_532(path):
    with rasterio.open(path) as src:
        b5 = src.read(5, resampling=Resampling.bilinear).astype("float32")
        b3 = src.read(3, resampling=Resampling.bilinear).astype("float32")
        b2 = src.read(2, resampling=Resampling.bilinear).astype("float32")
        chip = percentile_normalize(np.stack([b5, b3, b2], axis=0))
        return np.moveaxis(chip, 0, 2)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
extent = gpd.read_file(EXTENT_GPKG)
extent_union = union_all_safe(extent).buffer(0)

with rasterio.open(RGB_TIF) as src:
    if extent.crs != src.crs:
        extent = extent.to_crs(src.crs)
        extent_union = union_all_safe(extent).buffer(0)
    out, out_transform = rio_mask(src, [mapping(extent_union)], crop=True, filled=False, indexes=[1, 2, 3])
    out = np.where(out.mask, np.nan, out.data).astype("float32")
    _, H, W = out.shape
    scale = max(H / TARGET_MAX_DIM, W / TARGET_MAX_DIM, 1.0)
    if scale > 1.0:
        step = int(np.ceil(scale))
        out = out[:, ::step, ::step]
        out_transform = out_transform * Affine.scale(step, step)
    rast_crs = src.crs

rgb_01 = percentile_normalize(out, lo=2, hi=98)
alpha = np.where(np.any(np.isfinite(out), axis=0), 1.0, 0.0).astype(np.float32)
rgba_hwc = np.dstack([rgb_01[0], rgb_01[1], rgb_01[2], alpha])

# Plot boxes
plot_boxes = []
for i in range(1, 7):
    p = IOU_DIR / f"setA_plot{i}.gpkg"
    if p.exists():
        g = gpd.read_file(p).to_crs(rast_crs)
        g["__which__"] = i
        plot_boxes.append(g)
boxes_gdf = (gpd.GeoDataFrame(pd.concat(plot_boxes, ignore_index=True), crs=rast_crs)
             if plot_boxes else gpd.GeoDataFrame(geometry=[], crs=rast_crs))

# Set C extent
setc_extent_gdf = None
if Path(SETC_EXTENT_GPKG).exists():
    setc_extent_gdf = gpd.read_file(SETC_EXTENT_GPKG).to_crs(rast_crs)

# Big island
try:
    big_island = gpd.read_file(BIG_ISLAND_GPKG).to_crs(rast_crs)
except Exception:
    big_island = None

# Preview chips
order = [6, 3, 2, 1, 5, 4]
labels = {6: "Plot 6", 3: "Plot 3", 2: "Plot 2", 1: "Plot 1", 5: "Plot 5", 4: "Plot 4"}
preview_imgs, preview_titles = [], []
for i in order:
    tif = IOU_DIR / f"setA_plot{i}.tif"
    if tif.exists():
        preview_imgs.append(read_rgb_532(tif))
        preview_titles.append(labels[i])

PLOT_SHORT = {1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5", 6: "P6"}
setc_img = read_rgb_532(SETC_TIF)

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))

# Main map
ax_map = fig.add_axes([0.04, 0.04, 0.58, 0.92])
left, right, bottom, top = array_extent(out_transform, rgba_hwc.shape[1], rgba_hwc.shape[0])
ax_map.imshow(rgba_hwc, extent=(left, right, bottom, top), origin="upper")
extent.boundary.plot(ax=ax_map, color="black", linewidth=MAP_OUTLINE_LW, zorder=3)

# Plot outlines + larger labels
if not boxes_gdf.empty:
    boxes_gdf.boundary.plot(ax=ax_map, color="#333333", linewidth=PLOT_BOX_LW, zorder=4)
    map_w = right - left
    x_offset = 0.006 * map_w
    for which, gsub in boxes_gdf.groupby("__which__"):
        minx, miny, maxx, maxy = gsub.total_bounds
        ax_map.text(maxx + x_offset, (miny + maxy)/2,
                    PLOT_SHORT.get(which, f"P{which}"),
                    ha="left", va="center",
                    fontsize=PLOT_TAG_FONT, weight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=3.5, foreground="black")])

# Set C extent outline + label
if setc_extent_gdf is not None and not setc_extent_gdf.empty:
    setc_extent_gdf.boundary.plot(ax=ax_map, color="#111111", linewidth=SETC_BOX_LW,
                                  linestyle="--", zorder=5)
    minx, miny, maxx, maxy = setc_extent_gdf.total_bounds
    ax_map.text(maxx + 0.006*(right-left), (miny+maxy)/2,
                "Set C", ha="left", va="center",
                fontsize=PLOT_TAG_FONT, weight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=3.5, foreground="black")])

# Legend (bigger text + thicker handles)
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color="black", lw=LEGEND_HANDLE_LW, label="Pu‘u Wa‘awa‘a"),
    Line2D([0],[0], color="#333333", lw=LEGEND_HANDLE_LW, label="Set A plots"),
    Line2D([0],[0], color="#111111", lw=LEGEND_HANDLE_LW, linestyle="--", label="Set C extent"),
]
ax_map.legend(handles=handles, loc="upper left", frameon=False, prop={"size": LEGEND_FONT})

ax_map.set_axis_off()
add_scalebar(ax_map, where="center-bottom", fontsize=SCALEBAR_FONT)

# Inset island with larger labels
if big_island is not None and not big_island.empty:
    iax = inset_axes(ax_map, width="26%", height="26%", loc="lower left", borderpad=1)
    iax.set_facecolor("#E9F4FF")
    big_island.boundary.plot(ax=iax, color="black", linewidth=1.1)
    extent.boundary.plot(ax=iax, color="#222222", linewidth=1.3)
    ex_ct = union_all_safe(extent).centroid
    iax.plot(ex_ct.x, ex_ct.y, marker="*", markersize=10, color="#D81B60")
    iax.text(ex_ct.x, ex_ct.y, "  Pu‘u Wa‘awa‘a", fontsize=INSET_LABEL_FONT,
             va="center", ha="left", color="#D81B60",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    iax.text(0.5, 0.88, "Hawai‘i", transform=iax.transAxes,
             ha="center", va="center", fontsize=INSET_ISLAND_TITLE_FONT, color="#333333",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    iax.set_xticks([]); iax.set_yticks([])

# Right grid (unchanged geometry; only bigger titles)
right_left, right_width = 0.55, 0.33
setc_height, setc_bottom, gap_grid_setc = 0.28, 0.06, 0.012
grid_top = 0.95
grid_bottom = setc_bottom + setc_height + gap_grid_setc
grid_height = grid_top - grid_bottom
cols, rows, vgap, seam = 2, 3, 0.012, 0.004
cell_w = (right_width - seam) / 2.0
cell_h = (grid_height - (rows - 1) * vgap) / rows

# Set C panel (title inside, bigger font)
ax_setc = fig.add_axes([right_left, setc_bottom, right_width, setc_height], zorder=1)
ax_setc.imshow(setc_img, origin="upper")
ax_setc.axis("off")
ax_setc.text(0.5, 0.99, SETC_LABEL, ha="center", va="top",
             transform=ax_setc.transAxes,
             fontsize=SETC_TITLE_FONT, color="black",
             path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

# Small preview tiles (bigger titles, slightly thicker frames)
left_x_col0, left_x_col1 = right_left, right_left + cell_w + seam
for idx, (img, title) in enumerate(zip(preview_imgs, preview_titles)):
    r, c = idx // cols, idx % cols
    top_y = grid_top - r * (cell_h + vgap)
    bottom_y = top_y - cell_h
    left_x = left_x_col0 if c == 0 else left_x_col1
    axp = fig.add_axes([left_x, bottom_y, cell_w, cell_h], zorder=2)
    axp.imshow(img, origin="upper"); axp.axis("off")
    axp.text(0.5, 0.99, title, ha="center", va="top",
             transform=axp.transAxes,
             fontsize=SUBPLOT_TITLE_FONT, color="black",
             path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
    for s in axp.spines.values():
        s.set_visible(True); s.set_linewidth(1.4); s.set_color("#333333")

# Save
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print(f"[OK] Saved {OUT_FIG}")