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
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────
RGB_TIF          = "rgb.tif"
EXTENT_GPKG      = "extent.gpkg"
IOU_DIR          = Path("IoU")
BIG_ISLAND_GPKG  = "hawaii.gpkg"

SETC_EXTENT_GPKG = "area51__ss1_extent.gpkg"
SETC_BLOWUP_TIF  = Path("Set_C_blowup.tif")

# NEW: Set B (ground photos) points
TRAININGS_GPKG   = "trainings_filtered.gpkg"

TARGET_MAX_DIM    = 4000
OUT_FIG           = "figure1.png"

# ─────────────────────────────────────────────────────────────────────────────
# Typography
# ─────────────────────────────────────────────────────────────────────────────
BASE_FONT               = 14
MAP_OUTLINE_LW          = 2.0
PLOT_BOX_LW             = 1.8
SETC_BOX_LW             = 2.2

LEGEND_FONT             = 14
LEGEND_HANDLE_LW        = 3.2

SCALEBAR_FONT           = 14
INSET_ISLAND_TITLE_FONT = 16
PLOT_TAG_FONT           = 15
SUBPLOT_TITLE_FONT      = 14

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
    try: return gdf.geometry.union_all()
    except Exception: return unary_union(gdf.geometry.tolist())

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

def add_scalebar(ax,
                 where="center-bottom",            # "center-bottom" | "bottom-right"
                 length_px_frac=0.22,
                 height_frac=0.010,
                 pad_frac=0.030,
                 label_pad=0.010,
                 face="white",
                 edge="black",
                 text_color="black",
                 fontsize=12,
                 target_m=None,                    # override bar length in meters
                 units="km",                       # "km" or "m" label formatting
                 zorder=10):
    # Axis extents in *data units*
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width_m  = abs(x1 - x0)
    height_m = abs(y1 - y0)

    # Choose bar length
    if target_m is None:
        approx = width_m * length_px_frac
        bar_m = nice_round_scale(approx)
    else:
        bar_m = float(target_m)

    if bar_m <= 0:
        return

    # Geometry
    pad_x = width_m  * pad_frac
    pad_y = height_m * pad_frac
    bar_h = height_m * height_frac

    if where == "center-bottom":
        bl_x = (x0 + x1) / 2.0 - bar_m / 2.0
    elif where == "bottom-right":
        bl_x = x1 - pad_x - bar_m
    else:
        bl_x = (x0 + x1) / 2.0 - bar_m / 2.0

    bl_y = y0 + pad_y

    # Bar
    rect = Rectangle((bl_x, bl_y), bar_m, bar_h,
                     facecolor=face, alpha=0.95,
                     edgecolor=edge, linewidth=1.6, zorder=zorder)
    ax.add_patch(rect)

    # Label text (km/m)
    if units == "km":
        label_txt = f"{bar_m/1000:.0f} km" if bar_m >= 1000 else f"{bar_m:.0f} m"
    else:
        label_txt = f"{bar_m:.0f} m"

    txt = ax.text(bl_x + bar_m/2.0,
                  bl_y + bar_h + height_m * label_pad,
                  label_txt,
                  ha="center", va="bottom",
                  color=text_color, fontsize=fontsize, zorder=zorder+1)
    txt.set_path_effects([pe.withStroke(linewidth=3.8, foreground="white")])

def read_rgb_532(path, lo=2.0, hi=98.0, gamma=0.95, sat_boost=1.20):
    with rasterio.open(path) as src:
        b5 = src.read(5, out_dtype="float32", resampling=Resampling.bilinear)
        b3 = src.read(3, out_dtype="float32", resampling=Resampling.bilinear)
        b2 = src.read(2, out_dtype="float32", resampling=Resampling.bilinear)
    img = np.stack([b5, b3, b2], axis=-1)
    out = np.zeros_like(img, dtype="float32")
    for i in range(3):
        band = img[..., i]; finite = np.isfinite(band)
        if finite.any():
            p_lo, p_hi = np.percentile(band[finite], [lo, hi])
            if np.isfinite(p_lo) and np.isfinite(p_hi) and p_hi > p_lo:
                band = (band - p_lo) / max(p_hi - p_lo, 1e-6)
                out[..., i] = np.clip(band, 0, 1)
    if gamma is not None and gamma > 0: out = np.clip(out, 0, 1) ** gamma
    return out

def read_rgb_532_with_extent(path, lo=2.0, hi=98.0, gamma=0.95, sat_boost=1.20):
    """Return (img_float HxWx3 in [0,1], extent=(left,right,bottom,top)) using WV3 bands 5/3/2."""
    with rasterio.open(path) as src:
        b5 = src.read(5, out_dtype="float32", resampling=Resampling.bilinear)
        b3 = src.read(3, out_dtype="float32", resampling=Resampling.bilinear)
        b2 = src.read(2, out_dtype="float32", resampling=Resampling.bilinear)
        t  = src.transform
        h, w = src.height, src.width
        extent = (t.c, t.c + t.a*w, t.f + t.e*h, t.f)  # left, right, bottom, top

    img = np.stack([b5, b3, b2], axis=-1)

    out = np.zeros_like(img, dtype="float32")
    for i in range(3):
        band = img[..., i]
        finite = np.isfinite(band)
        if finite.any():
            p_lo, p_hi = np.percentile(band[finite], [lo, hi])
            if np.isfinite(p_lo) and np.isfinite(p_hi) and p_hi > p_lo:
                band = (band - p_lo) / max(p_hi - p_lo, 1e-6)
                out[..., i] = np.clip(band, 0, 1)
    if gamma and gamma > 0:
        out = np.clip(out, 0, 1) ** gamma
    return out, extent

# ─────────────────────────────────────────────────────────────────────────────
# Load base map
# ─────────────────────────────────────────────────────────────────────────────
extent = gpd.read_file(EXTENT_GPKG)
extent_union = union_all_safe(extent).buffer(0)

with rasterio.open(RGB_TIF) as src:
    if extent.crs != src.crs:
        extent = extent.to_crs(src.crs); extent_union = union_all_safe(extent).buffer(0)
    out, out_transform = rio_mask(src,[mapping(extent_union)],crop=True,filled=False,indexes=[1,2,3])
    out = np.where(out.mask, np.nan, out.data).astype("float32")
    _,H,W = out.shape
    scale = max(H/TARGET_MAX_DIM, W/TARGET_MAX_DIM, 1.0)
    if scale>1.0:
        step=int(np.ceil(scale)); out=out[:,::step,::step]
        out_transform = out_transform*Affine.scale(step,step)
    rast_crs = src.crs

rgb_01 = percentile_normalize(out,lo=2,hi=98)
alpha = np.where(np.any(np.isfinite(out),axis=0),1.0,0.0).astype(np.float32)
rgba_hwc = np.dstack([rgb_01[0],rgb_01[1],rgb_01[2],alpha])

# Load Set A plot outlines
plot_boxes = []
for i in range(1,7):
    p = IOU_DIR / f"setA_plot{i}.gpkg"
    if p.exists():
        g = gpd.read_file(p).to_crs(rast_crs)
        g["__which__"] = i; plot_boxes.append(g)
boxes_gdf = (gpd.GeoDataFrame(pd.concat(plot_boxes,ignore_index=True),crs=rast_crs)
             if plot_boxes else gpd.GeoDataFrame(geometry=[],crs=rast_crs))
PLOT_SHORT = {1:"P1",2:"P2",3:"P3",4:"P4",5:"P5",6:"P6"}

# Load Set C extent
setc_extent_gdf = None
if Path(SETC_EXTENT_GPKG).exists():
    setc_extent_gdf = gpd.read_file(SETC_EXTENT_GPKG).to_crs(rast_crs)

# Load Set B points (ground photos)
setb_points = None
if Path(TRAININGS_GPKG).exists():
    try:
        setb_points = gpd.read_file(TRAININGS_GPKG)
        if not setb_points.empty:
            if setb_points.crs != rast_crs:
                setb_points = setb_points.to_crs(rast_crs)
    except Exception:
        setb_points = None

# Right-panel blowup image
if SETC_BLOWUP_TIF.exists():
    blowup_img, blowup_extent = read_rgb_532_with_extent(SETC_BLOWUP_TIF)
else:
    blowup_img = np.ones((400,400,3),dtype="float32")*0.95
    blowup_extent = (0,400,400,0)

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15,10))

# Main map
ax_map = fig.add_axes([0.04,0.04,0.58,0.92])
l,r,b,t = array_extent(out_transform,rgba_hwc.shape[1],rgba_hwc.shape[0])
ax_map.imshow(rgba_hwc,extent=(l,r,b,t),origin="upper")
extent.boundary.plot(ax=ax_map,color="black",linewidth=MAP_OUTLINE_LW,zorder=3)


# --- Set B points (ground photos) ---
if (setb_points is not None) and (not setb_points.empty):
    setb_points.plot(ax=ax_map,
                     markersize=12, marker="o",
                     color="#FFD24D", edgecolor="black", linewidth=0.6,
                     alpha=0.95, zorder=4)   # BELOW labels/outlines

# --- Set A plot boxes ---
if not boxes_gdf.empty:
    boxes_gdf.boundary.plot(ax=ax_map, color="red",
                            linewidth=PLOT_BOX_LW, zorder=5)

# --- Set A plot labels (draw AFTER points, so they’re on top) ---
if not boxes_gdf.empty:
    map_w = r - l; x_offset = 0.006 * map_w
    for which, gsub in boxes_gdf.groupby("__which__"):
        minx, miny, maxx, maxy = gsub.total_bounds
        ax_map.text(maxx + x_offset, (miny + maxy) / 2,
                    PLOT_SHORT.get(which, f"P{which}"),
                    ha="left", va="center", fontsize=PLOT_TAG_FONT,
                    weight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=3.5,
                                                foreground="black")],
                    zorder=6)   # ABOVE points

# --- Set C extent + label (always TOP) ---
if setc_extent_gdf is not None and not setc_extent_gdf.empty:
    setc_extent_gdf.boundary.plot(ax=ax_map, color="red",
                                  linewidth=SETC_BOX_LW, linestyle="--",
                                  zorder=7)
    minx, miny, maxx, maxy = setc_extent_gdf.total_bounds
    ax_map.text(maxx + 0.006 * (r - l), (miny + maxy) / 2,
                "Set C", ha="left", va="center",
                fontsize=PLOT_TAG_FONT, weight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=3.5,
                                            foreground="black")],
                zorder=8)   # HIGHEST

# --- Legend (order: A → B → C) ---
handles=[
    Line2D([0],[0],color="black",lw=LEGEND_HANDLE_LW,label="Pu‘u Wa‘awa‘a"),
    Line2D([0],[0],color="red",lw=LEGEND_HANDLE_LW,label="Set A Segmentation Plots"),
]
if (setb_points is not None) and (not setb_points.empty):
    handles.append(Line2D([0],[0], marker='o', linestyle='None',
                          markersize=8, markerfacecolor="#FFD24D",
                          markeredgecolor="black", markeredgewidth=0.8,
                          label="Set B Species Inventory"))
handles.append(Line2D([0],[0],color="red",lw=LEGEND_HANDLE_LW,
                      linestyle="--",label="Set C Integrated Plot"))

ax_map.legend(
    handles=handles,loc="upper right",
    bbox_to_anchor=(1.05, 1.0),
    frameon=False,prop={"size":LEGEND_FONT}
)
ax_map.set_axis_off()

# Inset Hawai‘i
if Path(BIG_ISLAND_GPKG).exists():
    big_island = gpd.read_file(BIG_ISLAND_GPKG).to_crs(rast_crs)
    if not big_island.empty:
        iax = inset_axes(ax_map, width="26%", height="26%", loc="lower left")
        iax.set_facecolor("#E9F4FF")

        big_island.boundary.plot(ax=iax, color="black", linewidth=1.1)
        extent.boundary.plot(ax=iax, color="#222", linewidth=1.3)

        ex_ct = union_all_safe(extent).centroid
        iax.plot(ex_ct.x, ex_ct.y, marker="*", markersize=10, color="#D81B60")

        iax.text(0.5, 0.5, "Hawai‘i", transform=iax.transAxes,
                 ha="center", va="center", fontsize=INSET_ISLAND_TITLE_FONT,
                 color="#333", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        iax.set_xticks([]); iax.set_yticks([])

        # 🔑 Add full border around the inset
        for spine in iax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
            spine.set_color("black")

# Right column: full-height Set C detail (georeferenced)
zoom_left, zoom_bottom, zoom_width, zoom_height = 0.58, 0.00, 0.49, 1.00

ax_zoom = fig.add_axes([zoom_left, zoom_bottom, zoom_width, zoom_height], zorder=2)
ax_zoom.imshow(blowup_img, extent=blowup_extent, origin="upper", interpolation="nearest")
ax_zoom.axis("off")

# Title inside
ax_zoom.text(0.5, 0.99, "Set C (detail)", ha="center", va="top",
             transform=ax_zoom.transAxes, fontsize=SUBPLOT_TITLE_FONT+2, color="white",
             path_effects=[pe.withStroke(linewidth=3.2, foreground="black")], zorder=50)

# thin frame
for sp in ax_zoom.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.4); sp.set_color("#333333")

# *** scalebars ***
# main map: 4 km centered
add_scalebar(ax_map, where="center-bottom",
             fontsize=SCALEBAR_FONT, target_m=4000,
             units="km", zorder=60)

# set C detail: 500 m bottom-right (adjust if you need a different length)
add_scalebar(ax_zoom, where="bottom-right",
             fontsize=SCALEBAR_FONT+2,
             target_m=40, units="m", zorder=60)

fig.savefig(OUT_FIG,dpi=300,bbox_inches="tight")
print(f"[OK] Saved {OUT_FIG}")
