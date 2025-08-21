#!/usr/bin/env python3
"""
Figure 2 — uniform square tiles (centered), tight columns, shared CHM colorbar,
and a single height scale on the first point-cloud panel.

Rows:
  0: Titles (Species on top, class below)
  1: Point Cloud (HAG vs X)  ← left-most panel shows y-axis (0,10,20 m)
  2: RGB (WV3 5/3/2 or 1/2/3)
  3: CHM (shared vmin/vmax across all; single colorbar)
  4: Ground Photo

All RGB and CHM chips are center-square-cropped, then center-cropped so
every tile in the row has the exact same pixel side length (min across classes).
Tiles render WITHOUT geo extents so their on-page size is identical.
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import matplotlib.patheffects as pe
import matplotlib as mpl
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from shapely.geometry import box
from PIL import Image
from pyforestscan.handlers import read_lidar

# ── EDIT THESE ────────────────────────────────────────────────────────────────
SEGMENT_BY_CLASS = {
    1: 154789,
    2: 133906,
    3: 111325,
    4: 84270,
    5: 33578,
    6: 112807,
}
IMG_BY_CLASS = {
    1: "IMG_7264.jpeg",
    2: "IMG_1768.jpeg",
    3: "IMG_0899.jpeg",
    4: "IMG_6719.jpeg",
    5: "IMG_8896.jpeg",
    6: "IMG_6788.jpeg",
}
CLASSES = ["(METPOL)","(GREROB)","(ACAKOA)","(DIOSAN)","(PROPAL)","(OTHER)"]
SPECIES = ["M. polymorpha","G. robusta","A. koa","D. sandwicensis","P. pallida","Pooled others"]
# ─────────────────────────────────────────────────────────────────────────────

OBJECTS_GPKG = Path("objects.gpkg")
FULL_RGB     = Path("/Users/iosefa/Dropbox/MergedImage2.tif")
FULL_CHM     = Path("/Users/iosefa/Dropbox/chm_full.tif")
CHIPS_DIR    = Path("six_class_chips")
OUT_FIG      = "figure2.png"

EPT_JSON = "/Users/iosefa/Downloads/ept6635/ept.json"
EPT_SRS  = "EPSG:6635"
PAD_M    = 5.0

# Typography — publication-friendly
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
ROW_LABELS = ["Point Cloud", "RGB", "CHM", "Ground Photo"]
ROW_LABEL_FONT = 18

# ── helpers ──────────────────────────────────────────────────────────────────
def pct_stretch(img, lo=2, hi=98):
    if img.ndim == 2:
        a = img.astype("float32", copy=False)
        finite = np.isfinite(a)
        if not finite.any(): return np.zeros_like(a, dtype="float32")
        p_lo, p_hi = np.percentile(a[finite], [lo, hi]); p_hi = max(p_hi, p_lo + 1e-6)
        return np.clip((a - p_lo) / (p_hi - p_lo), 0, 1)
    out = np.zeros_like(img, dtype="float32")
    for i in range(img.shape[2]):
        b = img[..., i].astype("float32", copy=False)
        finite = np.isfinite(b)
        if finite.any():
            p_lo, p_hi = np.percentile(b[finite], [lo, hi]); p_hi = max(p_hi, p_lo + 1e-6)
            out[..., i] = np.clip((b - p_lo) / (p_hi - p_lo), 0, 1)
        else:
            out[..., i] = 0.0
    return out

def pick_rgb_bands(src):
    return (5, 3, 2) if src.count >= 5 else (1, 2, 3)

def reproject_geom(geom, src_crs, dst_crs):
    g = gpd.GeoDataFrame(geometry=[geom], crs=src_crs)
    if g.crs != dst_crs:
        g = g.to_crs(dst_crs)
    return g.geometry.iloc[0]

def square_pad(geom, pad_m):
    minx, miny, maxx, maxy = geom.bounds
    return box(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)

def center_square_crop(arr):
    """Center-crop to square (returns array only), preserving center."""
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W, _ = arr.shape
    side = min(H, W)
    r0 = (H - side) // 2
    c0 = (W - side) // 2
    if arr.ndim == 2:
        return arr[r0:r0+side, c0:c0+side], side
    return arr[r0:r0+side, c0:c0+side, :], side

def center_crop_to_side(arr, target_side):
    """Center-crop square array to exact target_side."""
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W, _ = arr.shape
    # H == W by construction; crop equally around center
    cut = max(0, H - target_side)
    r0 = cut // 2
    r1 = r0 + target_side
    if arr.ndim == 2:
        return arr[r0:r1, r0:r1]
    return arr[r0:r1, r0:r1, :]

def load_rgb_chip_or_full(current_cls, geom, objects_crs):
    """Return stretched float RGB HxWx3 (no extent)."""
    chip = CHIPS_DIR / f"class_{current_cls}.tif"
    if chip.exists():
        with rasterio.open(chip) as src:
            bands = pick_rgb_bands(src)
            arr = src.read(indexes=bands, out_dtype="float32", masked=True)
            img = np.moveaxis(np.ma.filled(arr, np.nan), 0, -1)
            return pct_stretch(img)
    # fallback: clip from full raster
    with rasterio.open(FULL_RGB) as src:
        bands = pick_rgb_bands(src)
        geom_r = reproject_geom(geom, objects_crs, src.crs)
        pad = square_pad(geom_r, PAD_M)
        out, _ = rio_mask(src, [pad.__geo_interface__], crop=True, indexes=bands, filled=False)
        img = np.moveaxis(np.ma.filled(out, np.nan), 0, -1)
        return pct_stretch(img)

def load_chm_chip_or_full(current_cls, geom, objects_crs):
    """
    Return CHM in raw meters as float32 (HxW), no stretching.
    """
    chip = CHIPS_DIR / f"class_{current_cls}_chm.tif"
    if chip.exists():
        with rasterio.open(chip) as src:
            a = src.read(1, out_dtype="float32", masked=True)
            return np.ma.filled(a, np.nan).astype("float32", copy=False)
    # fallback: clip from full raster
    with rasterio.open(FULL_CHM) as src:
        geom_r = reproject_geom(geom, objects_crs, src.crs)
        pad = square_pad(geom_r, PAD_M)
        out, _ = rio_mask(src, [pad.__geo_interface__], crop=True, indexes=1, filled=False)
        a = np.ma.filled(out[0], np.nan).astype("float32", copy=False)
        return a

def nice_ceil(x, step=5.0):
    if not np.isfinite(x): return step
    return float(np.ceil(x / step) * step)

def load_jpeg(path: Path):
    try:
        im = Image.open(path).convert("RGB")
        return np.asarray(im)
    except Exception:
        return None

# ── load objects ──────────────────────────────────────────────────────────────
if not OBJECTS_GPKG.exists():
    raise FileNotFoundError(f"Missing {OBJECTS_GPKG}")
objs = gpd.read_file(OBJECTS_GPKG)
if "segment_id" not in objs.columns:
    raise RuntimeError("objects.gpkg must contain a 'segment_id' column.")
objects_crs = objs.crs

classes, geoms = [], []
for cls in [1, 2, 3, 4, 5, 6]:
    seg = SEGMENT_BY_CLASS.get(cls)
    cand = objs[objs["segment_id"] == seg]
    if cand.empty:
        print(f"[WARN] No segment_id={seg} for class {cls}; skipping.")
        continue
    classes.append(cls)
    geoms.append(cand.iloc[0].geometry)

n_cols = len(classes)

# ── preload → square-crop → unify side per row ───────────────────────────────
rgb_list, chm_list = [], []
rgb_sides, chm_sides = [], []

for current_cls, geom in zip(classes, geoms):
    rgb = load_rgb_chip_or_full(current_cls, geom, objects_crs)
    rgb, s_r = center_square_crop(rgb); rgb_list.append(rgb); rgb_sides.append(s_r)
    chm = load_chm_chip_or_full(current_cls, geom, objects_crs)
    chm, s_c = center_square_crop(chm); chm_list.append(chm); chm_sides.append(s_c)

rgb_target = int(min(rgb_sides)) if rgb_sides else None
chm_target = int(min(chm_sides)) if chm_sides else None

if rgb_target is not None:
    rgb_list = [center_crop_to_side(a, rgb_target) for a in rgb_list]
if chm_target is not None:
    chm_list = [center_crop_to_side(a, chm_target) for a in chm_list]

# CHM shared normalization (single colorbar)
# chm_vmin = 0.0
# chm_vmax = nice_ceil(np.nanmax(chm_max_vals), step=5.0) if chm_max_vals else 20.0
# norm_chm = mpl.colors.Normalize(vmin=chm_vmin, vmax=chm_vmax)
# cmap_chm = mpl.cm.get_cmap("viridis")

# ── layout (tight columns) ───────────────────────────────────────────────────
fig = plt.figure(figsize=(3.8 * n_cols, 12.0))
gs = mgrid.GridSpec(
    nrows=5, ncols=n_cols + 1,            # +1 gutter
    figure=fig,
    width_ratios=[0.02] + [1] * n_cols,
    height_ratios=[0.15, 1, 1, 1, 1],
    wspace=0.0015, hspace=0.03
)

# gutter labels
for r, label in enumerate(ROW_LABELS, start=1):
    if label in ["Point Cloud", "CHM"]:
        continue
    ax_lbl = fig.add_subplot(gs[r, 0]); ax_lbl.set_axis_off()
    ax_lbl.text(0.98, 0.5, label, rotation=90, va="center", ha="right",
                fontsize=ROW_LABEL_FONT, color="#333",
                transform=ax_lbl.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Keep handles to CHM axes for colorbar placement
chm_axes = []

# ── render columns ───────────────────────────────────────────────────────────
for c, (cls, geom, rgb_img, chm_img) in enumerate(zip(classes, geoms, rgb_list, chm_list), start=1):
    # Titles (two lines)
    ax_title = fig.add_subplot(gs[0, c]); ax_title.set_axis_off()
    ax_title.text(0.5, 0.5, f"{SPECIES[cls-1]}\n{CLASSES[cls-1]}",
                  ha="center", va="center", fontsize=18, linespacing=1.0)

    # Row 1: Point Cloud
    ax_pc = fig.add_subplot(gs[1, c]); ax_pc.set_box_aspect(1)
    try:
        b = ([geom.bounds[0], geom.bounds[2]],[geom.bounds[1], geom.bounds[3]])
        pcs = read_lidar(EPT_JSON, EPT_SRS, b, crop_poly=True, poly=geom.wkt, hag=True)
        if pcs and len(pcs[0]) > 0:
            pts = pcs[0]
            ax_pc.scatter(pts["X"], pts["HeightAboveGround"], s=32,
                          c=pts["Intensity"], cmap="viridis",
                          alpha=0.9, marker=".", linewidths=0)
            ax_pc.set_ylim(0, 20)
        else:
            ax_pc.text(0.5, 0.5, "no points", ha="center", va="center",
                       fontsize=12, color="#666", transform=ax_pc.transAxes)
    except Exception as e:
        ax_pc.text(0.5, 0.5, f"PC error\n{e}", ha="center", va="center",
                   fontsize=12, color="#A33", transform=ax_pc.transAxes)

    # Show a single height scale on the FIRST column only
    if c == 1:
        # y-axis as scale bar
        ax_pc.set_ylabel("Point Cloud Height (m)")
        ax_pc.set_yticks([0, 10, 20])
        ax_pc.tick_params(axis='y', length=4, width=1.2)
        # hide x ticks; keep frame subtle
        ax_pc.set_xticks([])
        for spine in ["top", "right", "bottom"]:
            ax_pc.spines[spine].set_visible(False)
        ax_pc.spines["left"].set_linewidth(1.2)
    else:
        ax_pc.set_axis_off()

    # Row 2: RGB
    ax_rgb = fig.add_subplot(gs[2, c]); ax_rgb.set_axis_off(); ax_rgb.set_box_aspect(1)
    ax_rgb.imshow(rgb_img, interpolation="nearest")

    # Row 3: CHM (shared norm, remember axes)
    chm_vmin, chm_vmax = 0, 20
    # CHM row
    ax_chm = fig.add_subplot(gs[3, c])
    ax_chm.set_axis_off()
    ax_chm.set_box_aspect(1)
    im = ax_chm.imshow(chm_img, cmap="viridis", interpolation="nearest",
                       vmin=chm_vmin, vmax=chm_vmax)

    # add colorbar only once, on the first column
    if c == 1:
        cbar = fig.colorbar(im, ax=ax_chm, orientation="vertical",
                            fraction=0.05, pad=0.02)
        cbar.set_label("CHM (m)", fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        # move it to the left side
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        # shift the colorbar leftwards so it hugs the subplot
        pos = cbar.ax.get_position()
        cbar.ax.set_position([pos.x0 - 0.105, pos.y0, pos.width, pos.height])

    # Row 4: Photo
    ax_ph = fig.add_subplot(gs[4, c]); ax_ph.set_axis_off(); ax_ph.set_box_aspect(1)
    photo = IMG_BY_CLASS.get(cls, "")
    arr = None
    if photo:
        try:
            im = Image.open(photo).convert("RGB"); arr = np.asarray(im)
        except Exception:
            arr = None
    if arr is not None:
        ax_ph.imshow(arr)
    else:
        ax_ph.text(0.5, 0.5, "no photo", ha="center", va="center",
                   fontsize=12, color="#666", transform=ax_ph.transAxes)

fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print(f"[OK] Saved {OUT_FIG}")