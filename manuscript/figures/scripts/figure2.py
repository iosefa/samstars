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
from matplotlib.patches import Rectangle
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from shapely.geometry import box
from PIL import Image
from pyforestscan.handlers import read_lidar

# ── EDIT THESE ────────────────────────────────────────────────────────────────
CHIP_DIR = Path("manuscript/assets/six_class_chips")
OUTLINE_BY_CLASS = {cls: CHIP_DIR / f"class_{cls}_outline.gpkg" for cls in range(1, 7)}
RGB_BY_CLASS = {cls: CHIP_DIR / f"class_{cls}.tif" for cls in range(1, 7)}
CHM_BY_CLASS = {cls: CHIP_DIR / f"class_{cls}_chm.tif" for cls in range(1, 7)}
IMG_BY_CLASS = {
    1: "manuscript/data/ahupuaa/class_examples/IMG_7264.jpeg",
    2: "manuscript/data/ahupuaa/class_examples/IMG_1768.jpeg",
    3: "manuscript/data/ahupuaa/class_examples/IMG_0899.jpeg",
    4: "manuscript/data/ahupuaa/class_examples/IMG_6719.jpeg",
    5: "manuscript/data/ahupuaa/class_examples/IMG_8896.jpeg",
    6: "manuscript/data/ahupuaa/class_examples/IMG_6788.jpeg",
}
CLASSES = ["(METPOL)","(GREROB)","(ACAKOA)","(DIOSAN)","(PROPAL)","(OTHER)"]
SPECIES = ["M. polymorpha","G. robusta","A. koa","D. sandwicensis","P. pallida","Pooled others"]
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG      = OUT_DIR / "figure2.png"

EPT_JSON = "/Users/iosefa/repos/obia/docs/example_data/full/ept/ept.json"
EPT_SRS  = "EPSG:6635"

# Typography — publication-friendly
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
ROW_LABELS = ["Point Cloud", "CHM", "RGB", "Ground Photo"]
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
    return (1, 2, 3) if src.count >= 3 else (1, 1, 1)

def reproject_geom(geom, src_crs, dst_crs):
    g = gpd.GeoDataFrame(geometry=[geom], crs=src_crs)
    if g.crs != dst_crs:
        g = g.to_crs(dst_crs)
    return g.geometry.iloc[0]

def square_pad(geom, pad_m):
    minx, miny, maxx, maxy = geom.bounds
    return box(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)

def center_square_crop(arr):
    """Center-crop to square (returns array and offsets), preserving center."""
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W, _ = arr.shape
    side = min(H, W)
    r0 = (H - side) // 2
    c0 = (W - side) // 2
    if arr.ndim == 2:
        return arr[r0:r0+side, c0:c0+side], side, r0, c0
    return arr[r0:r0+side, c0:c0+side, :], side, r0, c0

def center_crop_to_side(arr, target_side):
    """Center-crop square array to exact target_side; returns array and offsets."""
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W, _ = arr.shape
    # H == W by construction; crop equally around center
    cut = max(0, H - target_side)
    r0 = cut // 2
    r1 = r0 + target_side
    if arr.ndim == 2:
        return arr[r0:r1, r0:r1], r0, r0
    return arr[r0:r1, r0:r1, :], r0, r0

def load_rgb_chip_or_full(current_cls):
    """Return (img HxWx3 float in [0,1], transform, width, height, crs) from six_class_chips."""
    chip = RGB_BY_CLASS.get(current_cls)
    if chip is None:
        raise FileNotFoundError(f"Missing RGB chip path for class {current_cls}")
    if not chip.exists():
        raise FileNotFoundError(f"Missing RGB chip: {chip}")
    with rasterio.open(chip) as src:
        bands = pick_rgb_bands(src)
        arr = src.read(indexes=bands, out_dtype="float32", masked=True)
        img = np.moveaxis(np.ma.filled(arr, np.nan), 0, -1)
        return pct_stretch(img), src.transform, src.width, src.height, src.crs

def load_chm_chip_or_full(current_cls):
    """Return (chm HxW float meters, transform, width, height, crs) from six_class_chips."""
    chip = CHM_BY_CLASS.get(current_cls)
    if chip is None:
        raise FileNotFoundError(f"Missing CHM chip path for class {current_cls}")
    if not chip.exists():
        raise FileNotFoundError(f"Missing CHM chip: {chip}")
    with rasterio.open(chip) as src:
        a = src.read(1, out_dtype="float32", masked=True)
        return np.ma.filled(a, np.nan).astype("float32", copy=False), src.transform, src.width, src.height, src.crs

def _polygon_exterior_rings(geom):
    try:
        from shapely.geometry import Polygon, MultiPolygon
    except Exception:
        return []
    rings = []
    if geom is None:
        return rings
    if geom.geom_type == 'Polygon':
        geoms = [geom]
    elif geom.geom_type == 'MultiPolygon':
        geoms = list(geom.geoms)
    else:
        return rings
    for g in geoms:
        try:
            x, y = g.exterior.xy
            rings.append((np.asarray(x), np.asarray(y)))
        except Exception:
            continue
    return rings

def geom_to_pixel_rings(geom, src_crs, chip_crs, transform, off_r=0, off_c=0):
    g = gpd.GeoSeries([geom], crs=src_crs)
    if chip_crs is not None and (g.crs != chip_crs):
        g = g.to_crs(chip_crs)
    rings_xy = _polygon_exterior_rings(g.iloc[0])
    px_rings = []
    for xs, ys in rings_xy:
        rr, cc = rasterio.transform.rowcol(transform, xs, ys)
        rr = np.asarray(rr, dtype=float) - float(off_r)
        cc = np.asarray(cc, dtype=float) - float(off_c)
        px_rings.append((cc, rr))
    return px_rings

def nice_ceil(x, step=5.0):
    if not np.isfinite(x): return step
    return float(np.ceil(x / step) * step)

def load_jpeg(path: Path):
    try:
        im = Image.open(path).convert("RGB")
        return np.asarray(im)
    except Exception:
        return None

# ── load outlines ─────────────────────────────────────────────────────────────
classes, geoms, geom_crs_list = [], [], []
for cls in [1, 2, 3, 4, 5, 6]:
    outline = OUTLINE_BY_CLASS.get(cls)
    if outline is None:
        print(f"[WARN] No outline path for class {cls}; skipping.")
        continue
    if not outline.exists():
        print(f"[WARN] Missing outline {outline}; skipping.")
        continue
    if cls == 6:
        gdf = gpd.read_file(outline, layer="objects")
    else:
        gdf = gpd.read_file(outline)
    if gdf.empty or gdf.geometry.isna().all():
        print(f"[WARN] Empty outline in {outline}; skipping.")
        continue
    classes.append(cls)
    geoms.append(gdf.geometry.dropna().iloc[0])
    geom_crs_list.append(gdf.crs)

n_cols = len(classes)

# ── preload → square-crop → unify side per row ───────────────────────────────
rgb_list, chm_list = [], []
rgb_meta, chm_meta = [], []   # (transform, w, h, crs)

for current_cls, geom in zip(classes, geoms):
    rgb, t_r, w_r, h_r, crs_r = load_rgb_chip_or_full(current_cls)
    rgb_list.append(rgb); rgb_meta.append((t_r, w_r, h_r, crs_r))
    chm, t_c, w_c, h_c, crs_c = load_chm_chip_or_full(current_cls)
    chm_list.append(chm); chm_meta.append((t_c, w_c, h_c, crs_c))

# Chips are already standardized to 30 m squares; no additional cropping/resizing.

# CHM shared normalization (single colorbar)
# chm_vmin = 0.0
# chm_vmax = nice_ceil(np.nanmax(chm_max_vals), step=5.0) if chm_max_vals else 20.0
# norm_chm = mpl.colors.Normalize(vmin=chm_vmin, vmax=chm_vmax)
# cmap_chm = mpl.cm.get_cmap("viridis")

# ── layout (tight columns) ───────────────────────────────────────────────────
# Choose figure width so square tiles nearly fill column width (minimal gaps)
_h_title = 0.15
_h_rows = 4.0  # four square rows: PC, CHM, RGB, Photo
_total_h_ratio = _h_title + _h_rows  # 0.15 + 4
fig_h = 12.0
fig_w = (fig_h / _total_h_ratio) * (n_cols + 0.02)
fig = plt.figure(figsize=(fig_w, fig_h))
gs = mgrid.GridSpec(
    nrows=5, ncols=n_cols + 1,            # +1 gutter
    figure=fig,
    width_ratios=[0.02] + [1] * n_cols,
    height_ratios=[0.15, 1, 1, 1, 1],
    wspace=0.0, hspace=0.03
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
for idx, (cls, geom, rgb_img, chm_img) in enumerate(zip(classes, geoms, rgb_list, chm_list), start=1):
    geom_crs = geom_crs_list[idx-1]
    # Titles (two lines)
    ax_title = fig.add_subplot(gs[0, idx]); ax_title.set_axis_off()
    ax_title.text(0.5, 0.5, f"{SPECIES[cls-1]}\n{CLASSES[cls-1]}",
                  ha="center", va="center", fontsize=18, linespacing=1.0)

    # Row 1: Point Cloud (top row)
    ax_pc = fig.add_subplot(gs[1, idx]); ax_pc.set_box_aspect(1)
    try:
        geom_pc = geom
        if geom_crs is not None and str(geom_crs) != str(EPT_SRS):
            geom_pc = reproject_geom(geom, geom_crs, EPT_SRS)
        b = ([geom_pc.bounds[0], geom_pc.bounds[2]],[geom_pc.bounds[1], geom_pc.bounds[3]])
        pcs = read_lidar(EPT_JSON, EPT_SRS, b, crop_poly=True, poly=geom_pc.wkt, hag=True)
        if pcs and len(pcs[0]) > 0:
            pts = pcs[0]
            ax_pc.scatter(pts["X"], pts["HeightAboveGround"], s=32,
                          c=pts["Intensity"], cmap="viridis",
                          alpha=0.9, marker=".", linewidths=0)
            ax_pc.set_ylim(0, 20)
            # Fix x-axis span to 30 m centered on the polygon center
            cx = (geom_pc.bounds[0] + geom_pc.bounds[2]) / 2.0
            ax_pc.set_xlim(cx - 15.0, cx + 15.0)
        else:
            ax_pc.text(0.5, 0.5, "no points", ha="center", va="center",
                       fontsize=12, color="#666", transform=ax_pc.transAxes)
    except Exception as e:
        ax_pc.text(0.5, 0.5, f"PC error\n{e}", ha="center", va="center",
                   fontsize=12, color="#A33", transform=ax_pc.transAxes)
    if idx == 1:
        ax_pc.set_ylabel("Point Cloud Height (m)")
        ax_pc.set_yticks([0, 10, 20])
        ax_pc.tick_params(axis='y', length=4, width=1.2)
        ax_pc.set_xticks([])
        for spine in ["top", "right", "bottom"]:
            ax_pc.spines[spine].set_visible(False)
        ax_pc.spines["left"].set_linewidth(1.2)
    else:
        ax_pc.set_axis_off()

    # Row 2: CHM
    chm_vmin, chm_vmax = 0, 20
    ax_chm = fig.add_subplot(gs[2, idx]); ax_chm.set_axis_off(); ax_chm.set_box_aspect(1)
    im = ax_chm.imshow(chm_img, cmap="viridis", interpolation="nearest",
                       vmin=chm_vmin, vmax=chm_vmax)
    # Overlay polygon outline on CHM
    t_c, w_c, h_c, crs_c = chm_meta[idx-1]
    rings_c = geom_to_pixel_rings(geom, geom_crs or crs_c, crs_c, t_c, off_r=0, off_c=0)
    for xs, ys in rings_c:
        ax_chm.plot(xs, ys, color='white', linewidth=2.0, alpha=0.9)
        ax_chm.plot(xs, ys, color='black', linewidth=0.8, alpha=0.9)
    # Add colorbar (overlapping) + 10 m scalebar on first column only
    if idx == 1:
        # Absolute-positioned colorbar overlay: does not alter CHM axes position
        ax_pos = ax_chm.get_position()
        cbar_w = ax_pos.width * 0.03
        cbar_h = ax_pos.height * 0.90
        cbar_x = ax_pos.x0 + ax_pos.width * 0.005
        # Nudge colorbar a bit further to the left
        cbar_x -= ax_pos.width * 0.015
        cbar_y = ax_pos.y0 + ax_pos.height * 0.05

        # Add a white square background with minimal padding behind the colorbar
        pad_x = cbar_w * 0.28  # increase horizontal padding
        pad_y = cbar_h * 0.07  # slight vertical increase
        bg = Rectangle(
            (cbar_x - pad_x, cbar_y - pad_y),
            cbar_w + 2 * pad_x,
            cbar_h + 2 * pad_y,
            linewidth=0,
            edgecolor="none",
            facecolor="white",
            transform=fig.transFigure,
            zorder=10,
        )
        fig.add_artist(bg)

        cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
        cax.set_zorder(11)
        cax.set_facecolor('white')
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label("CHM (m)", fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        cbar.outline.set_linewidth(1.0)
        # 10 m scalebar at bottom-left (in pixels)
        H, W = chm_img.shape[:2]
        res_x = abs(t_c.a)
        bar_px = 10.0 / res_x if res_x > 0 else 0
        x0, y0 = 8.0, float(H) - 10.0
        ax_chm.plot([x0, x0 + bar_px], [y0, y0], color='white', linewidth=5.0, alpha=0.95)
        ax_chm.plot([x0, x0 + bar_px], [y0, y0], color='black', linewidth=2.0, alpha=0.95)
        ax_chm.text(x0 + bar_px/2, y0 - 6, "10 m", ha='center', va='bottom', fontsize=12,
                    color='white', path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    # Row 3: RGB
    ax_rgb = fig.add_subplot(gs[3, idx]); ax_rgb.set_axis_off(); ax_rgb.set_box_aspect(1)
    ax_rgb.imshow(rgb_img, interpolation="nearest")
    t_r, w_r, h_r, crs_r = rgb_meta[idx-1]
    rings = geom_to_pixel_rings(geom, geom_crs or crs_r, crs_r, t_r, off_r=0, off_c=0)
    for xs, ys in rings:
        ax_rgb.plot(xs, ys, color='white', linewidth=2.0, alpha=0.9)
        ax_rgb.plot(xs, ys, color='black', linewidth=0.8, alpha=0.9)

    # Row 4: Photo
    ax_ph = fig.add_subplot(gs[4, idx]); ax_ph.set_axis_off(); ax_ph.set_box_aspect(1)
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
