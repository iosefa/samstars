#!/usr/bin/env python3
"""
Three exports for class 3 (segment_id=111325) using the same 5 m square pad:

  1) fig4_class3_rgb753.png      -- WV3 7/5/3 composite with crown outline
  2) fig4_class3_glcm_corr.png   -- GLCM correlation (local) with crown outline (grayscale)
  3) fig4_class3_pointcloud.png  -- LiDAR point cloud (X vs HAG), no scalebar

Robust to scikit-image "gray/grey" spelling and texture module layout.
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from shapely.geometry import box
import matplotlib.patheffects as pe

# ---- robust GLCM imports (gray/grey + texture submodule fallbacks) ----
try:
    from skimage.feature import greycomatrix, greycoprops
except Exception:
    try:
        from skimage.feature.texture import greycomatrix, greycoprops
    except Exception:
        try:
            from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops
        except Exception:
            from skimage.feature.texture import (  # type: ignore
                graycomatrix as greycomatrix, graycoprops as greycoprops
            )

from skimage.util import view_as_windows

# LiDAR helper
from pyforestscan.handlers import read_lidar

# ----------------------------- CONFIG -----------------------------
OBJECTS_GPKG = Path("objects.gpkg")          # polygons with 'segment_id'
SEGMENT_ID   = 111325                         # class 3 crown segment id (ACAKOA)
PAD_M        = 5.0                            # square pad (meters)

WV3_PATH     = Path("/Users/iosefa/Dropbox/MergedImage2.tif")
# band indices (1-based) for output (WV3: 7=NIR1, 5=Red, 3=Green)
RGB_BANDS    = (7, 5, 3)
COASTAL_BAND = 1                              # for GLCM

EPT_JSON     = "/Users/iosefa/Downloads/ept6635/ept.json"
EPT_SRS      = "EPSG:6635"

# Outputs
PNG_RGB      = "fig4_class3_rgb753.png"
PNG_GLCM     = "fig4_class3_glcm_corr.png"
PNG_PC       = "fig4_class3_pointcloud.png"

# GLCM params
GLCM_DISTANCE = 4
GLCM_ANGLE    = 0.0       # radians
GLCM_LEVELS   = 64        # quantization levels
WIN_SIZE      = 9         # local window size (odd)
# ------------------------------------------------------------------


def square_pad(geom, pad_m):
    minx, miny, maxx, maxy = geom.bounds
    return box(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def reproject_geom(geom, src_crs, dst_crs):
    g = gpd.GeoDataFrame(geometry=[geom], crs=src_crs)
    if g.crs != dst_crs:
        g = g.to_crs(dst_crs)
    return g.geometry.iloc[0]


def robust_stretch_per_band(img_hwc, lo=2, hi=98):
    """Percentile stretch to [0,1] for 3-band array (H,W,3)."""
    out = np.zeros_like(img_hwc, dtype=np.float32)
    for i in range(img_hwc.shape[2]):
        b = img_hwc[..., i].astype(np.float32)
        finite = np.isfinite(b)
        if not finite.any():
            continue
        p_lo, p_hi = np.percentile(b[finite], [lo, hi])
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
            continue
        out[..., i] = np.clip((b - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return out


def extent_from_transform(transform, w, h):
    """(left, right, bottom, top) for imshow extent with origin='upper'."""
    left = transform.c
    right = transform.c + transform.a * w
    top = transform.f
    bottom = transform.f + transform.e * h
    return (left, right, bottom, top)


def clip_wv3_bbox(rgb_bands, poly, objects_crs):
    """Clip WV3 around square-padded bbox, return (RGB float [0,1], extent, raster_crs)."""
    with rasterio.open(WV3_PATH) as src:
        geom = reproject_geom(poly, objects_crs, src.crs)
        pad  = square_pad(geom, PAD_M)
        out, transform = rio_mask(src, [pad.__geo_interface__],
                                  crop=True, filled=False,
                                  indexes=list(rgb_bands))
        # cast to float32 BEFORE filling with NaNs
        out = out.astype("float32", copy=False)
        img = np.moveaxis(np.ma.filled(out, np.nan), 0, -1)
        img = robust_stretch_per_band(img, lo=2, hi=98)
        extent = extent_from_transform(transform, img.shape[1], img.shape[0])
        return img, extent, src.crs


def glcm_correlation_local(coastal_clip):
    """
    Compute local GLCM correlation (distance=2, angle=0) over a sliding window.
    Input: 2D array (coastal band clip). Returns float32 image in [0,1] (scaled).
    """
    a = coastal_clip.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)

    # normalize to [0,1], then quantize to levels
    p_lo, p_hi = np.percentile(a[finite], [2, 98])
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        p_lo, p_hi = float(np.nanmin(a)), float(np.nanmax(a))
    a_norm = np.clip((a - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    a_q = np.floor(a_norm * (GLCM_LEVELS - 1)).astype(np.uint8)

    # pad for windows
    pad = WIN_SIZE // 2
    a_pad = np.pad(a_q, pad_width=pad, mode="edge")

    # windows
    windows = view_as_windows(a_pad, (WIN_SIZE, WIN_SIZE))
    H, W, _, _ = windows.shape
    out = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            win = windows[i, j]
            g = greycomatrix(
                win,
                distances=[2],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True
            )
            corr = greycoprops(g, 'correlation')[0, 0]  # [-1, 1]
            out[i, j] = np.float32(0.5 * (corr + 1.0))  # → [0, 1]

    return out


def clip_coastal_for_glcm(poly, objects_crs):
    """Clip coastal band (1) to padded bbox; return (2D float, extent, raster_crs)."""
    with rasterio.open(WV3_PATH) as src:
        geom = reproject_geom(poly, objects_crs, src.crs)
        pad  = square_pad(geom, PAD_M)
        out, transform = rio_mask(
            src, [pad.__geo_interface__],
            crop=True, filled=False, indexes=COASTAL_BAND
        )
        # Robustly cast to float32 BEFORE filling and handle 2D/3D outputs
        out = out.astype("float32", copy=False)
        if out.ndim == 3 and out.shape[0] == 1:   # (1, H, W)
            a = np.ma.filled(out[0], np.nan)
            h, w = a.shape
        elif out.ndim == 2:                       # (H, W)
            a = np.ma.filled(out, np.nan)
            h, w = a.shape
        else:
            # Unexpected shape; fallback to empty tile to avoid crashing
            a = np.zeros((256, 256), dtype="float32") * np.nan
            h, w = a.shape

        extent = extent_from_transform(transform, w, h)
        return a, extent, src.crs


def plot_rgb_with_outline(img, extent, outline_gdf, out_png):
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=300)
    ax.imshow(img, extent=extent, origin="upper", interpolation="nearest")
    if outline_gdf is not None and not outline_gdf.empty:
        outline_gdf.boundary.plot(ax=ax, color="white", linewidth=1.2, zorder=3)
        outline_gdf.boundary.plot(ax=ax, color="black", linewidth=2.0, alpha=0.45, zorder=2)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def plot_glcm_with_outline(corr_img, extent, outline_gdf, out_png):
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=300)
    ax.imshow(corr_img, cmap="gray", vmin=0, vmax=1, extent=extent, origin="upper")
    if outline_gdf is not None and not outline_gdf.empty:
        outline_gdf.boundary.plot(ax=ax, color="white", linewidth=1.2, zorder=3)
        outline_gdf.boundary.plot(ax=ax, color="black", linewidth=2.0, alpha=0.45, zorder=2)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def plot_pointcloud(poly, objects_crs, out_png):
    b = ([poly.bounds[0], poly.bounds[2]],
         [poly.bounds[1], poly.bounds[3]])
    pcs = read_lidar(EPT_JSON, EPT_SRS, b, crop_poly=True, poly=poly.wkt, hag=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=300)
    if pcs and len(pcs[0]) > 0:
        pts = pcs[0]
        ax.scatter(pts["X"], pts["HeightAboveGround"],
                   s=26, c=pts["Intensity"], cmap="viridis",
                   alpha=0.9, marker=".", linewidths=0)
        ax.set_ylim(0, 20)
        ax.set_ylabel("Height (m)")
        ax.set_yticks([0, 10, 20])
        ax.set_xticks([])
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.text(0.02, 0.98, f"segment_id = {SEGMENT_ID}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color="#222",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
                )
    else:
        ax.text(0.5, 0.5, "no points", ha="center", va="center",
                fontsize=12, color="#666", transform=ax.transAxes)
        ax.set_axis_off()
    plt.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def main():
    if not OBJECTS_GPKG.exists():
        raise FileNotFoundError(f"Missing {OBJECTS_GPKG}")
    objs = gpd.read_file(OBJECTS_GPKG)
    if "segment_id" not in objs.columns:
        raise RuntimeError("objects.gpkg must contain a 'segment_id' column.")
    objects_crs = objs.crs

    row = objs[objs["segment_id"] == SEGMENT_ID]
    if row.empty:
        raise RuntimeError(f"segment_id={SEGMENT_ID} not found in {OBJECTS_GPKG}")
    geom = row.iloc[0].geometry

    # 1) RGB 7/5/3 with outline
    rgb_img, rgb_extent, raster_crs = clip_wv3_bbox(RGB_BANDS, geom, objects_crs)
    outline_gdf = gpd.GeoDataFrame(geometry=[reproject_geom(geom, objects_crs, raster_crs)], crs=raster_crs)
    plot_rgb_with_outline(rgb_img, rgb_extent, outline_gdf, PNG_RGB)

    # 2) GLCM correlation with outline (grayscale)
    coastal, coastal_extent, raster_crs2 = clip_coastal_for_glcm(geom, objects_crs)
    glcm_corr = glcm_correlation_local(coastal)
    outline_gdf2 = gpd.GeoDataFrame(geometry=[reproject_geom(geom, objects_crs, raster_crs2)], crs=raster_crs2)
    plot_glcm_with_outline(glcm_corr, coastal_extent, outline_gdf2, PNG_GLCM)

    # 3) LiDAR point cloud (no scalebar)
    plot_pointcloud(geom, objects_crs, PNG_PC)


if __name__ == "__main__":
    main()