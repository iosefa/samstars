#!/usr/bin/env python3
"""
Figure 12 — Set C four-panel RGB with SAM and validation outlines

What it does
- Renders Set C imagery using RGB bands 5,3,2 (1-indexed) from set_c.tif.
- Overlays SAM–StarDist predicted crown polygons and reference validation crowns as outlines.
- Shows four zoomed panels clipped to specified test extents.

Inputs (relative to manuscript/)
- imagery: data/set_c/set_c.tif
- SAM preds: data/set_c/set_c_sam_predicted.gpkg
- reference: data/set_c/set_c_validation.gpkg
- panel extents (polygons):
  - Top-left:  data/set_c/FN_test_extent1.gpkg (falls back to FN_test_extent.gpkg if missing)
  - Top-right: data/set_c/FN_test_extent2.gpkg
  - Bottom-left: data/set_c/FN_test_extent3.gpkg
  - Bottom-right: data/set_c/TP_test_extent.gpkg

Output
- figures/output/figure12.png
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import box
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

try:
    import fiona
except Exception:  # pragma: no cover
    fiona = None


ROOT = Path(__file__).resolve().parents[2]  # manuscript
SETC = ROOT / "data" / "set_c"
IMG_PATH = SETC / "set_c.tif"
REF_PATH = SETC / "set_c_validation.gpkg"
SAM_PATH = SETC / "set_c_sam_predicted.gpkg"

EXT_TL_1 = SETC / "FN_test_extent1.gpkg"
EXT_TL_FALLBACK = SETC / "FN_test_extent.gpkg"
EXT_TR = SETC / "FN_test_extent2.gpkg"
EXT_BL = SETC / "FN_test_extent3.gpkg"
EXT_BR = SETC / "TP_test_extent.gpkg"

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure12.png"

# Diagnostics (FP/FN) from Figure 8
DIAG_DIR = OUT_DIR / "diags"
FP_PATH = DIAG_DIR / "figure8_sam_fp_pred.gpkg"
FN_PATH = DIAG_DIR / "figure8_sam_fn_ref.gpkg"


def list_layers(path: Path):
    if fiona is None:
        return []
    try:
        return fiona.listlayers(path)
    except Exception:
        return []


def pick_polygon_layer(path: Path) -> Optional[str]:
    layers = list_layers(path)
    if not layers:
        return None
    for ly in layers:
        with fiona.open(path, layer=ly) as src:
            if src.schema["geometry"] in ("Polygon", "MultiPolygon"):
                return ly
    return layers[0]


def load_gpkg_polygons(path: Path) -> gpd.GeoDataFrame:
    lyr = pick_polygon_layer(path)
    if lyr is None:
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(path, layer=lyr)
    return gdf


def ensure_crs(gdf: gpd.GeoDataFrame, target) -> gpd.GeoDataFrame:
    if (not gdf.crs) or (gdf.crs != target):
        return gdf.to_crs(target)
    return gdf


def read_rgb_bands(path: Path, bands_1idx=(5, 3, 2)) -> Tuple[np.ndarray, object, Tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        rgb = src.read(list(bands_1idx))  # (3,H,W)
        bounds = src.bounds
        crs = src.crs
    # Per-channel 2–98% scaling to 0..1
    scaled = []
    for c in range(rgb.shape[0]):
        band = rgb[c].astype(np.float32)
        finite = np.isfinite(band)
        if not np.any(finite):
            scaled.append(np.zeros_like(band, dtype=np.float32))
        else:
            lo, hi = np.percentile(band[finite], [2, 98])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0
            band = np.clip(band, lo, hi)
            band = (band - lo) / max(hi - lo, 1e-6)
            scaled.append(band)
    rgb01 = np.stack(scaled, axis=0)
    img = np.transpose(rgb01, (1, 2, 0))  # HxWx3
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return img, crs, extent


def get_bounds_from_gpkg(gpkg: Path, target_crs) -> Optional[Tuple[float, float, float, float]]:
    if not gpkg.exists():
        return None
    gdf = load_gpkg_polygons(gpkg)
    gdf = ensure_crs(gdf, target_crs)
    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx, miny, maxx, maxy)


def add_scale_bar(ax, bar_len: float = 2.0):
    import matplotlib.patches as mpatches
    # Use current visible limits (robust to inverted y)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xmin, xmax = (min(x0, x1), max(x0, x1))
    ymin, ymax = (min(y0, y1), max(y0, y1))
    width = float(xmax - xmin)
    height = float(ymax - ymin)

    # Fixed scalebar length (meters), styled cleanly
    pad_x = width * 0.03
    pad_y = height * 0.03
    bar_h = max(height * 0.006, height * 0.012)

    # Position at top-left of the axes (in data coords)
    top_y = ymax - pad_y
    left_x = xmin + pad_x

    # Background box fully covers label (below) and bar (above)
    # Increase gap between the bar and the label for maximum separation
    label_gap = max(pad_y * 0.60, bar_h * 1.00)
    text_h = max(pad_y * 0.80, bar_h * 0.80)
    bg_pad = max(pad_y * 0.30, bar_h * 0.30)
    bg_w = bar_len + 2 * bg_pad
    bg_h = bar_h + label_gap + text_h + 2 * bg_pad
    bg_x = left_x - bg_pad
    bg_y = top_y - bg_h
    ax.add_patch(mpatches.Rectangle((bg_x, bg_y), bg_w, bg_h,
                                    facecolor='white', edgecolor='black', linewidth=0.8, zorder=3.8))

    # Place label at bottom inside the background box (below the bar)
    bar_x = left_x
    label_y = bg_y + bg_pad
    # Bar sits above the label by label_gap + text_h
    bar_y = label_y + text_h + label_gap
    ax.add_patch(mpatches.Rectangle((bar_x, bar_y), bar_len, bar_h,
                                    facecolor='black', edgecolor='white', linewidth=0.8, zorder=4))

    # Draw label below the bar, inside the background box
    # Increase space below the bar by lowering the label within the text band
    # Maximize gap: place label at bottom of text band (larger distance to bar)
    ax.text(bar_x + bar_len/2, label_y, f"{int(bar_len)} m",
            ha='center', va='top', color='black', fontsize=13, zorder=4)


def main():
    # Imagery
    img, img_crs, full_extent = read_rgb_bands(IMG_PATH, (5, 3, 2))

    # Vectors (outlines)
    ref = ensure_crs(load_gpkg_polygons(REF_PATH), img_crs)
    sam = ensure_crs(load_gpkg_polygons(SAM_PATH), img_crs)

    # Load diagnostics (if present)
    fp = load_gpkg_polygons(FP_PATH) if FP_PATH.exists() else None
    fn = load_gpkg_polygons(FN_PATH) if FN_PATH.exists() else None
    if fp is not None:
        fp = ensure_crs(fp, img_crs)
    if fn is not None:
        fn = ensure_crs(fn, img_crs)

    # Extents (with TL fallback)
    tl_path = EXT_TL_1 if EXT_TL_1.exists() else EXT_TL_FALLBACK
    tl = get_bounds_from_gpkg(tl_path, img_crs)
    tr = get_bounds_from_gpkg(EXT_TR, img_crs)
    bl = get_bounds_from_gpkg(EXT_BL, img_crs)
    br = get_bounds_from_gpkg(EXT_BR, img_crs)

    panels = [
        (tl, None),
        (tr, None),
        (bl, None),
        (br, None),
    ]

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.5))
    axs = axs.ravel()

    # Desired panel titles
    panel_titles = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    ]

    for i, (bounds, _unused) in enumerate(panels):
        ax = axs[i]
        ax.imshow(img, extent=full_extent, origin='upper', zorder=1)
        ax.set_axis_off()

        # Clip and overlay FP/FN fills per extent
        if bounds is not None:
            minx, miny, maxx, maxy = bounds
            extent_poly = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=img_crs)
        else:
            extent_poly = None

        # Overlap suppression: hide labels whose label boxes overlap > threshold with existing ones
        placed_boxes = []  # list of (x0, y0, x1, y1) in data coords
        is_panel_a = (i == 0)
        extra_label_allowed = [True] if is_panel_a else [False]
        def _resolve_label_pos(x, y):
            try:
                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                w = abs(x1 - x0); h = abs(y1 - y0)
            except Exception:
                if full_extent:
                    w = abs(full_extent[1] - full_extent[0]); h = abs(full_extent[3] - full_extent[2])
                else:
                    w = h = 1.0
            # Approximate label box size as a fraction of view (smaller to keep more labels)
            bw = max(w * 0.040, 0.3)
            bh = max(h * 0.030, 0.3)
            x0b, x1b = x - bw/2.0, x + bw/2.0
            y0b, y1b = y - bh/2.0, y + bh/2.0
            # IoU-based overlap threshold (only hide when there is heavy overlap)
            thr = 0.80
            def iou_with(a):
                ax0, ay0, ax1, ay1 = a
                inter_w = max(0.0, min(x1b, ax1) - max(x0b, ax0))
                inter_h = max(0.0, min(y1b, ay1) - max(y0b, ay0))
                inter = inter_w * inter_h
                if inter <= 0:
                    return 0.0
                area_b = bw * bh
                area_a = (ax1 - ax0) * (ay1 - ay0)
                union = area_a + area_b - inter
                return inter / max(union, 1e-9)
            for idx, (ax0, ay0, ax1, ay1) in enumerate(placed_boxes):
                iou = iou_with((ax0, ay0, ax1, ay1))
                # In panel (a), always keep the left-most label when overlap occurs
                if is_panel_a and iou > 0.05:
                    placed_cx = 0.5 * (ax0 + ax1)
                    if x < placed_cx:
                        # Replace the existing (right-hand) label with this left-hand one
                        placed_boxes[idx] = (x0b, y0b, x1b, y1b)
                        return True, x, y
                    else:
                        # Candidate is to the right; allow exactly one extra overlapping label with a stronger jitter
                        if extra_label_allowed[0]:
                            # Try a stronger left/up jitter to separate visually
                            candidates = [(-bw*0.65, 0.0), (0.0, bh*0.65), (-bw*0.50, bh*0.40), (-bw*0.50, -bh*0.40)]
                            accepted = False
                            for dx, dy in candidates:
                                nx, ny = x + dx, y + dy
                                nx0, nx1 = nx - bw/2.0, nx + bw/2.0
                                ny0, ny1 = ny - bh/2.0, ny + bh/2.0
                                # bounds check
                                try:
                                    if bounds is not None:
                                        bminx, bminy, bmaxx, bmaxy = bounds
                                    else:
                                        ax_x0, ax_x1 = ax.get_xlim(); ax_y0, ax_y1 = ax.get_ylim()
                                        bminx, bmaxx = (min(ax_x0, ax_x1), max(ax_x0, ax_x1))
                                        bminy, bmaxy = (min(ax_y0, ax_y1), max(ax_y0, ax_y1))
                                except Exception:
                                    bminx = nx0; bmaxx = nx1; bminy = ny0; bmaxy = ny1
                                if not (bminx <= nx0 and nx1 <= bmaxx and bminy <= ny0 and ny1 <= bmaxy):
                                    continue
                                # check overlaps with existing boxes at new pos with lenient threshold
                                ok = True
                                for b in placed_boxes:
                                    inter_w = max(0.0, min(nx1, b[2]) - max(nx0, b[0]))
                                    inter_h = max(0.0, min(ny1, b[3]) - max(ny0, b[1]))
                                    inter = inter_w * inter_h
                                    if inter <= 0:
                                        continue
                                    area_b2 = bw * bh
                                    area_a2 = (b[2] - b[0]) * (b[3] - b[1])
                                    union2 = area_a2 + area_b2 - inter
                                    if inter / max(union2, 1e-9) > 0.45:
                                        ok = False; break
                                if ok:
                                    placed_boxes.append((nx0, ny0, nx1, ny1))
                                    extra_label_allowed[0] = False
                                    return True, nx, ny
                            # if no candidate worked, still place the original to ensure one extra label appears
                            placed_boxes.append((x0b, y0b, x1b, y1b))
                            extra_label_allowed[0] = False
                            return True, x, y
                        # No extra allowed; drop it
                        return False, x, y
                if iou > thr:
                    return False, x, y
            # If mild overlap remains, attempt small jitters to reduce
            offsets = [
                ( bw*0.45, 0.0), (-bw*0.45, 0.0),
                (0.0,  bh*0.45), (0.0, -bh*0.45),
                ( bw*0.35,  bh*0.35), (-bw*0.35,  bh*0.35),
                ( bw*0.35, -bh*0.35), (-bw*0.35, -bh*0.35),
            ]
            accept_box = (x0b, y0b, x1b, y1b)
            px, py = x, y
            # Accept original if no significant overlap; else try jitters to reduce IoU
            for (ax0, ay0, ax1, ay1) in placed_boxes:
                if iou_with((ax0, ay0, ax1, ay1)) > 0.05:
                    # try offsets
                    for dx, dy in offsets:
                        nx, ny = x + dx, y + dy
                        nx0, nx1 = nx - bw/2.0, nx + bw/2.0
                        ny0, ny1 = ny - bh/2.0, ny + bh/2.0
                        ok = True
                        for b in placed_boxes:
                            # recompute IoU for candidate box
                            x0b2, y0b2, x1b2, y1b2 = nx0, ny0, nx1, ny1
                            inter_w = max(0.0, min(x1b2, b[2]) - max(x0b2, b[0]))
                            inter_h = max(0.0, min(y1b2, b[3]) - max(y0b2, b[1]))
                            inter = inter_w * inter_h
                            if inter <= 0:
                                continue
                            area_b2 = bw * bh
                            area_a2 = (b[2] - b[0]) * (b[3] - b[1])
                            union2 = area_a2 + area_b2 - inter
                            iou2 = inter / max(union2, 1e-9)
                            if iou2 > 0.05:
                                ok = False; break
                        if ok:
                            px, py = nx, ny
                            accept_box = (nx0, ny0, nx1, ny1)
                            break
                    break
            # Ensure final position lies within current panel bounds
            try:
                if bounds is not None:
                    bminx, bminy, bmaxx, bmaxy = bounds
                else:
                    # fall back to current axes limits
                    ax_x0, ax_x1 = ax.get_xlim(); ax_y0, ax_y1 = ax.get_ylim()
                    bminx, bmaxx = (min(ax_x0, ax_x1), max(ax_x0, ax_x1))
                    bminy, bmaxy = (min(ax_y0, ax_y1), max(ax_y0, ax_y1))
                mx = max((bmaxx - bminx) * 0.01, bw * 0.10)
                my = max((bmaxy - bminy) * 0.01, bh * 0.10)
                if not (bminx + mx <= px <= bmaxx - mx and bminy + my <= py <= bmaxy - my):
                    return False, x, y
            except Exception:
                pass
            placed_boxes.append(accept_box)
            return True, px, py

        if fp is not None:
            try:
                fp_to_plot = gpd.clip(fp, extent_poly) if extent_poly is not None else fp
                if not fp_to_plot.empty:
                    # white halo underlay for visibility
                    fp_to_plot.plot(
                        ax=ax,
                        facecolor='none',
                        edgecolor='white',
                        linewidth=5.0,
                        linestyle=(0, (9, 5)),
                        zorder=3.29,
                    )
                    # colored dashed outline
                    fp_to_plot.plot(
                        ax=ax,
                        facecolor='none',
                        edgecolor=(0.8, 0.0, 0.0, 0.98),  # red
                        linewidth=3.2,
                        linestyle=(0, (9, 5)),  # longer dashes for visibility
                        zorder=3.3,
                    )
                    # Annotations: label mismatch (near overlap) vs low IoU — compute locally against refs
                    try:
                        # Compute max IoU to any reference polygon in this extent
                        A = fp_to_plot.reset_index(drop=True).copy(); A["aix"] = A.index
                        B = ref.reset_index(drop=True).copy(); B["bix"] = B.index
                        if extent_poly is not None:
                            B = gpd.clip(B, extent_poly)
                        if not B.empty:
                            inter = gpd.overlay(A[["aix","geometry"]], B[["bix","geometry"]], how='intersection')
                            if not inter.empty:
                                area_i = inter.geometry.area
                                area_a = A.set_index("aix").geometry.area
                                area_b = B.set_index("bix").geometry.area
                                inter = inter.merge(area_a.rename("aa"), left_on="aix", right_index=True)
                                inter = inter.merge(area_b.rename("ab"), left_on="bix", right_index=True)
                                inter["iou"] = area_i / (inter["aa"] + inter["ab"] - area_i + 1e-12)
                                max_iou = inter.groupby("aix")["iou"].max()
                            else:
                                max_iou = gpd.pd.Series(index=A["aix"], data=0.0)
                        else:
                            max_iou = gpd.pd.Series(index=A["aix"], data=0.0)
                        for idx, geom in A[["aix","geometry"]].itertuples(index=False):
                            try:
                                c = geom.representative_point(); v = float(max_iou.get(idx, 0.0))
                                txt = "Label\nmismatch" if v >= 0.5 else "IoU<0.5"
                                ok, px, py = _resolve_label_pos(c.x, c.y)
                                if ok:
                                    ax.text(
                                        px, py, txt,
                                        color='black', fontsize=14, fontweight='bold', ha='center', va='center', zorder=4,
                                        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.9),
                                        path_effects=[pe.withStroke(linewidth=2.6, foreground='white')]
                                    )
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
        if fn is not None:
            try:
                fn_to_plot = gpd.clip(fn, extent_poly) if extent_poly is not None else fn
                if not fn_to_plot.empty:
                    # white halo underlay for visibility
                    fn_to_plot.plot(
                        ax=ax,
                        facecolor='none',
                        edgecolor='white',
                        linewidth=5.0,
                        linestyle=(0, (2, 2)),
                        zorder=3.34,
                    )
                    # colored dashed outline (purple)
                    fn_to_plot.plot(
                        ax=ax,
                        facecolor='none',
                        edgecolor='#CC79A7',  # purple (colorblind-friendly)
                        linewidth=3.2,
                        linestyle=(0, (2, 2)),  # dotted-ish for contrast
                        zorder=3.35,
                    )
                    # Annotations: label mismatch (near overlap) vs low IoU — compute locally against preds
                    try:
                        A = fn_to_plot.reset_index(drop=True).copy(); A["aix"] = A.index
                        B = sam.reset_index(drop=True).copy(); B["bix"] = B.index
                        if extent_poly is not None:
                            B = gpd.clip(B, extent_poly)
                        if not B.empty:
                            inter = gpd.overlay(A[["aix","geometry"]], B[["bix","geometry"]], how='intersection')
                            if not inter.empty:
                                area_i = inter.geometry.area
                                area_a = A.set_index("aix").geometry.area
                                area_b = B.set_index("bix").geometry.area
                                inter = inter.merge(area_a.rename("aa"), left_on="aix", right_index=True)
                                inter = inter.merge(area_b.rename("ab"), left_on="bix", right_index=True)
                                inter["iou"] = area_i / (inter["aa"] + inter["ab"] - area_i + 1e-12)
                                max_iou = inter.groupby("aix")["iou"].max()
                            else:
                                max_iou = gpd.pd.Series(index=A["aix"], data=0.0)
                        else:
                            max_iou = gpd.pd.Series(index=A["aix"], data=0.0)
                        for idx, geom in A[["aix","geometry"]].itertuples(index=False):
                            try:
                                c = geom.representative_point(); v = float(max_iou.get(idx, 0.0))
                                txt = "Label\nmismatch" if v >= 0.5 else "IoU<0.5"
                                ok, px, py = _resolve_label_pos(c.x, c.y)
                                if ok:
                                    ax.text(
                                        px, py, txt,
                                        color='black', fontsize=14, fontweight='bold', ha='center', va='center', zorder=4,
                                        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.9),
                                        path_effects=[pe.withStroke(linewidth=2.6, foreground='white')]
                                    )
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

        # Overlays: predicted (blue) and reference (orange) outlines
        try:
            sam.plot(ax=ax, facecolor='none', edgecolor='#0072B2', linewidth=3.2, alpha=0.95, zorder=3)
        except Exception:
            pass
        try:
            ref.plot(ax=ax, facecolor='none', edgecolor='#E69F00', linewidth=3.2, alpha=0.95, zorder=3)
        except Exception:
            pass

        # Zoom to requested bounds if available
        if bounds is not None:
            minx, miny, maxx, maxy = bounds
            ax.set_xlim(minx, maxx)
            # keep north up (origin='upper' so invert y when setting limits)
            ax.set_ylim(maxy, miny)

        ax.set_title(panel_titles[i], fontsize=16)

        # Add a scale bar to each subplot with custom lengths
        try:
            lens = [5.0, 3.0, 2.0, 4.0]  # a=5m, b=3m, c=2m, d=4m
            add_scale_bar(ax, bar_len=lens[i] if i < len(lens) else 2.0)
        except Exception:
            pass

    # Legend back to prior style
    import matplotlib.patches as mpatches
    handles = [
        Line2D([], [], color='#0072B2', linewidth=3.2, label='SAM–StarDist (pred)'),
        Line2D([], [], color='#E69F00', linewidth=3.2, label='Reference (validation)'),
        Line2D([], [], color=(0.8, 0.0, 0.0, 0.95), linewidth=3.0, linestyle=(0, (6, 4)), label='False Positive'),
        Line2D([], [], color='#CC79A7', linewidth=3.0, linestyle=(0, (2, 2)), label='False Negative'),
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=4,
        frameon=True,
        prop={'size': 13},
        handlelength=3.0,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    fig.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
