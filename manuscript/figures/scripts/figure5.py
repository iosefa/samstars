#!/usr/bin/env python3
"""
Figure: Whole-extent predicted/reference + PQ + imagery (1 row × 4 cols)

Row:
  Semantic (geometry + class match; IoU ≥ 0.50)

Columns:
  1) Combined predicted + reference (by class)
  2) WorldView-3 / CHM (diagonal split)
  3) Drone image
  4) Panoptic Quality (semantic)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

# -------------------
# Hard-coded inputs
# -------------------
EXTENT_GPKG   = Path("/Users/iosefa/repos/sam/PQ_extent.gpkg")
PRED_GPKG     = Path("/Users/iosefa/repos/sam/PQ_predicted.gpkg")
REF_GPKG      = Path("/Users/iosefa/repos/sam/PQ_reference.gpkg")
WV_TIF        = Path("/Users/iosefa/repos/sam/PQ_worldview.tif")
CHM_TIF       = Path("/Users/iosefa/repos/sam/PQ_chm.tif")
DRONE_TIF     = Path("/Users/iosefa/repos/sam/PQ_drone.tif")

PRED_CLASS    = "predicted_class"   # field in predicted
REF_CLASS     = "class"             # field in reference
# Output path under figures/output relative to this script
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure5.png"

IOU_THR       = 0.50

# Typography — enlarge for publication
TITLE_SIZE = 20
LABEL_SIZE = 18
LEGEND_FONTSIZE = 16
ROW_LABEL_SIZE = 20

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 16,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -------------------
# Color palettes (color-blind friendly)
# -------------------
# PQ fills (bottom row)
COLS_PQ  = {"TP": "#0072B2", "FP": "#D55E00"}   # blue / orange
FN_EDGE  = "#B2182B"                             # dark red for FN outline
REF_EDGE = "#4D4D4D"                             # dark gray for matched refs

# Class mapping (ints -> codes) and outline colors
CLASS_LABELS  = {2: "GREROB", 4: "DIOSAN"}
CLASS_COLORS  = {"GREROB": "#0072B2", "DIOSAN": "#E69F00", "OTHER": "#999999"}
# Second-row class colors (distinct from blue/orange)
# Second-row class colors tuned for contrast over imagery
# GREROB: deep purple; DIOSAN: high-contrast gold (not green/blue/orange)
ROW2_CLASS_COLORS = {"GREROB": "#7E2F8E", "DIOSAN": "#FFD700"}
DEFAULT_CLASS_COLOR = "#999999"

# Imagery outline thickness
LW_PRED_IMAGERY = 3.2
LW_REF_IMAGERY  = 3.2

# Dataset-level fills (for combined panels)
PRED_FILL = "#4C78A8"  # blue
REF_FILL  = "#F58518"  # orange
PRED_ALPHA = 0.40
REF_ALPHA  = 0.28

# Dataset outline styles for combined/imagery (high-contrast overlay)
# Use vivid magenta with a white halo to stand out over drone imagery
OUTLINE_COLOR = "#FF00A6"  # vivid magenta
PRED_LINE_COLOR = OUTLINE_COLOR
REF_LINE_COLOR  = OUTLINE_COLOR
DATASET_LINE_LW = 2.0
PRED_LINE_STYLE = "--"  # predicted dashed
REF_LINE_STYLE  = "-"   # reference solid

# -------------------
# Helpers
# -------------------
def ensure_metric_crs(gdf: gpd.GeoDataFrame, target_crs):
    if gdf.crs is None:
        raise RuntimeError("Input layer has no CRS.")
    if target_crs is None or gdf.crs == target_crs:
        return gdf
    return gdf.to_crs(target_crs)

def greedy_match(df, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df.empty:
        return df
    out = df.sort_values(iou_col, ascending=False).copy()
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out

def compute_intersections(pred, ref):
    inter = gpd.overlay(pred[["pid","geometry"]], ref[["vid","geometry"]], how="intersection")
    if inter.empty:
        inter["iou"] = []
        return inter
    pred_area = pred.set_index("pid").geometry.area
    ref_area  = ref.set_index("vid").geometry.area
    inter["area_int"] = inter.geometry.area
    inter = inter.merge(pred_area.rename("area_p"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(ref_area.rename("area_v"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["area_int"] / (inter["area_p"] + inter["area_v"] - inter["area_int"] + 1e-12)
    return inter

def classify_geometry_only(pred, ref, inter):
    m50 = greedy_match(inter.loc[inter["iou"] >= IOU_THR, ["pid","vid","iou"]])
    pred_cat = {pid: "FP" for pid in pred["pid"].tolist()}
    for _, r in m50.iterrows():
        pred_cat[int(r["pid"])] = "TP"
    matched_ref = set(m50["vid"].tolist())
    ref_fn = {vid: (vid not in matched_ref) for vid in ref["vid"].tolist()}
    return pred_cat, ref_fn

def classify_semantic(pred, ref, inter, pred_class_field, ref_class_field):
    inter2 = inter.merge(pred[["pid", pred_class_field]], on="pid", how="left") \
                  .merge(ref[["vid", ref_class_field]],   on="vid", how="left")
    m50 = inter2[(inter2["iou"] >= IOU_THR) &
                 (inter2[pred_class_field] == inter2[ref_class_field])]
    m50 = greedy_match(m50[["pid","vid","iou"]])
    pred_cat = {pid: "FP" for pid in pred["pid"].tolist()}
    for _, r in m50.iterrows():
        pred_cat[int(r["pid"])] = "TP"
    matched_ref = set(m50["vid"].tolist())
    ref_fn = {vid: (vid not in matched_ref) for vid in ref["vid"].tolist()}
    return pred_cat, ref_fn

def _coerce_to_codes(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    return s.map(CLASS_LABELS).fillna("OTHER")

def _present_codes(series):
    return list(series.dropna().unique())

def _class_handles(codes_present):
    handles = []
    for code in codes_present:
        if code == "OTHER":
            continue
        col = CLASS_COLORS.get(code, DEFAULT_CLASS_COLOR)
        handles.append(mpatches.Patch(color=col, label=code))
    return handles

def _class_line_handles(codes_present):
    handles = []
    for code in codes_present:
        if code == "OTHER":
            continue
        col = CLASS_COLORS.get(code, DEFAULT_CLASS_COLOR)
        handles.append(mlines.Line2D([], [], color=col, linewidth=2.0, label=code))
    return handles

def _ensure_required_codes(codes_present, required=("GREROB", "DIOSAN")):
    # Ensure a consistent legend that always shows the key classes
    seen = set()
    ordered = []
    # add required first
    for r in required:
        if r not in seen:
            ordered.append(r)
            seen.add(r)
    # add whatever was present (preserve order)
    for c in codes_present:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered

def _dataset_handles():
    return [
        mpatches.Patch(facecolor=PRED_FILL, edgecolor='none', alpha=PRED_ALPHA, label='Predicted'),
        mpatches.Patch(facecolor=REF_FILL,  edgecolor='none', alpha=REF_ALPHA,  label='Reference'),
    ]

def _dataset_line_handles():
    return [
        mlines.Line2D([], [], color=PRED_LINE_COLOR, linewidth=DATASET_LINE_LW,
                       linestyle=PRED_LINE_STYLE, label='Predicted'),
        mlines.Line2D([], [], color=REF_LINE_COLOR,  linewidth=DATASET_LINE_LW,
                       linestyle=REF_LINE_STYLE, label='Reference'),
    ]

def legend_inside_top_right(ax, handles):
    # Per-subplot legends are disabled in favor of a single global legend.
    return

# --- raster helpers for imagery rows ---
def extent_from_transform(t: Affine, w: int, h: int):
    left = t.c
    right = t.c + t.a * w
    top = t.f
    bottom = t.f + t.e * h
    return (left, right, bottom, top)

def _square_bounds(minx, maxx, miny, maxy):
    """Pad the shorter dimension so the view is square around center.
    Returns (minx2, maxx2, miny2, maxy2)."""
    dx = float(maxx - minx)
    dy = float(maxy - miny)
    if dx == 0 or dy == 0:
        return minx, maxx, miny, maxy
    if abs(dx - dy) < 1e-9:
        return minx, maxx, miny, maxy
    if dx < dy:
        cx = 0.5 * (minx + maxx)
        half = 0.5 * dy
        return cx - half, cx + half, miny, maxy
    else:
        cy = 0.5 * (miny + maxy)
        half = 0.5 * dx
        return minx, maxx, cy - half, cy + half

def show_raster(ax, tif_path: Path, singleband_cmap: str = "gray"):
    with rasterio.open(tif_path) as src:
        img = src.read(out_dtype="float32")
        t = src.transform
        ext = extent_from_transform(t, src.width, src.height)
        if img.shape[0] >= 3:
            rgb = np.stack([img[0], img[1], img[2]], axis=-1)
            for k in range(3):
                b = rgb[..., k]
                if np.isfinite(b).any():
                    lo, hi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
                    if hi > lo:
                        b = (b - lo) / (hi - lo)
                rgb[..., k] = np.clip(b, 0, 1)
            ax.imshow(rgb, extent=ext, origin="upper")
        else:
            b = img[0]
            if np.isfinite(b).any():
                lo, hi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
                if hi > lo:
                    b = (b - lo) / (hi - lo)
            ax.imshow(np.clip(b, 0, 1), cmap=singleband_cmap, extent=ext, origin="upper")
        return src.crs

def draw_raster_clipped(ax, tif_path: Path, clip_xy, singleband_cmap: str = "gray"):
    """Draw raster clipped to a polygon defined in data coords (clip_xy: list of (x,y)). Returns CRS."""
    import numpy as np
    import matplotlib.patches as patches
    with rasterio.open(tif_path) as src:
        img = src.read(out_dtype="float32")
        t = src.transform
        ext = extent_from_transform(t, src.width, src.height)
        if img.shape[0] >= 3:
            rgb = np.stack([img[0], img[1], img[2]], axis=-1)
            for k in range(3):
                b = rgb[..., k]
                if np.isfinite(b).any():
                    lo, hi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
                    if hi > lo:
                        b = (b - lo) / (hi - lo)
                rgb[..., k] = np.clip(b, 0, 1)
            artist = ax.imshow(rgb, extent=ext, origin="upper")
        else:
            b = img[0]
            if np.isfinite(b).any():
                lo, hi = np.nanpercentile(b[np.isfinite(b)], [2, 98])
                if hi > lo:
                    b = (b - lo) / (hi - lo)
            artist = ax.imshow(np.clip(b, 0, 1), cmap=singleband_cmap, extent=ext, origin="upper")
        poly = patches.Polygon(clip_xy, closed=True, transform=ax.transData)
        artist.set_clip_path(poly)
        return src.crs

# -------------------
# Plot helpers (TOP 3 ROWS – unchanged logic)
# -------------------
def plot_pred_row(ax_left, ax_right, pred, have_classes):
    ax_left.set_axis_off()
    if not pred.empty:
        pred.plot(ax=ax_left, color="#CFCFCF", edgecolor="#8a8a8a", linewidth=0.3)
    ax_left.set_title("Predicted (no class)")

    ax_right.set_axis_off()
    if have_classes and not pred.empty:
        present = _ensure_required_codes(_present_codes(pred[PRED_CLASS]))
        for k, g in pred.groupby(PRED_CLASS):
            col = CLASS_COLORS.get(k, DEFAULT_CLASS_COLOR)
            g.plot(ax=ax_right, facecolor=col, edgecolor="white", linewidth=0.15, alpha=0.95)
        legend_inside_top_right(ax_right, _class_handles(present))
    else:
        if not pred.empty:
            pred.plot(ax=ax_right, color="#CFCFCF", edgecolor="#8a8a8a", linewidth=0.3)
    ax_right.set_title("Predicted (by class)")

def plot_ref_row(ax_left, ax_right, ref, have_classes):
    ax_left.set_axis_off()
    if not ref.empty:
        ref.plot(ax=ax_left, color="#E5E5E5", edgecolor="#7a7a7a", linewidth=0.3)
    ax_left.set_title("Reference (no class)")

    ax_right.set_axis_off()
    if have_classes and not ref.empty:
        present = _ensure_required_codes(_present_codes(ref[REF_CLASS]))
        for k, g in ref.groupby(REF_CLASS):
            col = CLASS_COLORS.get(k, DEFAULT_CLASS_COLOR)
            g.plot(ax=ax_right, facecolor=col, edgecolor="white", linewidth=0.15, alpha=0.95)
        legend_inside_top_right(ax_right, _class_handles(present))
    else:
        if not ref.empty:
            ref.plot(ax=ax_right, color="#E5E5E5", edgecolor="#7a7a7a", linewidth=0.3)
    ax_right.set_title("Reference (by class)")

def plot_pq_panel(ax, pred, ref, pred_cat_map, ref_fn_map, title):
    ax.set_axis_off()
    # predicted fills
    for cat, color in COLS_PQ.items():
        subset = pred.loc[[pred_cat_map.get(pid, "FP") == cat for pid in pred["pid"].tolist()]]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="none", alpha=0.85, linewidth=0)
    # reference outlines
    ref_matched = ref.loc[[not ref_fn_map.get(vid, True) for vid in ref["vid"].tolist()]]
    ref_unmatch = ref.loc[[ref_fn_map.get(vid, True) for vid in ref["vid"].tolist()]]
    if not ref_matched.empty:
        ref_matched.boundary.plot(ax=ax, color=REF_EDGE, linewidth=0.7, alpha=0.85, zorder=5)
    if not ref_unmatch.empty:
        ref_unmatch.boundary.plot(ax=ax, color=FN_EDGE, linewidth=1.8, alpha=0.95, zorder=6)
    ax.set_title(title)
    # Per-panel legends removed; added globally at bottom

# -------------------
# Imagery rows
# -------------------
def _imagery_style_handles(color="black"):
    return [
        mlines.Line2D([], [], color=color, linewidth=LW_PRED_IMAGERY, linestyle='--', label='Predicted'),
        mlines.Line2D([], [], color=color, linewidth=LW_REF_IMAGERY,  linestyle='-', label='Reference'),
    ]

def plot_imagery_row(
    ax_left,
    ax_right,
    tif_path,
    pred,
    ref,
    extent,
    title_left,
    title_right,
    singleband_cmap: str = "gray",
    left_outline_color: str = OUTLINE_COLOR,
    class_code: str = None,
):
    # Read/plot imagery once per axis – reproject outlines to the raster CRS
    ax_left.set_axis_off()
    r_crs = show_raster(ax_left, tif_path, singleband_cmap=singleband_cmap)
    pred_r = pred if pred.crs == r_crs else pred.to_crs(r_crs)
    ref_r  = ref  if ref.crs  == r_crs else ref.to_crs(r_crs)
    extent_r = extent if extent.crs == r_crs else extent.to_crs(r_crs)

    # Left: either neutral both (default) or predicted-only for class_code
    if class_code is not None:
        pred_sel = pred_r if pred_r.empty else pred_r[pred_r[PRED_CLASS] == class_code]
        if not pred_sel.empty:
            col = CLASS_COLORS.get(class_code, DEFAULT_CLASS_COLOR)
            pred_sel.boundary.plot(ax=ax_left, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
            pred_sel.boundary.plot(ax=ax_left, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')
    else:
        if not pred_r.empty:
            # white halo underlay for visibility (Pred dashed)
            pred_r.boundary.plot(ax=ax_left, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
            pred_r.boundary.plot(ax=ax_left, color=left_outline_color, linewidth=LW_PRED_IMAGERY, linestyle='--')
        if not ref_r.empty:
            # Reference solid
            ref_r.boundary.plot(ax=ax_left, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
            ref_r.boundary.plot(ax=ax_left, color=left_outline_color, linewidth=LW_REF_IMAGERY, linestyle='-')
    extent_r.boundary.plot(ax=ax_left, color="black", linewidth=1.0)
    # Force identical view to the extent bounds and equal aspect
    minx, miny, maxx, maxy = extent_r.total_bounds
    minx, maxx, miny, maxy = _square_bounds(minx, maxx, miny, maxy)
    ax_left.set_xlim(minx, maxx)
    ax_left.set_ylim(miny, maxy)
    ax_left.set_aspect('equal')
    legend_inside_top_right(ax_left, _imagery_style_handles(color=left_outline_color))
    ax_left.set_title(title_left)

    # Right: class-colored outlines, thick
    ax_right.set_axis_off()
    show_raster(ax_right, tif_path, singleband_cmap=singleband_cmap)
    if class_code is not None:
        ref_sel = ref_r if ref_r.empty else ref_r[ref_r[REF_CLASS] == class_code]
        if not ref_sel.empty:
            col = ROW2_CLASS_COLORS.get(class_code, CLASS_COLORS.get(class_code, DEFAULT_CLASS_COLOR))
            ref_sel.boundary.plot(ax=ax_right, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
            ref_sel.boundary.plot(ax=ax_right, color=col, linewidth=LW_REF_IMAGERY, linestyle='-')
    else:
        if not pred_r.empty:
            for k, g in pred_r.groupby(PRED_CLASS):
                col = CLASS_COLORS.get(k, DEFAULT_CLASS_COLOR)
                g.boundary.plot(ax=ax_right, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')
        if not ref_r.empty:
            for k, g in ref_r.groupby(REF_CLASS):
                col = CLASS_COLORS.get(k, DEFAULT_CLASS_COLOR)
                g.boundary.plot(ax=ax_right, color=col, linewidth=LW_REF_IMAGERY, linestyle='-')
    extent_r.boundary.plot(ax=ax_right, color="black", linewidth=1.0)
    # Force identical view to the extent bounds and equal aspect
    ax_right.set_xlim(minx, maxx)
    ax_right.set_ylim(miny, maxy)
    ax_right.set_aspect('equal')
    # Style legend (solid=pred, dashed=ref) + class legend (DIOSAN/GREROB always shown)
    present = []
    if not pred_r.empty and PRED_CLASS in pred_r.columns:
        present.extend(_present_codes(pred_r[PRED_CLASS]))
    if not ref_r.empty and REF_CLASS in ref_r.columns:
        present.extend(_present_codes(ref_r[REF_CLASS]))
    seen = set(); present = [x for x in present if not (x in seen or seen.add(x))]
    present = _ensure_required_codes(present)
    handles = _imagery_style_handles(color="black") + _class_line_handles(present)
    legend_inside_top_right(ax_right, handles)
    ax_right.set_title(title_right)

def plot_imagery_diagonal(ax_left, ax_right, wv_tif, chm_tif, pred, ref, extent, title_left, title_right, class_code: str = None):
    # Both axes: show WV in top-left triangle, CHM in bottom-right triangle
    ax_left.set_axis_off(); ax_right.set_axis_off()
    # Use WV CRS for extents
    with rasterio.open(wv_tif) as src:
        r_crs = src.crs
    pred_r = pred if pred.crs == r_crs else pred.to_crs(r_crs)
    ref_r  = ref  if ref.crs  == r_crs else ref.to_crs(r_crs)
    extent_r = extent if extent.crs == r_crs else extent.to_crs(r_crs)
    minx, miny, maxx, maxy = extent_r.total_bounds
    minx, maxx, miny, maxy = _square_bounds(minx, maxx, miny, maxy)
    tri_tl = [(minx, maxy), (minx, miny), (maxx, maxy)]
    tri_br = [(maxx, miny), (minx, miny), (maxx, maxy)]
    # Left: outlines
    draw_raster_clipped(ax_left, wv_tif, tri_tl, singleband_cmap="gray")
    draw_raster_clipped(ax_left, chm_tif, tri_br, singleband_cmap="viridis")
    if class_code is not None:
        pred_sel = pred_r if pred_r.empty else pred_r[pred_r[PRED_CLASS] == class_code]
        if not pred_sel.empty:
            col = CLASS_COLORS.get(class_code, DEFAULT_CLASS_COLOR)
            pred_sel.boundary.plot(ax=ax_left, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
            pred_sel.boundary.plot(ax=ax_left, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')
    else:
        if not pred_r.empty:
            pred_r.boundary.plot(ax=ax_left, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
            pred_r.boundary.plot(ax=ax_left, color=OUTLINE_COLOR, linewidth=LW_PRED_IMAGERY, linestyle='--')
        if not ref_r.empty:
            ref_r.boundary.plot(ax=ax_left, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
            ref_r.boundary.plot(ax=ax_left, color=OUTLINE_COLOR, linewidth=LW_REF_IMAGERY, linestyle='-')
    extent_r.boundary.plot(ax=ax_left, color="black", linewidth=1.0)
    ax_left.set_xlim(minx, maxx); ax_left.set_ylim(miny, maxy); ax_left.set_aspect('equal')
    ax_left.set_title(title_left)
    # Right: outlines
    draw_raster_clipped(ax_right, wv_tif, tri_tl, singleband_cmap="gray")
    draw_raster_clipped(ax_right, chm_tif, tri_br, singleband_cmap="viridis")
    if class_code is not None:
        ref_sel = ref_r if ref_r.empty else ref_r[ref_r[REF_CLASS] == class_code]
        if not ref_sel.empty:
            col = ROW2_CLASS_COLORS.get(class_code, CLASS_COLORS.get(class_code, DEFAULT_CLASS_COLOR))
            ref_sel.boundary.plot(ax=ax_right, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
            ref_sel.boundary.plot(ax=ax_right, color=col, linewidth=LW_REF_IMAGERY, linestyle='-')
    else:
        if not pred_r.empty:
            for k, g in pred_r.groupby(PRED_CLASS):
                col = CLASS_COLORS.get(k, DEFAULT_CLASS_COLOR)
                g.boundary.plot(ax=ax_right, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')

# ---- helpers for class-pair rendering on a single axis (second row) ----
def draw_class_pair_diagonal(ax, wv_tif, chm_tif, pred, ref, extent, class_code: str):
    with rasterio.open(wv_tif) as src:
        r_crs = src.crs
    pred_r = pred if pred.crs == r_crs else pred.to_crs(r_crs)
    ref_r  = ref  if ref.crs  == r_crs else ref.to_crs(r_crs)
    extent_r = extent if extent.crs == r_crs else extent.to_crs(r_crs)
    minx, miny, maxx, maxy = extent_r.total_bounds
    tri_tl = [(minx, maxy), (minx, miny), (maxx, maxy)]
    tri_br = [(maxx, miny), (minx, miny), (maxx, maxy)]
    ax.set_axis_off()
    draw_raster_clipped(ax, wv_tif, tri_tl, singleband_cmap="gray")
    draw_raster_clipped(ax, chm_tif, tri_br, singleband_cmap="viridis")
    col = ROW2_CLASS_COLORS.get(class_code, CLASS_COLORS.get(class_code, DEFAULT_CLASS_COLOR))
    pred_sel = pred_r if pred_r.empty else pred_r[pred_r[PRED_CLASS] == class_code]
    ref_sel  = ref_r  if ref_r.empty  else ref_r[ref_r[REF_CLASS] == class_code]
    if not ref_sel.empty:
        ref_sel.boundary.plot(ax=ax, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
        ref_sel.boundary.plot(ax=ax, color=col, linewidth=LW_REF_IMAGERY, linestyle='-')
    if not pred_sel.empty:
        pred_sel.boundary.plot(ax=ax, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
        pred_sel.boundary.plot(ax=ax, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')
    minx, miny, maxx, maxy = extent_r.total_bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_aspect('equal')
    extent_r.boundary.plot(ax=ax, color="black", linewidth=1.0)

def draw_class_pair_raster(ax, tif_path, pred, ref, extent, class_code: str):
    ax.set_axis_off()
    r_crs = show_raster(ax, tif_path, singleband_cmap="gray")
    pred_r = pred if pred.crs == r_crs else pred.to_crs(r_crs)
    ref_r  = ref  if ref.crs  == r_crs else ref.to_crs(r_crs)
    extent_r = extent if extent.crs == r_crs else extent.to_crs(r_crs)
    col = ROW2_CLASS_COLORS.get(class_code, CLASS_COLORS.get(class_code, DEFAULT_CLASS_COLOR))
    pred_sel = pred_r if pred_r.empty else pred_r[pred_r[PRED_CLASS] == class_code]
    ref_sel  = ref_r  if ref_r.empty  else ref_r[ref_r[REF_CLASS] == class_code]
    if not ref_sel.empty:
        ref_sel.boundary.plot(ax=ax, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
        ref_sel.boundary.plot(ax=ax, color=col, linewidth=LW_REF_IMAGERY, linestyle='-')
    if not pred_sel.empty:
        pred_sel.boundary.plot(ax=ax, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
        pred_sel.boundary.plot(ax=ax, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')
    extent_r.boundary.plot(ax=ax, color="black", linewidth=1.0)
    minx, miny, maxx, maxy = extent_r.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')
    
def draw_dual_class_pair_diagonal(ax, wv_tif, chm_tif, pred, ref, extent, class_codes):
    with rasterio.open(wv_tif) as src:
        r_crs = src.crs
    pred_r = pred if pred.crs == r_crs else pred.to_crs(r_crs)
    ref_r  = ref  if ref.crs  == r_crs else ref.to_crs(r_crs)
    extent_r = extent if extent.crs == r_crs else extent.to_crs(r_crs)
    minx, miny, maxx, maxy = extent_r.total_bounds
    tri_tl = [(minx, maxy), (minx, miny), (maxx, maxy)]
    tri_br = [(maxx, miny), (minx, miny), (maxx, maxy)]
    ax.set_axis_off()
    draw_raster_clipped(ax, wv_tif, tri_tl, singleband_cmap="gray")
    draw_raster_clipped(ax, chm_tif, tri_br, singleband_cmap="viridis")
    for code in class_codes:
        col = ROW2_CLASS_COLORS.get(code, CLASS_COLORS.get(code, DEFAULT_CLASS_COLOR))
        pred_sel = pred_r if pred_r.empty else pred_r[pred_r[PRED_CLASS] == code]
        ref_sel  = ref_r  if ref_r.empty  else ref_r[ref_r[REF_CLASS] == code]
        if not ref_sel.empty:
            ref_sel.boundary.plot(ax=ax, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
            ref_sel.boundary.plot(ax=ax, color=col, linewidth=LW_REF_IMAGERY, linestyle='-')
        if not pred_sel.empty:
            pred_sel.boundary.plot(ax=ax, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
            pred_sel.boundary.plot(ax=ax, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')
    extent_r.boundary.plot(ax=ax, color="black", linewidth=1.0)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_aspect('equal')

def draw_dual_class_pair_raster(ax, tif_path, pred, ref, extent, class_codes):
    ax.set_axis_off()
    r_crs = show_raster(ax, tif_path, singleband_cmap="gray")
    pred_r = pred if pred.crs == r_crs else pred.to_crs(r_crs)
    ref_r  = ref  if ref.crs  == r_crs else ref.to_crs(r_crs)
    extent_r = extent if extent.crs == r_crs else extent.to_crs(r_crs)
    for code in class_codes:
        col = ROW2_CLASS_COLORS.get(code, CLASS_COLORS.get(code, DEFAULT_CLASS_COLOR))
        pred_sel = pred_r if pred_r.empty else pred_r[pred_r[PRED_CLASS] == code]
        ref_sel  = ref_r  if ref_r.empty  else ref_r[ref_r[REF_CLASS] == code]
        if not ref_sel.empty:
            ref_sel.boundary.plot(ax=ax, color="white", linewidth=LW_REF_IMAGERY + 1.5, linestyle='-')
            ref_sel.boundary.plot(ax=ax, color=col, linewidth=LW_REF_IMAGERY, linestyle='-')
        if not pred_sel.empty:
            pred_sel.boundary.plot(ax=ax, color="white", linewidth=LW_PRED_IMAGERY + 1.5, linestyle='--')
            pred_sel.boundary.plot(ax=ax, color=col, linewidth=LW_PRED_IMAGERY, linestyle='--')
    extent_r.boundary.plot(ax=ax, color="black", linewidth=1.0)
    minx, miny, maxx, maxy = extent_r.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')

# -------------------
# Combined Pred + Ref columns
# -------------------
def plot_combined_geo(ax, pred, ref):
    ax.set_axis_off()
    # Draw dashed/solid outlines (no fills) with white halo for visibility
    if not ref.empty:
        ref.boundary.plot(ax=ax, color="white", linewidth=DATASET_LINE_LW + 1.2,
                          linestyle=REF_LINE_STYLE)
        ref.boundary.plot(ax=ax, color=REF_LINE_COLOR, linewidth=DATASET_LINE_LW,
                          linestyle=REF_LINE_STYLE)
    if not pred.empty:
        pred.boundary.plot(ax=ax, color="white", linewidth=DATASET_LINE_LW + 1.2,
                           linestyle=PRED_LINE_STYLE)
        pred.boundary.plot(ax=ax, color=PRED_LINE_COLOR, linewidth=DATASET_LINE_LW,
                           linestyle=PRED_LINE_STYLE)
    ax.set_title("Predicted + Reference (no class)")
    legend_inside_top_right(ax, _dataset_line_handles())

def plot_combined_class(ax, pred, ref):
    ax.set_axis_off()
    # Second row combined panel: outline by class with distinct colors,
    # predicted dashed and reference solid. Use ROW2_CLASS_COLORS for contrast.
    if not ref.empty and REF_CLASS in ref.columns:
        for code, g in ref.groupby(REF_CLASS):
            if code not in ("GREROB", "DIOSAN"):
                continue
            col = ROW2_CLASS_COLORS.get(code, CLASS_COLORS.get(code, DEFAULT_CLASS_COLOR))
            # Solid reference with optional halo
            g.boundary.plot(ax=ax, color="white", linewidth=DATASET_LINE_LW + 0.8, linestyle='-')
            g.boundary.plot(ax=ax, color=col, linewidth=DATASET_LINE_LW, linestyle='-')
    if not pred.empty and PRED_CLASS in pred.columns:
        for code, g in pred.groupby(PRED_CLASS):
            if code not in ("GREROB", "DIOSAN"):
                continue
            col = ROW2_CLASS_COLORS.get(code, CLASS_COLORS.get(code, DEFAULT_CLASS_COLOR))
            # Dashed predicted with optional halo
            g.boundary.plot(ax=ax, color="white", linewidth=DATASET_LINE_LW + 0.8, linestyle='--')
            g.boundary.plot(ax=ax, color=col, linewidth=DATASET_LINE_LW, linestyle='--')
    ax.set_title("Predicted + Reference (by class)")
    # Legend: Predicted/Reference per class (GREROB, DIOSAN)
    gr_col = ROW2_CLASS_COLORS.get("GREROB", CLASS_COLORS.get("GREROB", DEFAULT_CLASS_COLOR))
    di_col = ROW2_CLASS_COLORS.get("DIOSAN", CLASS_COLORS.get("DIOSAN", DEFAULT_CLASS_COLOR))
    handles = [
        mlines.Line2D([], [], color=gr_col, linewidth=DATASET_LINE_LW, linestyle='--', label='Predicted GREROB'),
        mlines.Line2D([], [], color=gr_col, linewidth=DATASET_LINE_LW, linestyle='-',  label='Reference GREROB'),
        mlines.Line2D([], [], color=di_col, linewidth=DATASET_LINE_LW, linestyle='--', label='Predicted DIOSAN'),
        mlines.Line2D([], [], color=di_col, linewidth=DATASET_LINE_LW, linestyle='-',  label='Reference DIOSAN'),
    ]
    legend_inside_top_right(ax, handles)

# -------------------
# Main
# -------------------
def main():
    # Load and align to extent CRS
    extent = gpd.read_file(EXTENT_GPKG); target_crs = extent.crs
    ref  = ensure_metric_crs(gpd.read_file(REF_GPKG),  target_crs).reset_index(drop=True)
    pred = ensure_metric_crs(gpd.read_file(PRED_GPKG), target_crs).reset_index(drop=True)
    extent = ensure_metric_crs(extent, target_crs)

    # IDs
    ref["vid"]  = np.arange(len(ref), dtype=int)
    pred["pid"] = np.arange(len(pred), dtype=int)

    # Convert classes to string codes for coloring
    if REF_CLASS in ref.columns:
        ref[REF_CLASS] = _coerce_to_codes(ref[REF_CLASS])
    if PRED_CLASS in pred.columns:
        pred[PRED_CLASS] = _coerce_to_codes(pred[PRED_CLASS])

    # Intersections and categories (keep working logic for top 3 rows)
    inter = compute_intersections(pred, ref)
    pred_cat_geo, ref_fn_geo = classify_geometry_only(pred, ref, inter)
    pred_cat_sem, ref_fn_sem = classify_semantic(pred, ref, inter, PRED_CLASS, REF_CLASS)

    # Figure layout: 1 row × 4 columns (semantic only)
    # Use GridSpec wspace/hspace to force near-touching columns
    # Choose figure size to make subplot cells nearly square with equal aspect
    fig_h = 6.0
    left_m, right_m, top_m, bottom_m = 0.03, 0.97, 0.92, 0.18
    usable_w = (right_m - left_m)
    usable_h = (top_m - bottom_m)
    fig_w = 1.0 * fig_h * (usable_h / usable_w)  # 4 columns vs 1 row
    fig, axes = plt.subplots(
        1, 4, figsize=(fig_w, fig_h),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0}
    )

    # New order: Col1 Combined, Col2 WV/CHM, Col3 UAV, Col4 PQ

    # Column 1: Combined Pred + Ref
    plot_combined_class(axes[0], pred, ref)

    # Column 4: PQ panels (moved last)
    plot_pq_panel(axes[3], pred, ref, pred_cat_sem, ref_fn_sem, "")

    # Column titles (top row only); ensure aligned at same vertical position
    axes[0].set_title("Predicted + Reference", y=1.02)
    axes[1].set_title("WorldView 3 / CHM", y=1.02)
    axes[2].set_title("UAV", y=1.02)
    axes[3].set_title("Panoptic Quality", y=1.02)
    # Apply class-specific rendering (second row only), using distinct colors and legends
    from matplotlib.lines import Line2D
    gr_col = ROW2_CLASS_COLORS.get("GREROB", CLASS_COLORS.get("GREROB", DEFAULT_CLASS_COLOR))
    di_col = ROW2_CLASS_COLORS.get("DIOSAN", CLASS_COLORS.get("DIOSAN", DEFAULT_CLASS_COLOR))

    # Col 2: BOTH classes (GREROB, DIOSAN) on WV/CHM diagonal
    draw_dual_class_pair_diagonal(axes[1], WV_TIF, CHM_TIF, pred, ref, extent, class_codes=["GREROB", "DIOSAN"])
    handles_wv = [
        Line2D([], [], color=gr_col, linewidth=LW_PRED_IMAGERY, linestyle='--', label='Predicted GREROB'),
        Line2D([], [], color=gr_col, linewidth=LW_REF_IMAGERY,  linestyle='-',  label='Reference GREROB'),
        Line2D([], [], color=di_col, linewidth=LW_PRED_IMAGERY, linestyle='--', label='Predicted DIOSAN'),
        Line2D([], [], color=di_col, linewidth=LW_REF_IMAGERY,  linestyle='-',  label='Reference DIOSAN'),
    ]
    legend_inside_top_right(axes[1], handles_wv)

    # Col 3: BOTH classes (GREROB, DIOSAN) on UAV raster
    draw_dual_class_pair_raster(axes[2], DRONE_TIF, pred, ref, extent, class_codes=["GREROB", "DIOSAN"])
    handles_uav = [
        Line2D([], [], color=gr_col, linewidth=LW_PRED_IMAGERY, linestyle='--', label='Predicted GREROB'),
        Line2D([], [], color=gr_col, linewidth=LW_REF_IMAGERY,  linestyle='-',  label='Reference GREROB'),
        Line2D([], [], color=di_col, linewidth=LW_PRED_IMAGERY, linestyle='--', label='Predicted DIOSAN'),
        Line2D([], [], color=di_col, linewidth=LW_REF_IMAGERY,  linestyle='-',  label='Reference DIOSAN'),
    ]
    legend_inside_top_right(axes[2], handles_uav)

    # Tighten spacing between columns only (keep visible row spacing)
    # Leave extra bottom space for a single, horizontal global legend
    bottom_m = 0.22
    plt.subplots_adjust(wspace=0.0, hspace=0.0,
                        left=left_m, right=right_m, top=top_m, bottom=bottom_m)

    # Build a single global legend across the bottom (horizontal)
    global_handles = []
    # 1) Class-specific colors (GREROB purple, DIOSAN gold)
    gr_col = ROW2_CLASS_COLORS.get("GREROB", CLASS_COLORS.get("GREROB", DEFAULT_CLASS_COLOR))
    di_col = ROW2_CLASS_COLORS.get("DIOSAN", CLASS_COLORS.get("DIOSAN", DEFAULT_CLASS_COLOR))
    cls_handles = [
        mlines.Line2D([], [], color=gr_col, linewidth=LW_PRED_IMAGERY, linestyle='--', label='Predicted GREROB'),
        mlines.Line2D([], [], color=gr_col, linewidth=LW_REF_IMAGERY,  linestyle='-',  label='Reference GREROB'),
        mlines.Line2D([], [], color=di_col, linewidth=LW_PRED_IMAGERY, linestyle='--', label='Predicted DIOSAN'),
        mlines.Line2D([], [], color=di_col, linewidth=LW_REF_IMAGERY,  linestyle='-',  label='Reference DIOSAN'),
    ]
    global_handles.extend(cls_handles)
    # 2) PQ items (TP, FP, FN) — placed last (far right)
    pq_handles = [
        mpatches.Patch(color=COLS_PQ["TP"], label="TP"),
        mpatches.Patch(color=COLS_PQ["FP"], label="FP"),
        mlines.Line2D([], [], color=FN_EDGE, linewidth=1.8, linestyle='-', label='FN'),
    ]
    global_handles.extend(pq_handles)

    # Deduplicate by label while keeping order
    seen = set(); unique_handles = []
    for h in global_handles:
        lab = h.get_label()
        if lab and lab not in seen:
            unique_handles.append(h)
            seen.add(lab)
    n_total = len(unique_handles)
    ncol = max(2, math.ceil(n_total / 2))
    fig.legend(handles=unique_handles, loc='lower center', ncol=ncol,
               frameon=True, facecolor='white', edgecolor='#444', fontsize=LEGEND_FONTSIZE,
               bbox_to_anchor=(0.5, 0.06))

    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {OUT_FIG}")

if __name__ == "__main__":
    main()
