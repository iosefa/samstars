#!/usr/bin/env python3
"""
Figure: Whole-extent predicted/reference + PQ + imagery (5 rows × 2 cols)

Columns:
  A (left)  : Class-agnostic (geometry only; IoU ≥ 0.50)
  B (right) : Semantic (geometry + class match; IoU ≥ 0.50)

Rows:
  1) Predicted crowns — left: no class (gray) | right: by class (color)
  2) Reference crowns — left: no class (gray) | right: by class (color)
  3) PQ illustration  — predicted fills (TP blue / FP orange), reference FN (dark red outline), matched refs (gray)
  4) WorldView-3 image (PQ_worldview.tif) — predicted (solid, thick) + reference (dashed, thick) outlines
  5) Drone image (PQ_drone.tif)          — predicted (solid, thick) + reference (dashed, thick) outlines
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -------------------
# Hard-coded inputs
# -------------------
EXTENT_GPKG   = Path("/Users/iosefa/repos/sam/PQ_extent.gpkg")
PRED_GPKG     = Path("/Users/iosefa/repos/sam/PQ_predicted.gpkg")
REF_GPKG      = Path("/Users/iosefa/repos/sam/PQ_reference.gpkg")
WV_TIF        = Path("/Users/iosefa/repos/sam/PQ_worldview.tif")
DRONE_TIF     = Path("/Users/iosefa/repos/sam/PQ_drone.tif")

PRED_CLASS    = "predicted_class"   # field in predicted
REF_CLASS     = "class"             # field in reference
OUT_FIG       = "figure5_full.png"

IOU_THR       = 0.50

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
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
DEFAULT_CLASS_COLOR = "#999999"

# Imagery outline thickness
LW_PRED_IMAGERY = 3.2
LW_REF_IMAGERY  = 3.2

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

def legend_inside_top_right(ax, handles):
    if not handles:
        return
    ax.legend(handles=handles,
              loc="upper right",
              bbox_to_anchor=(0.98, 0.98),
              bbox_transform=ax.transAxes,
              frameon=True, facecolor="white", framealpha=0.78,
              edgecolor="#444", borderaxespad=0.5,
              handlelength=1.4, handletextpad=0.6)

# --- raster helpers for imagery rows ---
def extent_from_transform(t: Affine, w: int, h: int):
    left = t.c
    right = t.c + t.a * w
    top = t.f
    bottom = t.f + t.e * h
    return (left, right, bottom, top)

def show_raster(ax, tif_path: Path):
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
            ax.imshow(np.clip(b, 0, 1), cmap="gray", extent=ext, origin="upper")
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
        present = _present_codes(pred[PRED_CLASS])
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
        present = _present_codes(ref[REF_CLASS])
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
    # legend inside
    patches = [
        mpatches.Patch(color=COLS_PQ["TP"], label="TP"),
        mpatches.Patch(color=COLS_PQ["FP"], label="FP"),
        mpatches.Patch(facecolor="none", edgecolor=FN_EDGE, linewidth=1.8, label="FN"),
    ]
    legend_inside_top_right(ax, patches)

# -------------------
# Imagery rows (ONLY MODIFICATION you requested)
# -------------------
def plot_imagery_row(ax_left, ax_right, tif_path, pred, ref, extent, title_left, title_right):
    # Read/plot imagery once per axis – reproject outlines to the raster CRS
    ax_left.set_axis_off()
    r_crs = show_raster(ax_left, tif_path)
    pred_r = pred if pred.crs == r_crs else pred.to_crs(r_crs)
    ref_r  = ref  if ref.crs  == r_crs else ref.to_crs(r_crs)
    extent_r = extent if extent.crs == r_crs else extent.to_crs(r_crs)

    # Left: neutral outlines, thick
    if not pred_r.empty:
        pred_r.boundary.plot(ax=ax_left, color="black", linewidth=LW_PRED_IMAGERY)
    if not ref_r.empty:
        ref_r.boundary.plot(ax=ax_left, color="black", linewidth=LW_REF_IMAGERY, linestyle="--")
    extent_r.boundary.plot(ax=ax_left, color="black", linewidth=1.0)
    ax_left.set_title(title_left)

    # Right: class-colored outlines, thick
    ax_right.set_axis_off()
    show_raster(ax_right, tif_path)
    if not pred_r.empty:
        for k, g in pred_r.groupby(PRED_CLASS):
            col = CLASS_COLORS.get(k, DEFAULT_CLASS_COLOR)
            g.boundary.plot(ax=ax_right, color=col, linewidth=LW_PRED_IMAGERY)
    if not ref_r.empty:
        for k, g in ref_r.groupby(REF_CLASS):
            col = CLASS_COLORS.get(k, DEFAULT_CLASS_COLOR)
            g.boundary.plot(ax=ax_right, color=col, linewidth=LW_REF_IMAGERY, linestyle="--")
    extent_r.boundary.plot(ax=ax_right, color="black", linewidth=1.0)
    ax_right.set_title(title_right)

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

    # Figure layout: 5 rows × 2 columns
    fig, axes = plt.subplots(5, 2, figsize=(14, 26), constrained_layout=True)

    # Row 1: Predicted (unchanged)
    plot_pred_row(axes[0, 0], axes[0, 1], pred, have_classes=True)

    # Row 2: Reference (unchanged)
    plot_ref_row(axes[1, 0], axes[1, 1], ref, have_classes=True)

    # Row 3: PQ panels (unchanged)
    plot_pq_panel(axes[2, 0], pred, ref, pred_cat_geo, ref_fn_geo,
                  "A) Class-agnostic PQ (geometry only)")
    plot_pq_panel(axes[2, 1], pred, ref, pred_cat_sem, ref_fn_sem,
                  "B) Semantic PQ (geometry + class)")

    # Row 4: WorldView-3 imagery with THICK outlines
    plot_imagery_row(axes[3, 0], axes[3, 1], WV_TIF, pred, ref, extent,
                     "WorldView-3 (outlines: predicted solid, reference dashed)",
                     "WorldView-3 (by class outlines)")

    # Row 5: Drone imagery with THICK outlines
    plot_imagery_row(axes[4, 0], axes[4, 1], DRONE_TIF, pred, ref, extent,
                     "Drone (outlines: predicted solid, reference dashed)",
                     "Drone (by class outlines)")

    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {OUT_FIG}")

if __name__ == "__main__":
    main()