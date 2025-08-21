#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean evaluation script:
- Class-agnostic segmentation metrics on 6 validation plots (Table 1 + Fig. 6)
- Height and density stratifications (Fig. 6c–d)
- Visual overlays for best/worst plots (Fig. 7)
- Semantic PQ on Set C (51 ha) with per-class & macro summaries

Outputs (saved to working dir):
  table1_segmentation_by_plot.csv
  summary_overall_class_agnostic.csv
  iou_by_heightclass_2bins.csv
  iou_by_densityclass_2bins.csv
  fig6_segmentation_metrics.png
  fig7_segmentation_overlays.png
  setC_semantic_metrics_per_class.csv
  setC_semantic_metrics_macro.csv
  setC_perclass_PQ_bar.png
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import fiona

# =============================================================================
# CONFIG
# =============================================================================
# Paths
ROOT_VALID = Path("IoU")                           # folder with validation1.gpkg … validation6.gpkg
PRED_PATH  = Path("classified_xgb.gpkg")           # full-region predicted crowns with 'predicted_class'
TRAIN_PATH = Path("training.gpkg")                 # (optional) points with field 'class' to label validation
SETC_VAL_PATH  = Path("/Users/iosefa/repos/sam/area51_labelled_validated_subset1.gpkg")
SETC_PRED_PATH = Path("/Users/iosefa/repos/sam/area51_predicted_subset1.gpkg")

# Layers/fields
FILES = range(1, 6+1)                              # validation plot indices you have
PRED_CLASS_FIELD   = "predicted_class"
VAL_CLASS_FIELD    = "val_class"
TRAIN_CLASS_FIELD  = "class"
HEIGHT_FIELD_NAME  = "ch"

# Metrics params
IOU_THR = 0.50
DENSITY_BUFFER_M = 10.0                            # match Methods text (10 m)
HEIGHT_SPLIT_M   = 10.0                            # two height classes split here

# Set C semantic PQ params
SETC_VAL_CLASS_FIELD  = "class"
SETC_PRED_CLASS_FIELD = "predicted_class"
SETC_IOU_THR = 0.50
SETC_OUT_PREFIX = "setC_semantic"

# Plot style
mpl.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

# =============================================================================
# Helpers
# =============================================================================
def list_layers(path: Path):
    try:
        return fiona.listlayers(path)
    except Exception:
        return []

def pick_point_layer(path: Path):
    layers = list_layers(path)
    for ly in layers:
        with fiona.open(path, layer=ly) as src:
            if src.schema["geometry"] in ("Point", "MultiPoint"):
                return ly
    return layers[0] if layers else None

def pick_polygon_layer(path: Path):
    lyrnames = list_layers(path)
    if not lyrnames:
        return None
    for ly in lyrnames:
        with fiona.open(path, layer=ly) as src:
            if src.schema["geometry"] in ("Polygon", "MultiPolygon"):
                return ly
    return lyrnames[0]

def ensure_metric_crs(gdf: gpd.GeoDataFrame, target=None):
    if target is None:
        if (not gdf.crs) or gdf.crs.is_geographic:
            return gdf.to_crs(6635)
        return gdf
    else:
        if (not gdf.crs) or (gdf.crs != target):
            return gdf.to_crs(target)
        return gdf

def greedy_match(df_pairs, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df_pairs.empty:
        return df_pairs
    out = df_pairs.sort_values(iou_col, ascending=False)
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out

def compute_counts_and_pq(n_pred, n_val, pairs_df, iou_col="iou"):
    TP = len(pairs_df)
    FP = n_pred - TP
    FN = n_val  - TP
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    SQ        = pairs_df[iou_col].mean() if TP else 0.0
    RQ        = TP / (TP + 0.5*FP + 0.5*FN) if (TP + FP + FN) else 0.0
    PQ        = SQ * RQ
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, SQ=SQ, RQ=RQ, PQ=PQ)

def load_setC_gpkg(path: Path, class_field: str, target_crs=None) -> gpd.GeoDataFrame:
    layer = pick_polygon_layer(path)
    if layer is None:
        raise FileNotFoundError(f"No layers found in {path}")
    gdf = gpd.read_file(path, layer=layer)
    if class_field not in gdf.columns:
        raise KeyError(f"'{class_field}' not found in {path} (layer '{layer}')")
    gdf = ensure_metric_crs(gdf, target=target_crs)
    gdf = gdf.dropna(subset=[class_field]).copy()
    gdf[class_field] = pd.to_numeric(gdf[class_field], errors="coerce").astype("Int64")
    gdf = gdf[gdf[class_field].notna()].copy()
    return gdf

# =============================================================================
# 1) LOAD VALIDATION (per-plot) + envelopes
# =============================================================================
val_parts, envelopes = [], {}
for i in FILES:
    vpath = ROOT_VALID / f"validation{i}.gpkg"
    layers = list_layers(vpath)
    layer  = f"validation{i}" if f"validation{i}" in layers else (layers[0] if layers else None)
    if layer is None:
        raise FileNotFoundError(f"No layers in {vpath}")
    v = gpd.read_file(vpath, layer=layer)
    v = ensure_metric_crs(v)  # set/convert CRS to EPSG:6635 if needed
    v = v.reset_index(drop=True)
    v["plot_id_val"] = i
    val_parts.append(v)
    envelopes[i] = v.geometry.unary_union.buffer(0)

validation = gpd.GeoDataFrame(pd.concat(val_parts, ignore_index=True),
                              crs=val_parts[0].crs)
validation = validation.reset_index(drop=True)
validation["vid"] = validation.index
print(f"Loaded validation: {validation.shape[0]} polygons across {len(FILES)} plots.")

# =============================================================================
# 2) LOAD PREDICTED FULL REGION, CLIP TO EACH PLOT ENVELOPE
# =============================================================================
layers = list_layers(PRED_PATH)
pred_layer = layers[0] if layers else None
predicted_full = gpd.read_file(PRED_PATH, layer=pred_layer)
predicted_full = ensure_metric_crs(predicted_full, target=validation.crs)

pred_parts = []
for i in FILES:
    env = envelopes[i]
    sub = predicted_full.loc[predicted_full.intersects(env)].copy()
    if sub.empty:
        warnings.warn(f"No predicted crowns overlap validation plot {i}.")
        continue
    sub["geometry"] = sub.geometry.intersection(env)  # clip
    sub["plot_id_pred"] = i
    pred_parts.append(sub)

if not pred_parts:
    raise RuntimeError("No predicted crowns intersect any validation plots.")
predicted = gpd.GeoDataFrame(pd.concat(pred_parts, ignore_index=True), crs=validation.crs)
predicted = predicted.reset_index(drop=True)
predicted["pid"] = predicted.index

HAVE_PRED_LABELS = PRED_CLASS_FIELD in predicted.columns
if not HAVE_PRED_LABELS:
    warnings.warn(f"'{PRED_CLASS_FIELD}' not found on predicted polygons; semantic metrics will be skipped.")

# =============================================================================
# 3) INTERSECTIONS & IoU (class-agnostic)
# =============================================================================
intersections = gpd.overlay(
    predicted[["pid", "geometry"]],
    validation[["vid", "plot_id_val", "geometry"]],
    how="intersection",
)

predicted["area_p"]  = predicted.area
validation["area_v"] = validation.area

intersections = (
    intersections
    .merge(predicted[["pid", "area_p"]],  on="pid", how="left")
    .merge(validation[["vid", "area_v"]], on="vid", how="left")
)
intersections["inter_area"] = intersections.area
intersections["union_area"] = intersections["area_p"] + intersections["area_v"] - intersections["inter_area"]
intersections["iou"]        = intersections["inter_area"] / intersections["union_area"]

# best IoU per predicted & per validation crown
predicted["iou_best"] = intersections.groupby("pid")["iou"].max().reindex(predicted["pid"], fill_value=0).values
validation["iou_best_from_pred"] = intersections.groupby("vid")["iou"].max().reindex(validation["vid"], fill_value=0).values

glob_iou = predicted.geometry.union_all().intersection(validation.geometry.union_all()).area / \
           predicted.geometry.union_all().union(validation.geometry.union_all()).area

print(f"\nGLOBAL IoU (predicted ↔ reference): {glob_iou:.4f}")
print(f"Mean IoU per predicted polygon ± SD: {predicted['iou_best'].mean():.4f} ± {predicted['iou_best'].std(ddof=1):.4f}")

# =============================================================================
# 4) CLASS-AGNOSTIC MATCH COUNTS + PQ (overall & per-plot)
# =============================================================================
pairs_any = intersections.loc[intersections["iou"] >= IOU_THR, ["pid","vid","plot_id_val","iou"]]
pairs_any = greedy_match(pairs_any)

overall_any = compute_counts_and_pq(predicted.shape[0], validation.shape[0], pairs_any)

print("\nPANOPTIC (class-agnostic, overall):")
print(f"  TP={overall_any['TP']}  FP={overall_any['FP']}  FN={overall_any['FN']}")
print(f"  precision={overall_any['precision']:.3f}  recall={overall_any['recall']:.3f}")
print(f"  SQ={overall_any['SQ']:.3f}  RQ={overall_any['RQ']:.3f}  PQ={overall_any['PQ']:.3f}")

pd.DataFrame([{"metric":"class_agnostic", **overall_any, "global_iou":glob_iou,
               "mean_pred_poly_iou": predicted['iou_best'].mean(),
               "sd_pred_poly_iou":   predicted['iou_best'].std(ddof=1)}]) \
  .to_csv("summary_overall_class_agnostic.csv", index=False)

# Per-plot using plot_id from VALIDATION side
per_plot_rows = []
print("\nPANOPTIC (class-agnostic) by plot:")
for pid_plot in sorted(validation["plot_id_val"].unique()):
    env = envelopes[pid_plot]
    sub_pred = predicted.loc[predicted.intersects(env)].copy()
    sub_val  = validation[validation["plot_id_val"] == pid_plot].copy()
    sub_int  = intersections[intersections["plot_id_val"] == pid_plot].copy()

    sub_pairs = greedy_match(sub_int.loc[sub_int["iou"] >= IOU_THR, ["pid","vid","iou"]])
    stats = compute_counts_and_pq(sub_pred.shape[0], sub_val.shape[0], sub_pairs)
    stats.update({
        "plot_id": pid_plot,
        "n_pred": sub_pred.shape[0],
        "n_ref": sub_val.shape[0],
        "mean_pred_poly_iou": sub_int.groupby("pid")["iou"].max().reindex(sub_pred["pid"], fill_value=0).mean(),
        "sd_pred_poly_iou":   sub_int.groupby("pid")["iou"].max().reindex(sub_pred["pid"], fill_value=0).std(ddof=1)
    })
    per_plot_rows.append(stats)
    print(f"  plot {pid_plot}: SQ={stats['SQ']:.3f}  RQ={stats['RQ']:.3f}  PQ={stats['PQ']:.3f} "
          f"(TP={stats['TP']}, FP={stats['FP']}, FN={stats['FN']})")

per_plot_df = pd.DataFrame(per_plot_rows).sort_values("plot_id")
per_plot_df.to_csv("table1_segmentation_by_plot.csv", index=False)

# =============================================================================
# 5) STRATIFICATIONS (Height + Density) + Overall IoU distribution
# =============================================================================
# Height
if HEIGHT_FIELD_NAME in predicted.columns:
    bins   = [-float("inf"), HEIGHT_SPLIT_M, float("inf")]
    labels_h = [f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"]
    predicted["height_class_2"] = pd.cut(predicted[HEIGHT_FIELD_NAME], bins=bins, labels=labels_h, right=True)
    hstats = (predicted.groupby("height_class_2", observed=False)["iou_best"]
                      .agg(n="count",
                           mean=lambda x: round(x.mean(), 4),
                           sd=lambda x: round(x.std(ddof=1), 4)))
    print("\nIoU by height class (2 bins, predicted polygons)")
    print(hstats.to_string())
    hstats.to_csv("iou_by_heightclass_2bins.csv")
else:
    labels_h = None
    warnings.warn(f"Height field '{HEIGHT_FIELD_NAME}' not found on predicted crowns; skipping height stratification.")

# Density (15→10 m buffer as per config)
buffer_area = np.pi * (DENSITY_BUFFER_M**2)
val_union_geom = validation.geometry.unary_union

def local_cover_fraction(geom):
    rp = geom.representative_point()
    buf = rp.buffer(DENSITY_BUFFER_M)
    return float(buf.intersection(val_union_geom).area) / buffer_area

validation["lcc_dm"] = validation.geometry.apply(local_cover_fraction)
validation["density_class_2"] = pd.qcut(validation["lcc_dm"], q=2, labels=["low","high"])

dens_stats = (validation.groupby("density_class_2", observed=False)["iou_best_from_pred"]
                        .agg(n="count",
                             mean=lambda x: round(x.mean(), 4),
                             sd=lambda x: round(x.std(ddof=1), 4)))
print("\nIoU by local canopy density (2 bins; validation crowns, "
      f"{int(DENSITY_BUFFER_M)} m buffer)")
print(dens_stats.to_string())
dens_stats.to_csv("iou_by_densityclass_2bins.csv")

# =============================================================================
# 6) FIGURE 6 — 4-panel metrics figure
# =============================================================================

# =============================================================================
# FIGURE 6 — Segmentation metrics (4 panels)
# (a) IoU distribution (all predicted crowns vs reference)
# (b) PQ, SQ, RQ by validation plot (grouped bars)
# (c) IoU by height (≤ / > HEIGHT_SPLIT_M)
# (d) IoU by local canopy density (low / high)
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

# guard rails for expected inputs
assert "iou_best" in predicted.columns, "predicted['iou_best'] is missing."
assert "plot_id_val" in validation.columns, "validation['plot_id_val'] is missing."
assert "iou_best_from_pred" in validation.columns, "validation['iou_best_from_pred'] is missing."
assert "density_class_2" in validation.columns, "validation['density_class_2'] is missing."
assert "plot_id" in per_plot_df.columns and {"PQ","SQ","RQ"}.issubset(per_plot_df.columns), \
    "per_plot_df must contain plot_id, PQ, SQ, RQ."

# prepare grouped bars (ensure plots sorted 1..N)
plot_ids = sorted(per_plot_df["plot_id"].unique())
bar_data = per_plot_df.set_index("plot_id").loc[plot_ids][["PQ","SQ","RQ"]]

# height groups (if present)
height_labels = []
height_groups = []
if "height_class_2" in predicted.columns:
    height_labels = list(predicted["height_class_2"].dropna().unique())
    # keep consistent ordering ≤… then >…
    height_labels = sorted(height_labels, key=lambda s: (str(s).startswith(">"), str(s)))
    for lbl in height_labels:
        height_groups.append(predicted.loc[predicted["height_class_2"] == lbl, "iou_best"].dropna().values)
else:
    # fallback: a single group with all IoUs
    height_labels = ["All"]
    height_groups = [predicted["iou_best"].dropna().values]

# density groups
dens_labels = ["low", "high"]
dens_groups = [
    validation.loc[validation["density_class_2"] == "low",  "iou_best_from_pred"].dropna().values,
    validation.loc[validation["density_class_2"] == "high", "iou_best_from_pred"].dropna().values
]

# build the figure
fig = plt.figure(figsize=(10, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
ax_a = fig.add_subplot(gs[0, 0])  # (a)
ax_b = fig.add_subplot(gs[0, 1])  # (b)
ax_c = fig.add_subplot(gs[1, 0])  # (c)
ax_d = fig.add_subplot(gs[1, 1])  # (d)

# (a) IoU distribution (violin + box)
vals_a = [predicted["iou_best"].dropna().values]
vp = ax_a.violinplot(vals_a, positions=[1], widths=0.8, showmeans=False, showmedians=False, showextrema=False)
for b in vp["bodies"]:
    b.set_alpha(0.25)
ax_a.boxplot(vals_a, positions=[1], widths=0.25, patch_artist=True,
             boxprops=dict(alpha=0.5, linewidth=1.0),
             medianprops=dict(color="black", linewidth=1.2),
             whiskerprops=dict(linewidth=1.0),
             capprops=dict(linewidth=1.0),
             showfliers=True)
ax_a.set_xticks([1]); ax_a.set_xticklabels(["Predicted vs reference"])
ax_a.set_ylabel("IoU per predicted crown")
ax_a.set_title("(a) IoU distribution")

# (b) PQ, SQ, RQ by plot (grouped bars)
x = np.arange(len(plot_ids))
w = 0.25
ax_b.bar(x - w, bar_data["PQ"].values, width=w, label="PQ")
ax_b.bar(x,      bar_data["SQ"].values, width=w, label="SQ")
ax_b.bar(x + w,  bar_data["RQ"].values, width=w, label="RQ")
ax_b.set_xticks(x); ax_b.set_xticklabels([str(i) for i in plot_ids])
ax_b.set_ylim(0, 1.0)
ax_b.set_xlabel("Validation plot")
ax_b.set_ylabel("Score")
ax_b.set_title("(b) PQ, SQ, RQ by plot")
ax_b.legend(frameon=False)

# (c) IoU by height (violin + box)
vp_c = ax_c.violinplot(height_groups, positions=range(1, len(height_groups)+1),
                       widths=0.8, showmeans=False, showmedians=False, showextrema=False)
for b in vp_c["bodies"]:
    b.set_alpha(0.25)
ax_c.boxplot(height_groups, positions=range(1, len(height_groups)+1), widths=0.25, patch_artist=True,
             boxprops=dict(alpha=0.5, linewidth=1.0),
             medianprops=dict(color="black", linewidth=1.2),
             whiskerprops=dict(linewidth=1.0),
             capprops=dict(linewidth=1.0),
             showfliers=True)
ax_c.set_xticks(range(1, len(height_groups)+1))
ax_c.set_xticklabels([str(l) for l in height_labels])
ax_c.set_xlabel(f"Canopy-height class (split at {int(HEIGHT_SPLIT_M)} m)")
ax_c.set_ylabel("IoU per predicted crown")
ax_c.set_title("(c) IoU by height")

# (d) IoU by local canopy density (violin + box)
vp_d = ax_d.violinplot(dens_groups, positions=[1, 2], widths=0.8,
                       showmeans=False, showmedians=False, showextrema=False)
for b in vp_d["bodies"]:
    b.set_alpha(0.25)
ax_d.boxplot(dens_groups, positions=[1, 2], widths=0.25, patch_artist=True,
             boxprops=dict(alpha=0.5, linewidth=1.0),
             medianprops=dict(color="black", linewidth=1.2),
             whiskerprops=dict(linewidth=1.0),
             capprops=dict(linewidth=1.0),
             showfliers=True)
ax_d.set_xticks([1, 2]); ax_d.set_xticklabels(dens_labels)
ax_d.set_xlabel("Local canopy density (15 m ref. cover)")
ax_d.set_ylabel("IoU per validation crown\n(best from predictions)")
ax_d.set_title("(d) IoU by local canopy density")

plt.savefig("fig6_segmentation_metrics.png", dpi=300)
plt.close()
print("Saved: fig6_segmentation_metrics.png")

# =========================
# Figure 7 – segmentation examples (clean GIS-style outlines + top scalebar)
# =========================
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------- PATHS (edit if needed) ----------
ROOT_VALID = Path("IoU")                      # contains validation1.gpkg, validation4.gpkg
VAL1_PATH  = ROOT_VALID / "validation1.gpkg"  # manual crowns, plot 1 (lower-PQ area)
VAL4_PATH  = ROOT_VALID / "validation4.gpkg"  # manual crowns, plot 4 (higher-PQ area)
PRED_PATH  = Path("classified_xgb.gpkg")      # predicted crowns (full region)
RAST1_PATH = Path("setA_plot1.tif")           # WV3 tile for plot 1 (use bands 5,4,3 as RGB)
RAST4_PATH = Path("setA_plot4.tif")           # WV3 tile for plot 4 (use bands 5,4,3 as RGB)
SUB4_PATH  = Path("plot4_subset.gpkg")        # polygon defining exact area to show in panel (b)

# ---------- SETTINGS ----------
BANDS_RGB = (5, 4, 3)        # (R,G,B) – 1-based band indices
LINEWIDTH = 2.2              # thick outline
DPI       = 300
FIGNAME   = "fig7_segmentation_examples.png"
SCALE_LEN_M = 50             # one global scalebar length (meters)

# ---------- helpers ----------
def list_layers(path: Path):
    try:
        import fiona
        return fiona.listlayers(path)
    except Exception:
        return []

def pick_polygon_layer(path: Path):
    import fiona
    layers = list_layers(path)
    if not layers:
        return None
    for ly in layers:
        with fiona.open(path, layer=ly) as src:
            if src.schema["geometry"] in ("Polygon", "MultiPolygon"):
                return ly
    return layers[0]

def read_polygons(path: Path, target_crs=None):
    lyr = pick_polygon_layer(path)
    if lyr is None:
        raise FileNotFoundError(f"No polygon layers in {path}")
    gdf = gpd.read_file(path, layer=lyr)
    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def percentile_stretch(img, p_low=2, p_high=98):
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        band = img[i].astype(np.float32)
        good = np.isfinite(band)
        lo, hi = np.percentile(band[good], [p_low, p_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(band), np.nanmax(band)
        band = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
        out[i] = band.astype(np.uint8)
    return np.transpose(out, (1, 2, 0))  # H×W×C

def read_rgb_window(raster_path: Path, bounds=None, bands=(5,4,3)):
    with rasterio.open(raster_path) as src:
        for b in bands:
            if b < 1 or b > src.count:
                raise ValueError(f"Band {b} out of range for {raster_path} (has {src.count} bands)")
        if bounds is None:
            data = src.read(list(bands))
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        else:
            w = from_bounds(*bounds, transform=src.transform)
            if w.width <= 0 or w.height <= 0:
                raise RuntimeError("Requested window has non-positive width/height.")
            data = src.read(list(bands), window=w)
            # window extent
            left   = src.transform * (w.col_off,          w.row_off)
            right  = src.transform * (w.col_off + w.width, w.row_off)
            bottom = src.transform * (w.col_off,          w.row_off + w.height)
            # extent = (minx, maxx, miny, maxy)
            extent = (left[0], right[0], bottom[1], left[1])
        img = percentile_stretch(data)
        return img, extent, src.crs

def match_panel_size(ax_ref, ax_target):
    """Make ax_target the same physical size as ax_ref (keeps target's position anchor)."""
    r = ax_ref.get_position()
    t = ax_target.get_position()
    new = [t.x0, t.y0 + (t.height - r.height)/2.0, r.width, r.height]
    # Avoid overflow to the right
    if new[0] + new[2] > 0.99:
        new[0] = 0.99 - new[2]
    ax_target.set_position(new)

# ---------- load vectors ----------
val1 = read_polygons(VAL1_PATH)
val4 = read_polygons(VAL4_PATH)
pred_full = read_polygons(PRED_PATH, target_crs=val1.crs)

# envelopes to subset predictions per plot
env1 = gpd.GeoSeries(val1.geometry.unary_union, crs=val1.crs).buffer(0).iloc[0]
env4 = gpd.GeoSeries(val4.geometry.unary_union, crs=val4.crs).buffer(0).iloc[0]
pred1 = pred_full.loc[pred_full.intersects(env1)].copy()
pred1["geometry"] = pred1.geometry.intersection(env1)
pred4 = pred_full.loc[pred_full.intersects(env4)].copy()
pred4["geometry"] = pred4.geometry.intersection(env4)

# clip plot 4 to provided subset polygon
subset4 = read_polygons(SUB4_PATH, target_crs=val4.crs)
subset4_union = subset4.geometry.unary_union
clip_bounds4 = subset4_union.bounds
val4_clip  = gpd.clip(val4,  subset4_union)
pred4_clip = gpd.clip(pred4, subset4_union)

# ---------- load rasters ----------
img1, ext1, crs1 = read_rgb_window(RAST1_PATH, bounds=None, bands=BANDS_RGB)
img4, ext4, crs4 = read_rgb_window(RAST4_PATH, bounds=clip_bounds4, bands=BANDS_RGB)

# reproject vectors to raster CRS if needed
if val1.crs != crs1:
    val1   = val1.to_crs(crs1)
    pred1  = pred1.to_crs(crs1)
if val4_clip.crs != crs4:
    val4_clip  = val4_clip.to_crs(crs4)
    pred4_clip = pred4_clip.to_crs(crs4)

# ---------- plot ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 6), dpi=DPI)
axA, axB = axs

# Panel (a) – Plot 1
axA.imshow(img1, extent=(ext1[0], ext1[1], ext1[2], ext1[3]), origin="upper", zorder=0)
val1.boundary.plot(ax=axA, color="#1f77b4", linewidth=LINEWIDTH, zorder=3)  # manual (blue)
pred1.boundary.plot(ax=axA, color="#d62728", linewidth=LINEWIDTH, zorder=4) # predicted (red)
axA.set_title("(a) Lower-PQ area (Plot 1)")
axA.set_xlim(ext1[0], ext1[1]); axA.set_ylim(ext1[2], ext1[3])
axA.set_xticks([]); axA.set_yticks([])

# Panel (b) – Plot 4 clipped to subset
axB.imshow(img4, extent=(ext4[0], ext4[1], ext4[2], ext4[3]), origin="upper", zorder=0)
val4_clip.boundary.plot(ax=axB, color="#1f77b4", linewidth=LINEWIDTH, zorder=3)
pred4_clip.boundary.plot(ax=axB, color="#d62728", linewidth=LINEWIDTH, zorder=4)
axB.set_title("(b) Higher-PQ area (Plot 4, subset)")
axB.set_xlim(ext4[0], ext4[1]); axB.set_ylim(ext4[2], ext4[3])
axB.set_xticks([]); axB.set_yticks([])

# Legend (top-center)
legend_handles = [
    Line2D([0], [0], color="#d62728", lw=LINEWIDTH, label="Predicted (this study)"),
    Line2D([0], [0], color="#1f77b4", lw=LINEWIDTH, label="Reference (manual)")
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2,
           frameon=True, framealpha=0.95, facecolor="white",
           borderpad=0.6, bbox_to_anchor=(0.5, 1.02))

# Ensure equal panel sizes on the page (no extent changes)
match_panel_size(axA, axB)

# ---- Global top scale bar (sized from panel A’s map scale) ----
# compute bar length in figure fraction based on panel A x-range
x0, x1 = axA.get_xlim()
xrange_m = (x1 - x0)
bar_width_fig = (SCALE_LEN_M / xrange_m) * axA.get_position().width
bar_height_fig = 0.012  # small fixed height
# place just under the legend
top_y = 0.93
bar_left = 0.5 - bar_width_fig/2.0
bar_ax = fig.add_axes([bar_left, top_y, bar_width_fig, bar_height_fig])
bar_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.7", lw=0.8, alpha=0.9))
bar_ax.plot([0.08, 0.92], [0.5, 0.5], color="black", lw=2.5, solid_capstyle="butt")
bar_ax.plot([0.08, 0.08], [0.32, 0.68], color="black", lw=2.0)
bar_ax.plot([0.92, 0.92], [0.32, 0.68], color="black", lw=2.0)
bar_ax.text(0.5, 0.72, f"{SCALE_LEN_M} m", ha="center", va="bottom", fontsize=10)
bar_ax.set_axis_off()

plt.tight_layout()
plt.subplots_adjust(top=0.88, wspace=0.03)  # space for legend + scalebar
plt.savefig(FIGNAME, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved Figure 7 → {FIGNAME}")
