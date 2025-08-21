from pathlib import Path
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fiona

# =============================================================================
# CONFIG
# =============================================================================
ROOT_VALID = Path("IoU")                 # folder with validation1.gpkg … validation6.gpkg
PRED_PATH  = Path("classified_xgb.gpkg") # full-region predicted crowns with 'predicted_class'
TRAIN_PATH = Path("training.gpkg")      # points with field 'class'
FILES = range(1, 6+1)                    # validation plot indices you have

PRED_CLASS_FIELD   = "predicted_class"
VAL_CLASS_FIELD    = "val_class"
TRAIN_CLASS_FIELD  = "class"
HEIGHT_FIELD_NAME  = "ch"

IOU_THR = 0.5
DENSITY_BUFFER_M = 15.0
HEIGHT_SPLIT_M   = 10.0   # two height classes split here

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
    v = ensure_metric_crs(v)  # set/convert CRS
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
    sub["plot_id_pred"] = i                           # debug only
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

print(f"GLOBAL IoU (predicted ↔ reference): {glob_iou:.4f}")
print(f"Mean IoU per predicted polygon ± SD: {predicted['iou_best'].mean():.4f} ± {predicted['iou_best'].std(ddof=1):.4f}")

# =============================================================================
# 4) CLASS-AGNOSTIC MATCH COUNTS + PQ (overall & per-plot)
# =============================================================================
pairs_any = intersections.loc[intersections["iou"] >= IOU_THR, ["pid","vid","plot_id_val","iou"]]
pairs_any = greedy_match(pairs_any)

overall_any = compute_counts_and_pq(predicted.shape[0], validation.shape[0], pairs_any)
# print nicely
print("\nPANOPTIC (class-agnostic, overall):")
print(f"  TP={overall_any['TP']}  FP={overall_any['FP']}  FN={overall_any['FN']}")
print(f"  precision={overall_any['precision']:.3f}  recall={overall_any['recall']:.3f}")
print(f"  SQ={overall_any['SQ']:.3f}  RQ={overall_any['RQ']:.3f}  PQ={overall_any['PQ']:.3f}")

pd.DataFrame([{"metric":"class_agnostic", **overall_any, "global_iou":glob_iou,
               "mean_pred_poly_iou": predicted['iou_best'].mean(),
               "sd_pred_poly_iou":   predicted['iou_best'].std(ddof=1)}]) \
  .to_csv("summary_overall_class_agnostic.csv", index=False)

# per-plot using plot_id from VALIDATION side
per_plot_rows = []
print("\nPANOPTIC (class-agnostic) by plot:")
for pid_plot in sorted(validation["plot_id_val"].unique()):
    sub_pred = predicted.loc[predicted.intersects(envelopes[pid_plot])]
    sub_val  = validation[validation["plot_id_val"] == pid_plot]
    sub_int  = intersections[intersections["plot_id_val"] == pid_plot]

    sub_pairs = greedy_match(sub_int.loc[sub_int["iou"] >= IOU_THR, ["pid","vid","iou"]])
    stats = compute_counts_and_pq(sub_pred.shape[0], sub_val.shape[0], sub_pairs)
    stats.update({
        "plot_id": pid_plot,
        "mean_pred_poly_iou": sub_int.groupby("pid")["iou"].max().reindex(sub_pred["pid"], fill_value=0).mean(),
        "sd_pred_poly_iou":   sub_int.groupby("pid")["iou"].max().reindex(sub_pred["pid"], fill_value=0).std(ddof=1)
    })
    per_plot_rows.append(stats)
    print(f"  plot {pid_plot}: SQ={stats['SQ']:.3f}  RQ={stats['RQ']:.3f}  PQ={stats['PQ']:.3f} "
          f"(TP={stats['TP']}, FP={stats['FP']}, FN={stats['FN']})")

per_plot_df = pd.DataFrame(per_plot_rows)
per_plot_df.to_csv("summary_by_plot_class_agnostic.csv", index=False)

# =============================================================================
# 5) TWO HEIGHT CLASSES (predicted side) — if 'ch' exists
# =============================================================================
if HEIGHT_FIELD_NAME in predicted.columns:
    bins   = [-float("inf"), HEIGHT_SPLIT_M, float("inf")]
    labels = [f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"]
    predicted["height_class_2"] = pd.cut(predicted[HEIGHT_FIELD_NAME], bins=bins, labels=labels, right=True)
    hstats = (predicted.groupby("height_class_2", observed=False)["iou_best"]
                      .agg(n="count",
                           mean=lambda x: round(x.mean(), 4),
                           sd=lambda x: round(x.std(ddof=1), 4)))
    print("\nIoU by height class (2 bins, predicted polygons)")
    print(hstats.to_string())
    hstats.to_csv("iou_by_heightclass_2bins.csv")

    # Plot IoU by height class (save)
    fig, ax = plt.subplots(figsize=(4.8, 5))
    groups = [predicted.loc[predicted["height_class_2"] == lbl, "iou_best"].values for lbl in labels]
    positions = range(1, len(labels) + 1)

    vparts = ax.violinplot(groups, positions=positions, widths=0.8,
                           showmeans=False, showmedians=False, showextrema=False)
    for pc in vparts["bodies"]:
        pc.set_alpha(0.3)

    ax.boxplot(groups, positions=positions, widths=0.25, patch_artist=True,
               boxprops   =dict(alpha=0.5, linewidth=1.0),
               medianprops=dict(color="black", linewidth=1.4),
               whiskerprops=dict(linewidth=1.0),
               capprops    =dict(linewidth=1.0),
               showfliers=True)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Canopy-height class")
    ax.set_ylabel("IoU per predicted crown")
    ax.set_title(f"IoU by height class (split at {HEIGHT_SPLIT_M:.0f} m)", pad=10)
    plt.tight_layout()
    fig.savefig("iou_heightclass_2bins.png", dpi=300)
    plt.close(fig)

else:
    warnings.warn(f"Height field '{HEIGHT_FIELD_NAME}' not found on predicted crowns; skipping height stratification.")

# Overall IoU distribution plot
fig, ax = plt.subplots(figsize=(3.5, 5))
data = [predicted["iou_best"].values]
parts = ax.violinplot(data, positions=[1], widths=0.8, showmeans=False, showmedians=False, showextrema=False)
for pc in parts['bodies']:
    pc.set_alpha(0.3)
ax.boxplot(data, positions=[1], widths=0.25, patch_artist=True,
           boxprops=dict(alpha=0.5, linewidth=1.0), medianprops=dict(color="black", linewidth=1.4),
           whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=True)
ax.set_xticks([1]); ax.set_xticklabels(["Predicted vs reference"])
ax.set_ylabel("IoU per predicted polygon")
ax.set_title("IoU distribution", pad=10)
plt.tight_layout(); fig.savefig("iou_distribution.png", dpi=300); plt.close(fig)

# =============================================================================
# 6) TWO DENSITY CLASSES (validation crowns) — 15 m buffer, median split
# =============================================================================
buffer_area = np.pi * (DENSITY_BUFFER_M**2)
val_union_geom = validation.geometry.unary_union

def local_cover_fraction(geom):
    rp = geom.representative_point()
    buf = rp.buffer(DENSITY_BUFFER_M)
    return float(buf.intersection(val_union_geom).area) / buffer_area

validation["lcc_15m"] = validation.geometry.apply(local_cover_fraction)

thr_med = float(validation["lcc_15m"].dropna().median())
print(f"\n[Density split] Median local canopy density within {DENSITY_BUFFER_M:.0f} m = {thr_med:.3f}")

validation["density_class_2"] = pd.qcut(validation["lcc_15m"], q=2, labels=["low","high"])

dens_stats = (validation.groupby("density_class_2", observed=False)["iou_best_from_pred"]
                        .agg(n="count",
                             mean=lambda x: round(x.mean(), 4),
                             sd=lambda x: round(x.std(ddof=1), 4)))
print("\nIoU by local canopy density (2 bins; validation crowns, 15 m buffer)")
print(dens_stats.to_string())
dens_stats.to_csv("iou_by_densityclass_2bins.csv")

# Density plot (2 bins)
fig, ax = plt.subplots(figsize=(4.5, 5))
dens_labels = ["low", "high"]
groups = [validation.loc[validation["density_class_2"] == lbl, "iou_best_from_pred"].values for lbl in dens_labels]
v = ax.violinplot(groups, positions=range(1,3), widths=0.8, showmeans=False, showmedians=False, showextrema=False)
for pc in v["bodies"]:
    pc.set_alpha(0.3)
ax.boxplot(groups, positions=range(1,3), widths=0.25, patch_artist=True,
           boxprops=dict(alpha=0.5, linewidth=1.0), medianprops=dict(color="black", linewidth=1.4),
           whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=True)
ax.set_xticks([1,2]); ax.set_xticklabels(dens_labels)
ax.set_xlabel("Local canopy density (reference crown cover within 15 m)")
ax.set_ylabel("IoU per validation crown (best from predictions)")
ax.set_title("IoU by local canopy density (2 bins)", pad=10)
plt.tight_layout(); fig.savefig("iou_density_class_2bins.png", dpi=300); plt.close(fig)

# =============================================================================
# 7) LABEL VALIDATION CROWNS FROM TRAINING POINTS (drop conflicts)
# =============================================================================
HAVE_VAL_LABELS = False
if TRAIN_PATH.exists():
    lyr = pick_point_layer(TRAIN_PATH)
    if lyr is None:
        warnings.warn("No point layer found in trainings.gpkg; skipping semantic metrics.")
    else:
        train_pts = gpd.read_file(TRAIN_PATH, layer=lyr)
        if TRAIN_CLASS_FIELD not in train_pts.columns:
            warnings.warn(f"'{TRAIN_CLASS_FIELD}' not in trainings.gpkg; skipping semantic metrics.")
        else:
            train_pts = ensure_metric_crs(train_pts, target=validation.crs)
            joined = gpd.sjoin(train_pts[[TRAIN_CLASS_FIELD, "geometry"]],
                               validation[["vid","geometry"]], how="inner", predicate="intersects")
            lbls = (joined.groupby("vid")[TRAIN_CLASS_FIELD]
                          .agg(lambda s: list(sorted(pd.unique(s)))))
            lab = pd.DataFrame({"vid": lbls.index, "classes": lbls.values})
            lab["n_unique"] = lab["classes"].apply(len)
            lab = lab[lab["n_unique"] == 1].copy()
            lab[VAL_CLASS_FIELD] = lab["classes"].str[0]
            validation = validation.merge(lab[["vid", VAL_CLASS_FIELD]], on="vid", how="left")
            HAVE_VAL_LABELS = True
            print(f"\nValidation crowns labeled (unambiguous): {validation[VAL_CLASS_FIELD].notna().sum()} / {validation.shape[0]}")
else:
    warnings.warn("trainings.gpkg not found; skipping semantic metrics.")

# ---- After HAVE_VAL_LABELS is set, add: ----
if HAVE_VAL_LABELS:
    # ensure numeric class dtype
    validation[VAL_CLASS_FIELD] = pd.to_numeric(validation[VAL_CLASS_FIELD], errors="coerce")

if HAVE_PRED_LABELS:
    predicted[PRED_CLASS_FIELD] = pd.to_numeric(predicted[PRED_CLASS_FIELD], errors="coerce")

# Align class code systems if needed (0–5 vs 1–6)
if HAVE_VAL_LABELS and HAVE_PRED_LABELS:
    vals = set(validation[VAL_CLASS_FIELD].dropna().unique().tolist())
    preds = set(predicted[PRED_CLASS_FIELD].dropna().unique().tolist())

    # Heuristic: if preds are 0-based and vals are 1-based, shift preds by +1
    if preds and vals and min(preds) == 0 and min(vals) >= 1 and max(preds) + 1 == max(vals):
        predicted[PRED_CLASS_FIELD] = predicted[PRED_CLASS_FIELD] + 1

    # --- Align class code systems (validation 0–5, predicted 1–6) ---
    # cast to ints (avoid float codes)
    if HAVE_VAL_LABELS:
        validation[VAL_CLASS_FIELD] = validation[VAL_CLASS_FIELD].astype("Int64")
    if HAVE_PRED_LABELS:
        predicted[PRED_CLASS_FIELD] = predicted[PRED_CLASS_FIELD].astype("Int64")

    vals = set(validation[VAL_CLASS_FIELD].dropna().unique().tolist())
    preds = set(predicted[PRED_CLASS_FIELD].dropna().unique().tolist())

    # If predictions are 1–6 and validation is 0–5, shift predictions DOWN by 1
    if vals and preds and min(vals) == 0 and min(preds) == 1 and max(preds) - 1 == max(vals):
        predicted[PRED_CLASS_FIELD] = (predicted[PRED_CLASS_FIELD] - 1).astype("Int64")
        print("Aligned class codes: shifted predicted classes by -1 to match validation (0–5).")

    # Reprint to confirm
    print("  Validation classes (post-align):", sorted(set(validation[VAL_CLASS_FIELD].dropna().unique())))
    print("  Predicted  classes (post-align):", sorted(set(predicted[PRED_CLASS_FIELD].dropna().unique())))

# # =============================================================================
# # 8) SEMANTIC METRICS (per-class precision/recall/F1 and PQ)
# # =============================================================================
# if HAVE_VAL_LABELS and HAVE_PRED_LABELS:
#     pred_lab = predicted.dropna(subset=[PRED_CLASS_FIELD]).copy()
#     val_lab  = validation.dropna(subset=[VAL_CLASS_FIELD]).copy()
#
#     inter_lab = (intersections.merge(pred_lab[["pid", PRED_CLASS_FIELD]], on="pid", how="inner")
#                                .merge(val_lab[["vid", VAL_CLASS_FIELD]], on="vid", how="inner"))
#
#     classes = sorted(val_lab[VAL_CLASS_FIELD].unique().tolist())
#     rows = []
#     for c in classes:
#         preds_c = pred_lab.loc[pred_lab[PRED_CLASS_FIELD] == c, ["pid"]]
#         refs_c  = val_lab.loc[val_lab[VAL_CLASS_FIELD] == c, ["vid"]]
#
#         cand = inter_lab[(inter_lab["iou"] >= IOU_THR) &
#                          (inter_lab[PRED_CLASS_FIELD] == c) &
#                          (inter_lab[VAL_CLASS_FIELD]  == c)][["pid","vid","iou"]]
#         cand = greedy_match(cand)
#
#         TPc = len(cand)
#         FPc = preds_c.shape[0] - TPc
#         FNc = refs_c.shape[0]  - TPc
#         SQc = cand["iou"].mean() if TPc else 0.0
#         RQc = TPc / (TPc + 0.5*FPc + 0.5*FNc) if (TPc + FPc + FNc) else 0.0
#         PQc = SQc * RQc
#         prec = TPc / (TPc + FPc) if (TPc + FPc) else 0.0
#         rec  = TPc / (TPc + FNc) if (TPc + FNc) else 0.0
#         f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
#
#         rows.append({"class": c, "TP":TPc, "FP":FPc, "FN":FNc,
#                      "precision":prec, "recall":rec, "F1":f1,
#                      "SQ":SQc, "RQ":RQc, "PQ":PQc})
#
#     per_class_df = pd.DataFrame(rows).sort_values("PQ", ascending=False)
#     per_class_df.to_csv("semantic_metrics_per_class.csv", index=False)
#
#     macro = {
#         "macro_precision": per_class_df["precision"].mean() if not per_class_df.empty else np.nan,
#         "macro_recall":    per_class_df["recall"].mean()    if not per_class_df.empty else np.nan,
#         "macro_F1":        per_class_df["F1"].mean()        if not per_class_df.empty else np.nan,
#         "macro_SQ":        per_class_df["SQ"].mean()        if not per_class_df.empty else np.nan,
#         "macro_RQ":        per_class_df["RQ"].mean()        if not per_class_df.empty else np.nan,
#         "macro_PQ":        per_class_df["PQ"].mean()        if not per_class_df.empty else np.nan
#     }
#     pd.DataFrame([macro]).to_csv("semantic_metrics_macro.csv", index=False)
#     print("\nSaved per-class semantic PQ and macro metrics.")
# else:
#     print("\nSemantic metrics skipped (missing predicted or validation labels).")


# =============================================================================
# 8a) SEMANTIC METRICS — LABEL-AVAILABILITY MASK (IoU for label match = 0.30)
#      - Evaluates only within the union of labeled reference crowns
#      - FNs only among labeled refs; FPs only among predictions intersecting the mask
#      - Optional: filter predictions by classifier confidence before evaluation
# =============================================================================
SEM_IOU_THR     = 0.30        # IoU threshold for semantic (label + geometry) match
PRED_CONF_FIELD = None        # e.g., "pred_prob_max"; set to None to disable filtering
CONF_THR        = 0.60        # used only if PRED_CONF_FIELD is not None

if HAVE_VAL_LABELS and HAVE_PRED_LABELS:
    # 1) Subset reference crowns to those with labels (mask)
    val_lab = validation.dropna(subset=[VAL_CLASS_FIELD]).copy()
    if val_lab.empty:
        print("\n[Masked PQ @IoU>=0.30] No labeled reference crowns; skipping masked semantic metrics.")
    else:
        # Union of labeled reference crowns (mask geometry)
        mask_geom = val_lab.geometry.union_all()

        # 2) Subset predicted crowns: must have predicted class and intersect mask
        pred_lab = predicted.dropna(subset=[PRED_CLASS_FIELD]).copy()
        pred_lab = pred_lab.loc[pred_lab.intersects(mask_geom)].copy()

        # Optional: filter by classifier confidence if available
        conf_suffix = ""
        if (PRED_CONF_FIELD is not None) and (PRED_CONF_FIELD in pred_lab.columns):
            before = len(pred_lab)
            pred_lab = pred_lab.loc[pred_lab[PRED_CONF_FIELD] >= CONF_THR].copy()
            conf_suffix = f"_conf{int(CONF_THR*100)}"
            print(f"[Masked PQ @IoU>=0.30] Filtered predictions by {PRED_CONF_FIELD} ≥ {CONF_THR}: "
                  f"{before} → {len(pred_lab)}")
        elif PRED_CONF_FIELD is not None:
            print(f"[Masked PQ @IoU>=0.30] Warning: {PRED_CONF_FIELD} not found; skipping confidence filter.")

        # 3) Build intersections restricted to mask participants
        inter_masked = (intersections
                        .merge(pred_lab[["pid", PRED_CLASS_FIELD]], on="pid", how="inner")
                        .merge(val_lab[["vid", VAL_CLASS_FIELD]],  on="vid", how="inner"))

        # 4) Compute per-class PQ/SQ/RQ within the mask at SEM_IOU_THR
        classes = sorted(val_lab[VAL_CLASS_FIELD].dropna().unique().tolist())
        rows = []
        for c in classes:
            preds_c = pred_lab.loc[pred_lab[PRED_CLASS_FIELD] == c, ["pid"]]
            refs_c  = val_lab.loc[val_lab[VAL_CLASS_FIELD]  == c, ["vid"]]

            cand = inter_masked[(inter_masked["iou"] >= SEM_IOU_THR) &
                                (inter_masked[PRED_CLASS_FIELD] == c) &
                                (inter_masked[VAL_CLASS_FIELD]  == c)][["pid","vid","iou"]]
            cand = greedy_match(cand)

            TPc = len(cand)
            FPc = preds_c.shape[0] - TPc
            FNc = refs_c.shape[0]  - TPc

            SQc = cand["iou"].mean() if TPc else 0.0
            RQc = TPc / (TPc + 0.5*FPc + 0.5*FNc) if (TPc + FPc + FNc) else 0.0
            PQc = SQc * RQc
            prec = TPc / (TPc + FPc) if (TPc + FPc) else 0.0
            rec  = TPc / (TPc + FNc) if (TPc + FNc) else 0.0
            f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

            rows.append({"class": c, "TP":TPc, "FP":FPc, "FN":FNc,
                         "precision":prec, "recall":rec, "F1":f1,
                         "SQ":SQc, "RQ":RQc, "PQ":PQc})

        per_class_masked = pd.DataFrame(rows).sort_values("PQ", ascending=False)

        # 5) Macro (unweighted) and overall (TP-weighted SQ) within the mask
        TP = per_class_masked["TP"].sum()
        FP = per_class_masked["FP"].sum()
        FN = per_class_masked["FN"].sum()
        overall_SQ_weighted = (per_class_masked["SQ"] * per_class_masked["TP"]).sum() / TP if TP else 0.0
        overall_RQ = TP / (TP + 0.5*FP + 0.5*FN) if (TP + FP + FN) else 0.0
        overall_PQ = overall_SQ_weighted * overall_RQ

        macro_masked = {
            "macro_precision": per_class_masked["precision"].mean() if not per_class_masked.empty else np.nan,
            "macro_recall":    per_class_masked["recall"].mean()    if not per_class_masked.empty else np.nan,
            "macro_F1":        per_class_masked["F1"].mean()        if not per_class_masked.empty else np.nan,
            "macro_SQ":        per_class_masked["SQ"].mean()        if not per_class_masked.empty else np.nan,
            "macro_RQ":        per_class_masked["RQ"].mean()        if not per_class_masked.empty else np.nan,
            "macro_PQ":        per_class_masked["PQ"].mean()        if not per_class_masked.empty else np.nan,
            "overall_SQ_weighted": overall_SQ_weighted,
            "overall_RQ": overall_RQ,
            "overall_PQ": overall_PQ,
            "TP": TP, "FP": FP, "FN": FN,
            "semantic_iou_threshold": SEM_IOU_THR,
            "conf_field": PRED_CONF_FIELD if PRED_CONF_FIELD is not None else "",
            "conf_threshold": CONF_THR if PRED_CONF_FIELD is not None else ""
        }

        # 6) Save with descriptive suffix
        suffix = f"_MASKED_iou{int(SEM_IOU_THR*100)}{conf_suffix}"
        per_class_masked.to_csv(f"semantic_metrics_per_class{suffix}.csv", index=False)
        pd.DataFrame([macro_masked]).to_csv(f"semantic_metrics_macro{suffix}.csv", index=False)

        print(f"\n[Masked PQ @IoU>=0.30] Saved per-class and macro metrics to "
              f"'semantic_metrics_per_class{suffix}.csv' and 'semantic_metrics_macro{suffix}.csv'.")

else:
    print("\n[Masked PQ @IoU>=0.30] Semantic metrics skipped (missing predicted or validation labels).")

# =============================================================================
# 8) SEMANTIC PQ ON SET C (51 ha) — species-aware evaluation on separate area
# =============================================================================
SETC_VAL_PATH  = Path("/Users/iosefa/repos/sam/area51_labelled_validated_subset1.gpkg")  # hand-drawn crowns + labels
SETC_PRED_PATH = Path("/Users/iosefa/repos/sam/area51_predicted_subset1.gpkg")           # predicted crowns + labels
SETC_VAL_CLASS_FIELD  = "class"
SETC_PRED_CLASS_FIELD = "predicted_class"
SETC_IOU_THR = 0.50
SETC_OUT_PREFIX = "setC_semantic"

def pick_polygon_layer(path: Path):
    lyrnames = list_layers(path)
    if not lyrnames:
        return None
    # prefer a polygon layer if multiple are present
    for ly in lyrnames:
        with fiona.open(path, layer=ly) as src:
            if src.schema["geometry"] in ("Polygon", "MultiPolygon"):
                return ly
    return lyrnames[0]

def load_setC_gpkg(path: Path, class_field: str, target_crs=None) -> gpd.GeoDataFrame:
    layer = pick_polygon_layer(path)
    if layer is None:
        raise FileNotFoundError(f"No layers found in {path}")
    gdf = gpd.read_file(path, layer=layer)
    if class_field not in gdf.columns:
        raise KeyError(f"'{class_field}' not found in {path} (layer '{layer}')")
    gdf = ensure_metric_crs(gdf, target=target_crs)
    # keep only valid polygons with a class
    gdf = gdf.dropna(subset=[class_field]).copy()
    # force integer labels (avoid float dtype)
    gdf[class_field] = pd.to_numeric(gdf[class_field], errors="coerce").astype("Int64")
    gdf = gdf[gdf[class_field].notna()].copy()
    return gdf

try:
    # 1) load validation first to define CRS
    setc_val = load_setC_gpkg(SETC_VAL_PATH, SETC_VAL_CLASS_FIELD, target_crs=None)
    setc_val = ensure_metric_crs(setc_val)  # if needed, push to 6635
    setc_pred = load_setC_gpkg(SETC_PRED_PATH, SETC_PRED_CLASS_FIELD, target_crs=setc_val.crs)

    # 2) ids and class sanity
    setc_val = setc_val.reset_index(drop=True);  setc_val["vid"] = setc_val.index
    setc_pred = setc_pred.reset_index(drop=True); setc_pred["pid"] = setc_pred.index

    # Report class codes present
    vals = sorted(set(setc_val[SETC_VAL_CLASS_FIELD].dropna().unique().tolist()))
    preds = sorted(set(setc_pred[SETC_PRED_CLASS_FIELD].dropna().unique().tolist()))
    print("\n[Set C] class codes")
    print("  Validation:", vals)
    print("  Predicted :", preds)

    # 3) intersections + IoU
    setc_int = gpd.overlay(
        setc_pred[["pid", "geometry"]],
        setc_val[["vid", "geometry"]],
        how="intersection",
    )
    if setc_int.empty:
        raise RuntimeError("[Set C] No overlaps between predicted and validation crowns.")

    setc_pred["area_p"]  = setc_pred.area
    setc_val["area_v"]   = setc_val.area

    setc_int = (setc_int
                .merge(setc_pred[["pid", "area_p", SETC_PRED_CLASS_FIELD]], on="pid", how="left")
                .merge(setc_val[["vid", "area_v", SETC_VAL_CLASS_FIELD]],   on="vid", how="left"))
    setc_int["inter_area"] = setc_int.area
    setc_int["union_area"] = setc_int["area_p"] + setc_int["area_v"] - setc_int["inter_area"]
    setc_int["iou"]        = setc_int["inter_area"] / setc_int["union_area"]

    # 4) per-class PQ/SQ/RQ at IoU >= SETC_IOU_THR with class agreement
    classes = sorted(set(setc_val[SETC_VAL_CLASS_FIELD].unique()).intersection(
                     set(setc_pred[SETC_PRED_CLASS_FIELD].unique())))
    rows = []
    for c in classes:
        preds_c = setc_pred.loc[setc_pred[SETC_PRED_CLASS_FIELD] == c, ["pid"]]
        refs_c  = setc_val.loc[setc_val[SETC_VAL_CLASS_FIELD] == c,  ["vid"]]

        cand = setc_int[(setc_int["iou"] >= SETC_IOU_THR) &
                        (setc_int[SETC_PRED_CLASS_FIELD] == c) &
                        (setc_int[SETC_VAL_CLASS_FIELD]  == c)][["pid","vid","iou"]]
        cand = greedy_match(cand)

        TPc = len(cand)
        FPc = preds_c.shape[0] - TPc
        FNc = refs_c.shape[0]  - TPc

        SQc = cand["iou"].mean() if TPc else 0.0
        RQc = TPc / (TPc + 0.5*FPc + 0.5*FNc) if (TPc + FPc + FNc) else 0.0
        PQc = SQc * RQc

        prec = TPc / (TPc + FPc) if (TPc + FPc) else 0.0
        rec  = TPc / (TPc + FNc) if (TPc + FNc) else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

        rows.append({"class": c, "TP":TPc, "FP":FPc, "FN":FNc,
                     "precision":prec, "recall":rec, "F1":f1,
                     "SQ":SQc, "RQ":RQc, "PQ":PQc})

    setc_per_class = pd.DataFrame(rows).sort_values("PQ", ascending=False)
    setc_per_class.to_csv(f"{SETC_OUT_PREFIX}_metrics_per_class.csv", index=False)

    # 5) macro (unweighted) and overall (TP-weighted SQ) summaries
    TP = setc_per_class["TP"].sum()
    FP = setc_per_class["FP"].sum()
    FN = setc_per_class["FN"].sum()

    overall_SQ_weighted = (setc_per_class["SQ"] * setc_per_class["TP"]).sum() / TP if TP else 0.0
    overall_RQ = TP / (TP + 0.5*FP + 0.5*FN) if (TP + FP + FN) else 0.0
    overall_PQ = overall_SQ_weighted * overall_RQ

    macro = {
        "macro_precision": setc_per_class["precision"].mean() if not setc_per_class.empty else np.nan,
        "macro_recall":    setc_per_class["recall"].mean()    if not setc_per_class.empty else np.nan,
        "macro_F1":        setc_per_class["F1"].mean()        if not setc_per_class.empty else np.nan,
        "macro_SQ":        setc_per_class["SQ"].mean()        if not setc_per_class.empty else np.nan,
        "macro_RQ":        setc_per_class["RQ"].mean()        if not setc_per_class.empty else np.nan,
        "macro_PQ":        setc_per_class["PQ"].mean()        if not setc_per_class.empty else np.nan,
        "overall_SQ_weighted": overall_SQ_weighted,
        "overall_RQ": overall_RQ,
        "overall_PQ": overall_PQ,
        "TP": TP, "FP": FP, "FN": FN,
        "iou_threshold": SETC_IOU_THR,
    }
    pd.DataFrame([macro]).to_csv(f"{SETC_OUT_PREFIX}_metrics_macro.csv", index=False)

    # 6) nice printout
    print("\n[Set C] Semantic PQ @ IoU ≥ {:.2f}".format(SETC_IOU_THR))
    print("  Overall: SQ={:.3f}  RQ={:.3f}  PQ={:.3f} (TP={}, FP={}, FN={})"
          .format(overall_SQ_weighted, overall_RQ, overall_PQ, TP, FP, FN))
    print("\n[Set C] Per-class (sorted by PQ):")
    if not setc_per_class.empty:
        for _, r in setc_per_class.iterrows():
            print("  class {:>2}: PQ={:.3f}  SQ={:.3f}  RQ={:.3f}  |  P={:.3f} R={:.3f} F1={:.3f} "
                  "(TP={}, FP={}, FN={})"
                  .format(int(r['class']), r['PQ'], r['SQ'], r['RQ'],
                          r['precision'], r['recall'], r['F1'],
                          int(r['TP']), int(r['FP']), int(r['FN'])))
    else:
        print("  (no overlapping classes found)")

except Exception as e:
    warnings.warn(f"[Set C] Semantic PQ evaluation failed: {e}")