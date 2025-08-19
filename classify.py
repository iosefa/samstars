import geopandas as gpd
import pandas as pd

from obia.classify import classify
from utils.utils import label_segments


regions_path = "regions.gpkg"
training_path = "trainings_filtered.gpkg"
objects_path = "objects.gpkg"

training_points = gpd.read_file(training_path)
training_points["class"] = (
    pd.to_numeric(training_points["class"], errors="coerce")
      .astype("Int64")
)
n_before = len(training_points)
training_points = training_points.dropna(subset=["class"])
print(f"Dropped {n_before - len(training_points)} rows with non-numeric classes")


objects = gpd.read_file(objects_path)
regions = gpd.read_file(regions_path).to_crs(6635)

def _parse_acceptables(cell):
    """
    Turn  "6,2,4,3,1"  →  [6, 2, 4, 3, 1]
    Handles NaNs, extra spaces, duplicates.
    """
    if pd.isna(cell) or cell == '':
        return []                         # no restriction
    return [int(tok) for tok in str(cell).split(',') if tok.strip().isdigit()]

regions['acceptable_classes'] = regions['acceptable_classes'].apply(_parse_acceptables)

# quick sanity-check
unique_codes = (
    regions['acceptable_classes']
    .explode()
    .dropna()
    .astype(int)
    .unique()
)
print(f"Unique acceptable class codes ({len(unique_codes)}):", unique_codes)


training_points = training_points[["class", "geometry"]]
training_points["feature_class"] = training_points["class"]

training, mixed = label_segments(objects, training_points)

print(f"Found {len(mixed)} mixed segments")
# regions['acceptable_classes'] = regions['acceptable_classes'].apply(
#     lambda x: [int(cls) for cls in x.split(',')]
# )


columns_tr = [
    'segment_id',
    'b0_mean',
    'b1_mean',
    'b2_mean',
    'b3_mean',
    'b4_mean',
    'b5_mean',
    'b6_mean',
    'b7_mean',
    'b0_min',
    'b1_min',
    'b2_min',
    'b3_min',
    'b4_min',
    'b5_min',
    'b6_min',
    'b7_min',
    'b0_max',
    'b1_max',
    'b2_max',
    'b3_max',
    'b4_max',
    'b5_max',
    'b6_max',
    'b7_max',
    # 'b0_kurtosis',
    'b0_energy',
    'b0_contrast',
    'b0_dissimilarity',
    'b0_ASM',
    'b0_correlation',
    'b0_homogeneity',
    # 'b0_skewness',
    # 'b0_variance',
    'pai',
    'fhd',
    'ch',
    'mean_intensity',
    'variance_intensity',
    'feature_class',
    'geometry'
]
tr = training[columns_tr].copy()

columns_ob = [
    'segment_id',
    'b0_mean',
    'b1_mean',
    'b2_mean',
    'b3_mean',
    'b4_mean',
    'b5_mean',
    'b6_mean',
    'b7_mean',
    'b0_min',
    'b1_min',
    'b2_min',
    'b3_min',
    'b4_min',
    'b5_min',
    'b6_min',
    'b7_min',
    'b0_max',
    'b1_max',
    'b2_max',
    'b3_max',
    'b4_max',
    'b5_max',
    'b6_max',
    'b7_max',
    # 'b0_kurtosis',
    'b0_energy',
    'b0_contrast',
    'b0_dissimilarity',
    'b0_ASM',
    'b0_correlation',
    'b0_homogeneity',
    # 'b0_skewness',
    # 'b0_variance',
    'pai',
    'fhd',
    'ch',
    'mean_intensity',
    'variance_intensity',
    'geometry'
]

objects = objects[columns_ob]

print("training", tr.shape)

print("\nRows per class BEFORE train/test split:")
print(tr["feature_class"].value_counts().sort_index().to_string())


classified_mlp = classify(
    objects, tr,
    acceptable_classes_gdf=regions,
    test_size=0.3,
    compute_reports=True,
    method='xgb',
    compute_shap=True,
    sample_shap=True
)

# classified_mlp.classified.to_file(
#     "classified_xgb.gpkg"
# )

print(classified_mlp.confusion_matrix)
print(classified_mlp.report)

import re

def parse_classification_report_text(report_text):
    """
    Parse sklearn.metrics.classification_report (string form) to a tidy DataFrame.
    Returns columns: class_name, precision, recall, f1-score, support
    Handles lines like:
      <label>   0.91  0.90  0.90   1234
      accuracy                        0.91  5678
      macro avg 0.90  0.91  0.91   5678
      weighted avg ...
    """
    lines = [ln.strip() for ln in report_text.splitlines() if ln.strip()]
    # Find header line
    header_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"\bprecision\b\s+\brecall\b\s+\bf1-score\b\s+\bsupport\b", ln):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header row in classification report text.")

    rows = []
    # Parse rows after header until we hit averages/accuracy; collect those too
    for ln in lines[header_idx+1:]:
        # tokens may be separated by variable whitespace; support label names with spaces by rsplit
        # Typical rows end with 4 numeric columns; capture them and the leading label
        m = re.match(r"(.+?)\s+([0-9.]+|[0-9]*\.[0-9]+)\s+([0-9.]+|[0-9]*\.[0-9]+)\s+([0-9.]+|[0-9]*\.[0-9]+)\s+([0-9]+)", ln)
        if m:
            label = m.group(1).strip()
            precision = float(m.group(2))
            recall    = float(m.group(3))
            f1        = float(m.group(4))
            support   = int(m.group(5))
            rows.append({
                "class_name": label,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support
            })
            continue

        # Handle the single-value "accuracy" line (e.g., "accuracy 0.91 5678")
        m_acc = re.match(r"(accuracy)\s+([0-9.]+)\s+([0-9]+)", ln)
        if m_acc:
            rows.append({
                "class_name": "accuracy",
                "precision": None,
                "recall": None,
                "f1-score": float(m_acc.group(2)),
                "support": int(m_acc.group(3))
            })
            continue

        # Handle macro avg / weighted avg lines (same 4-number pattern as classes)
        m_avg = re.match(r"(macro avg|weighted avg)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)", ln)
        if m_avg:
            rows.append({
                "class_name": m_avg.group(1),
                "precision": float(m_avg.group(2)),
                "recall": float(m_avg.group(3)),
                "f1-score": float(m_avg.group(4)),
                "support": int(m_avg.group(5))
            })
            continue

    if not rows:
        raise ValueError("No data rows parsed from classification report text.")

    return pd.DataFrame(rows)

# --- Save tidy per-class metrics CSV from text report ---
metrics_csv_path = "classification_metrics_by_class.csv"
try:
    df_report = parse_classification_report_text(classified_mlp.report)

    # Optional: map numeric codes to species tags for nicer figures
    label_map = {
        "0": "METPOL", "1": "GREROB", "2": "ACAKOA",
        "3": "DIOSAN", "4": "PROPAL", "5": "OTHER"
    }
    df_report["class_name"] = df_report["class_name"].map(lambda x: label_map.get(x, x))

    df_report.to_csv(metrics_csv_path, index=False)
    print(f"[OK] Saved per-class metrics to {metrics_csv_path}")
except Exception as e:
    print(f"[ERROR] Could not save classification metrics CSV: {e}")

# --- Save confusion matrix CSV (already working for you) ---
confusion_csv_path = "classification_confusion_matrix.csv"
try:
    if isinstance(classified_mlp.confusion_matrix, pd.DataFrame):
        classified_mlp.confusion_matrix.to_csv(confusion_csv_path, index=True)
    else:
        pd.DataFrame(classified_mlp.confusion_matrix).to_csv(confusion_csv_path, index=True)
    print(f"[OK] Saved confusion matrix to {confusion_csv_path}")
except Exception as e:
    print(f"[WARN] Could not save confusion matrix CSV: {e}")


labels = {
    "b0_mean": "Coastal Blue (mean)",
    "b0_variance": "Coastal Blue (variance)",
    "b0_skewness": "Coastal Blue (skewness)",
    "b0_kurtosis": "Coastal Blue (kurtosis)",
    "b0_min": "Coastal Blue (min)",
    "b0_max": "Coastal Blue (max)",
    #
    "b1_mean": "Blue (mean)",
    "b1_variance": "Blue (variance)",
    "b1_skewness": "Blue (skewness)",
    "b1_kurtosis": "Blue (kurtosis)",
    "b1_min": "Blue (min)",
    "b1_max": "Blue (max)",
    #
    "b2_mean": "Green (mean)",
    "b2_variance": "Green (variance)",
    "b2_skewness": "Green (skewness)",
    "b2_kurtosis": "Green (kurtosis)",
    "b2_min": "Green (min)",
    "b2_max": "Green (max)",
    #
    "b3_mean": "Yellow (mean)",
    "b3_variance": "Yellow (variance)",
    "b3_skewness": "Yellow (skewness)",
    "b3_kurtosis": "Yellow (kurtosis)",
    "b3_min": "Yellow (min)",
    "b3_max": "Yellow (max)",
    #
    "b4_mean": "Red (mean)",
    "b4_variance": "Red (variance)",
    "b4_skewness": "Red (skewness)",
    "b4_kurtosis": "Red (kurtosis)",
    "b4_min": "Red (min)",
    "b4_max": "Red (max)",
    #
    "b5_mean": "Red Edge (mean)",
    "b5_variance": "Red Edge (variance)",
    "b5_skewness": "Red Edge (skewness)",
    "b5_kurtosis": "Red Edge (kurtosis)",
    "b5_min": "Red Edge (min)",
    "b5_max": "Red Edge (max)",
    #
    "b6_mean": "NIR1 (mean)",
    "b6_variance": "NIR1 (variance)",
    "b6_skewness": "NIR1 (skewness)",
    "b6_kurtosis": "NIR1 (kurtosis)",
    "b6_min": "NIR1 (min)",
    "b6_max": "NIR1 (max)",
    #
    "b7_mean": "NIR2 (mean)",
    "b7_variance": "NIR2 (variance)",
    "b7_skewness": "NIR2 (skewness)",
    "b7_kurtosis": "NIR2 (kurtosis)",
    "b7_min": "NIR2 (min)",
    "b7_max": "NIR2 (max)",
    #
    "b0_contrast": "Coastal Blue (contrast)",
    "b0_dissimilarity": "Coastal Blue (dissimilarity)",
    "b0_homogeneity": "Coastal Blue (homogeneity)",
    "b0_correlation": "Coastal Blue (correlation)",
    #
    "b1_contrast": "Blue (contrast)",
    "b1_dissimilarity": "Blue (dissimilarity)",
    "b1_homogeneity": "Blue (homogeneity)",
    "b1_correlation": "Blue (correlation)",
    #
    "b2_contrast": "Green (contrast)",
    "b2_dissimilarity": "Green (dissimilarity)",
    "b2_homogeneity": "Green (homogeneity)",
    "b2_correlation": "Green (correlation)",
    #
    "b3_contrast": "Yellow (contrast)",
    "b3_dissimilarity": "Yellow (dissimilarity)",
    "b3_homogeneity": "Yellow (homogeneity)",
    "b3_correlation": "Yellow (correlation)",
    #
    "b4_contrast": "Red (contrast)",
    "b4_dissimilarity": "Red (dissimilarity)",
    "b4_homogeneity": "Red (homogeneity)",
    "b4_correlation": "Red (correlation)",
    #
    "b5_contrast": "Red Edge (contrast)",
    "b5_dissimilarity": "Red Edge (dissimilarity)",
    "b5_homogeneity": "Red Edge (homogeneity)",
    "b5_correlation": "Red Edge (correlation)",
    #
    "b6_contrast": "NIR1 (contrast)",
    "b6_dissimilarity": "NIR1 (dissimilarity)",
    "b6_homogeneity": "NIR1 (homogeneity)",
    "b6_correlation": "NIR1 (correlation)",
    #
    "b7_contrast": "NIR2 (contrast)",
    "b7_dissimilarity": "NIR2 (dissimilarity)",
    "b7_homogeneity": "NIR2 (homogeneity)",
    "b7_correlation": "NIR2 (correlation)",
    "fhd": "Foliage Height Diversity",
    "pai": "Plant Area Index",
    "ch": "Canopy Height (max)",
    "mean_intensity": "Intensity (mean)",
    "variance_intensity": "Intensity (variance)",
    "total_points": "Total Points",
}

species_labels = {
    0: "METPOL",
    1: "GREROB",
    2: "ACAKOA",
    3: "DIOSAN",
    4: "PROPAL",
    5: "OTHER",
}


vars = tr.drop(['feature_class', 'geometry', 'segment_id'], axis=1, errors='ignore')
shap_values = classified_mlp.shap_values
mapped_feature_names = [labels.get(col, col) for col in vars.columns]
class_names = [species_labels[class_ind] for class_ind in range(shap_values.shape[-1])]


# ================================
# SHAP roll-ups: sum vs mean-per-feature
# ================================
import numpy as np
import pandas as pd

# --- Expect these to exist from earlier steps:
# S (shap values; list-of-arrays or np.array), vars (X features df),
# mapped_feature_names (list of pretty names aligned to X.columns),
# class_names (list of class display names)

# Resolve SHAP to (N, F, C)
def to_3d_shap(S):
    if isinstance(S, list):
        return np.stack(S, axis=-1)  # list length = n_classes
    S = np.asarray(S)
    if S.ndim == 2:
        return S[:, :, None]         # binary case
    elif S.ndim == 3:
        return S
    raise ValueError(f"Unexpected SHAP dimensionality: {S.shape}")

S3 = to_3d_shap(classified_mlp.shap_values)  # (N, F, C)
X  = vars.copy()
pretty_names = pd.Series(mapped_feature_names, index=X.columns)

N, F, C = S3.shape
absS = np.abs(S3)

# ------------------------------------------------------------
# Build base tables: df_global (feature-level), df_pc (feature x class)
# ------------------------------------------------------------
global_mean_abs = absS.mean(axis=(0, 2))  # (F,)
global_pct = 100 * global_mean_abs / global_mean_abs.sum()

df_global = (
    pd.DataFrame({
        "feature": X.columns,
        "pretty_name": pretty_names.values,
        "mean_abs_shap": global_mean_abs,
        "percent_contrib": global_pct
    })
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
df_global.to_csv("shap_global_importance.csv", index=False)

# Per-class mean |SHAP|
rows = []
for ci in range(C):
    per_class_mean_abs = absS[:, :, ci].mean(axis=0)  # (F,)
    for fi, val in enumerate(per_class_mean_abs):
        rows.append({
            "class_index": ci,
            "class_name": class_names[ci] if ci < len(class_names) else f"class_{ci}",
            "feature": X.columns[fi],
            "pretty_name": pretty_names.iloc[fi],
            "mean_abs_shap": float(val)
        })
df_pc = pd.DataFrame(rows)
df_pc["rank_in_class"] = df_pc.groupby("class_index")["mean_abs_shap"].rank(ascending=False, method="first")
df_pc = df_pc.sort_values(["class_index", "rank_in_class"])
df_pc.to_csv("shap_per_class_importance.csv", index=False)

print("\n[SHAP] Global mean |SHAP| (top 20 features):")
print(df_global.head(20)[["pretty_name","mean_abs_shap","percent_contrib"]].to_string(index=False))

print("\n[SHAP] Per-class top 10 features:")
for ci in range(C):
    sub = df_pc[df_pc["class_index"] == ci].nsmallest(10, "rank_in_class")
    cname = class_names[ci] if ci < len(class_names) else f"class_{ci}"
    print(f"\n  Class {ci} ({cname}):")
    print(sub[["rank_in_class","pretty_name","mean_abs_shap"]].rename(columns={"rank_in_class":"rank"}).to_string(index=False))

# ------------------------------------------------------------
# Grouping helpers (bands & modalities)
# ------------------------------------------------------------
def feature_group(col: str) -> str:
    # Bands
    if col.startswith("b0_"): return "Coastal Blue"
    if col.startswith("b1_"): return "Blue"
    if col.startswith("b2_"): return "Green"
    if col.startswith("b3_"): return "Yellow"
    if col.startswith("b4_"): return "Red"
    if col.startswith("b5_"): return "Red Edge"
    if col.startswith("b6_"): return "NIR1"
    if col.startswith("b7_"): return "NIR2"
    # Texture (GLCM on b0)
    if col in {"b0_energy","b0_contrast","b0_dissimilarity","b0_ASM","b0_correlation","b0_homogeneity"}:
        return "Texture (Coastal Blue GLCM)"
    # LiDAR structure
    if col in {"pai","fhd","ch"}:
        return "LiDAR structure"
    # LiDAR intensity
    if col in {"mean_intensity","variance_intensity"}:
        return "LiDAR intensity"
    return "Other"

def feature_modality(group: str) -> str:
    if group in {"Coastal Blue","Blue","Green","Yellow","Red","Red Edge","NIR1","NIR2"}:
        return "Spectral (WV3)"
    if group == "Texture (Coastal Blue GLCM)":
        return "Texture"
    if group == "LiDAR structure":
        return "LiDAR structure"
    if group == "LiDAR intensity":
        return "LiDAR intensity"
    return "Other"

group_labels = pd.Series({col: feature_group(col) for col in X.columns})
modality_labels = group_labels.map(feature_modality)

df_global["group"]   = df_global["feature"].map(group_labels)
df_global["modality"]= df_global["group"].map(feature_modality)

df_pc["group"]    = df_pc["feature"].map(group_labels)
df_pc["modality"] = df_pc["group"].map(feature_modality)

# ------------------------------------------------------------
# GLOBAL roll-ups
#   - SUM: total contribution (favours large groups)
#   - MEAN: average per feature (size-normalized; fair for comparison)
# ------------------------------------------------------------
# Bands
band_global_sum = (
    df_global.groupby("group", as_index=False)
    .agg(mean_abs_shap_sum=("mean_abs_shap","sum"),
         n_features=("mean_abs_shap","size"))
    .sort_values("mean_abs_shap_sum", ascending=False)
)
band_global_sum["percent_contrib_sum"] = 100 * band_global_sum["mean_abs_shap_sum"] / band_global_sum["mean_abs_shap_sum"].sum()
band_global_sum.to_csv("shap_band_global_SUM.csv", index=False)

band_global_mean = (
    df_global.groupby("group", as_index=False)
    .agg(mean_abs_shap_mean=("mean_abs_shap","mean"),
         n_features=("mean_abs_shap","size"))
    .sort_values("mean_abs_shap_mean", ascending=False)
)
band_global_mean.to_csv("shap_band_global_MEAN.csv", index=False)

print("\n[SHAP] Global roll-up by band (SUM of |SHAP|):")
print(band_global_sum.head(10).to_string(index=False))

print("\n[SHAP] Global roll-up by band (MEAN |SHAP| per feature):")
print(band_global_mean.head(10).to_string(index=False))

# Modalities
modality_global_sum = (
    df_global.groupby("modality", as_index=False)
    .agg(mean_abs_shap_sum=("mean_abs_shap","sum"),
         n_features=("mean_abs_shap","size"))
    .sort_values("mean_abs_shap_sum", ascending=False)
)
modality_global_sum["percent_contrib_sum"] = 100 * modality_global_sum["mean_abs_shap_sum"] / modality_global_sum["mean_abs_shap_sum"].sum()
modality_global_sum.to_csv("shap_modality_global_SUM.csv", index=False)

modality_global_mean = (
    df_global.groupby("modality", as_index=False)
    .agg(mean_abs_shap_mean=("mean_abs_shap","mean"),
         n_features=("mean_abs_shap","size"))
    .sort_values("mean_abs_shap_mean", ascending=False)
)
modality_global_mean.to_csv("shap_modality_global_MEAN.csv", index=False)

print("\n[SHAP] Global roll-up by modality (SUM of |SHAP|):")
print(modality_global_sum.to_string(index=False))

print("\n[SHAP] Global roll-up by modality (MEAN |SHAP| per feature):")
print(modality_global_mean.to_string(index=False))

# ------------------------------------------------------------
# PER-CLASS roll-ups (sum & mean)
# ------------------------------------------------------------
# Bands per class
band_pc_sum = (
    df_pc.groupby(["class_index","class_name","group"], as_index=False)
    .agg(mean_abs_shap_sum=("mean_abs_shap","sum"),
         n_features=("mean_abs_shap","size"))
    .sort_values(["class_index","mean_abs_shap_sum"], ascending=[True, False])
)
band_pc_sum.to_csv("shap_band_per_class_SUM.csv", index=False)

band_pc_mean = (
    df_pc.groupby(["class_index","class_name","group"], as_index=False)
    .agg(mean_abs_shap_mean=("mean_abs_shap","mean"),
         n_features=("mean_abs_shap","size"))
    .sort_values(["class_index","mean_abs_shap_mean"], ascending=[True, False])
)
band_pc_mean.to_csv("shap_band_per_class_MEAN.csv", index=False)

# Modalities per class
modality_pc_sum = (
    df_pc.groupby(["class_index","class_name","modality"], as_index=False)
    .agg(mean_abs_shap_sum=("mean_abs_shap","sum"),
         n_features=("mean_abs_shap","size"))
    .sort_values(["class_index","mean_abs_shap_sum"], ascending=[True, False])
)
modality_pc_sum.to_csv("shap_modality_per_class_SUM.csv", index=False)

modality_pc_mean = (
    df_pc.groupby(["class_index","class_name","modality"], as_index=False)
    .agg(mean_abs_shap_mean=("mean_abs_shap","mean"),
         n_features=("mean_abs_shap","size"))
    .sort_values(["class_index","mean_abs_shap_mean"], ascending=[True, False])
)
modality_pc_mean.to_csv("shap_modality_per_class_MEAN.csv", index=False)

print("\n[SHAP] Saved roll-ups:")
print("  - shap_band_global_SUM.csv / MEAN.csv")
print("  - shap_modality_global_SUM.csv / MEAN.csv")
print("  - shap_band_per_class_SUM.csv / MEAN.csv")
print("  - shap_modality_per_class_SUM.csv / MEAN.csv")


# ================================
# SHAP: persist plotting-ready artifacts
# ================================
import json
from pathlib import Path

OUTDIR = Path(".")
OUTDIR.mkdir(parents=True, exist_ok=True)

# 1) Deterministic sample of rows for beeswarm-style plots
#    (keeps file sizes reasonable and guarantees features ↔ SHAP alignment)
rng = np.random.default_rng(42)
N_SAMP = min(1000, S3.shape[0])  # adjust if you want fewer rows
sample_idx = np.sort(rng.choice(S3.shape[0], size=N_SAMP, replace=False))

S3_sample = S3[sample_idx, :, :]           # (Ns, F, C)
X_sample  = X.iloc[sample_idx].copy()      # (Ns, F)
X_sample.index = range(len(X_sample))      # reset index for clean CSV

# 2) Save the sampled SHAP values + matching features
np.savez_compressed("shap_values_sampled.npz", S=S3_sample)           # (Ns, F, C)
X_sample.to_csv("shap_features_sampled.csv", index=False)             # (Ns, F)
pd.DataFrame({"row_index": sample_idx}).to_csv("shap_sample_indices.csv", index=False)

# 3) Save metadata: feature names (raw & pretty) and class names
#    - JSON for lists; CSV for mappings (human-readable)
feature_names_raw   = X.columns.tolist()
feature_names_pretty= pretty_names.reindex(X.columns).tolist()
class_names_out     = [class_names[i] if i < len(class_names) else f"class_{i}" for i in range(S3.shape[-1])]

with open("shap_feature_names.json", "w") as f:
    json.dump({"raw": feature_names_raw, "pretty": feature_names_pretty}, f, indent=2)

with open("shap_class_names.json", "w") as f:
    json.dump({"class_names": class_names_out}, f, indent=2)

pd.DataFrame({"feature": feature_names_raw, "pretty_name": feature_names_pretty}) \
  .to_csv("shap_feature_name_map.csv", index=False)

# 4) Save compact roll-ups for plotting scripts (already computed above, but export in tidy form)
#    4a) Global per-feature importance (for top-20 bars, etc.)
df_global.to_csv("shap_global_per_feature.csv", index=False)

#    4b) Per-class per-feature importance (for optional per-class panels)
df_pc.to_csv("shap_per_class_per_feature.csv", index=False)

#    4c) Band & modality roll-ups — both SUM and MEAN versions
band_global_sum.to_csv("shap_rollup_band_SUM.csv", index=False)
band_global_mean.to_csv("shap_rollup_band_MEAN.csv", index=False)
modality_global_sum.to_csv("shap_rollup_modality_SUM.csv", index=False)
modality_global_mean.to_csv("shap_rollup_modality_MEAN.csv", index=False)

band_pc_sum.to_csv("shap_rollup_band_per_class_SUM.csv", index=False)
band_pc_mean.to_csv("shap_rollup_band_per_class_MEAN.csv", index=False)
modality_pc_sum.to_csv("shap_rollup_modality_per_class_SUM.csv", index=False)
modality_pc_mean.to_csv("shap_rollup_modality_per_class_MEAN.csv", index=False)

# 5) Quick console ping so you know what to point your plotting script at
print("\n=== SHAP artifacts saved for plotting ===")
print("  - shap_values_sampled.npz (S: (Ns, F, C))")
print("  - shap_features_sampled.csv (Ns × F)  [columns match SHAP features]")
print("  - shap_sample_indices.csv (row indices from full X/S used in sample)")
print("  - shap_feature_names.json (raw & pretty lists)")
print("  - shap_class_names.json")
print("  - shap_feature_name_map.csv (raw ↔ pretty)")
print("  - shap_global_per_feature.csv")
print("  - shap_per_class_per_feature.csv")
print("  - shap_rollup_band_SUM.csv / MEAN.csv")
print("  - shap_rollup_modality_SUM.csv / MEAN.csv")
print("  - shap_rollup_band_per_class_SUM.csv / MEAN.csv")
print("  - shap_rollup_modality_per_class_SUM.csv / MEAN.csv")