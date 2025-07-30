import geopandas as gpd
import pandas as pd

from obia.classify import classify
from utils.utils import label_segments


regions_path = "regions.gpkg"
training_path = "trainings_filtered.gpkg"
# training_path = "training_lhs.gpkg"
# objects_path = "objects.gpkg"
# objects_path = "objects_pre_match.gpkg"
objects_path = "objects_post_match_spec.gpkg"

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
    # 'pai',
    # 'fhd',
    # 'ch',
    # 'mean_intensity',
    # 'variance_intensity',
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
    # 'pai',
    # 'fhd',
    # 'ch',
    # 'mean_intensity',
    # 'variance_intensity',
    'geometry'
]

objects = objects[columns_ob]

print("training", tr.shape)

classified_mlp = classify(
    objects, tr, acceptable_classes_gdf=regions, test_size=0.3, compute_reports=True,
    method='mlp', compute_shap=False, sample_shap=False,
    # hidden_layer_sizes=(200, 150, 100,),
    hidden_layer_sizes=(128, 64, 32),
    # learning_rate_init=0.001,
    # learning_rate='adaptive',
    # solver='lbfgs',
    solver='adam',
    max_iter=10000
)

# classified_mlp.classified.to_file(
#     "classified_new.gpkg"
# )

print(classified_mlp.confusion_matrix)
print(classified_mlp.report)


# import matplotlib.pyplot as plt
# import shap
#
#
# labels = {
#     "b0_mean": "Coastal Blue (mean)",
#     "b0_variance": "Coastal Blue (variance)",
#     "b0_skewness": "Coastal Blue (skewness)",
#     "b0_kurtosis": "Coastal Blue (kurtosis)",
#     "b0_min": "Coastal Blue (min)",
#     "b0_max": "Coastal Blue (max)",
#     #
#     "b1_mean": "Blue (mean)",
#     "b1_variance": "Blue (variance)",
#     "b1_skewness": "Blue (skewness)",
#     "b1_kurtosis": "Blue (kurtosis)",
#     "b1_min": "Blue (min)",
#     "b1_max": "Blue (max)",
#     #
#     "b2_mean": "Green (mean)",
#     "b2_variance": "Green (variance)",
#     "b2_skewness": "Green (skewness)",
#     "b2_kurtosis": "Green (kurtosis)",
#     "b2_min": "Green (min)",
#     "b2_max": "Green (max)",
#     #
#     "b3_mean": "Yellow (mean)",
#     "b3_variance": "Yellow (variance)",
#     "b3_skewness": "Yellow (skewness)",
#     "b3_kurtosis": "Yellow (kurtosis)",
#     "b3_min": "Yellow (min)",
#     "b3_max": "Yellow (max)",
#     #
#     "b4_mean": "Red (mean)",
#     "b4_variance": "Red (variance)",
#     "b4_skewness": "Red (skewness)",
#     "b4_kurtosis": "Red (kurtosis)",
#     "b4_min": "Red (min)",
#     "b4_max": "Red (max)",
#     #
#     "b5_mean": "Red Edge (mean)",
#     "b5_variance": "Red Edge (variance)",
#     "b5_skewness": "Red Edge (skewness)",
#     "b5_kurtosis": "Red Edge (kurtosis)",
#     "b5_min": "Red Edge (min)",
#     "b5_max": "Red Edge (max)",
#     #
#     "b6_mean": "NIR1 (mean)",
#     "b6_variance": "NIR1 (variance)",
#     "b6_skewness": "NIR1 (skewness)",
#     "b6_kurtosis": "NIR1 (kurtosis)",
#     "b6_min": "NIR1 (min)",
#     "b6_max": "NIR1 (max)",
#     #
#     "b7_mean": "NIR2 (mean)",
#     "b7_variance": "NIR2 (variance)",
#     "b7_skewness": "NIR2 (skewness)",
#     "b7_kurtosis": "NIR2 (kurtosis)",
#     "b7_min": "NIR2 (min)",
#     "b7_max": "NIR2 (max)",
#     #
#     "b0_contrast": "Coastal Blue (contrast)",
#     "b0_dissimilarity": "Coastal Blue (dissimilarity)",
#     "b0_homogeneity": "Coastal Blue (homogeneity)",
#     "b0_correlation": "Coastal Blue (correlation)",
#     #
#     "b1_contrast": "Blue (contrast)",
#     "b1_dissimilarity": "Blue (dissimilarity)",
#     "b1_homogeneity": "Blue (homogeneity)",
#     "b1_correlation": "Blue (correlation)",
#     #
#     "b2_contrast": "Green (contrast)",
#     "b2_dissimilarity": "Green (dissimilarity)",
#     "b2_homogeneity": "Green (homogeneity)",
#     "b2_correlation": "Green (correlation)",
#     #
#     "b3_contrast": "Yellow (contrast)",
#     "b3_dissimilarity": "Yellow (dissimilarity)",
#     "b3_homogeneity": "Yellow (homogeneity)",
#     "b3_correlation": "Yellow (correlation)",
#     #
#     "b4_contrast": "Red (contrast)",
#     "b4_dissimilarity": "Red (dissimilarity)",
#     "b4_homogeneity": "Red (homogeneity)",
#     "b4_correlation": "Red (correlation)",
#     #
#     "b5_contrast": "Red Edge (contrast)",
#     "b5_dissimilarity": "Red Edge (dissimilarity)",
#     "b5_homogeneity": "Red Edge (homogeneity)",
#     "b5_correlation": "Red Edge (correlation)",
#     #
#     "b6_contrast": "NIR1 (contrast)",
#     "b6_dissimilarity": "NIR1 (dissimilarity)",
#     "b6_homogeneity": "NIR1 (homogeneity)",
#     "b6_correlation": "NIR1 (correlation)",
#     #
#     "b7_contrast": "NIR2 (contrast)",
#     "b7_dissimilarity": "NIR2 (dissimilarity)",
#     "b7_homogeneity": "NIR2 (homogeneity)",
#     "b7_correlation": "NIR2 (correlation)",
#     "fhd": "Foliage Height Diversity",
#     "pai": "Plant Area Index",
#     "ch": "Canopy Height (max)",
#     "mean_intensity": "Intensity (mean)",
#     "variance_intensity": "Intensity (variance)",
#     "total_points": "Total Points",
# }
#
# species_labels = {
#     0: "METPOL",
#     1: "GREROB",
#     2: "ACAKOA",
#     3: "DIOSAN",
#     4: "PROPAL",
#     5: "OTHER",
# }
#
#
# vars = tr.drop(['feature_class', 'geometry', 'segment_id'], axis=1, errors='ignore')
# shap_values = classified_mlp.shap_values
# mapped_feature_names = [labels.get(col, col) for col in vars.columns]
# class_names = [species_labels[class_ind] for class_ind in range(shap_values.shape[-1])]
#
# import matplotlib.pyplot as plt
#
# # Data
# species = ["ōhi‘a", "silk oak", "koa", "lama", "kiawe", "other"]
# counts  = [166772, 26819, 56490, 16114, 44343, 22695]
#
# # Create the bar chart
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.bar(species, counts, width=0.6)
#
# # Labels and title (optional)
# ax.set_xlabel("Species")
# ax.set_ylabel("Count")
# ax.set_title("Tree Counts by Species")
#
# plt.tight_layout()
# plt.show()