import geopandas as gpd

from obia.classify import classify
from utils.utils import label_segments


def code_to_class(code):
    if code == "METPOL":
        return 1
    elif code == "GREROB":
        return 2
    elif code == "ACAKOA":
        return 3
    elif code == "DIOSAN":
        return 4
    else:
        return 6

regions_path = "regions.gpkg"
training_path = "training.gpkg"
objects_path = "objects.gpkg"

training_points = gpd.read_file(training_path)
objects = gpd.read_file(objects_path)
regions = gpd.read_file(regions_path).to_crs(6635)

training_points['class'] = training_points['code'].apply(code_to_class)
training_points = training_points[["class", "geometry"]]
training_points["feature_class"] = training_points["class"]

training, mixed = label_segments(objects, training_points)

regions['acceptable_classes'] = regions['acceptable_classes'].apply(
    lambda x: [int(cls) for cls in x.split(',')]
)


columns_tr = [
    'segment_id',
    'b2_mean',
    'b6_mean',
    'b7_mean',
    'b1_skewness',
    'b0_variance',
    'b3_max',
    'b5_max',
    'b0_min',
    'b0_kurtosis',
    'b0_energy',
    'b0_contrast',
    'b0_dissimilarity',
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
    'b2_mean',
    'b6_mean',
    'b7_mean',
    'b1_skewness',
    'b0_variance',
    'b3_max',
    'b5_max',
    'b0_min',
    'b0_kurtosis',
    'b0_energy',
    'b0_contrast',
    'b0_dissimilarity',
    'pai',
    'fhd',
    'ch',
    'mean_intensity',
    'variance_intensity',
    'geometry'
]

objects = objects[columns_ob]

print("training", tr.shape)

classified_mlp = classify(
    objects, tr, acceptable_classes_gdf=regions, test_size=0.3, compute_reports=True,
    method='mlp', compute_shap=False, sample_shap=True,
    # hidden_layer_sizes=(200, 150, 100,),
    hidden_layer_sizes=(256, 128,),
    # solver='lbfgs',
    solver='adam',
    max_iter=10000
)

classified_mlp.classified.to_file(
    "classified.gpkg"
)

print(classified_mlp.confusion_matrix)
print(classified_mlp.report)

