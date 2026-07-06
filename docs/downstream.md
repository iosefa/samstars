# Downstream Analysis

`samstars` writes individual crown polygons.
Those polygons can be used as crown-level units for later analysis, including species classification, biomass estimation, crown metrics, and change analysis.

The package does not assign species labels directly.
It supports downstream species classification by providing crown geometries that can be used for feature extraction, modeling, and map production.

## Crown Output

The main segmentation output is:

```text
out/crowns/crowns.gpkg
```

The GeoPackage layer is named `crowns`.
It contains one feature per predicted crown:

| Field | Description |
| --- | --- |
| `id` | Crown identifier within the output file. |
| `source` | Source tile or proposal record used to create the crown. |
| `label` | Internal instance label from polygon refinement. |
| `score` | Refinement confidence score when available. |
| `geometry` | Crown polygon in the output CRS. |

Keep these fields unchanged when adding downstream attributes.
If you combine outputs from multiple runs, add a run identifier before merging files so crown identifiers remain unambiguous.

## Species Classification

A common downstream workflow is:

1. Run `samstars.segment(...)` to produce `crowns.gpkg`.
2. Extract crown-level predictors from imagery, CHM, vegetation indices, texture layers, or other rasters.
3. Train or apply a species classifier outside `samstars`, for example with XGBoost, scikit-learn, or another modeling library.
4. Join predicted species labels and confidence scores back to the crown polygons.

Use a table with one row per crown and a stable key, usually `id` for one output file or `run_id` plus `id` for merged runs.
`samstars` does not provide species-classification or feature-extraction functions.

## Join Species Predictions

After classification, read the crown GeoPackage into a GeoPandas `GeoDataFrame` and join a prediction table to it.
The `.merge(...)` call is the standard Pandas table-join method available on GeoPandas data frames.

```python
from pathlib import Path

import geopandas as gpd
import pandas as pd

crowns_path = Path("/abs/path/segmentation_run/crowns/crowns.gpkg")
species_table = Path("/abs/path/species_predictions.csv")
out_path = Path("/abs/path/segmentation_run/crowns/crowns_with_species.gpkg")

crowns = gpd.read_file(crowns_path, layer="crowns")
species = pd.read_csv(species_table)

classified = crowns.merge(
    species[["id", "species", "species_score"]],
    on="id",
    how="left",
    validate="one_to_one",
)

classified.to_file(out_path, layer="crowns", driver="GPKG")
```

Recommended downstream fields:

| Field | Description |
| --- | --- |
| `species` | Predicted species name, code, or class label. |
| `species_score` | Classifier confidence or probability for the assigned class. |
| `species_model` | Optional classifier name or version. |
| `species_source` | Optional source of the species label, such as model, field label, or manual edit. |

## Feature Tables

Feature extraction is project-specific.
The crown polygons can be used with any raster or vector processing library that supports polygon summaries.

Typical crown-level predictors include:

- image-band means, medians, and standard deviations
- vegetation indices summarized inside each crown
- CHM height summaries, such as maximum, mean, and percentiles
- texture metrics computed from imagery or canopy-height rasters
- crown geometry metrics, such as area and perimeter

Store feature tables separately from the segmentation output when they are intermediate modeling data.
Write joined GeoPackages only for outputs intended for mapping, review, or downstream analysis.

## Quality Checks

Before using crown polygons for classification or other analyses, check:

- the output CRS matches the rasters used for feature extraction
- very small or low-confidence crowns are handled consistently
- training and prediction areas are kept separate during model assessment
- merged outputs have unique crown identifiers
- species labels are joined to the correct segmentation run
