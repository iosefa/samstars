# Training

Training creates a model bundle that can be reused by `samstars.segment(...)`.
The bundle contains the trained model files and the band settings used to prepare new imagery.

## Inputs

Provide imagery, a canopy height model, a first-return density raster, and crown polygons for a labeled training area.
The raster requirements and band choices are described in [Data Preparation](data_preparation.md).
Training also requires a base initialization checkpoint.

### Base Checkpoint

The current training implementation requires the SAM ViT-L checkpoint, [`sam_vit_l_0b3195.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth), available from the [official checkpoint list](https://github.com/facebookresearch/segment-anything#model-checkpoints).
Save it locally and pass its path as `sam_base_ckpt`.

## Prepare Crown Annotations

Delineate individual crown boundaries in GIS software and save them in a GeoPackage, with one feature per crown.
The layer should have a defined coordinate reference system and align with the training imagery.
Species labels and additional attribute fields are not required.

Use training areas with complete crown annotations.
During refinement training, pixels outside the supplied polygons are treated as background, so crop the training rasters to the annotated area.

For one training area, pass the GeoPackage path through `crowns`.
For tiled training, save the annotations for each tile as `crowns.gpkg` inside its tile directory.
Reserve separate areas for evaluating the trained model.

## Train From One Labeled Area

```python
from pathlib import Path

from samstars import train

model = train(
    img=Path("data/training/image.tif"),
    chm=Path("data/training/chm.tif"),
    den=Path("data/training/first_return_density.tif"),
    crowns=Path("data/training/crowns.gpkg"),
    sam_base_ckpt=Path("models/sam_vit_l_0b3195.pth"),
    out=Path("models/crowns"),
    rgb_bands=(1, 2, 3),
    storage="persistent",
    device="cpu",
)

print(model.root)
```

`train(...)` prepares the cost raster and seed points, builds training samples, and trains the proposal and refinement stages.
The example uses RGB imagery. For imagery with a near-infrared band, pass `nir_band` to include an NDVI cue in the cost surface.
Band indices are one-based; `rgb_bands` lists red, green, and blue in that order.

## Train From Labeled Tiles

```python
from pathlib import Path

from samstars import train

model = train(
    tiles_dir=Path("data/training_tiles"),
    sam_base_ckpt=Path("models/sam_vit_l_0b3195.pth"),
    out=Path("models/crowns"),
    rgb_bands=(1, 2, 3),
    storage="persistent",
    device="cpu",
)
```

Each subdirectory in `training_tiles` must use one of these layouts:

| Layout | Required files |
| --- | --- |
| Raw labeled tile | `img.tif`, `chm.tif`, `den.tif`, `crowns.gpkg` |
| Prepared labeled tile | `img.tif`, `chm.tif`, `cost.tif`, `seeds.gpkg`, `crowns.gpkg` |

Use the same layout for all tiles. Raw tiles are prepared automatically.
For prepared tiles, the three image channels and cost settings must already match the intended model inputs; training does not select their bands again.
When creating a bundle from prepared tiles, supply the original source `rgb_bands`, `nir_band`, and `texture_band` settings if the bundle will later be used to prepare source rasters.

## Training Controls

| Parameter | Purpose |
| --- | --- |
| `rgb_bands` | Source band indices in red, green, blue order; default `(1, 2, 3)`. |
| `nir_band` | Optional near-infrared band for the NDVI cost term. |
| `texture_band` | Optional band for image texture; otherwise use the mean RGB intensity. |
| `sam_epochs`, `sam_lr` | Proposal training duration and learning rate; defaults `15` and `5e-5`. |
| `stardist_epochs` | Refinement training duration; default `50`. |
| `chip_size`, `chip_stride` | Training window size and spacing in pixels; defaults `512` and `384`. |
| `val_fraction` | Fraction of refinement records used for validation; default `0.2`. |

The record split is not a spatially independent evaluation; use separate held-out areas for that purpose.
Seed controls are described in [Data Preparation](data_preparation.md#create-seed-points), and the full parameter list is in the [API reference](api/top_level.md#samstars.api.train).

## Model Bundle and Saved Outputs

The bundle contains `manifest.json`, `decoder.pth`, `config.json`, `weights.h5`, and, when available, `thresholds.json`.
Pass the bundle directory to `segment(model=...)`; see [Segmentation](segmentation.md).

With `storage="persistent"`, preparation, training samples, and proposal outputs are saved under `models/crowns/_training/`.
Use `storage="temp"` to retain only the finished bundle.
Use a new bundle directory for a new training experiment.

See [Examples](examples.md) for notebook walkthroughs and [Advanced Workflows](advanced_workflows.md) for inspecting or rerunning individual stages.
