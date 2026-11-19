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
| `fg_extra` | Maximum extra foreground points per seed; nonnegative integer, default `80`. Set `0` to use only the seed as foreground. |
| `stardist_epochs` | Refinement training duration; default `50`. |
| `sam_window_size_m` | SAM ground-window width and height in metres. Saved for prediction. |
| `chip_size` | Alternative to `sam_window_size_m`, in native pixels; default `512` when neither is supplied. |
| `chip_stride` | SAM window spacing in native pixels; default 75% of the window size (25% overlap). |
| `stardist_patch_size` | Square StarDist training patch in native pixels; positive multiple of 16, default `256`. |
| `fast` | Batch SAM prompts within each fixed window; default `True`. Set `False` to reduce batch memory. |
| `val_fraction` | Fraction of refinement records used for validation; default `0.2`. |

The record split is not a spatially independent evaluation; use separate held-out areas for that purpose.
Seed controls are described in [Data Preparation](data_preparation.md#create-seed-points), and the full parameter list is in the [API reference](api/top_level.md#samstars.api.train).

## Window Size, Resolution, and Padding

Choose the ground area SAM should see using `sam_window_size_m`. For example,
`sam_window_size_m=20` at 0.5 m resolution creates a 40 × 40 native-pixel window.
SAM resizes that window to its fixed 1024 × 1024 encoder input; resizing does not
add new spatial detail. Alternatively, specify `chip_size` in native pixels.
Supplying both controls is an error. Metre sizes are rounded to the nearest pixel;
the bundle records the actual size used.

New bundles save the native pixel size, SAM window and stride, encoder size,
padding policy, and RGB scaling in `manifest.json` under `preprocessing`.
Prediction automatically reuses those settings. Larger scenes are processed in
overlapping windows and blended back onto their original raster grid. For these
bundles, `fast=True` batches prompts within a window and `fast=False` decodes one
prompt at a time; both use the same ground context and image scaling.

All training imagery must have the same resolution, a projected CRS in metres,
and north-up square pixels. Prediction checks the saved resolution and asks you
to align/resample incompatible inputs. It does not silently resample them.

StarDist's patch size is independent of the SAM window. Its current architecture
requires a multiple of 16 native pixels. Every annotated training and validation
image must be at least that wide and high. This is checked before SAM training,
so a tiny input does not fail only after the decoder has finished training.
For example, a 20 m square at 0.5 m resolution is 40 × 40 pixels: use
`sam_window_size_m=20, stardist_patch_size=32`. The default StarDist patch of 256
pixels requires at least 128 m in each dimension at that resolution.

```python
# Add these to train(...) for a small-window experiment on 0.5 m imagery:
window_settings = dict(
    sam_window_size_m=20,
    stardist_patch_size=32,
    fg_extra=0,
)
# model = train(..., **window_settings)
```

The SAM window does not impose a minimum scene size. Smaller scenes are padded
on the bottom and right with zeros; training warns when this is necessary.
Padding and NoData are excluded from RGB percentile scaling, foreground/background
prompt sampling, and decoder loss. Real pixels use the same per-channel 2nd–98th
percentile scaling during training and prediction. Padded pixels stay black.
Training-chip validity masks are saved under `training_chips/valid/`.

A small patch is technically supported, but it supplies less real context and
may truncate crowns. Use multiple completely annotated areas and evaluate on
separate areas; a valid patch size does not establish model accuracy.

Bundles without `preprocessing` retain their legacy whole-scene/crop behavior.
The saved-window behavior requires this updated library at both training and
prediction time. Installing a released QGIS engine does not automatically include
unreleased local library changes.

## Foreground Prompts

Pass `fg_extra=0` to `train(...)` or `SamStars.train(...)` to use the seed as
the only foreground point. Positive values cap the number of extra points
selected from high CHM pixels within the prompt box; fewer are used when fewer
pixels qualify. The default remains `80`. Negative-point sampling and box
prompts are unchanged.

The same count is used for SAM decoder training and for the proposals used to
train StarDist. It is saved as `defaults.fg_extra` in `manifest.json` and reused
automatically by subsequent segmentation. The lower-level `train_decoder(...)`
also accepts `fg_extra`, and its CLI accepts `--fg-extra`.

## Model Bundle and Saved Outputs

The bundle contains `manifest.json`, `decoder.pth`, `config.json`, `weights.h5`, and, when available, `thresholds.json`.
New bundles also contain `scale_stats.json`, so refinement reuses the training
CHM scaling limits. The refinement normalization setting is saved and reused too.
Pass the bundle directory to `segment(model=...)`; see [Segmentation](segmentation.md).

With `storage="persistent"`, preparation, training samples, and proposal outputs are saved under `models/crowns/_training/`.
Use `storage="temp"` to retain only the finished bundle.
Use a new bundle directory for a new training experiment.

See [Examples](examples.md) for notebook walkthroughs and [Advanced Workflows](advanced_workflows.md) for inspecting or rerunning individual stages.
