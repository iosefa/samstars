# Segmentation

Segmentation combines imagery, canopy height, and first-return density with a trained model bundle to produce individual crown polygons.
`samstars.segment(...)` prepares automatic prompts, generates crown proposals, and refines them into separate crown instances.

## Inputs

For a source-raster run, provide:

- RGB or multiband imagery
- canopy height and first-return density rasters covering the same area
- a model bundle created with [Training](training.md)

See [Data Preparation](data_preparation.md) for raster alignment and band requirements.

## Segment Source Rasters

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("models/crowns"),
    img=Path("data/inference/image.tif"),
    chm=Path("data/inference/chm.tif"),
    den=Path("data/inference/first_return_density.tif"),
    out=Path("runs/segmentation"),
    storage="persistent",
    device="cpu",
)

print(result.crowns)
```

The bundle supplies the band settings used during training.
New bundles also supply their saved SAM ground window, stride, resolution, and
RGB preprocessing. The scene may be larger or smaller than a training image:
larger scenes use overlapping windows, and small scenes are padded. Inputs must
match the saved pixel size; align/resample them before running if necessary.
See [training window settings](training.md#window-size-resolution-and-padding).

For a bundle with `preprocessing` settings, `fast=True` batches prompts within
each fixed window and `fast=False` processes them individually. It does not
change how much ground SAM sees. With no override, segmentation uses the saved
`fast` setting; training defaults to `False`. Older bundles without these settings
retain their legacy behavior: fast mode embeds a whole prepared tile, while
normal mode embeds seed-centered crops.

Cached proposals made with different window settings cannot be reused. Use a new
output directory or `skip_existing_sam=False` to regenerate those intermediates.
If source bands have been reordered, pass the corresponding `rgb_bands`, and any `nir_band` or `texture_band`, explicitly while retaining the same channel content.
Omitted band settings reuse the bundle defaults.

## Foreground Prompts

`segment(...)` and `model.segment(...)` reuse the bundle's `fg_extra` setting
when it is omitted or `None`. Bundles created before this setting was added
continue to use `80` extra foreground points per seed.

Pass `fg_extra=0` to use only each seed as foreground, or a positive integer
to cap the additional foreground points. This works in both normal crop mode
and fast batched mode. Negative-point sampling and box prompts are unchanged;
other seeds are not automatically turned into background points.

Use a new output directory when changing the count so existing proposals are
not reused. An override affects that run and does not modify the model bundle.

## Outputs

The main output is `runs/segmentation/crowns/crowns.gpkg`, returned as `result.crowns`.
It contains one feature per predicted crown and can be opened in GIS software or read with GeoPandas.
See [Downstream Analysis](downstream.md#crown-output) for its fields and uses.

With `storage="persistent"`, the output directory also keeps prepared inputs, proposal rasters, and refinement checkpoints for inspection or resuming a run.
`proposals/proposal_records.json` indexes the rasters used by refinement and is managed automatically; see [Advanced Workflows](advanced_workflows.md#proposal-and-canopy-index-rasters) for its contents and reuse.
With `storage="temp"`, intermediate files are removed after the run and final crown outputs are retained.

Overlapping tiles can produce duplicate crowns near their boundaries.
See [Large Raster Runs](large_raster_runs.md#overlap-and-duplicate-crowns) for overlap and deduplication settings.

## Segment Prepared Tiles

A prepared tile directory contains subdirectories with `img.tif`, `chm.tif`, `cost.tif`, and `seeds.gpkg`.
Use the same image channels and cost settings as during training.

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("models/crowns"),
    tiles_dir=Path("prepared/tiles"),
    out=Path("runs/tiled_segmentation"),
    storage="persistent",
    device="cpu",
)
```

[Large Raster Runs](large_raster_runs.md) explains how to create the tiles and choose their size.
For partial reruns, intermediate rasters, and separate model files, see [Advanced Workflows](advanced_workflows.md).
