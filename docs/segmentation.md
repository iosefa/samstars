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
If source bands have been reordered, pass the corresponding `rgb_bands`, and any `nir_band` or `texture_band`, explicitly while retaining the same channel content.
Omitted band settings reuse the bundle defaults.

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
