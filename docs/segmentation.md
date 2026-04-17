# Segmentation

samstars uses a two-stage segmentation strategy:

1. Lidar-guided prompting for high-recall canopy detection.
2. Shape-aware instance separation to split adjacent crowns.

In practice, this means:

- proposal stage: prompt-guided crown likelihood inference
- refinement stage: instance delineation into crown polygons

## Goal

Given high-resolution imagery and a point cloud, delineate upper-canopy individual tree crowns as polygons.

## Before you run

You need trained model artifacts in addition to the input rasters.
The intended workflow is to train these for your own data.

At inference time, you provide:

- `sam_ckpt`: the direct path to your trained SAM decoder checkpoint
- `stardist_model`: the direct path to your trained StarDist model directory

See [Training](training.md) for how to produce both artifact types.

## Required inputs

### Source inputs

- Multiband imagery raster
- Point cloud data
- model checkpoints/weights available in expected paths

### Derived inputs used by segmentation

- imagery raster
- CHM raster
- Density raster
- seed prompts derived from the raster inputs

## Conceptual stages

### Stage 1: Proposal generation

- For a small scene, run directly from the input rasters.
- For a large scene, tile first and then run the same proposal stage per tile.
- Use imagery, CHM, and density-derived prompts to build a crown-likelihood raster.

### Stage 2: Instance separation and polygons

- Run the refinement model over the proposal raster.
- Polygonize and write crown instances.

This stage decoupling is intentional: detect canopy candidates first, then separate touching crowns.

## Standard segmentation run

```python
from pathlib import Path
from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    device="cpu",
)
```

Use `"cuda"` on supported Linux/NVIDIA systems.

## Prepared tiles variant

Use this when you already have a prepared tile directory.
Each tile folder should contain the per-tile segmentation inputs, including:

- `img.tif`
- `chm.tif`
- `cost.tif`
- `seeds.gpkg`

You can build this layout with `samstars.data.build_tiles(...)`.

```python
from pathlib import Path
from samstars.segmentation import run_segmentation

run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    device="cpu",
)
```

## Common variants

### Proposal stage only

```python
from pathlib import Path
from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    device="cpu",
    run_stardist=False,
)
```

### Refinement stage only (from existing proposal outputs)

```python
from pathlib import Path
from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    run_sam=False,
    run_stardist=True,
)
```

Use the prepared-tile path when you already have a tile directory and want to rerun the same scene without rebuilding the internal prepared inputs.

## Outputs

- `.../crowns/crowns.gpkg`

- `.../crowns/labels/*_labels.tif` (if `--write-labels`)

## Troubleshooting

- Partial completion:
  - rerun with same output paths; run records and proposal outputs are resumable

For parameter details, see [API Levels](reference/api_levels.md), [CLI Scripts](reference/cli.md), and the [API Reference](api/segmentation_proposals.md).
