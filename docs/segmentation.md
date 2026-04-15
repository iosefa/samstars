# Segmentation

This page covers end-to-end individual tree crown segmentation in samstars.

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
The intended workflow is to train these for your own data and store them wherever you want.

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

- CHM raster
- Density raster
- prepared tile inputs or tile records

These derived inputs are produced automatically for a small scene or during data preparation for large-scene runs.

## Conceptual stages

### Stage 1: Tile and prompt-guided proposals

- Build tiles and auto-generate seed prompts from CHM + density + cost surface.
- Build tile records for the prepared inputs.
- Run proposal inference from those tile records.
- Output a continuous crown-likelihood representation per tile/chip.

### Stage 2: Instance separation and polygons

- Build canopy index / auxiliary-normalized inputs.
- Run refinement model over proposal outputs.
- Polygonize and write crown instances.

This stage decoupling is intentional: detect canopy candidates first, then separate touching crowns.

## Standard segmentation run

```python
from pathlib import Path
from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
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
    img=Path("/abs/path/image_8band.tif"),
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
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    run_sam=False,
    run_stardist=True,
)
```

### CPU fallback for StarDist

```python
from pathlib import Path
from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    run_sam=False,
    run_stardist=True,
    stardist_cpu=True,
)
```

Use the prepared-tile path when you already have a tile directory and want to rerun the same scene without rebuilding the internal prepared inputs.

### Resume failed refinement robustly (per-record isolation)

```bash
PYTHONPATH=$PWD python examples/run_stardist_resumable.py \
  --proposal-records /abs/path/run_dir/segmentation_run/proposals/0474/proposal_records.json \
  --model-dir /abs/path/model_dir \
  --model-name stardist \
  --out-dir /abs/path/run_dir/segmentation_run/crowns \
  --input-key index \
  --use-aux \
  --scale-stats /abs/path/run_dir/segmentation_run/proposal_scale_stats.json \
  --stardist-cpu \
  --skip-existing-labels
```

## Outputs

### Primary output

- `.../crowns/crowns.gpkg`

### Intermediate/optional outputs (debug and experimentation)

- `.../run_records/0474/tiles.json`
- `.../run_records/0474/tiles.pending.json`
- `.../proposals/0474/rasters/*_sam_prob.tif`
- `.../proposals/0474/rasters/*_sam_index.tif`
- `.../proposals/0474/rasters/*_logit.tif`
- `.../proposals/0474/proposal_records.json`
- `.../crowns/labels/*_labels.tif` (if `--write-labels`)

## Key controls

### Run-level controls (`examples/run_tiled_segmentation.sh`)

- `--sam-only` / `--stardist-only`
- `--skip-existing-sam` / `--no-skip-existing-sam`
- `--stardist-cpu`
- `--size-id` (default `0474`)
- `--stardist-input-key` (`index|file_name|logit`)

### Proposal module controls (`samstars/segmentation/proposals.py`)

- `--tile-records`
- `--ckpt-path`
- `--size-id`
- `--device`
- `--fast`
- `--fast-batch-size`
- `--scale-stats`
- `--crop-smooth-sigma`
- `--smooth-sigma`
- `--post-smooth-sigma`

### Refinement module controls (`samstars/segmentation/refinement.py`)

- `--input-key`
- `--use-aux`
- `--scale-stats` (required when `--use-aux`)
- `--prob-thr`, `--nms-thr`
- `--edge-buffer-px`
- `--min-area-m2`

## Troubleshooting

- Native crash during refinement on a few tiles:
  - use `examples/run_stardist_resumable.py`
  - keep `--skip-existing-labels` enabled
- GPU instability for refinement:
  - run with `--stardist-cpu`
- Partial completion:
  - rerun with same output paths; run records and proposal outputs are resumable

For full function-level details, see [API Levels](reference/api_levels.md) and [API Reference](api/segmentation_proposals.md).
