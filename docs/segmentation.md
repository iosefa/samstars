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

## Required inputs

### Source inputs

- Multiband imagery raster
- Point cloud data
- model checkpoints/weights available in expected paths

### Derived inputs used by segmentation

- CHM raster
- Density raster
- prepared tile inputs or tile manifests

These derived inputs are produced automatically for a small scene or during data preparation for large-scene runs.

## Conceptual stages

### Stage 1: Tile and prompt-guided proposals

- Build tiles and auto-generate seed prompts from CHM + density + cost surface.
- Run proposal inference from manifests.
- Output a continuous crown-likelihood representation per tile/chip.

### Stage 2: Instance separation and polygons

- Build canopy index / auxiliary-normalized inputs.
- Run refinement model over proposal outputs.
- Polygonize and write crown instances.

This stage decoupling is intentional: detect canopy candidates first, then separate touching crowns.

## Standard segmentation run

```python
from pathlib import Path
from samstars.segmentation.engine import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/ckpt_root"),
    device="cpu",
)
```

Use `"cuda"` on supported Linux/NVIDIA systems.

## Prepared tiles variant

```python
from pathlib import Path
from samstars.segmentation.engine import run_segmentation

run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/ckpt_root"),
    device="cpu",
)
```

## Common variants

### Proposal stage only

```python
from pathlib import Path
from samstars.segmentation.engine import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/ckpt_root"),
    device="cpu",
    run_stardist=False,
)
```

### Refinement stage only (from existing proposal outputs)

```python
from pathlib import Path
from samstars.segmentation.engine import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/ckpt_root"),
    run_sam=False,
    run_stardist=True,
)
```

### CPU fallback for StarDist

```python
from pathlib import Path
from samstars.segmentation.engine import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/ckpt_root"),
    run_sam=False,
    run_stardist=True,
    stardist_cpu=True,
)
```

Use the prepared-tile path when you already have a tile directory and want to rerun the same scene without rebuilding the internal prepared inputs.

### Resume failed refinement robustly (per-record isolation)

```bash
PYTHONPATH=$PWD python examples/run_stardist_resumable.py \
  --probs-json /abs/path/run_dir/manifest_run/sam_probs/0474/train_probs.json \
  --model-dir /abs/path/model_dir \
  --model-name stardist \
  --out-dir /abs/path/run_dir/manifest_run/stardist_preds \
  --input-key index \
  --use-aux \
  --scale-stats /abs/path/run_dir/manifest_run/scale_2_98.json \
  --stardist-cpu \
  --skip-existing-labels
```

## Outputs

### Primary output

- `.../stardist_preds/preds.gpkg`

### Intermediate/optional outputs (debug and experimentation)

- `.../sam_probs/0474/train_probs/*_sam_prob.tif`
- `.../sam_probs/0474/train_probs/*_sam_index.tif`
- `.../sam_probs/0474/train_probs/*_logit.tif`
- `.../sam_probs/0474/train_probs.json`
- `.../stardist_preds/labels/*_labels.tif` (if `--write-labels`)

## Key controls

### Run-level controls (`examples/run_manifest_samstars.sh`)

- `--sam-only` / `--stardist-only`
- `--skip-existing-sam` / `--no-skip-existing-sam`
- `--stardist-cpu`
- `--size-id` (default `0474`)
- `--stardist-input-key` (`index|file_name|logit`)

### Proposal module controls (`samstars/segmentation/proposals.py`)

- `--device`
- `--fast`
- `--fast-batch-size`
- `--scale-stats`
- `--sizes`
- `--splits`

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
  - rerun with same output paths; pipeline is resumable for proposal outputs

For full function-level details, see [API Levels](reference/api_levels.md) and [API Reference](api/segmentation_proposals.md).
