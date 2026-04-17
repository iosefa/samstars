# Quick Start

This page shows the fastest path to run tree crown segmentation with `samstars`.
It assumes you are starting from a small input area.

## 1) Install and activate the environment

```bash
pip install samstars
```

Or use the pinned Conda environment from [Installation](installation.md) if you want the repo development setup.

## 2) Train or prepare model artifacts

`samstars` does not bundle pretrained model weights.
Train models for your own data first.
You will need:

- a trained SAM decoder checkpoint
- a trained StarDist model directory

See [Training](training.md) for how to produce these artifacts.

## 3) Run segmentation on a small input area

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

The run prepares any required intermediate files under `out_dir` automatically.

## Outputs

- crown polygons: `/abs/path/run_dir/segmentation_run/crowns/crowns.gpkg`

## Large rasters

See [Large Raster Runs](large_raster_runs.md).
