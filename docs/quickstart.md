# Quick Start

This page shows the fastest path to run tree crown segmentation with `samstars`.
It assumes you are starting from a small input area.

## 1) Install and activate the environment

```bash
pip install samstars
```

Or use the pinned Conda environment from [Installation](installation.md) if you want the repo development setup.

## 2) Train or prepare a model bundle

`samstars` does not bundle pretrained model weights.
Train a model bundle for your own data first, or prepare one separately.

See [Training](training.md) for how to produce a bundle directory with `samstars.train(...)`.

## 3) Run segmentation on a small input area

```python
from pathlib import Path

from samstars import segment

segment(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    model=Path("/abs/path/your_model_bundle"),
    out=Path("/abs/path/run_dir/segmentation_run"),
    storage="temp",
    device="cpu",
)
```

Use `"cuda"` on supported Linux/NVIDIA systems.

Use `storage="persistent"` when you want to keep proposals, prep artifacts, and other run state under `out`.
Use `storage="temp"` when you only want final outputs left in `out`.

## Outputs

- crown polygons: `/abs/path/run_dir/segmentation_run/crowns/crowns.gpkg`

## Large rasters

See [Large Raster Runs](large_raster_runs.md).
