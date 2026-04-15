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
The intended path is to train models for your own data first.

Before you run segmentation, make sure you have a trained SAM decoder and a trained StarDist model available somewhere on disk.
Use whatever storage location and organization make sense for your project.

See [Training](training.md) for how to produce these artifacts.

## 3) Run segmentation on a small input area

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

The run writes any required prepared inputs under `out_dir` automatically.

## 4) CPU fallback for refinement

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
    stardist_cpu=True,
)
```

## Outputs

- run records: `/abs/path/run_dir/segmentation_run/run_records/`
- proposal rasters and records: `/abs/path/run_dir/segmentation_run/proposals/`
- crown polygons: `/abs/path/run_dir/segmentation_run/crowns/crowns.gpkg`
- label rasters: `/abs/path/run_dir/segmentation_run/crowns/labels/`

## Large rasters

For large scenes, prepare tiles first and then run segmentation from the tile directory.
That is the scalable preprocessing path.

```python
from pathlib import Path

from samstars.data import build_tiles
from samstars.segmentation import run_segmentation

build_tiles(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out=Path("/abs/path/run_dir/tiles"),
    tile=500,
    overlap=20,
)

run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    device="cpu",
)
```

Large-raster runs usually need more attention to:

- tile size and overlap
- runtime and disk usage
- resumable run records and reruns

See [Data Preparation](data_preparation.md) and [Segmentation](segmentation.md) for the full workflow.

For command-line wrappers and resumable utilities, see [Examples](examples.md).
