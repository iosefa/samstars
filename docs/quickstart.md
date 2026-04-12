# Quick Start

This page shows the fastest path to run tree crown segmentation with `samstars`.
It assumes you are starting from a small input area.

## 1) Activate environment

```bash
conda activate samstars
```

## 2) Run segmentation on a small input area

```python
from pathlib import Path

from samstars.segmentation.engine import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/checkpoints"),
    device="cpu",
)
```

Use `"cuda"` on supported Linux/NVIDIA systems.

The run writes any required prepared inputs under `out_dir` automatically.

## 3) CPU fallback for refinement

```python
from pathlib import Path

from samstars.segmentation.engine import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/checkpoints"),
    device="cpu",
    stardist_cpu=True,
)
```

## Outputs

- SAM rasters: `/abs/path/run_dir/manifest_run/sam_probs/0474/train_probs/`
- StarDist polygons: `/abs/path/run_dir/manifest_run/stardist_preds/preds.gpkg`
- StarDist labels: `/abs/path/run_dir/manifest_run/stardist_preds/labels/`

## Large rasters

For large scenes, prepare tiles first and then run segmentation from the tile directory.
That is the scalable preprocessing path.

```python
from pathlib import Path

from samstars.data import build_tiles
from samstars.segmentation.engine import run_segmentation

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
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/checkpoints"),
    device="cpu",
)
```

Large-raster runs usually need more attention to:

- tile size and overlap
- runtime and disk usage
- resumable manifests and reruns

See [Data Preparation](data_preparation.md) and [Segmentation](segmentation.md) for the full workflow.

For command-line wrappers and resumable utilities, see [Examples](examples.md).
