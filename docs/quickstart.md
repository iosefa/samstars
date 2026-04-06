# Quick Start

This page shows the fastest path to run tree crown segmentation with `samstars`.

## 1) Activate environment

```bash
conda activate samstars
```

## 2) Build tiles

```python
from pathlib import Path

from samstars.data import build_tiles

build_tiles(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out=Path("/abs/path/run_dir/tiles"),
    tile=500,
    overlap=20,
)
```

## 3) Run segmentation

```python
from pathlib import Path

from samstars.segmentation.engine import run_segmentation

run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/checkpoints"),
    device="cuda",
)
```

## 4) CPU fallback for refinement

```python
from pathlib import Path

from samstars.segmentation.engine import run_segmentation

run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/checkpoints"),
    device="cuda",
    stardist_cpu=True,
)
```

## Outputs

- SAM rasters: `/abs/path/run_dir/manifest_run/sam_probs/0474/train_probs/`
- StarDist polygons: `/abs/path/run_dir/manifest_run/stardist_preds/preds.gpkg`
- StarDist labels: `/abs/path/run_dir/manifest_run/stardist_preds/labels/`

For command-line wrappers and resumable utilities, see [Examples](examples.md).
