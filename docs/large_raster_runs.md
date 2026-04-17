# Large Raster Runs

Use this workflow when the input area is too large for a single direct run.

## Overview

For large scenes, first build tiles and then run segmentation from the tile directory.

## 1) Build tiles

```python
from pathlib import Path

from samstars.data import build_tiles

build_tiles(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out=Path("/abs/path/run_dir/tiles"),
    tile=500,
    overlap=20,
)
```

## 2) Run segmentation from tiles

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

Use `"cuda"` on supported Linux/NVIDIA systems.

## Notes

- Tiling is the scalable path for large scenes.
- Overlap affects edge behavior and rerun cost.
- Large runs write intermediate run data under `out_dir`.
- The final polygon output is `.../crowns/crowns.gpkg`.

See [Data Preparation](data_preparation.md), [Segmentation](segmentation.md), and [Examples](examples.md) for lower-level controls and helper scripts.
