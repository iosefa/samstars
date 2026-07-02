# Large Raster Runs

Use prepared tiles when the inference scene is too large or expensive to prepare and segment as one raw scene.

The large-raster workflow has two steps:

1. build tile directories from source rasters
2. run segmentation from those prepared tiles

## Build Tiles

```python
from pathlib import Path

from samstars.data import build_tiles

build_tiles(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out=Path("/abs/path/run_dir/tiles"),
    tile=500.0,
    overlap=20.0,
)
```

Each tile directory contains the prepared inference inputs used by segmentation:

- `img.tif`
- `chm.tif`
- `cost.tif`
- `seeds.gpkg`

`tile` and `overlap` are expressed in raster CRS units. For projected data in meters, `tile=500.0` creates approximately 500-meter tiles.

## Segment Tiles With a Bundle

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("/abs/path/model_bundle"),
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out=Path("/abs/path/run_dir/segmentation_run"),
    storage="persistent",
    device="cpu",
)

print(result.crowns)
```

This is the recommended path when you trained with `samstars.train(...)`.

## Segment Tiles With Direct Artifacts

```python
from pathlib import Path

from samstars.segmentation import run_segmentation

run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/models/stardist"),
    device="cpu",
)
```

Use `"cuda"` on supported Linux/NVIDIA systems. Use `"mps"` on supported Apple Silicon environments.

## Choosing Tile Settings

Use tiles large enough to include useful crown context, but small enough to keep SAM proposal generation stable. Increase overlap when edge effects are visible near tile boundaries.

The outer tile size controls prepared inference units. StarDist can also run internal refinement windows through `stardist_window_size_m`, `stardist_window_overlap_m`, and `stardist_dedup_iou_thr`.

## Outputs

The final polygon output is:

```text
run_dir/segmentation_run/crowns/crowns.gpkg
```

Persistent runs also keep tile records, pending records, proposal rasters, proposal records, scale stats, and optional labels under the segmentation output directory.

See [Segmentation](segmentation.md), [Data Preparation](data_preparation.md), and [Examples](examples.md) for lower-level controls and runnable wrappers.
