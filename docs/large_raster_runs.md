# Large Raster Runs

Large areas can be prepared as overlapping tiles and processed with one trained model bundle.
This reduces the size of the rasters handled in each preparation and inference step.

## Build Tiles

Use the same band selections and cost settings as during training.
The example below assumes RGB imagery without a near-infrared band.

```python
from pathlib import Path

from samstars.data import build_tiles

build_tiles(
    img=Path("data/inference/image.tif"),
    chm=Path("data/inference/chm.tif"),
    den=Path("data/inference/first_return_density.tif"),
    out=Path("prepared/tiles"),
    rgb_bands=(1, 2, 3),
    tile=500.0,
    overlap=20.0,
)
```

`tile` and `overlap` use the imagery CRS units. With a CRS in meters, this prepares approximately 500-meter tiles with 20 meters of overlap.
Each tile contains the four files described in [Data Preparation](data_preparation.md#build-prepared-tiles).

## Segment Prepared Tiles

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("models/crowns"),
    tiles_dir=Path("prepared/tiles"),
    out=Path("runs/tiled_segmentation"),
    storage="persistent",
    device="cpu",
)

print(result.crowns)
```

The combined crown polygons are written to `runs/tiled_segmentation/crowns/crowns.gpkg`.

## Choosing Tile Settings

Use tiles large enough to include neighboring crowns, and reduce tile size if memory use becomes excessive.
Increase overlap if crowns are truncated at tile edges.
For memory-intensive refinement, `stardist_window_size_m` and `stardist_window_overlap_m` subdivide each prepared tile into smaller prediction windows.
These controls use map units, normally meters.

## Overlap and Duplicate Crowns

Crowns from overlapping tiles or prediction windows can overlap in the combined output.
Final deduplication is disabled by default (`stardist_dedup_iou_thr=0.0`).
Set this parameter to a value between 0 and 1, such as `0.6`, to retain the higher-scoring crown when a pair's intersection-over-union reaches the threshold.
Lower thresholds remove more overlapping predictions and can also remove distinct neighboring crowns.

Inspect tile boundaries when choosing this setting. Deduplication removes matching polygons; it does not trim all remaining overlaps or guarantee one prediction per tree.

For resuming long runs and inspecting intermediate outputs, see [Advanced Workflows](advanced_workflows.md).
