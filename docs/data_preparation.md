# Data Preparation

This page covers the data products samstars uses before training and segmentation.

Most users can start with `samstars.train(...)`, `samstars.segment(...)`, and `samstars.data.build_tiles(...)`. The lower-level helpers are useful when you want to inspect, resume, or customize intermediate files.

## Prepared Tiles

Prepared tiles are the processing units for tiled segmentation. Build them from imagery, CHM, and density rasters:

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

Each inference tile contains:

- `img.tif`
- `chm.tif`
- `cost.tif`
- `seeds.gpkg`

Training tiles also need `crowns.gpkg`. Raw labeled training tiles can contain `den.tif` instead of `cost.tif` and `seeds.gpkg`; `samstars.train(...)` prepares those derived files internally.

## Run Records

Run records are JSON indexes over prepared tiles and proposal rasters. They make tiled runs explicit and resumable.

Build tile records manually with:

```bash
python examples/build_tile_records.py \
  --tiles-dir /abs/path/run_dir/tiles \
  --out-root /abs/path/run_dir/segmentation_run/run_records
```

This writes:

```text
run_records/tiles.json
```

Build pending records when you want to rerun only tiles without complete proposal rasters:

```bash
python examples/build_pending_tile_records.py \
  --records /abs/path/run_records/tiles.json \
  --proposal-rasters /abs/path/proposals/rasters \
  --out /abs/path/run_records/tiles.pending.json
```

The high-level segmentation API creates these records automatically.

## Scale Stats

StarDist can use CHM as an auxiliary channel. Scale stats normalize that CHM channel consistently across proposal records.

```bash
python examples/build_scale_stats_from_proposal_records.py \
  --proposal-records /abs/path/proposals/proposal_records.json \
  --out /abs/path/proposal_scale_stats.json
```

High-level training and segmentation build these stats when needed.

## Training Chips

SAM decoder training uses chip arrays derived from labeled tiles:

```bash
python -m samstars.training.chips \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_chips
```

The output contains image chips, masks, CHM chips, and prompt metadata. See [Training](training.md) for the full model-training workflow.

## Entry Points

- `samstars.data.build_tiles(...)`
- `samstars.data.build_single_tile(...)`
- `samstars.data.build_tile_records(...)`
- `samstars.data.build_pending_tile_records(...)`
- `samstars.data.build_scale_stats_from_proposal_records(...)`
- `samstars.training.chips`
