# Data Preparation

This page covers data steps before segmentation/training.

## Core steps

1. Tile imagery + auxiliary rasters.
2. Build tile records for tiled runs.
3. Build pending tile records for resumable proposal runs.
4. Build CHM scale stats for auxiliary normalization.
5. Build training chips for SAM training.

## Entry points

- `samstars/data/tiling.py`
- `examples/build_tile_records.py`
- `examples/build_pending_tile_records.py`
- `examples/build_scale_stats_from_proposal_records.py`
- `samstars/training/chips.py`

## Why run records exist

Run records are explicit bookkeeping for tiled segmentation.
They are required for resumable proposal runs and for downstream refinement.
They are not an optional workflow mode.
