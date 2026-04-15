# Command-Line Entry Points

This repo includes runnable scripts around the `samstars` Python API.

These are repository-local entrypoints.
They are not installed as console commands by `pip install samstars`.

## Data preparation

- `samstars/data/tiling.py`
- `examples/build_tile_records.py`
- `examples/build_pending_tile_records.py`

## Proposal generation

- `samstars/segmentation/proposals.py`

## Refinement

- `examples/build_scale_stats_from_proposal_records.py`
- `samstars/segmentation/refinement.py`

## Training

- `samstars/training/sam_training.py`
- `samstars/training/stardist_training.py`

## End-to-end wrappers

- `examples/run_tiled_segmentation.sh`
