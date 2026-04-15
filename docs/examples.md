# Examples

The `examples/` directory contains runnable wrappers around the library API.

Use these when you want a ready-to-run command.
Use the Python API in [Quick Start](quickstart.md) and [Segmentation](segmentation.md) when you want to integrate `samstars` directly into your own code.

These wrappers are repository-local helper scripts.
They are useful when you are working from a checkout of this repo.
They are not installed as console commands by `pip install samstars`.

## Main wrappers

- `examples/run_tiled_segmentation.sh`
- `examples/run_stardist_resumable.py`

## Utility wrappers

- `examples/build_tile_records.py`
- `examples/build_pending_tile_records.py`
- `examples/build_scale_stats_from_proposal_records.py`

## Suggested sequence

1. Run [Quick Start](quickstart.md).
2. Use [Segmentation](segmentation.md) variants.
3. Use API-level functions when you need lower-level control.
