# Examples

The `examples/` directory contains runnable wrappers around the library API.

Use these when you want a ready-to-run command.
Use the Python API in [Quick Start](quickstart.md) and [Segmentation](segmentation.md) when you want to integrate `samstars` directly into your own code.

These wrappers are repository-local helper scripts.
They are useful when you are working from a checkout of this repo.
They are not installed as console commands by `pip install samstars`.

## Main wrappers

- `examples/run_tiled_segmentation.sh`

## End-to-end tutorial

Use this when you have one labeled area and want to:

1. prepare training chips
2. train the models
3. inspect SAM proposal outputs
4. run segmentation on the same area

This is a tutorial example, not a recommended evaluation setup.
For real model assessment, keep separate training and held-out areas.

See [Train and Segment One Area](train_and_segment_one_area.ipynb) for the rendered notebook walkthrough.

The notebook walks through:

1. preparing one labeled tile
2. building SAM training chips
3. training the SAM decoder
4. inspecting SAM proposal outputs
5. building StarDist training records
6. training StarDist
7. running segmentation

The final output path is:

- `/abs/path/run_dir/segmentation_run/crowns/crowns.gpkg`

## StarDist-only rerun

Use this when the SAM proposals from a one-area run already look good and you only want to:

1. rebuild StarDist training records with a different crown layer
2. train a new StarDist model in a separate folder
3. write new crowns without rerunning SAM

See [Retrain StarDist for One Area](retrain_stardist_one_area.ipynb) for the rendered notebook walkthrough.

## Tiled tutorial

Use this when you want to exercise the tiled large-raster workflow on one raster stack:

1. build tiles from the source rasters
2. attach training crowns to the intersecting tiles
3. build SAM training chips
4. train the models
5. inspect SAM proposal outputs
6. run segmentation across all prepared tiles

This is still a tutorial example, not a recommended evaluation setup.
For real model assessment, keep separate training and held-out areas.

See [Train and Segment a Tiled Area](train_and_segment_tiled_area.ipynb) for the rendered notebook walkthrough.

## Utility wrappers

- `examples/build_tile_records.py`
- `examples/build_pending_tile_records.py`
- `examples/build_scale_stats_from_proposal_records.py`

## Suggested sequence

1. Use [Train and Segment One Area](train_and_segment_one_area.ipynb) when you want one labeled area to exercise the full pipeline end to end without tiling.
2. Use [Retrain StarDist for One Area](retrain_stardist_one_area.ipynb) when you want to keep the saved SAM proposals and only rerun StarDist with a different crown layer.
3. Use [Train and Segment a Tiled Area](train_and_segment_tiled_area.ipynb) when you want the same kind of walkthrough using prepared tiles.
4. Run [Quick Start](quickstart.md) for the shorter inference-only path.
5. Use [Segmentation](segmentation.md) variants when you already have trained models.
6. Use API-level functions when you need lower-level control.
