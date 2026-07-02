# Examples

The `examples/` directory contains repository-local wrappers around the Python API. Use them when you are working from a checkout and want a runnable command instead of writing a small script.

These wrappers are not installed as console commands by `pip install samstars`.

## Main Wrappers

Train or segment from prepared tiles with the bundle API:

```bash
python examples/run_prepared_tiles_inference.py \
  --model /abs/path/model_bundle \
  --tiles-dir /abs/path/run_dir/tiles \
  --out /abs/path/run_dir/segmentation_run \
  --storage persistent \
  --device cpu
```

Train a new bundle from labeled tiles:

```bash
python examples/run_prepared_tiles_inference.py \
  --sam-base-ckpt /abs/path/sam_vit_l_0b3195.pth \
  --tiles-dir /abs/path/training_tiles \
  --out /abs/path/model_bundle \
  --storage persistent \
  --device cpu
```

Run the older direct-artifact tiled wrapper:

```bash
bash examples/run_tiled_segmentation.sh \
  --tiles-dir /abs/path/run_dir/tiles \
  --out-dir /abs/path/run_dir/segmentation_run \
  --sam-ckpt /abs/path/models/sam_crowns_decoder.pth \
  --stardist-model /abs/path/models/stardist
```

## Notebook Walkthroughs

Use [Train and Segment One Area](train_and_segment_one_area.ipynb) when you want to train a bundle from one labeled training package and segment a small inference scene.

The notebook walks through:

1. setting input and output paths
2. training a flat model bundle
3. inspecting persistent training outputs
4. segmenting the clipped inference rasters
5. reading the final crown path

Use [Train and Segment a Tiled Area](train_and_segment_tiled_area.ipynb) when you want the same bundle workflow with tiled inference inputs.

The tiled notebook walks through:

1. training one bundled model from the labeled training package
2. building tiled inference inputs from larger clipped rasters
3. segmenting the prepared tiles with the bundle
4. reading the final crown path

These notebooks are tutorials, not evaluation protocols. For model assessment, keep separate training and held-out areas.

Use [Retrain StarDist for One Area](retrain_stardist_one_area.ipynb) when existing SAM proposals look good and you only want to rebuild StarDist records with a different crown layer, train a new StarDist model, and rerun refinement.

## Utility Wrappers

Build tile records:

```bash
python examples/build_tile_records.py \
  --tiles-dir /abs/path/run_dir/tiles \
  --out-root /abs/path/run_dir/segmentation_run/run_records
```

Build pending records:

```bash
python examples/build_pending_tile_records.py \
  --records /abs/path/run_records/tiles.json \
  --proposal-rasters /abs/path/proposals/rasters \
  --out /abs/path/run_records/tiles.pending.json
```

Build CHM scale stats:

```bash
python examples/build_scale_stats_from_proposal_records.py \
  --proposal-records /abs/path/proposals/proposal_records.json \
  --out /abs/path/proposal_scale_stats.json
```

## Suggested Sequence

1. Start with [Quick Start](quickstart.md) for the shortest inference path.
2. Use [Training](training.md) to create a model bundle.
3. Use [Segmentation](segmentation.md) when you already have a trained bundle or direct artifacts.
4. Use [Large Raster Runs](large_raster_runs.md) for tiled inference scenes.
5. Use the notebooks when you want an end-to-end walkthrough in a local checkout.
