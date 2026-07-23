# Examples

The examples are notebook walkthroughs for the main `samstars` use cases.
Use them when you want to run the workflow step by step in a local checkout.

They are tutorials, not evaluation protocols.
For model assessment, keep separate training and held-out areas.

## Train and Segment One Area

Use [Train and Segment One Area](train_and_segment_one_area.ipynb) when you want the shortest end-to-end training and segmentation walkthrough.

The notebook uses one labeled training package to train a model bundle, then segments a small input area with that bundle.

It covers:

1. setting training and segmentation paths
2. choosing tutorial parameters
3. training a model bundle
4. inspecting persistent training outputs
5. segmenting an input area
6. reading the final crown output

## Train and Segment a Tiled Area

Use [Train and Segment a Tiled Area](train_and_segment_tiled_area.ipynb) when the segmentation area should be prepared and processed as tiles.

The notebook trains one model bundle from a labeled training package, builds prepared inference tiles from larger rasters, and segments those tiles.

It covers:

1. training one model bundle
2. building prepared tiles from larger segmentation rasters
3. running segmentation from the prepared tiles
4. reading the final crown output

## Retrain Refinement for One Area

Use [Retrain Refinement for One Area](retrain_stardist_one_area.ipynb) when proposal rasters already exist and you want to rebuild only the polygon-refinement stage with a different crown-label layer.

The notebook reuses saved proposal rasters from a persistent training run, builds new refinement records, trains a new refinement model, and writes a new crown output.

It covers:

1. preparing a labeled tile for refinement training
2. reusing existing proposal records
3. building refinement training records
4. training a replacement refinement model
5. rerunning refinement without rerunning proposal generation

## Suggested Sequence

1. Start with [Quick Start](quickstart.md) for the shortest inference path.
2. Use [Training](training.md) to create a model bundle.
3. Use [Segmentation](segmentation.md) when you already have a trained bundle or separate model files.
4. Use [Downstream Analysis](downstream.md) when you want to use crown polygons for species classification or other crown-level analyses.
5. Use [Large Raster Runs](large_raster_runs.md) for tiled inference scenes.
6. Use the notebooks when you want an end-to-end walkthrough in a local checkout.

For repository-local command-line wrappers and utility scripts, see [CLI Scripts](reference/cli.md).
