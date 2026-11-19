# Command-Line Entry Points

This repository includes runnable scripts around the `samstars` Python API. They are repository-local entry points, not installed console commands.

Run them from the repository root or set `PYTHONPATH` to the checkout.

## Prepared Tile Bundle Wrapper

Train a bundle from prepared or raw labeled tiles:

```bash
python examples/run_prepared_tiles_inference.py \
  --sam-base-ckpt /abs/path/sam_vit_l_0b3195.pth \
  --tiles-dir /abs/path/training_tiles \
  --out /abs/path/model_bundle \
  --storage persistent \
  --device cpu
```

Segment prepared tiles with an existing bundle:

```bash
python examples/run_prepared_tiles_inference.py \
  --model /abs/path/model_bundle \
  --tiles-dir /abs/path/inference_tiles \
  --out /abs/path/segmentation_run \
  --storage persistent \
  --device cpu
```

Useful options:

- `--rgb-bands R G B`: source bands for raw training tiles, using one-based indices
- `--sam-window-size-m M`: SAM ground window for a new model, saved for prediction
- `--chip-size N`: alternative SAM window in native pixels (default 512); exclusive with `--sam-window-size-m`
- `--chip-stride N`: window spacing in native pixels (default 75% of the selected size)
- `--stardist-patch-size N`: refinement training patch, a positive multiple of 16 (default 256)
- `--nir-band N` and `--texture-band N`: optional cost-surface band choices for raw training tiles
- `--stardist-cpu`: force polygon refinement to CPU
- `--stardist-input-key index|file_name|logit`
- `--stardist-prob-thr` and `--stardist-nms-thr`
- `--stardist-window-size-m`
- `--stardist-window-overlap-m`
- `--stardist-dedup-iou-thr`
- `--sam-only`
- `--stardist-only`
- `--fast` or `--slow`: batched or individual prompt decoding; new bundles keep the same saved SAM ground window in either mode

## Separate Model Files Wrapper

Use this when you have a SAM decoder checkpoint and a StarDist model directory instead of a model bundle:

```bash
bash examples/run_tiled_segmentation.sh \
  --tiles-dir /abs/path/inference_tiles \
  --out-dir /abs/path/segmentation_run \
  --sam-ckpt /abs/path/models/sam_crowns_decoder.pth \
  --stardist-model /abs/path/models/stardist \
  --device cpu
```

The wrapper builds tile records, skips complete proposal rasters by default, builds CHM scale stats when needed, and runs polygon refinement.

## Segmentation Run Utilities

Build tile records:

```bash
python examples/build_tile_records.py \
  --tiles-dir /abs/path/tiles \
  --out-root /abs/path/segmentation_run/run_records
```

Build pending tile records:

```bash
python examples/build_pending_tile_records.py \
  --records /abs/path/run_records/tiles.json \
  --proposal-rasters /abs/path/proposals/rasters \
  --out /abs/path/run_records/tiles.pending.json
```

Build CHM scale stats from proposal records:

```bash
python examples/build_scale_stats_from_proposal_records.py \
  --proposal-records /abs/path/proposals/proposal_records.json \
  --out /abs/path/proposal_scale_stats.json
```

## Module Entrypoints

Build prepared tiles:

```bash
python -m samstars.data.tiling \
  --img /abs/path/image.tif \
  --chm /abs/path/chm.tif \
  --den /abs/path/first_return_density.tif \
  --rgb-bands 1 2 3 \
  --out /abs/path/tiles
```

Run proposal generation:

```bash
python -m samstars.segmentation.proposals \
  --tile-records /abs/path/run_records/tiles.json \
  --ckpt-path /abs/path/models/sam_crowns_decoder.pth \
  --out-root /abs/path/proposals \
  --device cpu \
  --fast
```

Add `--fg-extra 0` to use only each seed as foreground, or a positive integer
to limit extra foreground points. The default is `80`; both normal and `--fast`
proposal generation honor this option. Negative points and box prompts remain.

Run polygon refinement:

```bash
python -m samstars.segmentation.refinement \
  --proposal-records /abs/path/proposals/proposal_records.json \
  --model-dir /abs/path/models \
  --model-name stardist \
  --out-dir /abs/path/crowns \
  --input-key index \
  --use-aux \
  --scale-stats /abs/path/proposal_scale_stats.json \
  --write-labels
```

Build proposal-decoder training chips:

```bash
python -m samstars.training.chips \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_chips
```

Train the proposal decoder:

```bash
python -m samstars.training.sam_training \
  --training-chips-dir /abs/path/training_chips \
  --out-path /abs/path/models/sam_crowns_decoder.pth \
  --ckpt /abs/path/sam_vit_l_0b3195.pth \
  --device cpu
```

The decoder-training CLI also accepts `--fg-extra` (default `80`, `0` keeps only
the seed as foreground). When using the separate training and proposal CLIs,
pass the same foreground count to both; a standalone checkpoint does not store it.

Build refinement training records:

```bash
python -m samstars.training.records \
  --proposal-records /abs/path/training_run/proposals/proposal_records.json \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_records
```

Train the refinement model:

```bash
python -m samstars.training.stardist_training \
  --train-records /abs/path/training_records/train_records.json \
  --val-records /abs/path/training_records/val_records.json \
  --patch-size 256 \
  --out-dir /abs/path/stardist_models \
  --model-name stardist \
  --use-aux \
  --scale-stats /abs/path/training_scale_stats.json
```
