# Quick Start

This page shows the fastest path to run tree crown segmentation with `samstars`.

## 1) Activate environment

```bash
conda activate samstars
```

## 2) Build tiles

```bash
PYTHONPATH=$PWD python samstars/data/tiling.py \
  --img /abs/path/image_8band.tif \
  --chm /abs/path/chm.tif \
  --den /abs/path/first_return_density.tif \
  --out /abs/path/run_dir/tiles \
  --tile 500 \
  --overlap 20
```

## 3) Run SAM segmentation

```bash
bash examples/run_manifest_samstars.sh \
  --tiles-dir /abs/path/run_dir/tiles \
  --out-dir /abs/path/run_dir/manifest_run \
  --device cuda \
  --sam-only
```

## 4) Run StarDist segmentation

```bash
bash examples/run_manifest_samstars.sh \
  --tiles-dir /abs/path/run_dir/tiles \
  --out-dir /abs/path/run_dir/manifest_run \
  --stardist-only
```

If your environment has GPU runtime instability for this step, run:

```bash
bash examples/run_manifest_samstars.sh \
  --tiles-dir /abs/path/run_dir/tiles \
  --out-dir /abs/path/run_dir/manifest_run \
  --stardist-only \
  --stardist-cpu
```

## Outputs

- SAM rasters: `/abs/path/run_dir/manifest_run/sam_probs/0474/train_probs/`
- StarDist polygons: `/abs/path/run_dir/manifest_run/stardist_preds/preds.gpkg`
- StarDist labels: `/abs/path/run_dir/manifest_run/stardist_preds/labels/`
