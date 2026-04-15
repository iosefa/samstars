# Training

This page covers the model-training path for samstars.
The intended workflow is to train your own SAM and StarDist artifacts for your data and then pass those outputs into the segmentation pipeline.

## Available training modules

- `samstars/training/sam_training.py`
- `samstars/training/stardist_training.py`

At the package level, these are exposed as:

- `samstars.training.sam_training`
- `samstars.training.stardist_training`

## Typical workflow

1. Prepare training data.
2. Train SAM components.
3. Train StarDist refinement.
4. Run evaluation/inference on held-out areas.

## SAM decoder training

Use `samstars.training.sam_training` to train the SAM decoder by size.

Example:

```bash
python -m samstars.training.sam_training \
  --chips-root /abs/path/chips_root \
  --ckpt /abs/path/sam_vit_l_0b3195.pth \
  --device cpu \
  --sizes 0474
```

This produces the SAM decoder artifact used later by `run_segmentation(...)` via `sam_ckpt`.
Store that trained model wherever it makes sense for your project.

This training path expects prepared chip directories under `--chips-root`, typically with per-size folders such as:

- `<chips_root>/0474/chips_train/`
- `<chips_root>/0474/chips_val/`

## StarDist training

Use `samstars.training.stardist_training` to train the refinement model from training and validation proposal-record files.

Example:

```bash
python -m samstars.training.stardist_training \
  --train-records /abs/path/training_records/0474/train_records.json \
  --val-records /abs/path/training_records/0474/val_records.json \
  --out-dir /abs/path/stardist_models \
  --use-aux \
  --scale-stats /abs/path/scale_2_98.json
```

This produces the StarDist artifact set used later by `run_segmentation(...)` via `stardist_model`.
Store that trained model wherever it makes sense for your project.

Each training record should point to:

- the primary proposal raster used for refinement training
- the CHM raster if `--use-aux` is enabled
- the instance mask raster in the `mask` field

## How training outputs connect to inference

After training, the segmentation call uses those artifact roots directly:

```python
from pathlib import Path

from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/your_sam_models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/your_stardist_models/stardist"),
    device="cpu",
)
```

## Notes

- Keep training runs in dedicated output directories.
- Pin environment versions for reproducibility.
- The package does not ship pretrained models.
- Use API docs for full parameter details.

See [API Levels](reference/api_levels.md) for which layer to call directly.
