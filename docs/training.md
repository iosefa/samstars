# Training

This page covers the model-training path for samstars.
The intended workflow is to train your own SAM and StarDist artifacts for your data and then pass those outputs into the segmentation pipeline.

## Recommended entry point

For the simplest public API, use `samstars.train(...)`.
It writes a single flat model bundle directory containing:

- `decoder.pth`
- `config.json`
- `thresholds.json`
- `weights.h5`
- a bundle manifest used later by `samstars.segment(...)`

Example:

```python
from pathlib import Path

from samstars import train

model = train(
    img=Path("/abs/path/training_img.tif"),
    chm=Path("/abs/path/training_chm.tif"),
    den=Path("/abs/path/training_den.tif"),
    crowns=Path("/abs/path/training_crowns.gpkg"),
    sam_base_ckpt=Path("/abs/path/sam_vit_l_0b3195.pth"),
    out=Path("/abs/path/model_bundle"),
    storage="persistent",
    device="cpu",
)
```

The lower-level modules documented below remain available when you need finer control.

## Available training modules

- `samstars/training/chips.py`
- `samstars/training/records.py`
- `samstars/training/sam_training.py`
- `samstars/training/stardist_training.py`

At the package level, these are exposed as:

- `samstars.training.chips`
- `samstars.training.records`
- `samstars.training.sam_training`
- `samstars.training.stardist_training`

## Typical workflow

1. Prepare training data.
2. Train SAM components.
3. Build StarDist training records.
4. Train StarDist refinement.
5. Run evaluation/inference on held-out areas.

## Build training chips for SAM

Use `samstars.training.chips` to build the SAM training-chip directory from labeled tiles.

This step expects prepared training tiles.
If you are starting from larger rasters, first build tiles and then add crown labels to each tile.
The training-chip directory is the output of this step, not the input.

Example:

```bash
python -m samstars.training.chips \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_chips
```

Each input tile should contain:

- `img.tif`: imagery raster
- `chm.tif`: canopy height raster
- `crowns.gpkg`: crown polygons for supervision

Use `samstars.data.build_tiles(...)` when you need to create the tile layout first.
If you are preparing tiles from a multiband raster, choose the three image bands you want to use at tile-building time and keep that choice consistent through training and segmentation.

The output directory contains:

- `images/`: image chips as `.npy`
- `masks/`: binary crown masks as `.npy`
- `chm/`: CHM chips as `.npy`
- `prompts/`: prompt metadata per chip

## SAM decoder training

Use `samstars.training.sam_training` to train the SAM decoder from a prepared training-chip directory.

Example:

```bash
python -m samstars.training.sam_training \
  --training-chips-dir /abs/path/training_chips \
  --out-path /abs/path/models/sam_crowns_decoder.pth \
  --ckpt /abs/path/sam_vit_l_0b3195.pth \
  --device cpu
```

This produces the SAM decoder artifact used later by `run_segmentation(...)` via `sam_ckpt`.

The chip directory should contain:

- `images/`: image chips as `.npy`
- `masks/`: binary crown masks as `.npy`
- `chm/`: CHM chips as `.npy`
- `prompts/`: prompt metadata per chip

Key arguments:

- `--training-chips-dir`: prepared training-chip directory
- `--out-path`: output checkpoint path for the trained decoder
- `--ckpt`: base SAM checkpoint used to initialize training
- `--device`: training device
- `--scale-stats`: optional global imagery scaling
- `--epochs`: number of epochs
- `--min-iters`: minimum number of training iterations

## StarDist training

Before training StarDist, build train/val proposal-record files with rasterized crown masks.

Example:

```bash
python -m samstars.training.records \
  --proposal-records /abs/path/training_run/proposals/proposal_records.json \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_records
```

This writes:

- `train_records.json`
- `val_records.json`
- rasterized instance masks under `masks/`

Use `samstars.training.stardist_training` to train the refinement model from training and validation proposal-record files.

Example:

```bash
python -m samstars.training.stardist_training \
  --train-records /abs/path/training_records/train_records.json \
  --val-records /abs/path/training_records/val_records.json \
  --out-dir /abs/path/stardist_models \
  --model-name stardist \
  --use-aux \
  --scale-stats /abs/path/scale_2_98.json
```

This produces the StarDist artifact set used later by `run_segmentation(...)` via `stardist_model`.

Each training record should point to:

- the primary proposal raster used for refinement training
- the CHM raster if `--use-aux` is enabled
- the instance mask raster in the `mask` field

Key arguments:

- `--train-records`: training proposal-record JSON
- `--val-records`: validation proposal-record JSON
- `--out-dir`: output directory for the trained StarDist model
- `--model-name`: model subdirectory name created under `--out-dir`
- `--use-aux`: include CHM as an auxiliary input channel
- `--scale-stats`: required with `--use-aux`; used to scale the CHM input
- `--input-key`: choose which raster field to use as the primary model input; the default is `index`
- `--epochs`: number of epochs
- `--min-iters`: minimum number of training iterations

## How training outputs connect to inference

After training, the segmentation call uses those artifact roots directly:

```python
from pathlib import Path

from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image.tif"),
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
