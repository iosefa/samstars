# Training

This page covers training a samstars model bundle for later segmentation. The intended workflow is to train SAM and StarDist artifacts for your own imagery, CHM, density, and crown-label data.

The recommended public entry point is `samstars.train(...)`. It runs the lower-level preparation, SAM training, proposal generation, StarDist record creation, and StarDist training steps, then writes a flat model bundle for `samstars.segment(...)`.

## Input Expectations

Training requires:

- imagery raster
- CHM raster
- first-return density raster
- crown polygons for supervision
- base SAM checkpoint, such as `sam_vit_l_0b3195.pth`

Training can start from one labeled scene or from a directory of labeled tiles.

## Model Bundle Training

Train from one labeled scene:

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

print(model.root)
```

Train from labeled tiles:

```python
from pathlib import Path

from samstars import train

model = train(
    tiles_dir=Path("/abs/path/training_tiles"),
    sam_base_ckpt=Path("/abs/path/sam_vit_l_0b3195.pth"),
    out=Path("/abs/path/model_bundle"),
    storage="persistent",
    device="cpu",
)
```

Each training tile can be fully prepared:

- `img.tif`
- `chm.tif`
- `cost.tif`
- `seeds.gpkg`
- `crowns.gpkg`

or raw labeled:

- `img.tif`
- `chm.tif`
- `den.tif`
- `crowns.gpkg`

For raw labeled tiles, `train(...)` derives `cost.tif` and `seeds.gpkg` internally before continuing. Do not mix fully prepared and raw labeled tile layouts in the same `tiles_dir`.

## Bundle Contents

A trained bundle contains:

- `manifest.json`
- `decoder.pth`
- `config.json`
- `weights.h5`
- `thresholds.json`, when StarDist threshold optimization produced one

The manifest records the bundle format, artifact filenames, and defaults used later by `segment(...)`.

With `storage="persistent"`, training intermediates are kept under:

```text
model_bundle/_training/
```

This includes prepared tiles, SAM training chips, SAM proposal outputs, StarDist training records, scale stats, and the temporary StarDist training directory.

## Lower-Level Training Steps

Use the lower-level modules when you need to inspect or replace part of the training workflow.

## SAM Training Chips

Build SAM training chips from labeled tiles:

```bash
python -m samstars.training.chips \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_chips
```

Each input tile should contain:

- `img.tif`
- `chm.tif`
- `crowns.gpkg`

The output directory contains:

- `images/`: image chips as `.npy`
- `masks/`: binary crown masks as `.npy`
- `chm/`: CHM chips as `.npy`
- `prompts/`: prompt metadata per chip

## SAM Decoder Training

Train the SAM decoder from prepared chips:

```bash
python -m samstars.training.sam_training \
  --training-chips-dir /abs/path/training_chips \
  --out-path /abs/path/models/sam_crowns_decoder.pth \
  --ckpt /abs/path/sam_vit_l_0b3195.pth \
  --device cpu
```

Key arguments:

- `--training-chips-dir`: prepared training-chip directory
- `--out-path`: output checkpoint path for the trained decoder
- `--ckpt`: base SAM checkpoint
- `--device`: training device
- `--scale-stats`: optional global imagery scaling
- `--epochs`: number of epochs
- `--min-iters`: minimum number of training iterations

## StarDist Training Records

StarDist training records are built from SAM proposal records plus crown labels:

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

With the high-level persistent training API, the proposal records live under:

```text
model_bundle/_training/training_run/proposals/proposal_records.json
```

## StarDist Model Training

Train the refinement model from train/validation records:

```bash
python -m samstars.training.stardist_training \
  --train-records /abs/path/training_records/train_records.json \
  --val-records /abs/path/training_records/val_records.json \
  --out-dir /abs/path/stardist_models \
  --model-name stardist \
  --use-aux \
  --scale-stats /abs/path/scale_2_98.json
```

Each training record should point to:

- the primary proposal raster used for refinement training
- the CHM raster if `--use-aux` is enabled
- the rasterized instance mask in the `mask` field

Key arguments:

- `--train-records`: training proposal-record JSON
- `--val-records`: validation proposal-record JSON
- `--out-dir`: parent directory for the trained StarDist model
- `--model-name`: model subdirectory name under `--out-dir`
- `--use-aux`: include CHM as an auxiliary input channel
- `--scale-stats`: required with `--use-aux`
- `--input-key`: primary raster field, default `index`
- `--epochs`: number of epochs
- `--min-iters`: minimum number of training iterations

If no validation records are available, the trainer reuses the training data as validation data.

## Using the Trained Model

Use the bundle directly:

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("/abs/path/model_bundle"),
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out=Path("/abs/path/segmentation_run"),
    storage="persistent",
    device="cpu",
)
```

Use lower-level artifacts directly when needed:

```python
from pathlib import Path

from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/segmentation_run"),
    sam_ckpt=Path("/abs/path/models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/stardist_models/stardist"),
    device="cpu",
)
```

For examples, see [Train and Segment One Area](train_and_segment_one_area.ipynb) and [Train and Segment a Tiled Area](train_and_segment_tiled_area.ipynb).
