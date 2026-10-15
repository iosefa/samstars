# Quick Start

This example trains a crown-segmentation model and applies it to a small area.
Install the package using [Installation](installation.md), then replace the example paths with your own files.

## 1. Prepare Inputs

Training requires imagery, a canopy height model, a first-return density raster, and crown polygons.
See [Data Preparation](data_preparation.md) for raster requirements and [Prepare Crown Annotations](training.md#prepare-crown-annotations) for labeling guidance.
Download the [base checkpoint](training.md#base-checkpoint) before training.

The example assumes a three-band image in red, green, blue order. Set `rgb_bands` to match your image; see [Image Bands](data_preparation.md#image-bands).

## 2. Train a Model

```python
from pathlib import Path

from samstars import train

model = train(
    img=Path("data/training/image.tif"),
    chm=Path("data/training/chm.tif"),
    den=Path("data/training/first_return_density.tif"),
    crowns=Path("data/training/crowns.gpkg"),
    sam_base_ckpt=Path("models/sam_vit_l_0b3195.pth"),
    out=Path("models/crowns"),
    rgb_bands=(1, 2, 3),
    storage="persistent",
    device="cpu",
)
```

Training creates a reusable model directory at `models/crowns`. The [Training](training.md) guide covers labeled tiles and training controls.

## 3. Segment an Area

```python
from samstars import segment

result = segment(
    model=model,
    img=Path("data/inference/image.tif"),
    chm=Path("data/inference/chm.tif"),
    den=Path("data/inference/first_return_density.tif"),
    out=Path("runs/segmentation"),
    storage="persistent",
    device="cpu",
)

print(result.crowns)
```

The input imagery should have the same band order as the training imagery; segmentation reuses the bundle's band settings.
For later runs, replace `model=model` with `model=Path("models/crowns")`.

Open `runs/segmentation/crowns/crowns.gpkg` in your GIS to inspect the predicted crown polygons.
See [Large Raster Runs](large_raster_runs.md) for tiled segmentation and [Downstream Analysis](downstream.md) for species classification and other uses.
