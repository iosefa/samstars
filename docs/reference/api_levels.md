# API Levels

samstars has three practical API layers.

## Bundle API

Use this layer for most training and segmentation runs:

- `samstars.train(...)`
- `samstars.segment(...)`
- `samstars.SamStars`
- `samstars.SamStarsResult`

`train(...)` writes a flat model bundle and returns a `SamStars` runtime wrapper. `segment(...)` accepts either a bundle path or an existing `SamStars` instance and returns a `SamStarsResult`.

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
)
```

Use this layer when you want stable model packaging, stored defaults, and fewer artifact paths to manage.

## Pipeline API

Use this layer when you want direct control over prepared tiles, run records, SAM checkpoints, or StarDist model directories:

- `samstars.data`
- `samstars.segmentation.run_segmentation(...)`
- `samstars.training.chips`
- `samstars.training.records`
- `samstars.training.sam_training`
- `samstars.training.stardist_training`

`run_segmentation(...)` uses a direct SAM decoder checkpoint path and a direct StarDist model path.

```python
from pathlib import Path

from samstars.segmentation import run_segmentation

run_segmentation(
    tiles_dir=Path("/abs/path/tiles"),
    out_dir=Path("/abs/path/segmentation_run"),
    sam_ckpt=Path("/abs/path/models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/models/stardist"),
)
```

Use this layer for experiments, partial reruns, and custom training workflows.

## Building Blocks

Use these modules when you need custom feature preparation or utility behavior:

- `samstars.features`
- `samstars.utils`

These expose cost-surface, seed-generation, raster, and geometry helpers. They are lower-level than the training and segmentation workflows.

## Choosing a Layer

Start with the bundle API. Move to the pipeline API when you need direct artifact paths or partial stage control. Move to building blocks only when you are replacing part of the pipeline.
