# API Guide

`samstars` exposes a small training and segmentation interface, plus component functions for prepared tiles, records, and custom reruns.

## Model Bundle Workflow

Use these objects for most training and segmentation runs:

- `samstars.train(...)`
- `samstars.segment(...)`
- `samstars.SamStars`
- `samstars.SamStarsResult`

`train(...)` writes a model bundle and returns a `SamStars` runtime wrapper.
`segment(...)` accepts either a bundle path or an existing `SamStars` instance and returns a `SamStarsResult`.

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

Use the model bundle workflow when you want stored defaults and a single model directory to pass between training and segmentation.

## Component Functions

Use these functions when you want direct control over prepared tiles, run records, SAM checkpoints, or StarDist model directories:

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

Use these functions for experiments, partial reruns, and custom training workflows.

## Feature Helpers

Use these modules when you need custom feature preparation or utility behavior:

- `samstars.features`
- `samstars.utils`

These expose cost-surface, seed-generation, raster, and geometry helpers.

## Choosing an Interface

Start with `train(...)` and `segment(...)`.
Use component functions when you need separate model-file paths, prepared-tile records, or partial stage control.
Use feature helpers when you are replacing or inspecting part of the data-preparation workflow.
