# Segmentation

This page covers the standard inference case: imagery, canopy-height and density rasters, trained samstars model artifacts, and crown polygons as output.

samstars uses two stages:

1. SAM proposal generation from image, CHM, density-derived cost, and seed prompts.
2. StarDist refinement to split touching crowns and write instance polygons.

The recommended public API is the model-bundle interface. The lower-level `run_segmentation(...)` API remains available when you need to pass a SAM decoder checkpoint and StarDist model directory directly.

## Input Expectations

For raw-scene inference, samstars expects:

- a multiband imagery raster
- a CHM raster aligned or alignable to the imagery
- a first-return density raster aligned or alignable to the imagery
- a trained model bundle, or direct SAM and StarDist artifacts

Point-cloud processing is upstream of this package. Convert point clouds to CHM and density rasters before running segmentation.

For prepared-tile inference, each tile directory should contain:

- `img.tif`
- `chm.tif`
- `cost.tif`
- `seeds.gpkg`

Prepared tiles can be created with `samstars.data.build_tiles(...)`.

## Model Bundle Segmentation

Use this path when you trained or received a flat samstars model bundle containing `manifest.json`, `decoder.pth`, `config.json`, and `weights.h5`.

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("/abs/path/model_bundle"),
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out=Path("/abs/path/run_dir/segmentation_run"),
    storage="persistent",
    device="cpu",
)

print(result.crowns)
```

Use `"cuda"` on supported Linux/NVIDIA systems. Use `"mps"` on supported Apple Silicon environments.

`storage="persistent"` keeps run records, prepared inputs, proposal rasters, scale stats, labels, and crown outputs under `out`. `storage="temp"` uses temporary intermediate storage and copies durable final outputs back to `out`.

## Prepared Tiles

Use prepared tiles when the scene is too large for a single raw-scene preparation step or when you want explicit control over tiling.

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("/abs/path/model_bundle"),
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out=Path("/abs/path/run_dir/segmentation_run"),
    storage="persistent",
    device="cpu",
)
```

For the tiling step, see [Large Raster Runs](large_raster_runs.md).

## Direct Artifact Segmentation

Use `run_segmentation(...)` when you want to pass lower-level model artifacts directly instead of a flat model bundle.

```python
from pathlib import Path

from samstars.segmentation import run_segmentation

run_segmentation(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/models/stardist"),
    device="cpu",
)
```

The direct StarDist path should be the model subdirectory containing `config.json`, for example `/abs/path/models/stardist`.

## Stage Control

Run SAM proposals only:

```python
run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/models/sam_crowns_decoder.pth"),
    run_stardist=False,
)
```

Rerun StarDist from existing proposal outputs:

```python
run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/models/sam_crowns_decoder.pth"),
    stardist_model=Path("/abs/path/models/stardist"),
    run_sam=False,
    run_stardist=True,
)
```

Use the same `out_dir` that already contains `proposals/proposal_records.json`.

## Resumability

SAM proposal runs are resumable by default. Existing complete proposal rasters are skipped unless `skip_existing_sam=False`.

StarDist refinement has additional controls:

- `stardist_resume=True`: reuse completed refinement checkpoints.
- `stardist_resume_from_labels=True`: rebuild completed checkpoints from existing label rasters; requires `stardist_resume=True`.
- `stardist_skip_sources="tile_a,tile_b"`: skip named sources.
- `stardist_skip_sources_file=Path(...)`: skip sources listed in a file.
- `stardist_cpu=True`: run StarDist in a subprocess with `CUDA_VISIBLE_DEVICES=""`.

Windowed refinement is controlled with `stardist_window_size_m`, `stardist_window_overlap_m`, and `stardist_dedup_iou_thr`.

## Outputs

The main polygon output is:

```text
out/crowns/crowns.gpkg
```

Persistent direct-artifact runs also keep:

- `out/run_records/tiles.json`
- `out/run_records/tiles.pending.json`
- `out/proposals/proposal_records.json`
- `out/proposals/rasters/`
- `out/proposal_scale_stats.json`
- `out/crowns/labels/`

For parameter details, see [API Levels](reference/api_levels.md), [CLI Scripts](reference/cli.md), and the [API Reference](api/segmentation_engine.md).
