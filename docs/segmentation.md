# Segmentation

Segmentation in `samstars` takes imagery, canopy-height and first-return density rasters plus a trained model bundle, then writes crown polygons.

The run prepares or reuses canopy-derived tile products, then runs two model stages:

1. Proposal generation from image tiles, seed prompts, CHM, and boundary/gap costs.
2. Polygon refinement that separates touching crowns and writes non-overlapping crown instances.

Start with `samstars.segment(...)`. It can segment from source rasters or from prepared tiles.
If you have separate SAM decoder and StarDist model files instead of a model bundle, use `run_segmentation(...)`.

## Inputs

To segment from source rasters, provide:

- a multiband imagery raster
- a canopy height model raster aligned or alignable to the imagery
- a first-return density raster aligned or alignable to the imagery
- a trained `samstars` model bundle

Point-cloud processing is upstream of this package. Convert point clouds to canopy-height and first-return density rasters before running segmentation.

For tiled runs, provide a directory created by `samstars.data.build_tiles(...)`.
Each tile directory should contain:

- `img.tif`
- `chm.tif`
- `cost.tif`
- `seeds.gpkg`

For details on `cost.tif` and `seeds.gpkg`, see [Data Preparation](data_preparation.md).

## Segment Source Rasters

Use `samstars.segment(...)` with `img`, `chm`, and `den` when you want `samstars` to prepare the derived tile products and run segmentation in one call.

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

## Segment Prepared Tiles

Use prepared tiles when the area should be tiled before segmentation, for example for large rasters or reusable inputs.

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

## Proposal and Canopy Index Rasters

During segmentation, proposal generation writes rasters under:

```text
out/proposals/rasters/
```

Persistent runs also write:

```text
out/proposals/proposal_records.json
```

The proposal records index the rasters for each tile or chip.
Important record keys include:

- `file_name`: smoothed proposal probability raster, written as `*_sam_prob.tif`.
- `index`: proposal index raster, written as `*_sam_index.tif`.
- `chm`: CHM raster aligned to the proposal raster, written as `*_chm.tif`.
- `cost`: cost raster aligned to the proposal raster, written as `*_cost.tif`.
- `logit`: cached pre-smoothing proposal response raster, written as `*_logit.tif`.

The proposal index raster, written as `*_sam_index.tif` and recorded under the `index` key, is the canopy index used as the default refinement input.
It combines smoothed proposal probability with normalized CHM, the cost raster, and a ground-height weight.
Pixels with stronger proposal support, higher canopy height, lower cost, and above-ground CHM values receive higher index values.

## Run Records

Run records are JSON indexes over prepared tiles and proposal rasters.
They make tiled runs explicit and resumable.
`samstars.segment(...)` creates these records automatically.

The main record files are:

- `out/run_records/tiles.json`
- `out/run_records/tiles.pending.json`
- `out/proposals/proposal_records.json`

Build tile records manually with:

```bash
python examples/build_tile_records.py \
  --tiles-dir /abs/path/run_dir/tiles \
  --out-root /abs/path/run_dir/segmentation_run/run_records
```

Build pending records when you want to rerun only tiles without complete proposal rasters:

```bash
python examples/build_pending_tile_records.py \
  --records /abs/path/run_records/tiles.json \
  --proposal-rasters /abs/path/proposals/rasters \
  --out /abs/path/run_records/tiles.pending.json
```

## CHM Scale Stats

Polygon refinement can use CHM as an auxiliary channel.
Scale stats normalize that CHM channel consistently across proposal records.
`samstars.segment(...)` and `samstars.train(...)` build these stats when needed.

```bash
python examples/build_scale_stats_from_proposal_records.py \
  --proposal-records /abs/path/proposals/proposal_records.json \
  --out /abs/path/proposal_scale_stats.json
```

## Segment With Separate Model Files

Use `run_segmentation(...)` if you have a SAM decoder checkpoint and a StarDist model directory instead of a `samstars` model bundle.

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

The StarDist path should point to the model subdirectory containing `config.json`, for example `/abs/path/models/stardist`.

## Run One Stage at a Time

Long runs can be split into proposal generation and polygon refinement.

Run proposals only:

```python
run_segmentation(
    tiles_dir=Path("/abs/path/run_dir/tiles"),
    out_dir=Path("/abs/path/run_dir/segmentation_run"),
    sam_ckpt=Path("/abs/path/models/sam_crowns_decoder.pth"),
    run_stardist=False,
)
```

Rerun polygon refinement from existing proposal outputs:

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

## Resume Runs

Proposal runs resume by default. Existing complete proposal rasters are skipped unless `skip_existing_sam=False`.

Polygon refinement can also resume or skip selected tiles:

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

For crown-level species classification or other analyses after segmentation, see [Downstream Analysis](downstream.md).

When intermediate files are kept, the output directory can also include:

- `out/run_records/tiles.json`
- `out/run_records/tiles.pending.json`
- `out/proposals/proposal_records.json`
- `out/proposals/rasters/`
- `out/proposal_scale_stats.json`
- `out/crowns/labels/`

For parameter details, see the [API Guide](reference/api_levels.md), [CLI Scripts](reference/cli.md), and the [API Reference](api/segmentation_engine.md).
