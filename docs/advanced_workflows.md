# Advanced Workflows

These workflows support inspection, partial reruns, and direct use of component models.
Use `storage="persistent"` to retain the files needed for them.
The basic model-bundle workflow is covered in [Training](training.md) and [Segmentation](segmentation.md).

## Resume a Segmentation Run

Reuse the same model, inputs, and output directory:

```python
from pathlib import Path

from samstars import segment

result = segment(
    model=Path("models/crowns"),
    tiles_dir=Path("prepared/tiles"),
    out=Path("runs/tiled_segmentation"),
    storage="persistent",
    stardist_resume=True,
    device="cpu",
)
```

Complete proposal rasters are skipped by default. `stardist_resume=True` also reuses completed refinement checkpoints.
Resuming assumes that inputs and settings are unchanged. Use a new output directory when changing the model, band selections, or processing parameters.

Additional controls include:

| Parameter | Purpose |
| --- | --- |
| `skip_existing_sam=False` | Recompute proposal rasters. |
| `stardist_resume_from_labels=True` | Rebuild missing refinement checkpoints from saved label rasters; also requires `stardist_resume=True`. Recovered crowns have no confidence score. |
| `stardist_skip_sources="tile_a,tile_b"` | Skip named proposal sources. |
| `stardist_skip_sources_file=Path(...)` | Read source names to skip from a text file. |

## Run One Stage at a Time

Use a persistent output directory to share proposals between stages:

```python
from pathlib import Path

from samstars import segment

run = dict(
    model=Path("models/crowns"),
    tiles_dir=Path("prepared/tiles"),
    out=Path("runs/tiled_segmentation"),
    storage="persistent",
    device="cpu",
)

segment(**run, run_stardist=False)
segment(**run, run_sam=False)
```

The first call writes proposals; the second refines those proposals into crown polygons.
For retraining refinement with new annotations, see [Retrain Refinement for One Area](retrain_stardist_one_area.ipynb).

## Proposal and Canopy Index Rasters

The proposal stage produces raster maps of likely crown regions for refinement into individual crown polygons.
Persistent segmentation runs save these maps and their supporting rasters in `out/proposals/rasters/`, indexed by `out/proposals/proposal_records.json`.

The index is a JSON list with one record per processed tile or chip.
Each record groups the corresponding raster paths below and may include spatial metadata in `meta`.

| Record key | Raster | Contents |
| --- | --- | --- |
| `file_name` | `*_sam_prob.tif` | Smoothed crown-proposal probability. |
| `index` | `*_sam_index.tif` | Canopy index used as the default refinement input. |
| `chm` | `*_chm.tif` | Aligned canopy heights. |
| `cost` | `*_cost.tif` | Aligned boundary and gap cues. |
| `logit` | `*_logit.tif` | Cached proposal response before mosaic smoothing. |

The canopy index combines proposal probability, normalized canopy height, boundary cost, and a ground-height weight.
Higher values indicate stronger crown support.

`train(...)` and `segment(...)` create and use the index automatically.
It connects proposal generation to refinement and is also used to build refinement training samples.
To rerun refinement or retrain from saved proposals, retain the index and its referenced rasters at the recorded paths.
Final crown polygons can be used independently of these intermediate files.

## Run Records and CHM Scaling

`out/run_records/tiles.json` indexes the prepared tiles.
`tiles.pending.json` identifies tiles without complete proposal outputs.
The high-level API creates both files automatically; [CLI Scripts](reference/cli.md#segmentation-run-utilities) provides commands for rebuilding them individually.

When CHM is used as an auxiliary refinement channel, `out/proposal_scale_stats.json` stores the scaling limits used across proposal records.
Training stores its corresponding statistics under `model_bundle/_training/`.
These files are generated automatically.

## Inspect Training Chips

Training chips contain imagery arrays, crown masks, CHM arrays, and prompts.
To rebuild them from labeled tiles independently:

```bash
python -m samstars.training.chips \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_chips
```

Each input tile needs `img.tif`, `chm.tif`, and `crowns.gpkg`.
The output contains `images/`, `masks/`, `chm/`, and `prompts/` subdirectories.
`train(...)` performs this step automatically.

## Segment With Separate Model Files

`run_segmentation(...)` accepts a proposal-decoder checkpoint and a refinement model directory directly:

```python
from pathlib import Path

from samstars.segmentation import run_segmentation

run_segmentation(
    tiles_dir=Path("prepared/tiles"),
    out_dir=Path("runs/segmentation"),
    sam_ckpt=Path("models/decoder.pth"),
    stardist_model=Path("models/refinement"),
    device="cpu",
)
```

`sam_ckpt` is the trained proposal checkpoint.
`stardist_model` is the model subdirectory containing `config.json` and its weights.
This interface also accepts source `img`, `chm`, and `den` rasters with explicit band settings in place of `tiles_dir`.

## Multispectral Cost Settings

`rgb_bands` selects the three image channels in red, green, blue order.
For standard eight-band WorldView-3 imagery, true-color RGB is `(5, 3, 2)`.
An optional near-infrared band adds the NDVI gap cue; a separate texture band can also be selected:

```python
from pathlib import Path

from samstars.features import build_cost_surface, write_cost_surface

cost, profile = build_cost_surface(
    img_path=Path("data/worldview3.tif"),
    chm_path=Path("data/chm.tif"),
    rgb_bands=(5, 3, 2),
    nir_band=7,
    texture_band=1,
)
write_cost_surface(cost, profile, Path("prepared/cost.tif"))
```

These explicit cost settings reproduce the earlier WorldView-3 height, NDVI, and coastal-band texture combination.
The same band arguments are accepted by `build_tiles(...)`, `train(...)`, and `segment(...)`.
Red and near-infrared values should use comparable radiometric units for NDVI.

Earlier versions inferred `(7, 5, 3)` model channels from band count and assumed WorldView-3 bands for cost preparation.
New preparation defaults to `(1, 2, 3)` and adds NDVI only when `nir_band` is specified.
Existing models trained on other channel combinations need matching prepared inputs, or retraining with the new RGB settings.
The former `wv3_path` keyword remains an alias for `img_path`; it no longer implies a band layout.
