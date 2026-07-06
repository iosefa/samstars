# Training

Training creates a `samstars` model bundle for later crown segmentation.
The bundle stores the trained proposal decoder, polygon-refinement model files, and run defaults needed by `samstars.segment(...)`.

Training uses the same input and prepared-tile products described in [Data Preparation](data_preparation.md): imagery, canopy height, first-return density, seed points, and cost rasters.
It also requires crown polygons for supervision and a base SAM checkpoint.

## Inputs

Training requires:

- imagery raster
- canopy height model raster
- first-return density raster
- crown polygons in a GeoPackage or other vector file readable by GeoPandas
- base SAM checkpoint, such as `sam_vit_l_0b3195.pth`

You can train from one labeled scene or from a directory of labeled tiles.

## Train From One Labeled Area

Use `samstars.train(...)` with `img`, `chm`, `den`, and `crowns` when the labeled training area is a single set of aligned rasters and crown polygons.

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

During training, `samstars` prepares a tile-style training directory, builds seed points and cost rasters, trains the proposal decoder, generates proposal rasters, builds refinement records, and trains the polygon-refinement model.

## Train From Labeled Tiles

Use `tiles_dir` when the training area is already split into tile directories.

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

Each tile can be fully prepared:

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

For raw labeled tiles, `train(...)` derives `cost.tif` and `seeds.gpkg` before training.
Do not mix fully prepared and raw labeled tile layouts in the same `tiles_dir`.

## Model Bundle

A trained bundle contains:

- `manifest.json`
- `decoder.pth`
- `config.json`
- `weights.h5`
- `thresholds.json`, when threshold optimization produced one

`manifest.json` records the bundle format, artifact filenames, and defaults used later by `segment(...)`.
The bundle directory is the path passed to `segment(model=...)`.

## Persistent Training Outputs

With `storage="persistent"`, training intermediates are kept under:

```text
model_bundle/_training/
```

These files are useful for inspection, debugging, and partial reruns.
They include:

- prepared training tiles
- training chips
- proposal rasters and proposal records
- refinement training records
- CHM scale stats
- temporary refinement model training outputs

Use `storage="temp"` when you only want the final model bundle.

## Training Chips

Decoder training uses chip arrays derived from labeled tiles:

```bash
python -m samstars.training.chips \
  --tiles-dir /abs/path/training_tiles \
  --out-dir /abs/path/training_chips
```

Each input tile should contain:

- `img.tif`
- `chm.tif`
- `crowns.gpkg`

The output contains:

- `images/`: image chips as `.npy`
- `masks/`: binary crown masks as `.npy`
- `chm/`: CHM chips as `.npy`
- `prompts/`: prompt metadata per chip

The top-level `train(...)` function builds training chips automatically.
Use this command when you want to inspect or rebuild only the chip arrays.

## Training Controls

Common controls include:

- `rgb_bands`: image bands used for the three-band proposal image.
- `chip_size` and `chip_stride`: training chip size and stride in pixels.
- `sam_epochs`, `sam_min_iters`, and `sam_lr`: proposal-decoder training controls.
- `stardist_epochs` and `stardist_min_iters`: polygon-refinement training controls.
- `val_fraction`: validation fraction for refinement training records.
- `seed_d_min`, `seed_min_dist_px`, `seed_gauss_sigma`, `seed_merge_radius`, and `seed_min_canopy_height`: seed generation controls.
- `stardist_input_key`: primary refinement input, usually `index`.
- `stardist_use_aux`: whether to provide CHM as an auxiliary refinement channel.
- `stardist_cpu`: run refinement training on CPU.

See the [API reference for `samstars.api`](api/top_level.md) for the full function signature.

## Use the Trained Bundle

After training, pass the bundle to `segment(...)`:

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

print(result.crowns)
```

For runnable walkthroughs, see [Train and Segment One Area](train_and_segment_one_area.ipynb) and [Train and Segment a Tiled Area](train_and_segment_tiled_area.ipynb).
For component commands, see [CLI Scripts](reference/cli.md).
