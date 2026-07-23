# Data Preparation

`samstars` starts from imagery, canopy-height, and first-return density rasters.
It aligns those inputs and derives the prepared tile products that guide crown prompts, boundary and gap costs, proposal scoring, and polygon refinement.

For small scenes, `samstars.segment(...)` creates the derived files during the run.
For large scenes or reusable inputs, build prepared tiles first with `samstars.data.build_tiles(...)`.
Run segmentation with `storage="persistent"` to keep prepared files on disk.

## Source Rasters

Start with three aligned or alignable geospatial rasters:

- `img`: multiband imagery. Tile preparation writes the image bands used for segmentation.
- `chm`: canopy height model. Heights guide crown-center prompts, proposal scoring, and optional refinement input.
- `den`: first-return density raster. Density peaks are combined with CHM peaks to find candidate crown centers.

Point clouds are not direct inputs to `samstars`.
Convert point clouds to CHM and first-return density rasters before training or segmentation.

## Prepared Tiles

Prepared tiles are directories of aligned imagery and canopy-derived features.
Build them from imagery, CHM, and density rasters:

```python
from pathlib import Path

from samstars.data import build_tiles

build_tiles(
    img=Path("/abs/path/image.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out=Path("/abs/path/run_dir/tiles"),
    tile=500.0,
    overlap=20.0,
)
```

Each inference tile contains:

- `img.tif`
- `chm.tif`
- `cost.tif`
- `seeds.gpkg`

Training tiles also need `crowns.gpkg`.
Raw labeled training tiles can contain `den.tif` instead of `cost.tif` and `seeds.gpkg`; `samstars.train(...)` prepares those derived files internally.

## Tile Files

| File | Description |
| --- | --- |
| `img.tif` | Three image bands clipped to the tile footprint. This is the image input for proposal generation. |
| `chm.tif` | Canopy heights clipped to the same tile footprint. The CHM places prompt points, emphasizes canopy pixels in proposal scoring, and can provide an auxiliary channel during polygon refinement. |
| `cost.tif` | Boundary and gap cost surface generated from the CHM and imagery. Higher values discourage merging or scoring across likely crown edges and canopy gaps. |
| `seeds.gpkg` | Candidate crown-center points. The proposal stage uses these points as prompts for crown proposals. |
| `crowns.gpkg` | Training labels for crown polygons. This file is required only for training tiles. |

## Cost Raster

`cost.tif` marks likely crown boundaries and canopy gaps.
Higher values indicate pixels that are harder to cross when grouping seed candidates or scoring proposal pixels.
Lower values indicate pixels that are easier to keep within the same crown neighborhood.

The default cost surface combines:

- CHM gradient, which highlights sharp canopy-height changes.
- `1 - NDVI`, a low-vegetation or gap term derived from red and near-infrared imagery.
- Image texture entropy.
- Optional SLIC segment edges, when a SLIC raster or polygon layer is supplied to tile preparation.

The cost raster affects two steps:

- Seed generation uses it to avoid merging nearby seed candidates across high-cost pixels.
- Proposal generation uses it to reduce the proposal score in high-cost areas.

## Seed Points

`seeds.gpkg` stores candidate crown-center points.
`samstars` builds these points by detecting local peaks in the CHM and first-return density rasters, filtering out low-canopy points, and merging nearby candidates into one prompt point per likely crown center.

The main seed controls exposed by training and tiling are:

- `seed_d_min`: minimum peak value for CHM and density seed candidates.
- `seed_min_dist_px`: local-maximum suppression radius in pixels.
- `seed_gauss_sigma`: smoothing applied before peak detection.
- `seed_merge_radius`: distance used when merging nearby candidates.
- `seed_min_canopy_height`: minimum CHM value retained as a seed.
- `thin_seeds`: whether to thin dense seed sets after generation.

Segmentation creates proposal rasters, canopy index rasters, run records, and CHM scale stats.
See [Segmentation](segmentation.md) for those run outputs.
Training creates chip arrays from labeled tiles.
See [Training](training.md) for training products.
