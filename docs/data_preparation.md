# Data Preparation

Data preparation requires imagery, a canopy height model (CHM), and a first-return density raster.
`samstars` uses these to generate crown prompts and boundary cues that help separate neighboring crowns.

`train(...)` and `segment(...)` can create the cost surface and seed points automatically from these rasters.
The examples below show how to create them separately for inspection, or [prepare them together as tiles](#build-prepared-tiles).

## Required Inputs

| Input | What to provide |
| --- | --- |
| `img` | A georeferenced RGB or multiband imagery raster. |
| `chm` | A single-band raster of canopy height above ground, in meters. |
| `den` | A single-band raster of first lidar returns per cell, or per square meter. |
| `crowns` | One polygon per labeled tree crown, required only for training. |

## Image Bands

`rgb_bands` identifies the red, green, and blue bands using one-based indices.
The default `(1, 2, 3)` assumes the source image already has those bands in RGB order.
For a BGR image, use `rgb_bands=(3, 2, 1)`.
For a multiband image, select the three corresponding source bands explicitly; the sensor is not inferred from band count.

An optional `nir_band` adds a near-infrared band for the NDVI cost cue.
`texture_band` selects a separate band for image texture; by default, texture uses the mean of the selected RGB channels.
These arguments are available in `build_cost_surface(...)`, `build_tiles(...)`, `train(...)`, and `segment(...)`.
See [Advanced Workflows](advanced_workflows.md#multispectral-cost-settings) for a multispectral example.

Prepared images contain the three selected RGB channels.
Use the same channel content and cost settings during training and segmentation.

## Prepare Lidar Rasters

The CHM and first-return density raster can be derived from LAS/LAZ point clouds with any suitable processing tool:

1. **Height above ground.** Normalize point heights using classified ground points or a terrain model.
2. **Canopy height model.** Calculate the maximum height above ground in each raster cell.
3. **First-return density.** Select points with `ReturnNumber == 1` and count them in each raster cell. For returns per square meter, divide each count by the cell area in square meters.

Export both outputs as single-band GeoTIFFs.
Keep density units and cell size consistent between training and segmentation, since seed detection uses the raster values directly.

For implementation details, see PDAL's [height normalization](https://pdal.io/en/stable/stages/filters.hag_delaunay.html) and [raster output](https://pdal.io/en/stable/stages/writers.gdal.html) tools, or the [PyForestScan Python example](https://pyforestscan.sefa.ai/examples/getting-started-importing-preprocessing-dtm-chm/).

## Align and Check the Rasters

- Use a projected coordinate system in meters and north-up imagery with square pixels.
- Cover the same area in all three rasters and mark missing data as NoData.
- Check that canopy features line up between the imagery and lidar rasters.

`build_tiles(...)`, and the preparation performed by `train(...)` and `segment(...)`, reproject and resample the CHM and density raster to the imagery grid.
This matches coordinate systems and pixels; it does not correct positional offsets between surveys.

For the individual cost-surface and seed examples below, first align the rasters to the same CRS, pixel size, grid origin, and extent using your GIS software or [Rasterio](https://rasterio.readthedocs.io/en/stable/topics/reproject.html).
These lower-level functions expect already aligned inputs.

## Create a Cost Surface

A cost surface is a raster with values from 0 to 1 that highlights likely crown boundaries.
Higher values discourage grouping seed points across those areas and reduce segmentation scores there.
For RGB imagery, it combines canopy-height changes and image texture.
When `nir_band` is supplied, it also includes an NDVI cue for canopy gaps and non-vegetated areas.
The weights of the available terms are rescaled to sum to one.

Create `cost.tif` from an aligned image and CHM:

```python
from pathlib import Path

from samstars.features import build_cost_surface, write_cost_surface

cost, profile = build_cost_surface(
    img_path=Path("data/image.tif"),
    chm_path=Path("data/chm.tif"),
    rgb_bands=(1, 2, 3),
)

write_cost_surface(cost, profile, Path("prepared/cost.tif"))
```

## Create Seed Points

Seed points are candidate tree-crown centers used as automatic prompts.
The detector finds peaks in the CHM and first-return density raster, filters low-canopy points, and groups nearby candidates using the cost surface.

Using the cost raster created above:

```python
from pathlib import Path

from samstars.features import SeedParams, canonical_seeds

params = SeedParams(
    chm_raster=Path("data/chm.tif"),
    den_raster=Path("data/first_return_density.tif"),
    cost_raster=Path("prepared/cost.tif"),
    min_canopy_height=2.0,
)

seeds = canonical_seeds(params)
seeds.to_file("prepared/seeds.gpkg", driver="GPKG")
```

`min_canopy_height` is in meters. Other useful `SeedParams` controls are:

| Parameter | Meaning |
| --- | --- |
| `d_min` | Minimum peak value, applied to both height and density rasters; default `4.5`. Interpret this against the units of each raster. |
| `min_dist_px` | Peak-search radius in pixels; default `4`. |
| `gauss_sigma` | Smoothing scale in pixels before finding peaks; default `2`. |
| `merge_radius` | Controls grouping of nearby candidates using distance and cost; default `3.0` in projected coordinate units. |

Overlay `cost.tif` and `seeds.gpkg` on the imagery in your GIS to inspect the boundary cues and candidate crown centers.

## Build Prepared Tiles

For large scenes or reusable inputs, `build_tiles(...)` aligns the rasters and generates the cost surface and seeds for each tile in one call.
You can use this directly without running the individual examples above.

```python
from pathlib import Path

from samstars.data import build_tiles

build_tiles(
    img=Path("data/image.tif"),
    chm=Path("data/chm.tif"),
    den=Path("data/first_return_density.tif"),
    out=Path("prepared/tiles"),
    rgb_bands=(1, 2, 3),
    tile=500.0,
    overlap=20.0,
)
```

`tile` and `overlap` use the imagery CRS units: with a CRS in meters, this creates approximately 500-meter tiles with 20 meters of overlap.
The example assumes RGB imagery. If training used `nir_band` or `texture_band`, supply the same settings when building inference tiles.

Each tile directory contains:

| File | Contents |
| --- | --- |
| `img.tif` | The selected red, green, and blue bands, in that order. |
| `chm.tif` | Canopy heights on the tile grid. |
| `cost.tif` | Boundary cues and, when NIR is supplied, an NDVI gap cue. |
| `seeds.gpkg` | Candidate crown-center points. |

Training tiles also need `crowns.gpkg`; see [Training](training.md) for preparing labeled tiles.
See [Large Raster Runs](large_raster_runs.md) to run segmentation from the prepared tiles.
