# samstars Documentation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](installation.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/iosefa/samstars/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-0A7F46)](index.md)
[![Repo](https://img.shields.io/badge/github-iosefa%2Fsamstars-black?logo=github)](https://github.com/iosefa/samstars)

**Multimodal individual tree crown segmentation from imagery and lidar-derived rasters.**

## Overview

`samstars` is a Python package for individual tree crown segmentation from high-resolution imagery and lidar-derived rasters.

It combines:

- high-resolution imagery (satellite, aerial, or drone)
- lidar-derived raster layers (e.g., canopy height and density)

The package prepares canopy-derived seed points, boundary and gap cost rasters, and proposal index rasters before writing crown polygons.

The workflow was developed with WorldView-3 imagery and airborne lidar-derived rasters, but the library interface accepts any aligned or alignable imagery, canopy-height, and first-return density rasters.

## First Segmentation

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

Use `"cuda"` on supported Linux/NVIDIA systems.

See [Installation](installation.md) to install the library.
For a minimal run path, see [Quick Start](quickstart.md).
For task-focused documentation, see [Data Preparation](data_preparation.md), [Training](training.md), [Segmentation](segmentation.md), and [Downstream Analysis](downstream.md).
