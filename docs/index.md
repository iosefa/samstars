# samstars Documentation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](installation.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/iosefa/samstars/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-0A7F46)](index.md)
[![Repo](https://img.shields.io/badge/github-iosefa%2Fsamstars-black?logo=github)](https://github.com/iosefa/samstars)

**Multimodal instance segmentation of individual trees.**

## Overview

samstars is a Python package for individual tree crown segmentation from high-resolution imagery and lidar-derived rasters.

It combines:

- high-resolution imagery (satellite, aerial, or drone)
- lidar-derived raster layers (e.g., canopy height and density)

These inputs are used for individual tree crown instance segmentation.

The project was developed with WorldView + lidar data and is designed to generalize to other high-resolution imagery and point-cloud products.

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

See [Installation](installation.md) to set up the environment.
For a minimal run path, see [Quick Start](quickstart.md).
For task-focused documentation, see [Segmentation](segmentation.md), [Training](training.md), and [Data Preparation](data_preparation.md).
