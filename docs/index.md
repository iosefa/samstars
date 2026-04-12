# samstars Documentation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](installation.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/iosefa/sam/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-0A7F46)](index.md)
[![Repo](https://img.shields.io/badge/github-iosefa%2Fsam-black?logo=github)](https://github.com/iosefa/sam)

**Multimodal instance segmentation of individual trees.**

## Overview

samstars provides multimodal segmentation of individual tree crowns at scale.

It combines:

- high-resolution imagery (satellite, aerial, or drone)
- lidar-derived raster layers (e.g., canopy height and density)

These inputs are used for individual tree crown instance segmentation.

The current implementation follows a two-stage segmentation pipeline for crown instance extraction.

The project was developed with WorldView + lidar data and is designed to generalize to other high-resolution imagery and point-cloud products.

## Quick Example

```python
from pathlib import Path

from samstars.segmentation.engine import run_segmentation

run_segmentation(
    img=Path("/abs/path/image_8band.tif"),
    chm=Path("/abs/path/chm.tif"),
    den=Path("/abs/path/first_return_density.tif"),
    out_dir=Path("/abs/path/run_dir/manifest_run"),
    ckpt_root=Path("/abs/path/checkpoints"),
    device="cpu",
)
```

Use `"cuda"` on supported Linux/NVIDIA systems.

See [Installation](installation.md) to set up the environment.
For a minimal run path, see [Quick Start](quickstart.md).
For task-focused documentation, see [Segmentation](segmentation.md), [Training](training.md), and [Data Preparation](data_preparation.md).
