# samstars Documentation

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](installation.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/iosefa/samstars/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-samstars.sefa.ai-0A7F46)](https://samstars.sefa.ai)
[![Repo](https://img.shields.io/badge/github-iosefa%2Fsamstars-black?logo=github)](https://github.com/iosefa/samstars)

**Multimodal individual tree crown segmentation from imagery and lidar-derived rasters.**

![SAM-STARS graphical abstract showing satellite imagery and lidar inputs, automatic crown prompts, canopy fusion, individual crown delineation, species classification, and annotation efficiency.](assets/figures/sam-stars-graphical-abstract.png)

## Overview

`samstars` is a Python package for instance segmentation of individual tree crowns from high-resolution imagery and lidar-derived rasters. The resulting crown polygons can be used for downstream [species classification](downstream.md#species-classification).

It combines:

- high-resolution imagery (satellite, aerial, or drone)
- lidar-derived raster layers (e.g., canopy height and density)

It automatically derives prompts from lidar structure and spectral cues to segment individual tree crowns with limited training annotations.

## First Segmentation

Use the model bundle created by [Training](training.md) with imagery and lidar rasters prepared as described in [Data Preparation](data_preparation.md).

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

See [Installation](installation.md) to install the library.
For a minimal training and segmentation workflow, see [Quick Start](quickstart.md).
For task-focused documentation, see [Data Preparation](data_preparation.md), [Training](training.md), [Segmentation](segmentation.md), and [Downstream Analysis](downstream.md).
