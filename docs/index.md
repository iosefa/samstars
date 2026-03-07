# SAMStars Documentation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](installation.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/iosefa/sam/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-0A7F46)](index.md)
[![Repo](https://img.shields.io/badge/github-iosefa%2Fsam-black?logo=github)](https://github.com/iosefa/sam)

**Multimodal instance segmentation of individual trees.**

## Overview

SAMStars segments individual tree crowns at scale.

It combines:

- high-resolution imagery (satellite, aerial, or drone)
- point-cloud-derived raster layers (CHM and density)
- Segment Anything Model (SAM) + StarDist for crown instance extraction

The project was developed with WorldView + lidar data and is designed to generalize to other high-resolution imagery and point-cloud products.

## Quick Example

```bash
bash examples/run_manifest_samstars.sh \
  --tiles-dir /abs/path/tiles \
  --out-dir /abs/path/manifest_run \
  --device cuda \
  --sam-only
```

Use [Installation](installation.md) to set up first.
For a minimal run path, see [Quick Start](quickstart.md).
For task-focused documentation, see [Segmentation](segmentation.md), [Training](training.md), and [Data Preparation](data_preparation.md).
