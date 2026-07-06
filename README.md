# samstars

`samstars` is a Python library for instance segmentation of individual trees
using high-resolution imagery and lidar-derived rasters.

## Installation

For users, install the published library from PyPI:

```bash
pip install samstars
```

`samstars` does not bundle trained model files.
See [`docs/training.md`](docs/training.md) and
[`docs/segmentation.md`](docs/segmentation.md) to train a model bundle and run
segmentation.

## Developer setup

For local development from a clone:

```bash
pip install -e .
```

For documentation development:

```bash
pip install -e ".[docs]"
mkdocs serve
```

For reproducible development environments, use the platform Conda environment
files in this repository:

```bash
conda env create -f environment.ubuntu.samstars.yml
conda activate samstars
```

On macOS, use `environment.macos.samstars.yml` instead.

## Notes

- `pip install samstars` is the simplest install path when your platform resolves the dependencies cleanly.
- Segmentation runs require a trained `samstars` model bundle.
- The intended workflow is to train a model bundle for your data and then use it for segmentation.
