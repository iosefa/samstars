# samstars

`samstars` is a Python library for instance segmentation of individual trees
using lidar-derived rasters and WorldView imagery.

## Installation

Install the full library:

```bash
pip install samstars
```

Editable source install:

```bash
pip install -e .
```

If pip wheels are not available for your platform, or if you want the most
reproducible setup, use the platform Conda environment files documented in
[`docs/installation.md`](docs/installation.md).

## Notes

- `pip install samstars` is the simplest install path when your platform resolves the dependencies cleanly.
- `pip install -e .` installs the current checkout in editable mode.
- If some dependencies do not install cleanly with pip on your system, use the Conda environment files instead.
- Segmentation runs also require trained model artifacts.
- The intended workflow is to train your own models and then use those artifacts for inference.
- See [`docs/training.md`](docs/training.md) and [`docs/segmentation.md`](docs/segmentation.md) for the library-facing usage contract.
