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

- `pip install samstars` installs the full segmentation stack, including SAM and StarDist runtime dependencies.
- `pip install -e .` installs the current checkout in editable mode.
- Linux/NVIDIA users who want CUDA-enabled PyTorch may still prefer the Conda environment files.
- Apple Silicon TensorFlow packages are selected automatically by dependency markers.
