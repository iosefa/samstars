# samstars

`samstars` is a Python library for instance segmentation of individual trees
using high-resolution imagery and lidar-derived rasters.

## Installation

For users, install the published library from PyPI:

```bash
python -m pip install samstars
```

`samstars` supports Python 3.10 through 3.12 on Linux x86-64 and Apple
Silicon macOS. CPU execution is the baseline. Apple Silicon users can install
optional TensorFlow Metal support with `python -m pip install "samstars[metal]"`; Linux
CUDA users should install a matched PyTorch/Torchvision build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing `samstars`.

`samstars` does not bundle trained model files.
See [`docs/training.md`](docs/training.md) and
[`docs/segmentation.md`](docs/segmentation.md) to train a model bundle and run
segmentation.

## Developer setup

For local development from a clone:

```bash
python -m pip install -e .
```

Install the test dependencies and run the functional suite with:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

For documentation development:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

For platform-specific development environments, use the Conda environment
files in this repository:

```bash
conda env create -f environment.ubuntu.samstars.yml
conda activate samstars
python -m pip install -e .
```

On Apple Silicon macOS, use `environment.macos.samstars.yml` instead. These
environment files provide Python 3.10 CPU baselines; accelerator setup is
described in [`docs/installation.md`](docs/installation.md).

## Notes

- Python 3.10 through 3.12 is supported on Linux x86-64 and Apple Silicon macOS.
- CPU execution is the default; CUDA, MPS, and TensorFlow Metal are opt-in accelerators.
- Segmentation runs require a trained `samstars` model bundle.
- The intended workflow is to train a model bundle for your data and then use it for segmentation.
