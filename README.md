# samstars

[![PyPI](https://img.shields.io/pypi/v/samstars.svg)](https://pypi.org/project/samstars/)
[![Python](https://img.shields.io/pypi/pyversions/samstars.svg)](https://pypi.org/project/samstars/)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/samstars/ci.yml?branch=main&label=tests)](https://github.com/iosefa/samstars/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/iosefa/samstars/docs.yml?branch=main&label=docs)](https://samstars.sefa.ai/)
[![Contributors](https://img.shields.io/github/contributors/iosefa/samstars.svg?label=contributors)](https://github.com/iosefa/samstars/graphs/contributors)
[![PyPI Downloads](https://static.pepy.tech/badge/samstars)](https://pepy.tech/projects/samstars)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`samstars` is a Python library for instance segmentation of individual trees
using high-resolution imagery and lidar-derived rasters.

Read the full documentation at [samstars.sefa.ai](https://samstars.sefa.ai).

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
See the [training](https://samstars.sefa.ai/training/) and
[segmentation](https://samstars.sefa.ai/segmentation/) guides to train a model
bundle and run segmentation.

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
