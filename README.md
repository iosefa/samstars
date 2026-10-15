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

![SAM-STARS graphical abstract showing satellite imagery and lidar inputs, automatic crown prompts, canopy fusion, individual crown delineation, species classification, and annotation efficiency.](docs/assets/figures/sam-stars-graphical-abstract.png)

Read the full documentation at [samstars.sefa.ai](https://samstars.sefa.ai).

## Installation

Install from PyPI with Python 3.10–3.12:

```bash
python -m pip install samstars
```

See the [installation guide](https://samstars.sefa.ai/installation/) for
supported platforms and optional GPU setup.

## Developer setup

From a local clone:

```bash
python -m pip install -e .
```

To run the tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

For documentation development:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Conda setup is covered in the
[installation guide](docs/installation.md#conda-development-environments).
