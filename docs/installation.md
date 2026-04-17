# Installation

## Choose an install path

Use one of these:

1. **PyPI install**: install the library with pip.
2. **GitHub/source install**: install from this repository.
3. **Developer/reproducible setup**: use the pinned conda environment files.

`samstars` uses geospatial and ML dependencies.
Depending on your platform, some of these may need to be installed separately.
If you want the most predictable setup, use the Conda environment files.

## Option 1: pip install the library

```bash
pip install samstars
```

Use this when your Python environment can resolve the required dependencies cleanly.

If some native or platform-specific dependencies do not install with pip on your system, install those separately or use the Conda environment files instead.

`samstars` does not bundle trained model files.
See [Training](training.md) and [Segmentation](segmentation.md).

## Option 2: install from GitHub or from a local clone

Latest source from GitHub:

```bash
pip install "git+https://github.com/iosefa/sam.git"
```

Editable development install from a local clone:

```bash
git clone https://github.com/iosefa/sam.git
cd sam
pip install -e .
```

Verify:

```bash
python -c "import samstars; print('samstars import ok')"
pip show samstars
```

## Option 3 (Developer/Reproducible): pinned conda environment files

Use these when you want the repo development environment with the project dependencies pinned.

### Ubuntu

```bash
mamba env create -f environment.ubuntu.samstars.yml
conda activate samstars
```

### macOS

```bash
mamba env create -f environment.macos.samstars.yml
conda activate samstars
```

Update an existing environment:

```bash
mamba env update -n samstars -f environment.ubuntu.samstars.yml
```

## CUDA and GPU notes

- For NVIDIA acceleration on Ubuntu/Linux, you need a working NVIDIA driver on the host.
- The provided Ubuntu environment file includes the CUDA-enabled packages used by this project.
- A pip install may still require additional platform-specific setup for GPU use.

Check CUDA availability:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

- On some newer NVIDIA GPUs, specific kernels may be unstable due to runtime/PTX compatibility.
- If that happens, run the affected step on CPU (`CUDA_VISIBLE_DEVICES=""`), which is supported by the example run commands.

Check visible accelerators in the environment:

```bash
python - <<'PY'
import tensorflow as tf
print("tf:", tf.__version__)
print("built with cuda:", tf.test.is_built_with_cuda())
print("visible GPUs:", tf.config.list_physical_devices("GPU"))
PY
```
