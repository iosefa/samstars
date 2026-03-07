# Installation

## Choose an install path

Use one of these:

1. **Recommended**: create a conda base environment (`python`, `gdal`, `pdal`) and install `samstars` from PyPI with pip.
2. **GitHub**: install latest source version.
3. **Developer/reproducible setup**: use the pinned conda environment files.

`samstars` uses geospatial and ML dependencies (`gdal`, `pdal`, and GPU/CPU runtime libraries), so conda is usually the smoothest setup.

## Option 1 (Recommended): conda base env + pip install

Create and activate an environment with native geospatial dependencies:

```bash
conda create -n samstars -c conda-forge python=3.10 gdal pdal
conda activate samstars
```

Install the package:

```bash
pip install samstars
```

Verify:

```bash
python -c "import samstars; print('samstars import ok')"
pip show samstars
```

## Option 2: install from GitHub

Latest source:

```bash
pip install git+https://github.com/iosefa/sam.git
```

Editable development install:

```bash
git clone https://github.com/iosefa/sam.git
cd sam
pip install -e .
```

## Option 3 (Developer/Reproducible): pinned conda environment files

Use these when you want the exact repo development environment.

### Ubuntu (CUDA + SAM + StarDist)

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
- Some environments run fully on GPU, while some steps may need CPU fallback depending on card/runtime compatibility.
- The provided Ubuntu environment file includes CUDA-enabled packages used by this project.

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

## Serve documentation

From repo root:

```bash
conda activate samstars
mkdocs serve
```

Open `http://127.0.0.1:8000`.
