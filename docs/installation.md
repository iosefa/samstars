# Installation

## Choose an install path

Use one of these:

1. **PyPI install**: install the library and segmentation stack with pip.
2. **GitHub/source install**: install from this repository.
3. **Developer/reproducible setup**: use the pinned conda environment files.

`samstars` uses geospatial and ML dependencies.
Many of them have pip wheels on common platforms, but Conda is still the smoothest path when you need a fully reproducible environment or platform-specific GPU stacks.

## Option 1: pip install the library

```bash
pip install samstars
```

This installs the full segmentation stack, including the SAM proposal runtime and the StarDist refinement runtime.

It does not bundle trained model checkpoints or StarDist model weights.
The expected workflow is to train your own models for your data and then supply those model paths at run time.
See [Training](training.md) and [Segmentation](segmentation.md).

On Apple Silicon, the TensorFlow package selection is handled automatically by platform markers.
On Linux/NVIDIA, pip will typically install CPU PyTorch wheels by default; if you want a CUDA-specific stack, the Conda environment remains the simpler path.

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

Editable development install with docs tooling:

```bash
pip install -e ".[docs]"
```

Verify:

```bash
python -c "import samstars; print('samstars import ok')"
pip show samstars
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
- The pip install does not try to select CUDA-specific PyTorch wheels automatically.

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
