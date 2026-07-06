# Installation

## Install the library

```bash
pip install samstars
```

`samstars` uses geospatial and machine-learning dependencies. On most supported Python environments, pip will install the required Python packages directly.

If a native or platform-specific dependency does not install cleanly on your system, install that dependency using the package manager recommended for your platform and then rerun `pip install samstars`.

`samstars` does not bundle trained model files.
See [Training](training.md) and [Segmentation](segmentation.md).

Verify:

```bash
python -c "import samstars; print('samstars import ok')"
pip show samstars
```

## CUDA and GPU notes

- For NVIDIA acceleration on Ubuntu/Linux, you need a working NVIDIA driver on the host.
- GPU use may require additional platform-specific setup for PyTorch and TensorFlow.
- CPU execution is supported for the documented workflows and is the safest installation target when GPU packages are difficult to resolve.

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
