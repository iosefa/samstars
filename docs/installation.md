# Installation

## Supported environments

The initial `samstars` release supports Python 3.10 through 3.12 on these
64-bit platforms:

| Platform | Baseline | Optional acceleration |
| --- | --- | --- |
| Ubuntu 20.04 or newer on x86-64 | CPU | NVIDIA CUDA |
| macOS 14 or newer on Apple Silicon | CPU | PyTorch MPS and TensorFlow Metal |

CPU execution is the supported baseline and is used by the examples unless
stated otherwise. Windows, Linux ARM64, and Intel macOS are not currently
tested release targets.

## Install the library from PyPI

```bash
python -m pip install samstars
```

For optional GPU support, see [NVIDIA CUDA on Linux](#nvidia-cuda-on-linux)
or [Apple Silicon acceleration](#apple-silicon-acceleration).

On Linux, use PyTorch's CPU-only package index when you do not want pip to
install NVIDIA runtime libraries:

```bash
python -m pip install "torch>=2.3,<3" "torchvision>=0.18,<1" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install samstars
```

Verify:

```bash
python -c "import samstars, torch, tensorflow as tf; print('samstars import ok'); print('torch', torch.__version__); print('tensorflow', tf.__version__)"
pip show samstars
```

## NVIDIA CUDA on Linux

CUDA builds of PyTorch and Torchvision must be installed as a matched pair.
Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
to install a pair within the supported ranges (`torch>=2.3,<3` and
`torchvision>=0.18,<1`), then install `samstars`:

```bash
# Run the command produced by the PyTorch selector first.
python -m pip install "tensorflow[and-cuda]>=2.15,<2.19"
python -m pip install samstars
```

The TensorFlow command follows the
[official TensorFlow pip guide](https://www.tensorflow.org/install/pip) and
requires a compatible NVIDIA driver. If TensorFlow GPU support is not needed,
omit that command and let `samstars` install standard TensorFlow.
PyTorch and TensorFlow detect accelerators independently: `device="cuda"`
controls proposal generation, while TensorFlow controls refinement. Set
`stardist_cpu=True` when only proposal generation should use CUDA.

Check CUDA availability:

```bash
python - <<'PY'
import torch
import tensorflow as tf

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("tensorflow:", tf.__version__)
print("tensorflow GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

Use `device="cuda"` only when `torch.cuda.is_available()` returns `True`.

## Apple Silicon acceleration

The standard Apple Silicon PyTorch build can use MPS when it is available.
TensorFlow Metal support is optional; see Apple's
[TensorFlow acceleration guide](https://developer.apple.com/metal/tensorflow-plugin/)
for host requirements:

```bash
python -m pip install "samstars[metal]"
```

Check both frameworks after installation:

```bash
python - <<'PY'
import torch
import tensorflow as tf

print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
print("tensorflow:", tf.__version__)
print("tensorflow GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

Use `device="mps"` only when `torch.backends.mps.is_available()` returns
`True`. Otherwise, keep `device="cpu"`. TensorFlow Metal availability does not
change the PyTorch device setting.

## Conda development environments

The repository environment files provide platform-specific Python 3.10 CPU
development baselines. From the repository root, use the file for your
platform and then install the checkout in editable mode:

```bash
# Ubuntu x86-64
conda env create -f environment.ubuntu.samstars.yml
conda activate samstars
python -m pip install -e .
```

```bash
# Apple Silicon macOS
conda env create -f environment.macos.samstars.yml
conda activate samstars
python -m pip install -e .
```

For Apple Silicon development with TensorFlow Metal, replace the final command
with `python -m pip install -e ".[metal]"`. The Ubuntu environment pins CPU
builds of PyTorch and Torchvision. For CUDA development, uninstall those two
packages after creating the environment, run the matched installation command
from the PyTorch selector, and then install the checkout:

```bash
python -m pip uninstall -y torch torchvision
# Run the matched CUDA command produced by the PyTorch selector.
python -m pip install -e .
```

Verify that `torch.version.cuda` is non-empty and
`torch.cuda.is_available()` is `True` before selecting `device="cuda"`.
