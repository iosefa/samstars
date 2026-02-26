# Ubuntu Setup (RTX Desktop)

This guide sets up the two environments used by this repo on Ubuntu and verifies NVIDIA GPU access before running wrappers.

## 1) Create Conda Environments

From repo root:

```bash
conda env create -f docs/environment.ubuntu.sam_star.yml
conda env create -f docs/environment.ubuntu.stardist-env.yml
```

## 2) Verify NVIDIA Driver + GPU

On Ubuntu host:

```bash
nvidia-smi
```

Expected: your GPU (for example RTX 5090) is listed and driver is loaded.

## 3) Verify `sam_star` CUDA Access (PyTorch)

```bash
conda activate sam_star
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
PY
```

Expected: `cuda available: True`.

## 4) Verify `stardist-env` TensorFlow GPU

```bash
conda activate stardist-env
python - <<'PY'
import tensorflow as tf
print("tensorflow:", tf.__version__)
print("physical GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

If no GPU is shown, StarDist will still run on CPU, but slower.

## 5) Wrapper Calls

Use absolute paths for large raster inputs.

### A) End-to-end wrapper (tile + SAM + StarDist)

```bash
conda activate sam_star
bash scripts/run_sam_star_pipeline.sh \
  --img /abs/path/image_8band.tif \
  --chm /abs/path/chm.tif \
  --den /abs/path/first_return_density.tif \
  --run-dir /abs/path/output_run_dir
```

### B) Two-stage run (SAM first, StarDist later)

Step 1: SAM only

```bash
conda activate sam_star
bash scripts/run_sam_star_pipeline.sh \
  --img /abs/path/image_8band.tif \
  --chm /abs/path/chm.tif \
  --den /abs/path/first_return_density.tif \
  --run-dir /abs/path/output_run_dir \
  --skip-stardist
```

Step 2: StarDist only (reuse `run-dir/tiles`)

```bash
conda activate stardist-env
bash scripts/run_stardist_on_tiles.sh \
  --run-dir /abs/path/output_run_dir
```

Optional polygons in step 2:

```bash
bash scripts/run_stardist_on_tiles.sh \
  --run-dir /abs/path/output_run_dir \
  --write-polygons
```

### C) Manifest wrapper (recommended for explicit SAM device control)

This wrapper defaults SAM device to `mps`, so on Ubuntu/NVIDIA pass `--device cuda`.

```bash
conda activate sam_star
bash scripts/run_manifest_sam_stardist.sh \
  --tiles-dir /abs/path/output_run_dir/tiles \
  --out-dir /abs/path/output_manifest_run \
  --device cuda
```

SAM-only or StarDist-only with manifest wrapper:

```bash
# SAM only
bash scripts/run_manifest_sam_stardist.sh \
  --tiles-dir /abs/path/output_run_dir/tiles \
  --out-dir /abs/path/output_manifest_run \
  --device cuda \
  --sam-only

# StarDist only (assumes SAM outputs already exist in out-dir/sam_probs)
bash scripts/run_manifest_sam_stardist.sh \
  --tiles-dir /abs/path/output_run_dir/tiles \
  --out-dir /abs/path/output_manifest_run \
  --stardist-only
```

## 6) Notes

- SAM checkpoints (`.pth`) are not git-tracked; copy them to expected paths before running.
- StarDist model weights (`weights_best.h5` / `weights_last.h5`) are expected under your chosen model dir.
- Wrapper help:

```bash
bash scripts/run_sam_star_pipeline.sh --help
bash scripts/run_stardist_on_tiles.sh --help
bash scripts/run_manifest_sam_stardist.sh --help
```
