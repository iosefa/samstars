#!/usr/bin/env python3
"""Train polygon refinement from proposal rasters and crown instance masks.

The default primary input is the fused canopy index. Optional normalized CHM
adds a second channel. Training supports per-channel normalization and geometric
and intensity augmentation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from csbdeep.utils import normalize
from stardist.models import Config2D, StarDist2D
from stardist import fill_label_holes

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from samstars.data.scaling import load_scale_stats, scale_to_01


def _align_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Internal helper function."""
    arr = np.asarray(arr)
    th, tw = target_shape
    arr = arr[:th, :tw]
    ah, aw = arr.shape
    if ah < th or aw < tw:
        pad_h = max(0, th - ah)
        pad_w = max(0, tw - aw)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=0)
    return arr


def _scale_aux(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Internal helper function."""
    return scale_to_01(arr, lo, hi)


# ───────────── refinement augmentation ──────────────────────────────────
def random_fliprot(img: np.ndarray, mask: np.ndarray):
    """Random fliprot."""
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img: np.ndarray):
    """Random intensity change."""
    img = img * np.random.uniform(0.6, 2.0) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x: np.ndarray, y: np.ndarray):
    """Augmenter."""
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y
# ───────────────────────────────────────────────────────────────────────────


def _read_float_raster(path: str | Path) -> np.ndarray:
    """Internal helper function."""
    path = str(path)
    if path.endswith(".tif") or path.endswith(".tiff"):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    return np.load(path).astype(np.float32)


def _read_u16_mask(path: str | Path) -> np.ndarray:
    """Internal helper function."""
    path = str(path)
    if path.endswith(".tif") or path.endswith(".tiff"):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.uint16)
    return np.load(path).astype(np.uint16)


def load_pairs(
    records_path: Path,
    *,
    input_key: str = "index",
    use_aux: bool = False,
    scale_stats: dict | None = None,
    chm_key: str = "chm",
    do_normalize: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load pairs."""
    data = json.load(open(records_path))
    X, Y = [], []
    skipped = 0

    chm_lo = chm_hi = None
    if scale_stats:
        chm_lo = float(scale_stats.get("chm_01", {}).get("lo", 0.0))
        chm_hi = float(scale_stats.get("chm_01", {}).get("hi", 1.0))

    axis_norm = (0, 1)  # normalize channels independently

    for rec in data:
        fname = rec.get(input_key, "")
        mpath = rec.get("mask", "")
        if not fname or not mpath or not Path(mpath).exists():
            continue
        if not Path(fname).exists():
            skipped += 1
            continue

        x = _read_float_raster(fname)
        y = _read_u16_mask(mpath)

        # Align shapes to mask
        x = _align_to_shape(x, y.shape)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if use_aux:
            chm_path = rec.get(chm_key, "")
            if not chm_path:
                skipped += 1
                continue
            if not Path(chm_path).exists():
                skipped += 1
                continue

            chm = _read_float_raster(chm_path)
            chm = _align_to_shape(chm, y.shape)

            # scale aux to ~[0,1] using global stats (still useful even if we normalize later)
            if scale_stats is not None:
                chm = _scale_aux(chm, chm_lo, chm_hi)

            x = np.stack([x, chm.astype(np.float32)], axis=-1).astype(np.float32)
        else:
            x = x[..., None].astype(np.float32)

        if do_normalize:
            x = normalize(x, 1, 99.8, axis=axis_norm).astype(np.float32)

        X.append(x)
        Y.append(y)

    if skipped:
        print(f"skipped {skipped} pairs (missing inputs/aux)")
    return X, Y


def count_pairs(
    records_path: Path,
    *,
    input_key: str = "index",
    use_aux: bool = False,
    chm_key: str = "chm",
) -> int:
    """Count pairs."""
    data = json.load(open(records_path))
    count = 0
    for rec in data:
        fname = rec.get(input_key, "")
        mpath = rec.get("mask", "")
        if not fname or not mpath or not Path(mpath).exists():
            continue
        if not Path(fname).exists():
            continue
        if use_aux:
            chm_path = rec.get(chm_key, "")
            if not chm_path or not Path(chm_path).exists():
                continue
        count += 1
    return count


def train_stardist_model(
    *,
    train_records: Path,
    val_records: Path,
    out_dir: Path,
    model_name: str = "stardist",
    epochs: int = 50,
    min_iters: int = 0,
    input_key: str = "index",
    use_aux: bool = False,
    scale_stats_path: Path | None = None,
    chm_key: str = "chm",
    normalize_inputs: bool = True,
    augment: bool = True,
) -> None:
    """Train a StarDist model from train/val proposal-record files."""
    scale_stats = load_scale_stats(scale_stats_path) if scale_stats_path else None
    if use_aux and scale_stats is None:
        raise ValueError("Provide scale_stats_path when use_aux=True.")

    Xtr, Ytr = load_pairs(
        train_records,
        input_key=input_key,
        use_aux=use_aux,
        scale_stats=scale_stats,
        chm_key=chm_key,
        do_normalize=normalize_inputs,
    )
    Xval, Yval = load_pairs(
        val_records,
        input_key=input_key,
        use_aux=use_aux,
        scale_stats=scale_stats,
        chm_key=chm_key,
        do_normalize=normalize_inputs,
    )

    Ytr = [fill_label_holes(y) for y in Ytr]
    Yval = [fill_label_holes(y) for y in Yval]

    if not Xtr or not Ytr:
        raise RuntimeError("No training data with masks found.")

    iters_per_epoch = max(1, len(Xtr))
    if min_iters > 0:
        min_epochs = int((min_iters + iters_per_epoch - 1) // iters_per_epoch)
        epochs = max(epochs, min_epochs)
    print(f"epochs={epochs} (iters/epoch={iters_per_epoch})")

    n_channels = Xtr[0].shape[-1]
    conf = Config2D(n_rays=32, grid=(2, 2), use_gpu=False, n_channel_in=n_channels)
    model = StarDist2D(conf, name=model_name, basedir=str(out_dir))

    val_data = (Xval, Yval) if (len(Xval) > 0 and len(Yval) > 0) else None
    aug = augmenter if augment else None
    if val_data is None:
        print("no validation records found; reusing training data as validation_data")
        val_data = (Xtr, Ytr)

    model.train(
        Xtr,
        Ytr,
        validation_data=val_data,
        epochs=epochs,
        augmenter=aug,
    )

    if val_data is not None:
        try:
            opt = model.optimize_thresholds(Xval, Yval)
            print(f"✓ optimized thresholds: prob={opt.get('prob'):.3f}, nms={opt.get('nms'):.3f}")
        except Exception as exc:
            print(f"[warn] threshold optimization failed: {exc}")

    print(f"✓ trained refinement model -> {out_dir / model_name}")


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Train the polygon-refinement model from proposal-record files.")
    pa.add_argument("--train-records", type=Path, required=True, help="Training proposal-record JSON with masks")
    pa.add_argument("--val-records", type=Path, required=True, help="Validation proposal-record JSON with masks")
    pa.add_argument("--out-dir", type=Path, required=True, help="Output directory for the trained refinement model")
    pa.add_argument("--model-name", type=str, default="stardist", help="Model subdir name to create under --out-dir")
    pa.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    pa.add_argument("--min-iters", type=int, default=0, help="Minimum training iterations (converted to epochs)")

    # Original-matching knobs
    pa.add_argument(
        "--input-key",
        type=str,
        default="index",
        help="Proposal-record key to use as StarDist input (default: index). Use file_name for raw SAM prob.",
    )
    pa.add_argument("--normalize", action="store_true", help="Enable csbdeep percentile normalization")
    pa.add_argument("--no-normalize", action="store_true", help="Disable csbdeep percentile normalization")
    pa.add_argument("--no-augment", action="store_true", help="Disable training augmentation.")

    # Aux channels
    pa.add_argument("--scale-stats", type=Path, default=None, help="CHM scaling stats JSON; required with --use-aux")
    pa.add_argument("--use-aux", action="store_true", help="Add CHM as an auxiliary input channel")
    pa.add_argument("--chm-key", type=str, default="chm", help="Proposal-record key for CHM path")
    args = pa.parse_args()

    if args.normalize and args.no_normalize:
        raise SystemExit("Use only one of --normalize or --no-normalize.")
    if args.normalize:
        do_normalize = True
    elif args.no_normalize:
        do_normalize = False
    else:
        do_normalize = True

    train_stardist_model(
        train_records=args.train_records,
        val_records=args.val_records,
        out_dir=args.out_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        min_iters=args.min_iters,
        input_key=args.input_key,
        use_aux=args.use_aux,
        scale_stats_path=args.scale_stats,
        chm_key=args.chm_key,
        normalize_inputs=do_normalize,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
