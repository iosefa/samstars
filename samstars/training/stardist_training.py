#!/usr/bin/env python3
"""
Train Keras StarDist2D on per-chip inputs derived from proposal outputs.

This version restores the *original* StarDist training behavior that worked well:
- Use canopy index (default: record key "index") rather than raw SAM prob
- Normalize inputs with csbdeep.normalize(x, 1, 99.8) (per-channel if multi-channel)
- Use the original augmentation (flip/rot + intensity jitter + Gaussian noise)

You can still train on raw SAM prob by passing --input-key file_name.

If --use-aux is set, the network input becomes [primary, chm01].
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


# ───────────── original StarDist augmenter ──────────────────────────────────
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

    axis_norm = (0, 1)  # normalize channels independently (original)

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


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser()
    pa.add_argument("--train-records", type=Path, required=True, help="Training proposal-record JSON with masks")
    pa.add_argument("--val-records", type=Path, required=True, help="Validation proposal-record JSON with masks")
    pa.add_argument("--out-dir", type=Path, required=True)
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--min-iters", type=int, default=0, help="Minimum iterations (converted to epochs)")
    pa.add_argument("--auto-epochs-ref-size", type=str, default=None, help="Reference size for auto-epochs (e.g., 0100)")
    pa.add_argument("--auto-epochs-ref-epochs", type=float, default=None, help="Reference epochs for auto-epochs")
    pa.add_argument(
        "--auto-epochs-records-root",
        type=Path,
        default=None,
        help="Root containing <size>/train_records.json (defaults to train-records parent)",
    )

    # Original-matching knobs
    pa.add_argument(
        "--input-key",
        type=str,
        default="index",
        help="Proposal-record key to use as StarDist input (default: index). Use file_name for raw SAM prob.",
    )
    pa.add_argument("--normalize", action="store_true", help="Enable csbdeep percentile normalization")
    pa.add_argument("--no-normalize", action="store_true", help="Disable csbdeep percentile normalization (original had normalization ON).")
    pa.add_argument("--no-augment", action="store_true", help="Disable augmentation (original had augmentation ON).")

    # Aux channels
    pa.add_argument("--scale-stats", type=Path, default=None, help="Global scaling stats JSON for CHM/COST")
    pa.add_argument("--use-aux", action="store_true", help="Use CHM aux channel")
    pa.add_argument("--chm-key", type=str, default="chm", help="Proposal-record key for CHM path")
    args = pa.parse_args()

    scale_stats = load_scale_stats(args.scale_stats) if args.scale_stats else None
    if args.use_aux and scale_stats is None:
        raise SystemExit("--scale-stats is required when --use-aux is set")

    if args.normalize and args.no_normalize:
        raise SystemExit("Use only one of --normalize or --no-normalize.")
    if args.normalize:
        do_normalize = True
    elif args.no_normalize:
        do_normalize = False
    else:
        do_normalize = True

    Xtr, Ytr = load_pairs(
        args.train_records,
        input_key=args.input_key,
        use_aux=args.use_aux,
        scale_stats=scale_stats,
        chm_key=args.chm_key,
        do_normalize=do_normalize,
    )
    Xval, Yval = load_pairs(
        args.val_records,
        input_key=args.input_key,
        use_aux=args.use_aux,
        scale_stats=scale_stats,
        chm_key=args.chm_key,
        do_normalize=do_normalize,
    )

    Ytr = [fill_label_holes(y) for y in Ytr]
    Yval = [fill_label_holes(y) for y in Yval]

    if not Xtr or not Ytr:
        raise RuntimeError("No training data with masks found.")

    iters_per_epoch = max(1, len(Xtr))
    if args.auto_epochs_ref_size and args.auto_epochs_ref_epochs:
        records_root = args.auto_epochs_records_root
        if records_root is None:
            records_root = args.train_records.parent.parent
        ref_records = records_root / args.auto_epochs_ref_size / "train_records.json"
        if not ref_records.exists():
            raise FileNotFoundError(f"Missing ref train_records.json: {ref_records}")
        ref_n = count_pairs(
            ref_records,
            input_key=args.input_key,
            use_aux=args.use_aux,
            chm_key=args.chm_key,
        )
        cur_n = len(Xtr)
        if ref_n <= 0 or cur_n <= 0:
            raise RuntimeError("Auto-epochs requires non-zero training sample counts.")
        target_steps = ref_n * float(args.auto_epochs_ref_epochs)
        auto_epochs = int(np.ceil(target_steps / max(cur_n, 1)))
        args.epochs = max(args.epochs, auto_epochs)
        print(f"auto-epochs -> {auto_epochs} (ref_n={ref_n}, cur_n={cur_n})")
    if args.min_iters > 0:
        min_epochs = int((args.min_iters + iters_per_epoch - 1) // iters_per_epoch)
        args.epochs = max(args.epochs, min_epochs)
    print(f"epochs={args.epochs} (iters/epoch={iters_per_epoch})")

    n_channels = Xtr[0].shape[-1]
    conf = Config2D(n_rays=32, grid=(2, 2), use_gpu=False, n_channel_in=n_channels)
    model = StarDist2D(conf, name="stardist", basedir=str(args.out_dir))

    val_data = (Xval, Yval) if (len(Xval) > 0 and len(Yval) > 0) else None
    aug = None if args.no_augment else augmenter
    model.train(Xtr, Ytr, validation_data=val_data, epochs=args.epochs, augmenter=aug)

    if val_data is not None:
        try:
            opt = model.optimize_thresholds(Xval, Yval)
            print(f"✓ optimized thresholds: prob={opt.get('prob'):.3f}, nms={opt.get('nms'):.3f}")
        except Exception as exc:
            print(f"[warn] threshold optimization failed: {exc}")

    print(f"✓ trained StarDist -> {args.out_dir}")


if __name__ == "__main__":
    main()
