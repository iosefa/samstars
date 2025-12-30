#!/usr/bin/env python3
"""
Train Keras StarDist2D on smoothed SAM probability rasters (npys) with mask labels.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import rasterio
from stardist.models import Config2D, StarDist2D
from stardist import fill_label_holes
from csbdeep.utils import normalize
from tqdm import tqdm


def load_pairs(manifest):
    data = json.load(open(manifest))
    X, Y = [], []
    skipped = 0
    for rec in data:
        fname = rec.get("file_name", "")
        mpath = rec.get("mask", "")
        if not fname or not mpath or not Path(mpath).exists():
            continue
        # load prob
        if fname.endswith(".tif"):
            with rasterio.open(fname) as src:
                x = src.read(1).astype(np.float32)
        else:
            x = np.load(fname).astype(np.float32)
        # load mask if present
        if mpath.endswith(".tif"):
            with rasterio.open(mpath) as src:
                y = src.read(1).astype(np.uint16)
        else:
            y = np.load(mpath).astype(np.uint16)
        if x.shape != y.shape:
            skipped += 1
            continue
        X.append(x)
        Y.append(y)
    if skipped:
        print(f"skipped {skipped} pairs with shape mismatch")
    return X, Y


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--train-probs", type=Path, required=True)
    pa.add_argument("--val-probs", type=Path, required=True)
    pa.add_argument("--out-dir", type=Path, required=True)
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--min-iters", type=int, default=0, help="Minimum iterations (converted to epochs)")
    args = pa.parse_args()

    Xtr, Ytr = load_pairs(args.train_probs)
    Xval, Yval = load_pairs(args.val_probs)

    axis_norm = (0, 1)
    Xtr = [normalize(x, 1, 99.8, axis=axis_norm)[..., None] for x in tqdm(Xtr, desc="norm train")]
    Xval = [normalize(x, 1, 99.8, axis=axis_norm)[..., None] for x in tqdm(Xval, desc="norm val")]
    Ytr = [fill_label_holes(y) for y in Ytr]
    Yval = [fill_label_holes(y) for y in Yval]

    if not Xtr or not Ytr:
        raise RuntimeError("No training data with masks found.")

    iters_per_epoch = max(1, len(Xtr))
    if args.min_iters > 0:
        min_epochs = int((args.min_iters + iters_per_epoch - 1) // iters_per_epoch)
        args.epochs = max(args.epochs, min_epochs)
    print(f"epochs={args.epochs} (iters/epoch={iters_per_epoch})")

    conf = Config2D(n_rays=32, grid=(2, 2), use_gpu=False, n_channel_in=1)
    model = StarDist2D(conf, name="stardist", basedir=str(args.out_dir))
    val_data = (Xval, Yval) if (len(Xval) > 0 and len(Yval) > 0) else None
    model.train(Xtr, Ytr, validation_data=val_data, epochs=args.epochs)
    print(f"✓ trained StarDist -> {args.out_dir}")


if __name__ == "__main__":
    main()
