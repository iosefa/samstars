#!/usr/bin/env python3
"""
predict_stardist.py  –  use an existing StarDist-2D model to segment a
canopy-index raster and write the labelled crowns as a GeoTIFF.

Expected model files
────────────────────
models/
└── stardist/
    ├── config.json
    ├── weights_best.h5      (or weights_last.h5, etc.)
    └── ...

The script will raise an error if they’re missing (so it can’t silently
re-train).

Outputs
───────
labels_pred.tif   –  uint16 label image, same CRS/transform as input
preview.png       –  quick side-by-side PNG (index & labels)
"""

from __future__ import annotations
from pathlib import Path
import argparse

import rasterio
import numpy as np
from tifffile import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist import random_label_cmap

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
matplotlib.rcParams["image.interpolation"] = "none"

LABEL_CMAP = random_label_cmap()


def plot_img_label(img, lbl, outfile: Path):
    fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5),
                                 gridspec_kw=dict(width_ratios=(1.25, 1)))
    im = ai.imshow(img, cmap="gray", vmin=0, vmax=1)
    ai.set_title("canopy index (norm.)")
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=LABEL_CMAP)
    al.set_title("StarDist labels")
    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--model-dir", default="models", help="folder containing the model sub-dir")
    pa.add_argument("--model-name", default="stardist", help="sub-directory with config/weights")
    pa.add_argument("--input", required=True, help="canopy_index.tif to segment")
    pa.add_argument("--aux", default=None, help="optional aux raster (e.g., CHM) for multi-channel models")
    pa.add_argument("--out-tif",  default="labels_pred.tif", help="output GeoTIFF label image")
    pa.add_argument("--out-png",  default="preview.png",    help="optional quick-look PNG")
    pa.add_argument("--prob-thr", type=float, default=0.15, help="StarDist prob_thresh")
    pa.add_argument("--nms-thr",  type=float, default=0.20, help="StarDist nms_thresh")
    args = pa.parse_args()

    model_path = Path(args.model_dir) / args.model_name
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"✗ pretrained model not found under {model_path}\n"
            "  → train once with your original script to create it."
        )

    model = StarDist2D(None, name=args.model_name, basedir=args.model_dir)
    n_channel_in = int(getattr(model.config, "n_channel_in", 1))

    img = imread(args.input)
    if img.ndim > 2:
        img = img[..., 0]
    img = img.astype("float32")

    if n_channel_in == 1:
        img_norm = normalize(img, 1, 99.8, axis=(0, 1))[..., None]
    elif n_channel_in == 2:
        if not args.aux:
            raise ValueError("This StarDist model expects 2 channels; pass --aux (e.g., chm.tif).")
        aux = imread(args.aux)
        if aux.ndim > 2:
            aux = aux[..., 0]
        aux = aux.astype("float32")

        h = min(img.shape[0], aux.shape[0])
        w = min(img.shape[1], aux.shape[1])
        x = np.stack([img[:h, :w], aux[:h, :w]], axis=-1).astype("float32")
        img_norm = normalize(x, 1, 99.8, axis=(0, 1)).astype("float32")
    else:
        raise ValueError(f"Unsupported n_channel_in={n_channel_in} for helpers/predict_stars.py")

    labels, _ = model.predict_instances(
        img_norm,
        n_tiles=model._guess_n_tiles(img_norm),
        prob_thresh=args.prob_thr,
        nms_thresh=args.nms_thr,
        show_tile_progress=False,
    )

    with rasterio.open(args.input) as src:
        meta = src.meta.copy()
        # If we cropped to match aux size, trim labels back to input size via padding.
        if labels.shape != (src.height, src.width):
            out = np.zeros((src.height, src.width), dtype=labels.dtype)
            hh = min(src.height, labels.shape[0])
            ww = min(src.width, labels.shape[1])
            out[:hh, :ww] = labels[:hh, :ww]
            labels = out
        meta.update(count=1, dtype="uint16", compress="lzw")
        with rasterio.open(args.out_tif, "w", **meta) as dst:
            dst.write(labels.astype("uint16"), 1)

    plot_img_label(img_norm[..., 0], labels, Path(args.out_png))


if __name__ == "__main__":
    main()
