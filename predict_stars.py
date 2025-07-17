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

    img = imread(args.input)
    img_norm = normalize(img, 1, 99.8, axis=(0, 1))[..., None]

    labels, _ = model.predict_instances(
        img_norm,
        n_tiles=model._guess_n_tiles(img_norm),
        prob_thresh=args.prob_thr,
        nms_thresh=args.nms_thr,
        show_tile_progress=False,
    )

    with rasterio.open(args.input) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint16", compress="lzw")
        with rasterio.open(args.out_tif, "w", **meta) as dst:
            dst.write(labels.astype("uint16"), 1)

    plot_img_label(img_norm[..., 0], labels, Path(args.out_png))


if __name__ == "__main__":
    main()
