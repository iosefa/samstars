#!/usr/bin/env python3
"""
End-to-end training driver:
 1) Train SAM decoder (optional)
 2) Run SAM over tiles to produce crown_probability.tif
 3) Rasterize crown polygons to instance masks aligned to each probability raster
 4) Train StarDist on those SAM probability rasters
"""
from __future__ import annotations
import argparse, subprocess, sys, shutil
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from tifffile import imread
from csbdeep.utils import normalize
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D
from stardist.matching import matching

LABEL_CMAP = random_label_cmap()
ROOT = Path(__file__).resolve().parent.parent


# ─── utilities ───────────────────────────────────────────────────────────────
def run(cmd: list[str]) -> None:
    """Run a subprocess command, raising on failure."""
    print(" ", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def list_tiles(tiles_dir: Path) -> list[Path]:
    """Return sorted list of tile directories under tiles_dir."""
    return sorted(p for p in tiles_dir.iterdir() if p.is_dir())


def rasterize_crowns(crowns_gpkg: Path, reference_tif: Path, out_path: Path) -> int:
    """Rasterize crowns to match reference_tif grid; return crown count."""
    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint16", compress="lzw")
        transform, shape, crs = src.transform, (src.height, src.width), src.crs

    crowns = gpd.read_file(crowns_gpkg).to_crs(crs)
    crowns = crowns[crowns.geometry.notnull() & crowns.geometry.is_valid]
    if crowns.empty:
        raise RuntimeError(f"No crowns found in {crowns_gpkg}")

    shapes = [(geom, idx + 1) for idx, geom in enumerate(crowns.geometry)]
    mask = rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint16")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype("uint16"), 1)
    return len(crowns)


# ─── stardist training helpers ───────────────────────────────────────────────
def random_fliprot(img, mask):
    """Randomly permute axes and flip image/mask together."""
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


def random_intensity_change(img):
    """Random brightness/contrast jitter."""
    return img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)


def augmenter(x, y):
    """Augmentation pipeline for StarDist training."""
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


def train_stardist(x_dir: Path, y_dir: Path, model_dir: Path, model_name: str, val_frac=0.15):
    """Train StarDist on prepared X/Y TIFFs and report quick validation stats."""
    x_files = sorted(x_dir.glob("*.tif"))
    y_files = sorted(y_dir.glob("*.tif"))
    assert len(x_files) == len(y_files) and all(f.stem == y_files[i].stem for i, f in enumerate(x_files)), \
        "X/Y files must align by stem"
    assert len(x_files) > 1, "not enough training data"

    axis_norm = (0, 1)
    X = [normalize(imread(x), 1, 99.8, axis=axis_norm) for x in x_files]
    Y = [fill_label_holes(imread(y)) for y in y_files]

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(val_frac * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print(f"StarDist dataset: {len(X_trn)} train / {len(X_val)} val")

    n_channel = 1 if X_trn[0].ndim == 2 else X_trn[0].shape[-1]
    n_rays = 32
    use_gpu = False and gputools_available()
    grid = (2, 2)

    conf = Config2D(n_rays=n_rays, grid=grid, use_gpu=use_gpu, n_channel_in=n_channel)
    model = StarDist2D(conf, name=model_name, basedir=str(model_dir))

    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap("YX"))
    print(f"median object size: {median_size}")
    print(f"network field of view: {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than StarDist FOV.")

    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    print("✓ StarDist training complete.")

    pred_lbl, _ = model.predict_instances(normalize(X_val[0], 1, 99.8, axis=axis_norm))
    stats = matching(Y_val[0], pred_lbl)
    print(f"Validation (first val tile) F1: {stats.f1:.3f}, TP={stats.tp}, FP={stats.fp}, FN={stats.fn}")


# ─── main orchestration ──────────────────────────────────────────────────────
def parse_args():
    """CLI options for end-to-end SAM + StarDist training."""
    pa = argparse.ArgumentParser(description="Train SAM and StarDist over tiles")
    pa.add_argument("--tiles-dir", default=ROOT / "tiles", type=Path, help="Directory containing tile subfolders")
    pa.add_argument("--image-name", default="img.tif")
    pa.add_argument("--seeds-name", default="seeds.gpkg")
    pa.add_argument("--chm-name", default="chm.tif")
    pa.add_argument("--cost-name", default="cost.tif")
    pa.add_argument("--prob-name", default="crown_probability.tif", help="SAM output filename inside each tile")
    pa.add_argument("--crowns-name", default="crowns.gpkg", help="Reference crown polygons inside each tile")
    pa.add_argument("--sam-ckpt", default=ROOT / "sam" / "sam_vit_l_0b3195.pth", type=Path, help="SAM base checkpoint")
    pa.add_argument("--sam-dec-out", default=ROOT / "sam" / "sam_crowns_decoder.pth", type=Path, help="Fine-tuned SAM decoder weights")
    pa.add_argument("--sam-train-script", default=ROOT / "sam" / "train_sam_end_to_end.py", type=Path)
    pa.add_argument("--chip-dir", default=ROOT / "sam" / "data" / "chips", type=Path, help="Chip dir for SAM training")
    pa.add_argument("--stardist-x", default=ROOT / "stardist_train" / "x", type=Path, help="Where to write StarDist inputs")
    pa.add_argument("--stardist-y", default=ROOT / "stardist_train" / "y", type=Path, help="Where to write StarDist labels")
    pa.add_argument("--stardist-model-dir", default=ROOT / "models", type=Path)
    pa.add_argument("--stardist-model-name", default="stardist")
    pa.add_argument("--force-sam-infer", action="store_true", help="Re-run SAM even if prob file exists")
    pa.add_argument("--force-sam-train", action="store_true", help="Retrain SAM decoder even if weights exist")
    pa.add_argument("--skip-stardist-train", action="store_true")
    pa.add_argument("--train-kan", action="store_true", help="Train KAN classifier after StarDist")
    pa.add_argument("--kan-dataset", default=ROOT / "models" / "kan_dataset.npz", type=Path)
    pa.add_argument("--kan-model-dir", default=ROOT / "models" / "kan", type=Path)
    pa.add_argument("--kan-class-field", default="class", help="Crowns attribute with class label")
    pa.add_argument("--kan-feature-layer", default=None, help="Optional Keras layer name for encoder features")
    return pa.parse_args()


def main():
    """Entry point for end-to-end training."""
    args = parse_args()
    tiles = list_tiles(args.tiles_dir)
    if not tiles:
        raise SystemExit(f"No tiles found under {args.tiles_dir}")

    # 1) Train SAM decoder (optional)
    if args.force_sam_train or not Path(args.sam_dec_out).exists():
        cmd = [
            sys.executable, str(args.sam_train_script),
            "--tiles-dir", str(args.tiles_dir),
            "--image-name", args.image_name,
            "--crowns-name", args.crowns_name,
            "--ckpt", str(args.sam_ckpt),
            "--out", str(args.sam_dec_out),
            "--chip-dir", str(args.chip_dir),
        ]
        run(cmd)
    else:
        print(f"✓ SAM decoder weights present at {args.sam_dec_out}; skipping SAM training.")

    # 2) Run SAM over each tile to produce crown_probability.tif
    for tile in tiles:
        prob = tile / args.prob_name
        if prob.exists() and not args.force_sam_infer:
            print(f"✓ {prob} exists; skipping SAM inference.")
        else:
            img = tile / args.image_name
            seeds = tile / args.seeds_name
            chm = tile / args.chm_name
            cost = tile / args.cost_name
            missing = [p for p in (img, seeds, chm) if not p.exists()]
            if missing:
                print(f"Skipping {tile.name}: missing {', '.join(map(str, missing))}")
                continue
            cmd = [
                sys.executable, str(ROOT / "helpers" / "predict_sam.py"),
                "--img", str(img),
                "--seeds", str(seeds),
                "--chm", str(chm),
                "--cost", str(cost),
                "--sam-dec", str(args.sam_dec_out),
                "--sam-meta", str(args.sam_ckpt),
                "--out", str(prob),
            ]
            run(cmd)

    # 3) Prepare StarDist training pairs
    args.stardist_x.mkdir(parents=True, exist_ok=True)
    args.stardist_y.mkdir(parents=True, exist_ok=True)
    prepared = 0
    for tile in tiles:
        prob = tile / args.prob_name
        crowns = tile / args.crowns_name
        if not prob.exists() or not crowns.exists():
            print(f"Skipping {tile.name} for StarDist prep (missing prob or crowns)")
            continue
        x_out = args.stardist_x / f"{tile.name}.tif"
        y_out = args.stardist_y / f"{tile.name}.tif"
        if not x_out.exists():
            shutil.copyfile(prob, x_out)
        try:
            n_crowns = rasterize_crowns(crowns, prob, y_out)
            print(f"{tile.name}: labels {n_crowns}")
            prepared += 1
        except Exception as exc:
            print(f"Skipping {tile.name}: {exc}")
            continue
    if prepared == 0:
        raise SystemExit("No StarDist training pairs prepared.")

    # 4) Train StarDist
    if not args.skip_stardist_train:
        train_stardist(args.stardist_x, args.stardist_y, args.stardist_model_dir, args.stardist_model_name)
    else:
        print("StarDist training skipped.")

    # 5) KAN training (optional)
    if args.train_kan:
        cmd_feat = [
            sys.executable, str(ROOT / "helpers" / "kan_feature_builder.py"),
            "--tiles-dir", str(args.tiles_dir),
            "--stardist-model-dir", str(args.stardist_model_dir),
            "--stardist-model-name", args.stardist_model_name,
            "--class-field", args.kan_class_field,
            "--out", str(args.kan_dataset),
        ]
        if args.kan_feature_layer:
            cmd_feat += ["--feature-layer", args.kan_feature_layer]
        run(cmd_feat)

        cmd_kan = [
            sys.executable, str(ROOT / "helpers" / "train_kan_classifier.py"),
            "--dataset", str(args.kan_dataset),
            "--out-dir", str(args.kan_model_dir),
        ]
        run(cmd_kan)


if __name__ == "__main__":
    main()
