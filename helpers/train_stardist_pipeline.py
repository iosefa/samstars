#!/usr/bin/env python3
"""
StarDist (+optional KAN) pipeline:
 Assumes SAM prob rasters already exist per tile.
 1) Prepare StarDist training pairs (prob raster + rasterized crowns)
 2) Train StarDist
 3) (Optional) Build features + train KAN classifier

Use this when StarDist/TF needs its own environment separate from SAM.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from train_sam_and_stardist import rasterize_crowns, train_stardist  # reuse existing helpers

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print(" ", " ".join(map(str, cmd)))
    sys.stdout.flush()
    import subprocess
    subprocess.run(cmd, check=True)


def parse_args():
    pa = argparse.ArgumentParser(description="Train StarDist (and optional KAN) using existing SAM probability rasters.")
    pa.add_argument("--tiles-dir", required=True, type=Path, help="Directory containing tile subfolders")
    pa.add_argument("--prob-name", default="crown_probability.tif", help="SAM output filename inside each tile")
    pa.add_argument("--crowns-name", default="crowns.gpkg", help="Reference crown polygons inside each tile")
    pa.add_argument("--stardist-x", default=ROOT / "stardist_train" / "x", type=Path, help="Where to write StarDist inputs")
    pa.add_argument("--stardist-y", default=ROOT / "stardist_train" / "y", type=Path, help="Where to write StarDist labels")
    pa.add_argument("--stardist-model-dir", default=ROOT / "models", type=Path)
    pa.add_argument("--stardist-model-name", default="stardist")
    pa.add_argument("--skip-stardist-train", action="store_true")
    pa.add_argument("--train-kan", action="store_true", help="Train KAN classifier after StarDist")
    pa.add_argument("--kan-dataset", default=ROOT / "models" / "kan_dataset.npz", type=Path)
    pa.add_argument("--kan-model-dir", default=ROOT / "models" / "kan", type=Path)
    pa.add_argument("--kan-class-field", default="class", help="Crowns attribute with class label")
    pa.add_argument("--kan-feature-layer", default=None, help="Optional Keras layer name for encoder features")
    return pa.parse_args()


def main():
    args = parse_args()
    tiles = [p for p in sorted(args.tiles_dir.iterdir()) if p.is_dir()]
    if not tiles:
        raise SystemExit(f"No tiles found under {args.tiles_dir}")

    # Prepare StarDist training pairs
    args.stardist_x.mkdir(parents=True, exist_ok=True)
    args.stardist_y.mkdir(parents=True, exist_ok=True)
    prepared = 0
    for tile in tiles:
        prob = tile / args.prob_name
        crowns = tile / args.crowns_name
        if not prob.exists() or not crowns.exists():
            print(f"Skipping {tile.name} (missing prob or crowns)")
            continue
        x_out = args.stardist_x / f"{tile.name}.tif"
        y_out = args.stardist_y / f"{tile.name}.tif"
        if not x_out.exists():
            import shutil
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

    # Train StarDist
    if not args.skip_stardist_train:
        train_stardist(args.stardist_x, args.stardist_y, args.stardist_model_dir, args.stardist_model_name)
    else:
        print("StarDist training skipped.")

    # KAN training (optional)
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

    print("✓ StarDist pipeline complete.")


if __name__ == "__main__":
    main()
