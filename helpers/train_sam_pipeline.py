#!/usr/bin/env python3
"""
SAM-only pipeline:
 1) Train SAM decoder (optional)
 2) Run SAM over tiles to produce crown_probability.tif

Use this when SAM (torch) must run in a separate environment from StarDist.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    pa = argparse.ArgumentParser(description="Train SAM decoder and run SAM inference on tiles.")
    pa.add_argument("--tiles-dir", required=True, type=Path, help="Directory containing tile subfolders")
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
    pa.add_argument("--force-sam-infer", action="store_true", help="Re-run SAM even if prob file exists")
    pa.add_argument("--force-sam-train", action="store_true", help="Retrain SAM decoder even if weights exist")
    return pa.parse_args()


def run(cmd):
    print(" ", " ".join(map(str, cmd)))
    sys.stdout.flush()
    import subprocess
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    tiles = [p for p in sorted(args.tiles_dir.iterdir()) if p.is_dir()]
    if not tiles:
        raise SystemExit(f"No tiles found under {args.tiles_dir}")

    # Train SAM decoder if needed
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

    # SAM inference over tiles
    for tile in tiles:
        prob = tile / args.prob_name
        if prob.exists() and not args.force_sam_infer:
            print(f"✓ {prob} exists; skipping SAM inference.")
            continue
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

    print("✓ SAM pipeline complete.")


if __name__ == "__main__":
    main()
