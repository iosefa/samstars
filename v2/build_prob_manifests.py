#!/usr/bin/env python3
"""
Build prob manifests (train/val) from existing SAM prob TIFFs.
Looks for *_prob.tif under <probs_root>/<size>/<split>_probs and writes
<split>_probs.json listing the prob file and matching mask (if found) from chips.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_manifest(prob_dir: Path, split: str, chips_root: Path, size_dir: Path, fixed_val_size: str | None):
    recs = []
    for prob_tif in sorted(prob_dir.glob("*_prob.tif")):
        stem = prob_tif.stem.replace("_prob", "")
        if split == "train":
            mask_path = chips_root / size_dir.name / "chips_train" / "msk" / f"{stem}.npy"
        else:
            val_size = fixed_val_size or size_dir.name
            mask_path = chips_root / val_size / "chips_val" / "msk" / f"{stem}.npy"
        recs.append({"file_name": str(prob_tif), "mask": str(mask_path) if mask_path.exists() else ""})
    out = prob_dir.parent / f"{split}_probs.json"
    json.dump(recs, open(out, "w"))
    print(f"✓ wrote {out} ({len(recs)} records)")


def main():
    pa = argparse.ArgumentParser(description="Build prob manifests from SAM prob TIFFs.")
    pa.add_argument("--probs-root", type=Path, default=Path("v2/sam_probs"))
    pa.add_argument("--chips-root", type=Path, default=Path("v2/chips"))
    pa.add_argument("--fixed-val-size", type=str, default=None, help="Use masks from this size for val")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated size list; default is all under probs-root")
    args = pa.parse_args()

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
        size_dirs = [args.probs_root / s for s in sizes]
    else:
        size_dirs = sorted(p for p in args.probs_root.iterdir() if p.is_dir())

    for size_dir in size_dirs:
        if not size_dir.exists():
            print(f"skip missing: {size_dir}")
            continue
        tdir = size_dir / "train_probs"
        vdir = size_dir / "val_probs"
        if tdir.exists():
            build_manifest(tdir, "train", args.chips_root, size_dir, args.fixed_val_size)
        if vdir.exists():
            build_manifest(vdir, "val", args.chips_root, size_dir, args.fixed_val_size)


if __name__ == "__main__":
    main()
