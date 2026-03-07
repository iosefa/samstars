#!/usr/bin/env python3
"""
Build prob manifests (train/val) from existing SAM prob TIFFs.
Looks for *_sam_prob.tif under <probs_root>/<size>/<split>_probs and writes
<split>_probs.json listing the prob file + aux channels and matching mask.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _infer_stem(path: Path) -> str:
    """Internal helper function."""
    stem = path.stem
    if stem.endswith("_sam_prob"):
        return stem[:-9]
    if stem.endswith("_prob"):
        return stem[:-5]
    return stem


def build_manifest(prob_dir: Path, split: str, chips_root: Path, size_dir: Path, fixed_val_size: str | None):
    """Build manifest."""
    recs = []
    in_manifest = prob_dir.parent / f"{split}_probs.json"
    if in_manifest.exists():
        base_recs = json.load(open(in_manifest))
    else:
        base_recs = [{"file_name": str(p)} for p in sorted(prob_dir.glob("*_sam_prob.tif"))]
        if not base_recs:
            base_recs = [{"file_name": str(p)} for p in sorted(prob_dir.glob("*_prob.tif"))]

    for rec in base_recs:
        fname = rec.get("file_name") or ""
        if not fname:
            continue
        prob_path = Path(fname)
        if not prob_path.exists():
            continue
        stem = _infer_stem(prob_path)

        chm_path = rec.get("chm") or str(prob_path.with_name(f"{stem}_chm.tif"))
        cost_path = rec.get("cost") or str(prob_path.with_name(f"{stem}_cost.tif"))

        if split == "train":
            mask_path = chips_root / size_dir.name / "chips_train" / "msk" / f"{stem}.npy"
        else:
            val_size = fixed_val_size or size_dir.name
            mask_path = chips_root / val_size / "chips_val" / "msk" / f"{stem}.npy"

        out_rec = {
            "file_name": str(prob_path),
            "chm": chm_path if Path(chm_path).exists() else "",
            "cost": cost_path if Path(cost_path).exists() else "",
            "logit": rec.get("logit", ""),
            "index": rec.get("index", ""),
            "mask": str(mask_path) if mask_path.exists() else "",
            "meta": rec.get("meta", {}),
        }
        recs.append(out_rec)

    out = prob_dir.parent / f"{split}_probs.json"
    json.dump(recs, open(out, "w"))
    print(f"✓ wrote {out} ({len(recs)} records)")


def main():
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Build prob manifests from SAM prob TIFFs.")
    pa.add_argument("--probs-root", type=Path, default=Path("data/sam_probs"))
    pa.add_argument("--chips-root", type=Path, default=Path("data/chips"))
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
