#!/usr/bin/env python3
"""
Create a manifests root where each size dir has:
  - train_manifest.json from its own split
  - val_manifest.json from a fixed size split
This is useful for running SAM over a fixed validation set across sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    pa = argparse.ArgumentParser(description="Build per-size manifests with fixed val set.")
    pa.add_argument("--splits-root", type=Path, required=True)
    pa.add_argument("--out-root", type=Path, required=True)
    pa.add_argument("--fixed-val-size", type=str, required=True)
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default all under splits-root")
    args = pa.parse_args()

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = [p.name for p in sorted(args.splits_root.iterdir()) if p.is_dir()]

    fixed_val = args.splits_root / args.fixed_val_size / "val_manifest.json"
    if not fixed_val.exists():
        raise SystemExit(f"Missing fixed val manifest: {fixed_val}")
    fixed_val_data = json.load(open(fixed_val))

    args.out_root.mkdir(parents=True, exist_ok=True)
    for sz in sizes:
        train_manifest = args.splits_root / sz / "train_manifest.json"
        if not train_manifest.exists():
            print(f"skip missing train manifest: {train_manifest}")
            continue
        out_dir = args.out_root / sz
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(json.load(open(train_manifest)), open(out_dir / "train_manifest.json", "w"))
        json.dump(fixed_val_data, open(out_dir / "val_manifest.json", "w"))
        print(f"✓ wrote {out_dir}/train_manifest.json + fixed val")


if __name__ == "__main__":
    main()
