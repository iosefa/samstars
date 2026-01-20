#!/usr/bin/env python3
"""
Add a single missing size to an existing splits root by slicing an existing
train_manifest.json (typically the largest size).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _chip_key(rec: dict) -> tuple[str, int]:
    return (rec.get("tile", ""), int(rec.get("chip_id", -1)))


def _find_sizes(root: Path) -> list[int]:
    sizes = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.isdigit() and len(p.name) == 4:
            sizes.append(int(p.name))
    return sorted(sizes)


def main() -> None:
    pa = argparse.ArgumentParser(description="Add a single size split from an existing train_manifest.json.")
    pa.add_argument("--splits-root", type=Path, required=True)
    pa.add_argument("--new-size", type=int, required=True, help="New size to add (e.g., 120)")
    pa.add_argument("--source-size", type=int, default=None, help="Existing size to slice from (default: max size)")
    pa.add_argument("--fixed-val-manifest", type=Path, default=None, help="Optional fixed val manifest to exclude")
    pa.add_argument("--allow-overlap", action="store_true", help="Allow train/val overlap when fixed val exists")
    pa.add_argument("--write-per-size-val", action="store_true", help="Write empty val_manifest.json")
    pa.add_argument("--overwrite", action="store_true", help="Overwrite existing size dir if present")
    args = pa.parse_args()

    sizes = _find_sizes(args.splits_root)
    if not sizes:
        raise SystemExit("No existing size directories found under --splits-root.")

    source_size = args.source_size if args.source_size is not None else max(sizes)
    source_dir = args.splits_root / f"{source_size:04d}"
    source_manifest = source_dir / "train_manifest.json"
    if not source_manifest.exists():
        raise SystemExit(f"Missing source train_manifest.json: {source_manifest}")

    out_dir = args.splits_root / f"{args.new_size:04d}"
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists. Use --overwrite to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = json.load(open(source_manifest))

    fixed_val_manifest = args.fixed_val_manifest
    if fixed_val_manifest is None:
        candidate = args.splits_root / "fixed_val_manifest.json"
        if candidate.exists():
            fixed_val_manifest = candidate

    if fixed_val_manifest and fixed_val_manifest.exists() and not args.allow_overlap:
        fixed_recs = json.load(open(fixed_val_manifest))
        fixed_keys = {_chip_key(r) for r in fixed_recs}
        recs = [r for r in recs if _chip_key(r) not in fixed_keys]

    if len(recs) < args.new_size:
        raise SystemExit(
            f"Source manifest has {len(recs)} chips after filtering; "
            f"cannot create size {args.new_size}."
        )

    subset = recs[: args.new_size]
    json.dump(subset, open(out_dir / "train_manifest.json", "w"))
    if args.write_per_size_val:
        json.dump([], open(out_dir / "val_manifest.json", "w"))

    print(f"✓ wrote {out_dir / 'train_manifest.json'} ({len(subset)} chips)")


if __name__ == "__main__":
    main()
