#!/usr/bin/env python3
import argparse
from pathlib import Path

from samstars.data import build_inference_manifest_from_tiles


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(
        description="Build manifest-style train/val JSONs from tile folders for inference-only runs."
    )
    pa.add_argument("--tiles-dir", type=Path, required=True, help="Directory containing tile_* folders")
    pa.add_argument("--out-root", type=Path, required=True, help="Output root for manifests")
    pa.add_argument("--size-id", type=str, default="0474", help="Size folder name under out-root")
    pa.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split filename to write")
    pa.add_argument("--write-both", action="store_true", help="Write both train_manifest.json and val_manifest.json")
    args = pa.parse_args()

    recs, outs = build_inference_manifest_from_tiles(
        tiles_dir=args.tiles_dir,
        out_root=args.out_root,
        size_id=args.size_id,
        split=args.split,
        write_both=args.write_both,
    )
    for out in outs:
        print(f"✓ wrote {out} ({len(recs)} records)")


if __name__ == "__main__":
    main()
