#!/usr/bin/env python3
import argparse
from pathlib import Path

from samstars.data import build_tile_records


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(
        description="Build tile-level run records from a prepared tile directory."
    )
    pa.add_argument("--tiles-dir", type=Path, required=True, help="Directory containing tile_* folders")
    pa.add_argument("--out-root", type=Path, required=True, help="Output root for run records")
    pa.add_argument("--size-id", type=str, default="0474", help="Size folder name under out-root")
    pa.add_argument("--filename", type=str, default="tiles.json", help="Records filename to write")
    args = pa.parse_args()

    recs, out = build_tile_records(
        tiles_dir=args.tiles_dir,
        out_root=args.out_root,
        size_id=args.size_id,
        filename=args.filename,
    )
    print(f"✓ wrote {out} ({len(recs)} records)")


if __name__ == "__main__":
    main()
