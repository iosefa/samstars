#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from samstars.data import build_pending_tile_records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    pa = argparse.ArgumentParser(
        description="Build pending tile records by skipping chips with existing proposal rasters."
    )
    pa.add_argument("--records", type=Path, required=True, help="Input tile records JSON (for example tiles.json)")
    pa.add_argument(
        "--proposal-rasters",
        type=Path,
        required=True,
        help="Directory containing proposal rasters (*_sam_index.tif, *_sam_prob.tif, *_logit.tif)",
    )
    pa.add_argument("--out", type=Path, required=True, help="Output pending tile records JSON")
    return pa.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    total, done, pending = build_pending_tile_records(
        records_path=args.records,
        proposal_rasters_dir=args.proposal_rasters,
        out_path=args.out,
    )
    print(f"input_records: {total}")
    print(f"already_done: {done}")
    print(f"pending: {pending}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
