#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from samstars.data import build_pending_manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    pa = argparse.ArgumentParser(
        description="Build a pending-only SAM manifest by skipping records with existing SAM outputs."
    )
    pa.add_argument("--manifest", type=Path, required=True, help="Input manifest JSON (e.g., train_manifest.json)")
    pa.add_argument(
        "--sam-out-dir",
        type=Path,
        required=True,
        help="Directory containing SAM outputs (*_sam_index.tif, *_sam_prob.tif, *_logit.tif)",
    )
    pa.add_argument("--out", type=Path, required=True, help="Output pending manifest JSON")
    return pa.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    total, done, pending = build_pending_manifest(
        manifest_path=args.manifest,
        sam_out_dir=args.sam_out_dir,
        out_path=args.out,
    )
    print(f"input_records: {total}")
    print(f"already_done: {done}")
    print(f"pending: {pending}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
