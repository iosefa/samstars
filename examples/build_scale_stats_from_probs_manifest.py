#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from samstars.data import build_scale_stats_from_probs_manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    pa = argparse.ArgumentParser(
        description="Build StarDist scale-stats JSON (chm_01 lo/hi) from a SAM probs manifest."
    )
    pa.add_argument("--probs-json", type=Path, required=True, help="Path to train_probs.json / val_probs.json")
    pa.add_argument("--out", type=Path, required=True, help="Output JSON path (e.g., scale_2_98.json)")
    pa.add_argument(
        "--percentiles",
        type=str,
        default="2,98",
        help="Low/high percentiles as 'low,high' (default: 2,98)",
    )
    pa.add_argument("--chm-key", type=str, default="chm", help="Manifest key containing CHM raster path")
    pa.add_argument(
        "--nodata-values",
        type=str,
        default="-9999,-9999.0,-32768,-32768.0",
        help="Comma-separated literal CHM nodata sentinel values to exclude",
    )
    return pa.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()

    try:
        p_lo, p_hi = [float(x.strip()) for x in args.percentiles.split(",")]
    except Exception as exc:
        raise SystemExit(f"Invalid --percentiles '{args.percentiles}': {exc}") from exc
    if not (0 <= p_lo < p_hi <= 100):
        raise SystemExit(f"Percentiles must satisfy 0 <= low < high <= 100, got {p_lo},{p_hi}")

    try:
        nodata_values = [float(x.strip()) for x in args.nodata_values.split(",") if x.strip()]
    except Exception as exc:
        raise SystemExit(f"Invalid --nodata-values '{args.nodata_values}': {exc}") from exc

    out = build_scale_stats_from_probs_manifest(
        probs_json=args.probs_json,
        out=args.out,
        percentiles=(p_lo, p_hi),
        chm_key=args.chm_key,
        nodata_values=tuple(nodata_values),
    )
    lo = out["chm_01"]["lo"]
    hi = out["chm_01"]["hi"]
    print(f"wrote: {args.out}")
    print(f"chm_01 lo/hi: {lo:.6f}, {hi:.6f}")
    print(
        "records: "
        f"total={out['records_total']} "
        f"missing_or_unreadable_chm={out['records_missing_or_unreadable_chm']}"
    )


if __name__ == "__main__":
    main()
