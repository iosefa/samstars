#!/usr/bin/env python3
"""
Summarize StarDist eval metrics (PQ/SQ/RQ, IoU, Dice) across sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _count_train_instances(chips_root: Path, size: str) -> int:
    """Internal helper function."""
    manifest = chips_root / size / "chips_train" / "detectron_manifest.json"
    if not manifest.exists():
        return 0
    data = json.load(open(manifest))
    total = 0
    for rec in data:
        gpkg = Path(rec["gpkg_path"])
        if gpkg.exists():
            total += len(gpd.read_file(gpkg))
    return total


def _load_metrics(eval_dir: Path) -> dict | None:
    """Internal helper function."""
    metrics_json = eval_dir / "metrics.json"
    if metrics_json.exists():
        return json.load(open(metrics_json))
    metrics_csv = eval_dir / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        return {str(r.metric): float(r.value) if str(r.value) != "" else None for r in df.itertuples()}
    return None


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Summarize StarDist metrics over sizes.")
    pa.add_argument("--eval-root", type=Path, required=True)
    pa.add_argument("--chips-root", type=Path, required=True)
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated size list; default auto-detect")
    pa.add_argument("--out-csv", type=Path, default=None)
    args = pa.parse_args()

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = sorted([p.name for p in args.eval_root.iterdir() if p.is_dir()])

    rows = []
    for sz in sizes:
        eval_dir = args.eval_root / sz
        metrics = _load_metrics(eval_dir)
        if not metrics:
            print(f"[skip] missing metrics: {eval_dir}")
            continue
        n_train = _count_train_instances(args.chips_root, sz)
        rows.append({
            "size": sz,
            "n_train_instances": n_train,
            "pq": metrics.get("pq"),
            "sq": metrics.get("sq"),
            "rq": metrics.get("rq"),
            "iou": metrics.get("iou"),
            "dice": metrics.get("dice"),
            "val_instances": metrics.get("val_instances"),
        })
        print(f"[ok] {sz}: n_train={n_train}, PQ={metrics.get('pq')}, SQ={metrics.get('sq')}, RQ={metrics.get('rq')}")

    df = pd.DataFrame(rows).sort_values("n_train_instances")
    out_csv = args.out_csv if args.out_csv else (args.eval_root / "summary_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
