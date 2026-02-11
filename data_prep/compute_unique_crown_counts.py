#!/usr/bin/env python3
"""
Compute unique crown counts per training size by de-duplicating crowns that
appear in overlapping chips. Uses bounds from train_manifest.json and counts
unique crowns by (crowns_path, index) across all chips for that size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


def _load_records(path: Path):
    return json.load(open(path))


def _count_unique_crowns(records) -> int:
    if not records:
        return 0
    cache = {}
    sindex_cache = {}
    seen = set()
    for rec in records:
        crowns_path = rec.get("crowns_path")
        bounds = rec.get("bounds")
        if not crowns_path or not bounds:
            continue
        crowns_path = str(crowns_path)
        if crowns_path not in cache:
            gdf = gpd.read_file(crowns_path)
            cache[crowns_path] = gdf
            sindex_cache[crowns_path] = gdf.sindex
        gdf = cache[crowns_path]
        sindex = sindex_cache[crowns_path]
        geom = box(*bounds)
        candidates = list(sindex.intersection(geom.bounds))
        if not candidates:
            continue
        sub = gdf.iloc[candidates]
        hits = sub[sub.intersects(geom)]
        for idx in hits.index:
            seen.add((crowns_path, int(idx)))
    return len(seen)


def main() -> None:
    pa = argparse.ArgumentParser(description="Compute unique crown counts per size.")
    pa.add_argument("--splits-root", type=Path, required=True)
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default all in splits-root")
    pa.add_argument("--out", type=Path, required=True)
    args = pa.parse_args()

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = [p.name for p in args.splits_root.iterdir() if p.is_dir() and p.name.isdigit()]
        sizes = sorted(sizes)

    rows = []
    for sz in sizes:
        manifest = args.splits_root / sz / "train_manifest.json"
        if not manifest.exists():
            continue
        records = _load_records(manifest)
        n_unique = _count_unique_crowns(records)
        rows.append({"size": sz, "n_train_unique": n_unique})

    if not rows:
        raise SystemExit("No sizes found with train_manifest.json.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
