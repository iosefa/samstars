#!/usr/bin/env python3
"""
Ensure fixed-val chips cover all species classes by moving chips from train->val.

Updates:
  - data/chip_splits_*/fixed_val_manifest.json
  - per-size train_manifest.json (nested subsets rebuilt from updated train pool)

This runs after chip splits/builds; you should rebuild chips after changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import geopandas as gpd
from shapely.geometry import box


def _rec_key(rec: dict) -> Tuple[str, int]:
    return rec.get("tile", ""), int(rec.get("chip_id") or 0)


def _load_manifest(path: Path) -> List[dict]:
    return json.load(open(path))


def _load_crowns(tiles_dir: Path, class_field: str) -> Dict[str, gpd.GeoDataFrame]:
    out: Dict[str, gpd.GeoDataFrame] = {}
    for tile_dir in sorted(p for p in tiles_dir.iterdir() if p.is_dir()):
        gpkg = tile_dir / "crowns.gpkg"
        if not gpkg.exists():
            continue
        gdf = gpd.read_file(gpkg)
        if gdf.empty:
            continue
        if class_field not in gdf.columns:
            raise RuntimeError(f"Missing '{class_field}' in {gpkg}")
        out[tile_dir.name] = gdf
    if not out:
        raise RuntimeError("No crowns.gpkg files found.")
    return out


def _classes_for_rec(rec: dict, crowns_by_tile: Dict[str, gpd.GeoDataFrame], class_field: str) -> Set[int]:
    tile = rec.get("tile", "")
    gdf = crowns_by_tile.get(tile)
    if gdf is None or gdf.empty:
        return set()
    win = box(*rec["bounds"])
    sub = gdf[gdf.intersects(win)]
    if sub.empty:
        return set()
    vals = sub[class_field]
    vals = gpd.pd.to_numeric(vals, errors="coerce").dropna().astype(int)
    return set(vals.tolist())


def _class_counts_for_rec(rec: dict, crowns_by_tile: Dict[str, gpd.GeoDataFrame], class_field: str) -> Dict[int, int]:
    tile = rec.get("tile", "")
    gdf = crowns_by_tile.get(tile)
    if gdf is None or gdf.empty:
        return {}
    win = box(*rec["bounds"])
    sub = gdf[gdf.intersects(win)]
    if sub.empty:
        return {}
    vals = sub[class_field]
    vals = gpd.pd.to_numeric(vals, errors="coerce").dropna().astype(int)
    return vals.value_counts().to_dict()


def _get_all_classes(crowns_by_tile: Dict[str, gpd.GeoDataFrame], class_field: str) -> Set[int]:
    classes = set()
    for gdf in crowns_by_tile.values():
        vals = gpd.pd.to_numeric(gdf[class_field], errors="coerce").dropna().astype(int)
        classes.update(vals.tolist())
    return classes


def _max_size_dir(splits_root: Path) -> Path:
    size_dirs = []
    for p in splits_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.isdigit():
            size_dirs.append(int(p.name))
    if not size_dirs:
        raise RuntimeError("No size dirs found under splits root.")
    return splits_root / f"{max(size_dirs):04d}"


def main() -> None:
    pa = argparse.ArgumentParser(description="Add fixed-val chips to cover all classes.")
    pa.add_argument("--splits-root", type=Path, required=True)
    pa.add_argument("--tiles-dir", type=Path, required=True)
    pa.add_argument("--class-field", type=str, default="class")
    pa.add_argument(
        "--avoid-train-manifest",
        type=Path,
        default=None,
        help="Train manifest to avoid when selecting new val chips (prevents leakage).",
    )
    pa.add_argument(
        "--remove-existing-overlap",
        action="store_true",
        help="Remove any fixed-val chips that are present in --avoid-train-manifest.",
    )
    pa.add_argument(
        "--skip-train-update",
        action="store_true",
        help="Only update fixed_val_manifest.json; do not rewrite per-size train manifests.",
    )
    pa.add_argument(
        "--add-classes",
        type=str,
        default=None,
        help="Comma-separated class IDs to add to val even if already present.",
    )
    pa.add_argument(
        "--min-class-count",
        type=int,
        default=1,
        help="Minimum number of crowns of a class required to select a chip.",
    )
    pa.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    args = pa.parse_args()

    fixed_val_path = args.splits_root / "fixed_val_manifest.json"
    if not fixed_val_path.exists():
        raise SystemExit(f"Missing fixed val manifest: {fixed_val_path}")

    fixed_val = _load_manifest(fixed_val_path)
    max_dir = _max_size_dir(args.splits_root)
    train_manifest = max_dir / "train_manifest.json"
    if not train_manifest.exists():
        raise SystemExit(f"Missing train_manifest.json in {max_dir}")
    train_pool = _load_manifest(train_manifest)

    crowns_by_tile = _load_crowns(args.tiles_dir, args.class_field)
    all_classes = _get_all_classes(crowns_by_tile, args.class_field)

    avoid_keys = set()
    if args.avoid_train_manifest:
        if not args.avoid_train_manifest.exists():
            raise SystemExit(f"Missing avoid train manifest: {args.avoid_train_manifest}")
        avoid_keys = {_rec_key(r) for r in _load_manifest(args.avoid_train_manifest)}

    if args.remove_existing_overlap and avoid_keys:
        before = len(fixed_val)
        fixed_val = [r for r in fixed_val if _rec_key(r) not in avoid_keys]
        removed = before - len(fixed_val)
        if removed:
            print(f"Removed {removed} fixed-val chips that overlapped avoid-train-manifest.")

    fixed_keys = {_rec_key(r) for r in fixed_val}
    val_classes = set()
    for rec in fixed_val:
        val_classes.update(_classes_for_rec(rec, crowns_by_tile, args.class_field))

    missing = sorted(all_classes - val_classes)
    add_classes = []
    if args.add_classes:
        add_classes = [int(s.strip()) for s in args.add_classes.split(",") if s.strip()]

    if not missing and not add_classes:
        print("Fixed val already covers all classes.")
        return

    selected = []
    train_pool_keys = {_rec_key(r) for r in train_pool}
    counts_cache: Dict[Tuple[str, int], Dict[int, int]] = {}

    def _counts(rec: dict) -> Dict[int, int]:
        k = _rec_key(rec)
        if k in counts_cache:
            return counts_cache[k]
        counts_cache[k] = _class_counts_for_rec(rec, crowns_by_tile, args.class_field)
        return counts_cache[k]

    for cls in missing:
        found = None
        best_cnt = -1
        for rec in train_pool:
            k = _rec_key(rec)
            if k in fixed_keys:
                continue
            if k in avoid_keys:
                continue
            if k not in train_pool_keys:
                continue
            cnt = _counts(rec).get(cls, 0)
            if cnt < args.min_class_count:
                continue
            if cnt > best_cnt:
                found = rec
                best_cnt = cnt
        if found is None:
            print(f"[warn] no train chip found for missing class {cls}")
            continue
        selected.append(found)
        fixed_keys.add(_rec_key(found))
        val_classes.update(_classes_for_rec(found, crowns_by_tile, args.class_field))

    for cls in add_classes:
        found = None
        best_cnt = -1
        for rec in train_pool:
            k = _rec_key(rec)
            if k in fixed_keys:
                continue
            if k in avoid_keys:
                continue
            if k not in train_pool_keys:
                continue
            cnt = _counts(rec).get(cls, 0)
            if cnt < args.min_class_count:
                continue
            if cnt > best_cnt:
                found = rec
                best_cnt = cnt
        if found is None:
            print(f"[warn] no train chip found for add class {cls}")
            continue
        selected.append(found)
        fixed_keys.add(_rec_key(found))

    if not selected:
        print("No chips selected; nothing to update.")
        return

    if missing:
        print(f"Missing classes: {missing}")
    if add_classes:
        print(f"Added classes requested: {add_classes}")
    print("Selected chips to move to val:")
    for rec in selected:
        counts = _counts(rec)
        print(
            f"- {rec.get('tile')} chip_id={rec.get('chip_id')} n_crowns={rec.get('n_crowns')} "
            f"class_counts={counts}"
        )

    if args.dry_run:
        print("Dry-run only; no files written.")
        return

    # Update fixed val manifest
    fixed_val_updated = fixed_val + selected
    json.dump(fixed_val_updated, open(fixed_val_path, "w"))
    print(f"✓ updated {fixed_val_path} (+{len(selected)} chips)")

    if args.skip_train_update:
        print(
            "Note: train manifests were NOT updated. Selected chips remain in training subsets, "
            "so train/val overlap exists until you rebuild splits."
        )
        print("Next: rebuild fixed_val chips from the updated fixed_val_manifest.json.")
        return

    # Rebuild train pool (remove moved chips) and rewrite per-size manifests
    moved_keys = {_rec_key(r) for r in selected}
    train_pool_updated = [r for r in train_pool if _rec_key(r) not in moved_keys]

    for size_dir in sorted(p for p in args.splits_root.iterdir() if p.is_dir() and p.name.isdigit()):
        n = int(size_dir.name)
        if n > len(train_pool_updated):
            print(f"[warn] size {n} exceeds train pool ({len(train_pool_updated)}); skipping")
            continue
        subset = train_pool_updated[:n]
        json.dump(subset, open(size_dir / "train_manifest.json", "w"))
        val_path = size_dir / "val_manifest.json"
        if val_path.exists():
            json.dump([], open(val_path, "w"))
        print(f"✓ updated {size_dir / 'train_manifest.json'} ({len(subset)} chips)")

    print("Note: rebuild chips from manifests to refresh fixed_val and per-size chips.")


if __name__ == "__main__":
    main()
