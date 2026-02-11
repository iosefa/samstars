#!/usr/bin/env python3
"""
Build a short reference data card (markdown) for the current train/val design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(x) for x in row.tolist()) + " |")
    return "\n".join([header, sep] + rows)


def _load_crowns(tiles_dir: Path, class_field: str = "class") -> gpd.GeoDataFrame:
    gdfs = []
    for tile_dir in sorted(p for p in tiles_dir.iterdir() if p.is_dir()):
        gpkg = tile_dir / "crowns.gpkg"
        if not gpkg.exists():
            continue
        gdf = gpd.read_file(gpkg)
        if gdf.empty:
            continue
        if gdf.crs is None:
            img = tile_dir / "img.tif"
            if img.exists():
                with rasterio.open(img) as src:
                    gdf = gdf.set_crs(src.crs)
        gdf = gdf.copy()
        gdf["tile"] = tile_dir.name
        gdf["crown_id"] = [f"{tile_dir.name}_{i}" for i in range(len(gdf))]
        gdfs.append(gdf)
    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    out = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    if class_field in out.columns:
        out[class_field] = pd.to_numeric(out[class_field], errors="coerce").astype("Int64")
    return out


def _split_by_tile(crowns: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    out: dict[str, gpd.GeoDataFrame] = {}
    if crowns.empty:
        return out
    for tile, sub in crowns.groupby("tile"):
        out[str(tile)] = sub.copy()
    return out


def _load_fixed_val_windows(fixed_val_manifest: Path) -> gpd.GeoDataFrame:
    recs = json.load(open(fixed_val_manifest))
    polys = [box(*r["bounds"]) for r in recs]
    crs = recs[0].get("crs") if recs else None
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
    return gdf


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


def _per_chip_class_counts(
    train_pool: list[dict],
    crowns_by_tile: dict[str, gpd.GeoDataFrame],
    class_field: str,
    classes: list[int],
) -> pd.DataFrame:
    rows = []
    for rec in train_pool:
        tile = rec.get("tile", "")
        gdf = crowns_by_tile.get(tile)
        if gdf is None or gdf.empty:
            rows.append({c: 0 for c in classes})
            continue
        geom = box(*rec["bounds"])
        try:
            hits = list(gdf.sindex.query(geom, predicate="intersects"))
            sub = gdf.iloc[hits]
            sub = sub[sub.intersects(geom)]
        except Exception:
            sub = gdf[gdf.intersects(geom)]
        vals = gpd.pd.to_numeric(sub[class_field], errors="coerce").dropna().astype(int)
        counts = vals.value_counts().to_dict()
        rows.append({c: int(counts.get(c, 0)) for c in classes})
    return pd.DataFrame(rows)


def main() -> None:
    pa = argparse.ArgumentParser(description="Create reference data card markdown.")
    pa.add_argument("--tiles-dir", type=Path, default=Path("data/train-sam"))
    pa.add_argument("--fixed-val-manifest", type=Path, default=Path("data/chip_splits_128_100/fixed_val_manifest.json"))
    pa.add_argument("--splits-root", type=Path, default=Path("data/chip_splits_128_100"))
    pa.add_argument("--out-md", type=Path, default=Path("docs/reference_data_card.md"))
    args = pa.parse_args()

    crowns = _load_crowns(args.tiles_dir)
    if crowns.empty:
        raise SystemExit("No crowns found. Check --tiles-dir.")

    if "class" not in crowns.columns:
        raise SystemExit("Missing 'class' field in crowns.gpkg.")

    fixed_val = _load_fixed_val_windows(args.fixed_val_manifest)
    if fixed_val.empty:
        raise SystemExit("Fixed val manifest is empty.")

    if crowns.crs and fixed_val.crs and crowns.crs != fixed_val.crs:
        fixed_val = fixed_val.to_crs(crowns.crs)
    elif crowns.crs and fixed_val.crs is None:
        fixed_val = fixed_val.set_crs(crowns.crs)

    val_union = fixed_val.unary_union
    crowns = crowns.copy()
    crowns["split"] = crowns.geometry.intersects(val_union).map(lambda x: "val" if x else "train")

    total = int(len(crowns))
    val_total = int((crowns["split"] == "val").sum())
    train_total = int((crowns["split"] == "train").sum())
    val_pct = (val_total / total * 100.0) if total else 0.0

    # Species counts
    species = (
        crowns.groupby(["class", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"class": "species_code"})
    )
    if "train" not in species.columns:
        species["train"] = 0
    if "val" not in species.columns:
        species["val"] = 0
    species["total"] = species["train"] + species["val"]
    species = species.sort_values("species_code")

    # Tile/plot counts
    plots = (
        crowns.groupby(["tile", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"tile": "plot"})
    )
    if "train" not in plots.columns:
        plots["train"] = 0
    if "val" not in plots.columns:
        plots["val"] = 0
    plots["total"] = plots["train"] + plots["val"]
    plots = plots.sort_values("plot")

    fixed_val_recs = json.load(open(args.fixed_val_manifest))
    fixed_val_chip_counts = pd.Series([r["tile"] for r in fixed_val_recs]).value_counts().sort_index()
    fixed_val_crowns_sum = int(sum(int(r.get("n_crowns") or 0) for r in fixed_val_recs))

    # Per-size training class counts
    classes = sorted(set(crowns["class"].dropna().astype(int).tolist()))
    max_dir = _max_size_dir(args.splits_root)
    train_pool = json.load(open(max_dir / "train_manifest.json"))
    crowns_by_tile = _split_by_tile(crowns)
    counts_per_chip = _per_chip_class_counts(train_pool, crowns_by_tile, "class", classes)
    cum_counts = counts_per_chip.cumsum()

    sizes = sorted(int(p.name) for p in args.splits_root.iterdir() if p.is_dir() and p.name.isdigit())
    rows = []
    for n in sizes:
        if n < 1 or n > len(cum_counts):
            continue
        row = {"size": f"{n:04d}"}
        for c in classes:
            row[f"class_{c}"] = int(cum_counts.iloc[n - 1][c])
        row["total"] = int(sum(cum_counts.iloc[n - 1][c] for c in classes))
        rows.append(row)
    size_class_df = pd.DataFrame(rows)

    # Build markdown
    lines = []
    lines.append("# Reference Dataset Data Card\n")

    lines.append("## Reference dataset definition\n")
    lines.append(
        "Reference crowns are the union of all `crowns.gpkg` geometries in "
        f"`{args.tiles_dir}`. Each crown polygon carries a numeric species label in the "
        "`class` field (EPSG:6635). This combined geometry+label set is the authoritative "
        "reference dataset for training/validation.\n"
    )
    lines.append(
        "Plot tiles were selected along precipitation and elevation gradients to capture "
        "the major species composition.\n"
    )

    lines.append("## Split design (current run)\n")
    lines.append(
        f"- Fixed validation chips are read from `{args.fixed_val_manifest}` "
        f"({len(fixed_val_recs)} chips; tiles: {', '.join(sorted(set(r['tile'] for r in fixed_val_recs)))}).\n"
        "- The fixed validation set is selected **randomly at the chip level** (not spatially "
        "stratified), using a shuffled chip list (seeded in `build_chip_splits.py`) and taking chips "
        "until a target crown count is reached (if `--fixed-val-crowns` was used). "
        "Training chips are all remaining chips.\n"
        "- Per-size training subsets are **nested** and ordered by ascending crowns-per-chip "
        "(`--order-by instances-asc`).\n"
    )

    lines.append("## Summary totals\n")
    lines.append(f"- Total crowns: {total}\n")
    lines.append(f"- Train crowns: {train_total}\n")
    lines.append(f"- Val crowns (unique, by intersection with fixed-val windows): {val_total} ({val_pct:.1f}%)\n")
    lines.append(f"- Fixed-val crowns summed from chip metadata (n_crowns): {fixed_val_crowns_sum}\n")

    lines.append("\n## Fixed-val chips by plot\n")
    fv_df = fixed_val_chip_counts.reset_index()
    fv_df.columns = ["plot", "n_chips"]
    lines.append(_df_to_md(fv_df))
    lines.append("")

    lines.append("\n## Species counts (train vs val)\n")
    lines.append(_df_to_md(species[["species_code", "train", "val", "total"]]))
    lines.append("")

    lines.append("\n## Plot/stratum counts (train vs val)\n")
    lines.append(_df_to_md(plots[["plot", "train", "val", "total"]]))
    lines.append("")

    lines.append("\n## Training class counts by size\n")
    lines.append(_df_to_md(size_class_df))
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines))
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
