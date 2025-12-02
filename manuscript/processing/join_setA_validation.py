#!/usr/bin/env python3
"""
Join Set A validation crowns into a single GeoPackage.

Reads:
  manuscript/data/set_a/setA_validation<i>.gpkg for i in [1..6]

Writes:
  manuscript/data/set_a/validation_crowns.gpkg (layer: validation_crowns)
"""
from pathlib import Path
import sys
import warnings
import geopandas as gpd

try:
    import fiona
except Exception:
    fiona = None


def list_layers(path: Path):
    if fiona is None:
        return []
    try:
        return fiona.listlayers(path)
    except Exception:
        return []


def pick_polygon_layer(path: Path):
    layers = list_layers(path)
    if not layers:
        return None
    for ly in layers:
        with fiona.open(path, layer=ly) as src:
            if src.schema["geometry"] in ("Polygon", "MultiPolygon"):
                return ly
    return layers[0]


def main():
    repo_root = Path(__file__).resolve().parents[2]
    set_a_dir = repo_root / "manuscript" / "data" / "set_a"
    out_path = set_a_dir / "validation_crowns.gpkg"

    parts = []
    base_crs = None
    for i in range(1, 7):
        p = set_a_dir / f"setA_validation{i}.gpkg"
        if not p.exists():
            warnings.warn(f"Missing {p.name}; skipping")
            continue
        lyr = pick_polygon_layer(p)
        if lyr is None:
            warnings.warn(f"No polygon layer in {p.name}; skipping")
            continue
        g = gpd.read_file(p, layer=lyr)
        if g.empty:
            continue
        if base_crs is None:
            base_crs = g.crs
        elif g.crs and base_crs and g.crs != base_crs:
            g = g.to_crs(base_crs)
        parts.append(g)

    if not parts:
        print("No validation inputs found; nothing to write.")
        sys.exit(0)

    merged = gpd.pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.unlink(missing_ok=True)
    except Exception:
        pass
    merged.to_file(out_path, layer="validation_crowns", driver="GPKG")
    print(f"[OK] Wrote {out_path} | crowns={len(merged)} | parts={len(parts)}")


if __name__ == "__main__":
    main()

