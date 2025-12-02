#!/usr/bin/env python3
"""
Clip SAM–StarDist crowns (classified_xgb.gpkg) to Set A plot extents and
write a single GeoPackage with the merged subset.

Inputs (defaults):
  - crowns:   classified_xgb.gpkg (repo root)
  - extents:  manuscript/data/set_a/setA_plot{i}_extent.gpkg for i in [1..6]
               Fallback if not found: manuscript/data/set_a/setA_plot{i}.gpkg

Output:
  - manuscript/data/set_a/sam_crowns.gpkg (layer: sam_crowns)
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


def ensure_same_crs(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if a.crs is None or b.crs is None:
        raise RuntimeError("Both GeoDataFrames must have a valid CRS.")
    return b if a.crs == b.crs else b.to_crs(a.crs)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    crowns_path = repo_root / "classified_xgb.gpkg"
    out_path = repo_root / "manuscript" / "data" / "set_a" / "sam_crowns.gpkg"

    # Locate extent files (prefer *_extent.gpkg; fallback to setA_plot{i}.gpkg)
    set_a_dir = repo_root / "manuscript" / "data" / "set_a"
    extent_paths = []
    for i in range(1, 7):
        p = set_a_dir / f"setA_plot{i}_extent.gpkg"
        if not p.exists():
            p2 = set_a_dir / f"setA_plot{i}.gpkg"
            if p2.exists():
                p = p2
        if p.exists():
            extent_paths.append(p)
        else:
            warnings.warn(f"Extent not found for plot {i}: {set_a_dir}/setA_plot{i}_extent.gpkg (or fallback {set_a_dir}/setA_plot{i}.gpkg)")

    if not extent_paths:
        print("No extent files found. Nothing to do.")
        sys.exit(0)

    # Read crowns
    lyr_c = pick_polygon_layer(crowns_path)
    if lyr_c is None:
        raise FileNotFoundError(f"No polygon layers found in {crowns_path}")
    crowns = gpd.read_file(crowns_path, layer=lyr_c)
    if crowns.empty:
        print("Crowns file is empty; nothing to clip.")
        sys.exit(0)

    # Build union of extents
    extents = []
    for ep in extent_paths:
        lyr_e = pick_polygon_layer(ep)
        if lyr_e is None:
            warnings.warn(f"No polygon layer in {ep}; skipping")
            continue
        g = gpd.read_file(ep, layer=lyr_e)
        if g.empty:
            continue
        g = ensure_same_crs(crowns, g)
        # Clean invalid geometries
        try:
            if not g.geometry.is_valid.all():
                g["geometry"] = g.geometry.buffer(0)
        except Exception:
            pass
        # Use the envelope if geometry is not polygonal
        if not all(t in ("Polygon","MultiPolygon") for t in g.geometry.geom_type.unique()):
            g = g.copy(); g["geometry"] = g.geometry.envelope
        extents.append(g)

    if not extents:
        print("No valid extent geometries found; nothing to do.")
        sys.exit(0)

    # Union all extents into a single geometry
    try:
        union_geom = gpd.GeoSeries(pd.concat(extents, ignore_index=True).geometry).unary_union
    except Exception:
        # Fallback without pandas import
        geoms = []
        for g in extents:
            geoms.extend(list(g.geometry))
        from shapely.ops import unary_union as shp_union
        union_geom = shp_union(geoms)

    # Clean crowns geometries
    try:
        if not crowns.geometry.is_valid.all():
            crowns["geometry"] = crowns.geometry.buffer(0)
    except Exception:
        pass

    # Clip
    try:
        crowns_clipped = gpd.clip(crowns, union_geom)
    except Exception as e:
        print(f"Clip failed with error: {e}; attempting intersection overlay")
        crowns_clipped = gpd.overlay(crowns, gpd.GeoDataFrame(geometry=[union_geom], crs=crowns.crs), how="intersection")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.unlink(missing_ok=True)
    except Exception:
        pass
    crowns_clipped.to_file(out_path, layer="sam_crowns", driver="GPKG")
    print(f"[OK] Wrote {out_path} | crowns={len(crowns_clipped)}")


if __name__ == "__main__":
    main()

