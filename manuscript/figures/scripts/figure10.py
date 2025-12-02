#!/usr/bin/env python3
"""
Figure 10 — Landscape-scale crown map and species composition (Set C AOI)

Panel A: AOI crown map (representative points) colored by species
Panel B: Species composition by canopy area share

Inputs (repo root unless otherwise noted)
- crowns:  classified_xgb.gpkg (polygons with 'predicted_class') [EPSG:6635]
- AOI:     extent.gpkg                                            [EPSG:6635]

Output
- manuscript/figures/output/figure10.png
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Paths
ROOT = Path(__file__).resolve().parents[2]
# Prefer robust path resolution: several candidates exist in this repo
def _resolve_crowns_path():
    cands = [
        (ROOT.parent / "classified_xgb.gpkg", None),
        (ROOT / "classified_xgb.gpkg", None),
        (ROOT.parent / "classified.gpkg", "classified"),
        (ROOT.parent / "classified_new.gpkg", "classified_new"),
    ]
    for p, layer in cands:
        try:
            if p.exists() and p.stat().st_size > 0:
                return p, layer
        except Exception:
            continue
    raise FileNotFoundError(
        "Could not locate crowns GeoPackage. Tried: " + 
        ", ".join(str(p) for p, _ in cands)
    )

# Resolve at import time so CLI failures surface early
_CROWNS_PATH, _CROWNS_LAYER = _resolve_crowns_path()
# AOI extent moved under manuscript/data/ahupuaa
AOI_PATH    = ROOT / "data" / "ahupuaa" / "extent.gpkg"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure10.png"

# Fields and labels
CROWN_CLASS_FIELD = "predicted_class"
SPECIES_LABELS = {1:"METPOL", 2:"GREROB", 3:"ACAKOA", 4:"DIOSAN", 5:"PROPAL", 6:"OTHER"}
SPECIES_ORDER  = ["METPOL","ACAKOA","GREROB","DIOSAN","PROPAL","OTHER"]

# Distinct categorical palette (ColorBrewer Set1-inspired)
# Ensure classes 1 & 2 (METPOL, GREROB) are NOT blue/green since they co-occur often.
PALETTE = {
    "METPOL": "#ff7f00",  # strong orange
    "ACAKOA": "#377eb8",  # blue
    "GREROB": "#984ea3",  # purple
    "DIOSAN": "#e41a1c",  # red
    "PROPAL": "#4daf4a",  # green
    "OTHER":  "#a65628",  # brown
}

plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})


def _to_int_codes(s):
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf[gdf.geometry.notna()].copy()
    out = out[~out.geometry.is_empty]
    try:
        out = out[out.geometry.is_valid]
    except Exception:
        pass
    return out


def main():
    # Load crowns using resolved path/layer
    if _CROWNS_LAYER:
        crowns = gpd.read_file(_CROWNS_PATH, layer=_CROWNS_LAYER)
    else:
        crowns = gpd.read_file(_CROWNS_PATH)
    aoi    = gpd.read_file(AOI_PATH)
    if crowns.crs is None:
        raise RuntimeError("Crowns have no CRS; expected EPSG:6635")
    if aoi.crs is None:
        raise RuntimeError("AOI has no CRS; expected EPSG:6635")
    if crowns.crs != aoi.crs:
        aoi = aoi.to_crs(crowns.crs)

    crowns = clean_geoms(crowns)
    aoi = clean_geoms(aoi)
    crowns[CROWN_CLASS_FIELD] = _to_int_codes(crowns[CROWN_CLASS_FIELD])
    crowns["species"] = crowns[CROWN_CLASS_FIELD].map(SPECIES_LABELS).fillna(crowns[CROWN_CLASS_FIELD].astype(str))
    crowns["species"] = pd.Categorical(crowns["species"], categories=SPECIES_ORDER, ordered=True)

    crowns["area_m2"] = crowns.geometry.area
    sp = (crowns.groupby("species", dropna=False)["area_m2"]
                 .agg(count="size", area_m2="sum")
                 .reset_index())
    total_area_m2 = float(sp["area_m2"].sum())
    sp["share_of_crown_area_pct"] = 100.0 * sp["area_m2"]/total_area_m2 if total_area_m2 else 0.0
    sp = sp.set_index("species").reindex(SPECIES_ORDER).fillna(0.0).reset_index()

    # Plot thin polygon outlines per species for a fine map
    crowns_lines = crowns[["species","geometry"]].copy()
    crowns_lines = clean_geoms(crowns_lines)

    # Figure layout: left panel spans two rows; right side has two stacked subplots
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12.0, 6.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1, 1], wspace=0.03, hspace=0.18)
    ax0 = fig.add_subplot(gs[:, 0])    # (a) map spans both rows
    ax1 = fig.add_subplot(gs[0, 1])    # (b) crown count (top-right)
    ax2 = fig.add_subplot(gs[1, 1])    # (c) crown area share (bottom-right)

    # Panel A
    aoi.boundary.plot(ax=ax0, color="black", linewidth=0.6)
    for sp_name, sub in crowns_lines.groupby("species", observed=True):
        if sub.empty: continue
        color = PALETTE.get(sp_name, "#999999")
        sub.plot(ax=ax0, facecolor='none', edgecolor=color, linewidth=0.12, alpha=0.85)
    ax0.set_axis_off()
    # Title handled later via fig.text for cross-panel alignment
    handles = [Line2D([0],[0], marker="o", linestyle="none", markerfacecolor=PALETTE[s], markeredgecolor="none", markersize=11, label=s) for s in SPECIES_ORDER]
    # Place legend near the right edge, inside the axes; larger, publication-ready
    leg = ax0.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.005),
        borderaxespad=0.0,
        title="Species",
        frameon=True,
        prop={"size": 12},
        title_fontsize=13,
    )
    try: leg._legend_box.align = "left"
    except Exception: pass

    # Scale bar (bottom-left, opaque white background)
    try:
        minx, miny, maxx, maxy = aoi.total_bounds
        width = float(maxx - minx)
        height = float(maxy - miny)
        # Force a 5 km scale bar
        bar_len = 5000.0
        pad_x = width * 0.03
        pad_y = height * 0.03
        bar_h = max(height * 0.006, bar_len * 0.01)
        # Place the scale bar much further right and slightly lower near the bottom edge
        bx = maxx - pad_x - bar_len
        # Place much lower below the visible bottom edge (use AOI height fraction)
        by = miny - height * 0.08
        # background box
        bg_pad = bar_h * 0.4
        bg_h = bar_h * 1.8
        bg = mpatches.Rectangle((bx - bg_pad, by - bg_pad/2), bar_len + 2*bg_pad, bg_h + bg_pad,
                                facecolor='white', edgecolor='none', linewidth=0.0, zorder=3.8, clip_on=False)
        ax0.add_patch(bg)
        # bar
        rect = mpatches.Rectangle((bx, by), bar_len, bar_h, facecolor='black', edgecolor='white', linewidth=0.8, zorder=4, clip_on=False)
        ax0.add_patch(rect)
        # label: show in km if >= 1000 m
        if bar_len >= 1000:
            lab_val = bar_len / 1000.0
            lab_unit = 'km'
        else:
            lab_val = int(bar_len)
            lab_unit = 'm'
        ax0.text(bx + bar_len/2, by + bar_h + bar_h*0.25, f"{lab_val:g} {lab_unit}",
                 ha='center', va='bottom', color='black', fontsize=11, zorder=4, clip_on=False)
    except Exception:
        pass

    # Panel B (top-right) — crown count
    x = np.arange(len(sp))
    colors = [PALETTE.get(s, "#999999") for s in sp["species"]]
    bars_b = ax1.bar(x, sp["count"].values, width=0.55, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels([])  # hide labels on top plot; show on bottom
    ax1.set_ylabel("Count")
    # Title handled later via fig.text for cross-panel alignment
    ax1.yaxis.grid(True, linestyle='-', alpha=0.15)
    # annotate counts above bars
    for b in bars_b:
        h = b.get_height()
        ax1.annotate(f"{int(h):,}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 5),
                     textcoords='offset points', ha='center', va='bottom', fontsize=9)

    # Panel C (bottom-right) — composition (by crown area)
    bars_c = ax2.bar(x, sp["share_of_crown_area_pct"].values, width=0.55, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sp["species"], rotation=15, ha="right")
    ax2.set_ylabel("Share (%)")
    ymax = float(sp[["share_of_crown_area_pct"]].to_numpy().max())
    ax2.set_ylim(0, max(ymax*1.15, 5))
    ax2.set_title("(c) Species composition (by crown area %)", pad=6, fontsize=14)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.15)
    # annotate percent above bars
    for b in bars_c:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 5),
                     textcoords='offset points', ha='center', va='bottom', fontsize=9)

    # Tight layout, then add aligned panel titles in figure coordinates
    plt.tight_layout()
    try:
        pos0 = ax0.get_position()
        pos1 = ax1.get_position()
        # Use a fixed figure Y so both titles sit on the exact same line
        y_title = 0.98
        fig.text(pos0.x0, y_title, "(a) Crown map colored by species", fontsize=15, va='top', ha='left')
        fig.text(pos1.x0, y_title, "(b) Crown count", fontsize=15, va='top', ha='left')
    except Exception:
        pass
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
