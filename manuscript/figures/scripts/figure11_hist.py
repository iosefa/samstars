#!/usr/bin/env python3
"""
Figure 11 (supplement) — Histogram comparison in mid‑elevation forests (CAH vs crowns)

Goal
- Show distribution of 30 m canopy fraction for CAH vs crown‑derived maps, focusing on
  mid‑elevation heterogeneous forests (operationalized via CAH classes containing
  the keyword 'mesic'). Also summarize species composition (crowns) within the same
  mid‑elevation footprint to illustrate presence of diverse classes (e.g., DIOSAN/lama,
  GREROB/silky oak, OTHER, etc.).

Outputs (under manuscript/figures/output):
- figure11_hist.png
- figure11_hist_summary.csv (optional, binned histogram counts + means)
- figure11_hist_species_counts.csv (species counts within mid‑elevation footprint)

Notes
- Uses the same inputs as figure11.py: CAH_LandCover.gpkg (EPSG:26904), AOI extent,
  and crown polygons (classified_xgb.gpkg; EPSG:6635).
- "Mid‑elevation" is approximated by CAH class names that include 'mesic' (case‑insensitive),
  including both native/mixed/alien mesic forests and open/closed mesic variants.
  The script prints the exact DN codes and class names used.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

# ---------------- Paths (mirrors figure11.py) ----------------
ROOT = Path(__file__).resolve().parents[2]
# Prefer files located within manuscript/, with fallback to repo root
def _first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    # return the first candidate even if missing; caller will error clearly
    return Path(candidates[0])

CROWNS_PATH = _first_existing(
    ROOT / "classified_xgb.gpkg",      # manuscript-local (preferred)
    ROOT.parent / "classified_xgb.gpkg" # repo root (legacy)
)
AOI_PATH    = ROOT / "data" / "ahupuaa" / "extent.gpkg"
JACOBI_PATH = _first_existing(
    ROOT / "CAH_LandCover.gpkg",       # manuscript-local (preferred)
    ROOT.parent / "CAH_LandCover.gpkg"  # repo root (legacy)
)
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure11_hist.png"

# Fields and config
CROWN_CLASS_FIELD = "predicted_class"
JACOBI_CODE_FIELD = "DN"
GRID_RES_M = 30.0

# CAH forest classes used in Figure 11 (subset + names)
DN_TO_CLASS = {
    1:"Closed ohia wet forest", 2:"Closed koa-ohia wet forest", 3:"Low-stature ohia wet forest",
    4:"Open ohia wet forest", 5:"Open koa-ohia wet forest", 6:"Closed ohia mesic forest",
    7:"Closed hala forest", 8:"Closed koa-ohia mesic forest", 9:"Open ohia mesic forest",
    10:"Open koa-ohia mesic forest", 11:"Closed ohia dry forest",
    12:"Mixed mamane-naio-native trees dry woodland", 13:"Open ohia dry forest",
    14:"Open koa-mamane dry forest", 15:"Native mesic to dry forest and shrubland",
    16:"Mixed native-alien wet forest", 17:"Mixed native-alien mesic forest",
    18:"Mixed native-alien dry forest", 19:"Alien wet forest", 20:"Alien mesic forest",
    21:"Alien dry forest", 22:"Kiawe dry forest and shrubland", 23:"Plantation forest",
}
FOREST_DN = list(DN_TO_CLASS.keys())  # same forest subset used in figure11.py

# Optional detail AOI (same as figure11.py bottom-row focus)
DETAIL_PATH = ROOT / "data" / "landcover_detail.gpkg"  # EPSG:6635

def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None or (gdf.crs.to_epsg() or -1) != epsg:
        return gdf.to_crs(epsg)
    return gdf

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

def _union_all_safe(geom_series: gpd.GeoSeries):
    """Return a unary union using GeoPandas 1.0+ union_all() if available,
    else fall back to unary_union attribute for older versions."""
    try:
        if hasattr(geom_series, "union_all"):
            return geom_series.union_all()
    except Exception:
        pass
    return geom_series.unary_union

def make_grid(aoi: gpd.GeoDataFrame, res: float) -> gpd.GeoDataFrame:
    aoi_union = _union_all_safe(aoi.geometry)
    minx, miny, maxx, maxy = aoi_union.bounds
    nx = math.ceil((maxx - minx) / res)
    ny = math.ceil((maxy - miny) / res)
    cells = [box(minx + i*res, miny + j*res,
                 minx + (i+1)*res, miny + (j+1)*res)
             for i in range(nx) for j in range(ny)]
    grid = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=aoi.crs)
    grid = gpd.overlay(grid, aoi, how="intersection")
    grid["cell_area"] = grid.geometry.area
    return grid.reset_index(drop=True)

def canopy_fraction_by_cell(grid: gpd.GeoDataFrame, crowns: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    inter = gpd.overlay(crowns[["geometry"]], grid[["cell_id","geometry","cell_area"]], how="intersection")
    if inter.empty:
        grid["canopy_frac"] = 0.0
        return grid
    inter["part_area"] = inter.geometry.area
    add = inter.groupby("cell_id")["part_area"].sum().rename("canopy_area")
    grid = grid.merge(add, on="cell_id", how="left")
    grid["canopy_area"] = grid["canopy_area"].fillna(0.0)
    grid["canopy_frac"] = grid["canopy_area"] / grid["cell_area"]
    return grid

def jacobi_fraction_by_cell(grid: gpd.GeoDataFrame, jacobi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    inter = gpd.overlay(grid[["cell_id","geometry","cell_area"]], jacobi[["geometry"]], how="intersection")
    if inter.empty:
        grid["jacobi_frac"] = 0.0
        return grid
    inter["part_area"] = inter.geometry.area
    add = inter.groupby("cell_id")["part_area"].sum().rename("jacobi_area")
    grid = grid.merge(add, on="cell_id", how="left")
    grid["jacobi_area"] = grid["jacobi_area"].fillna(0.0)
    grid["jacobi_frac"] = grid["jacobi_area"] / grid["cell_area"]
    return grid

def main():
    # 1) Read inputs
    crowns = ensure_crs(clean_geoms(gpd.read_file(CROWNS_PATH)), 6635)
    aoi    = ensure_crs(clean_geoms(gpd.read_file(AOI_PATH)), 6635)
    jacobi = ensure_crs(gpd.read_file(JACOBI_PATH), 26904).to_crs(6635)
    if JACOBI_CODE_FIELD not in jacobi.columns:
        raise RuntimeError("Jacobi layer missing DN field")
    jacobi = gpd.overlay(jacobi[[JACOBI_CODE_FIELD, "geometry"]], aoi, how="intersection")
    jacobi = jacobi[jacobi[JACOBI_CODE_FIELD].isin(FOREST_DN)].copy()
    jacobi = clean_geoms(jacobi)

    # 2) Define footprint = detail AOI (same enlargement area as Figure 11)
    try:
        detail = gpd.read_file(DETAIL_PATH)
        if detail.crs is None or (detail.crs.to_epsg() or -1) != 6635:
            detail = detail.to_crs(6635)
        detail = clean_geoms(detail)
        if detail.empty:
            detail = None
    except Exception:
        detail = None

    # If detail AOI not available, fall back to entire AOI
    footprint = detail if (detail is not None and not detail.empty) else aoi

    # Filter CAH to footprint
    jacobi_fp = gpd.overlay(jacobi, footprint, how="intersection")
    jacobi_fp = clean_geoms(jacobi_fp)

    # Diagnostic: list CAH classes present in footprint
    used = [(int(dn), DN_TO_CLASS.get(int(dn), str(dn))) for dn in sorted(set(jacobi_fp[JACOBI_CODE_FIELD].unique().tolist()))]
    if used:
        print("[CAH forest classes in footprint] (DN → name):")
        for dn, nm in used:
            print(f"  {dn} → {nm}")
    else:
        print("[CAH forest classes in footprint] None found")

    # 3) Build 30 m grid and compute canopy fractions
    grid_all = make_grid(aoi, GRID_RES_M)
    # Clip grid to footprint and recompute cell area within footprint
    grid = gpd.overlay(grid_all[["cell_id","geometry"]], footprint, how="intersection")
    grid["cell_area"] = grid.geometry.area
    # Clip crowns to footprint as well to match denominators
    crowns_fp = gpd.overlay(crowns[[CROWN_CLASS_FIELD, "geometry"]], footprint, how="intersection")
    grid = canopy_fraction_by_cell(grid, crowns_fp)
    grid = jacobi_fraction_by_cell(grid, jacobi_fp)
    grid["canopy_frac"] = pd.to_numeric(grid["canopy_frac"], errors="coerce").fillna(0.0).clip(0, 1)
    grid["jacobi_frac"] = pd.to_numeric(grid["jacobi_frac"], errors="coerce").fillna(0.0).clip(0, 1)

    # Keep all cells inside the footprint (including where CAH has 0 cover)
    g_mid = grid.copy()

    # 4) Histogram overlay: canopy_frac (crowns) vs jacobi_frac (CAH) within mid‑elevation footprint
    bins = np.linspace(0.0, 1.0, 21)  # 0–1 in 0.05 steps
    # Area-weighted distributions: weight by canopy area per cell so curves reflect
    # accumulated canopy cover, not just cell frequency.
    cf_frac = pd.to_numeric(g_mid["canopy_frac"], errors="coerce").to_numpy()
    jf_frac = pd.to_numeric(g_mid["jacobi_frac"], errors="coerce").to_numpy()
    ca_area = pd.to_numeric(g_mid.get("canopy_area", 0.0), errors="coerce").fillna(0.0).to_numpy()
    ja_area = pd.to_numeric(g_mid.get("jacobi_area", 0.0), errors="coerce").fillna(0.0).to_numpy()
    m_cf = np.isfinite(cf_frac) & (cf_frac > 0.0) & (cf_frac <= 1.0) & (ca_area > 0.0)
    m_jf = np.isfinite(jf_frac) & (jf_frac > 0.0) & (jf_frac <= 1.0) & (ja_area > 0.0)
    cf_frac_nz = cf_frac[m_cf]
    jf_frac_nz = jf_frac[m_jf]
    ca_area_nz = ca_area[m_cf]
    ja_area_nz = ja_area[m_jf]
    total_ca_area = float(ca_area_nz.sum())
    total_ja_area = float(ja_area_nz.sum())
    # Weighted means of fraction (by canopy area)
    mean_cf = float((cf_frac_nz * ca_area_nz).sum() / total_ca_area) if total_ca_area > 0 else float("nan")
    mean_jf = float((jf_frac_nz * ja_area_nz).sum() / total_ja_area) if total_ja_area > 0 else float("nan")

    # 5) Species composition within mid‑elevation footprint (crowns clipped to jacobi_mid union)
    mid_union = _union_all_safe(footprint.geometry) if (footprint is not None and not footprint.empty) else None
    species_counts = None
    if mid_union is not None and not crowns.empty:
        crowns_mid = crowns[crowns.geometry.intersects(mid_union)].copy()
        # Coerce class codes to integer where possible, then harmonize to 0..5
        cls = pd.to_numeric(crowns_mid[CROWN_CLASS_FIELD], errors="coerce").astype("Int64")
        # Determine indexing scheme: 0..5 or 1..6; shift if needed
        uniq = sorted([int(v) for v in cls.dropna().unique().tolist()])
        if uniq and min(uniq) >= 1 and max(uniq) <= 6:
            cls = (cls - 1).astype("Int64")
        # Keep only 0..5
        cls = cls.where(cls.isin([0,1,2,3,4,5]))
        species_counts = cls.dropna().astype(int).value_counts().sort_index()
    # Map class codes to names for plotting
    CLASS_CODE_TO_NAME = {0:"METPOL",1:"GREROB",2:"ACAKOA",3:"DIOSAN",4:"PROPAL",5:"OTHER"}
    if species_counts is not None and len(species_counts) > 0:
        idx = [int(i) for i in species_counts.index.tolist()]
        labels_sp = [CLASS_CODE_TO_NAME.get(i, str(i)) for i in idx]
        counts_sp = [int(species_counts.loc[i]) for i in idx]
    else:
        labels_sp, counts_sp = [], []

    # 6) Plot
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # (a) KDE + CDF: Area‑scaled KDE on left axis; cumulative curves on right axis
    frac_cah = pd.to_numeric(g_mid["jacobi_frac"], errors="coerce").to_numpy()
    frac_crn = pd.to_numeric(g_mid["canopy_frac"], errors="coerce").to_numpy()
    w_cah = pd.to_numeric(g_mid.get("jacobi_area", 0.0), errors="coerce").fillna(0.0).to_numpy()
    w_crn = pd.to_numeric(g_mid.get("canopy_area", 0.0), errors="coerce").fillna(0.0).to_numpy()

    # Weighted KDE scaled to total canopy area (so integral equals total area)
    xs = np.linspace(0.0, 1.0, 512)
    def weighted_kde_scaled(x, w, xs_grid):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x > 0) & (x <= 1.0)
        if not np.any(m):
            return np.zeros_like(xs_grid), 0.0
        x, w = x[m], w[m]
        total_area = float(w.sum())
        try:
            kde = gaussian_kde(x, weights=w)
            y = kde(xs_grid) * total_area
        except Exception:
            nb = 100
            bins = np.linspace(0.0, 1.0, nb+1)
            hist, edges = np.histogram(x, bins=bins, weights=w)
            bw = edges[1] - edges[0]
            dens = hist / max(bw, 1e-9)
            dens = gaussian_filter1d(dens, sigma=1.0)
            centers = (edges[:-1] + edges[1:]) / 2
            y = np.interp(xs_grid, centers, dens, left=0.0, right=0.0)
        return y, total_area

    y_kde_cah_m2, tot_cah_m2 = weighted_kde_scaled(frac_cah, w_cah, xs)
    y_kde_crn_m2, tot_crn_m2 = weighted_kde_scaled(frac_crn, w_crn, xs)
    y_kde_cah_ha = y_kde_cah_m2 / 1e4
    y_kde_crn_ha = y_kde_crn_m2 / 1e4

    ax1.fill_between(xs, y_kde_cah_ha, color="#1f77b4", alpha=0.20)
    ax1.plot(xs, y_kde_cah_ha, color="#1f77b4", linewidth=1.8, label="CAH (KDE)")
    ax1.fill_between(xs, y_kde_crn_ha, color="#ff7f0e", alpha=0.20)
    ax1.plot(xs, y_kde_crn_ha, color="#ff7f0e", linewidth=1.8, label="Crowns (KDE)")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel("Canopy fraction (30 m)")
    ax1.set_ylabel("Canopy area density (ha per fraction)")
    ax1.set_title("(a) Detail AOI: canopy cover distribution and cumulative area")
    leg1 = ax1.legend(frameon=False, loc='upper left')
    ax1.add_artist(leg1)

    # CDF on right axis
    def cdf_curve(v, w):
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(m):
            return np.array([0.0]), np.array([0.0])
        v, w = v[m], w[m]
        idx = np.argsort(v)
        v, w = v[idx], w[idx]
        cum = np.cumsum(w) / 1e4  # ha
        return v, cum
    x_cdf_cah, y_cdf_cah = cdf_curve(frac_cah, w_cah)
    x_cdf_crn, y_cdf_crn = cdf_curve(frac_crn, w_crn)
    ax1b = ax1.twinx()
    ax1b.plot(x_cdf_cah, y_cdf_cah, color="#1f77b4", linestyle=":", linewidth=1.3, alpha=0.95, label="CAH (CDF)")
    ax1b.plot(x_cdf_crn, y_cdf_crn, color="#ff7f0e", linestyle=":", linewidth=1.3, alpha=0.95, label="Crowns (CDF)")
    ymax = max(y_cdf_cah[-1] if y_cdf_cah.size else 0.0,
               y_cdf_crn[-1] if y_cdf_crn.size else 0.0)
    ax1b.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
    ax1b.set_ylabel("Cumulative canopy area (ha)")
    ax1b.legend(frameon=False, loc='lower right')

    # Totals (hectares)
    ha = 1e4
    ax1.text(0.02, 0.98,
             f"Total cover (ha)\nCAH={tot_cah_m2/ha:.1f}\nCrowns={tot_crn_m2/ha:.1f}",
             transform=ax1.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.85, linewidth=0.4))

    # (b) Species composition within mid‑elevation footprint
    if counts_sp:
        x = np.arange(len(labels_sp))
        bars = ax2.bar(x, counts_sp, color="#6a3d9a", alpha=0.9)
        ax2.set_xticks(x); ax2.set_xticklabels(labels_sp, rotation=0)
        ax2.set_ylabel("Crown count")
        ax2.set_title("(b) Species composition (detail AOI)")
        # annotate counts
        for xi, v in zip(x, counts_sp):
            ax2.annotate(str(v), xy=(xi, v), xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No crowns in mid‑elevation footprint", ha='center', va='center')
        ax2.set_axis_off()

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")

    # 7) Write simple summaries
    try:
        # Histogram counts (for reproducibility)
        dfh = pd.DataFrame({
            "kde_x_frac": xs,
            "kde_area_density_cah_ha": y_kde_cah_ha,
            "kde_area_density_crowns_ha": y_kde_crn_ha,
            "cdf_x_cah_frac": x_cdf_cah,
            "cdf_y_cah_ha": y_cdf_cah,
            "cdf_x_crowns_frac": x_cdf_crn,
            "cdf_y_crowns_ha": y_cdf_crn,
            "total_cover_area_cah_m2": tot_cah_m2,
            "total_cover_area_crowns_m2": tot_crn_m2,
        })
        dfh.to_csv(OUT_DIR / "figure11_hist_summary.csv", index=False)
        if counts_sp:
            pd.DataFrame({"class_code": [int(k) for k in species_counts.index],
                          "class_name": labels_sp,
                          "count": counts_sp}).to_csv(OUT_DIR / "figure11_hist_species_counts.csv", index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
