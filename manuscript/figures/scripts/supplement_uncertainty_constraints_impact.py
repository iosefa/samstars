#!/usr/bin/env python3
"""
Supplement — Uncertainty vs. constrained regions impact

Purpose:
- Quantify how many XGBoost crown predictions fall inside regions that specify
  acceptable classes (constraints), and among those, how many are low-confidence
  (prediction_margin < 0.25).
- Provide per-class breakdowns in/out of constrained regions.

Inputs:
- Root `classified.gpkg` (table: `classified`; columns: `predicted_class`, `prediction_margin`, `geometry`).
- Root `regions.gpkg` (layer: `combined_classes`; columns: `acceptable_classes`, `geometry`).

Outputs (under manuscript/figures/output):
- supp_uncertainty_constraints_overall.csv
- supp_uncertainty_constraints_by_class.csv
- supp_uncertainty_constraints.png (two panels: overall in/out low-conf rate, and per-class grouped rates)
"""

from pathlib import Path
import warnings
import sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
CROWNS = REPO_ROOT / "classified.gpkg"
REGIONS = REPO_ROOT / "regions.gpkg"
OUT_DIR = ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ID_TO_NAME = {1: "METPOL", 2: "GREROB", 3: "ACAKOA", 4: "DIOSAN", 5: "PROPAL", 6: "OTHER"}
CLASS_ORDER = [1, 2, 3, 4, 5, 6]


def ensure_metric_crs(gdf: gpd.GeoDataFrame, target=None):
    if target is None:
        if (not gdf.crs) or gdf.crs.is_geographic:
            return gdf.to_crs(6635)
        return gdf
    else:
        if (not gdf.crs) or (gdf.crs != target):
            return gdf.to_crs(target)
    return gdf


def read_crowns(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    # Prefer Fast path via fiona/geopandas; fallback to sqlite if needed is out-of-scope here
    g = gpd.read_file(path, layer="classified")
    g = ensure_metric_crs(g)
    # Basic validity fix
    try:
        if not g.geometry.is_valid.all():
            g["geometry"] = g.geometry.buffer(0)
    except Exception:
        pass
    # Keep only necessary fields
    keep = [c for c in ("predicted_class", "prediction_margin", g.geometry.name) if c in g.columns]
    g = g[keep].copy()
    g["predicted_class"] = pd.to_numeric(g["predicted_class"], errors="coerce").astype("Int64")
    g["prediction_margin"] = pd.to_numeric(g["prediction_margin"], errors="coerce")
    g = g.dropna(subset=["predicted_class", "prediction_margin"]).copy()
    g["predicted_class"] = g["predicted_class"].astype(int)
    return g


def _parse_acceptables(cell):
    if pd.isna(cell) or (isinstance(cell, str) and not cell.strip()):
        return []
    if isinstance(cell, list):
        return [int(v) for v in cell if str(v).isdigit()]
    return [int(tok) for tok in str(cell).split(',') if tok.strip().isdigit()]


def read_regions(path: Path, layer: str = "combined_classes") -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    r = gpd.read_file(path, layer=layer)
    r = ensure_metric_crs(r)
    # Normalize acceptable_classes column if present
    if "acceptable_classes" in r.columns:
        r["acceptable_classes"] = r["acceptable_classes"].apply(_parse_acceptables)
    # Keep minimal columns
    keep = [c for c in ("acceptable_classes", r.geometry.name) if c in r.columns]
    return r[keep].copy()


def summarize_in_out(crowns: gpd.GeoDataFrame, regions: gpd.GeoDataFrame, thr: float = 0.25):
    # Spatial join to flag crowns inside any region
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        j = gpd.sjoin(crowns, regions[[regions.geometry.name]], how="left", predicate="intersects")
    in_region = j["index_right"].notna().values
    crowns = crowns.copy()
    crowns["in_region"] = in_region
    crowns["low_conf"] = crowns["prediction_margin"] < thr

    # Overall
    def agg(df):
        n = int(len(df))
        l = int((df["low_conf"]).sum())
        rate = (l / n) if n else 0.0
        return n, l, rate

    total_n, total_l, total_rate = agg(crowns)
    in_n, in_l, in_rate = agg(crowns[crowns["in_region"]])
    out_n, out_l, out_rate = agg(crowns[~crowns["in_region"]])

    overall = pd.DataFrame([
        dict(group="overall_total", n_total=total_n, n_lowconf=total_l, rate=total_rate),
        dict(group="in_region",     n_total=in_n,    n_lowconf=in_l,    rate=in_rate),
        dict(group="out_region",    n_total=out_n,   n_lowconf=out_l,   rate=out_rate),
    ])

    # Per-class breakdown in/out
    crowns["class_name"] = crowns["predicted_class"].map(CLASS_ID_TO_NAME)
    rows = []
    for cls in CLASS_ORDER:
        sub = crowns[crowns["predicted_class"] == cls]
        n_all, l_all, r_all = agg(sub)
        n_in, l_in, r_in = agg(sub[sub["in_region"]])
        n_out, l_out, r_out = agg(sub[~sub["in_region"]])
        rows.append(dict(class_id=cls, class_name=CLASS_ID_TO_NAME.get(cls, str(cls)),
                         n_total=n_all, n_lowconf=n_all and l_all or 0, rate=r_all,
                         n_total_in=n_in, n_lowconf_in=l_in, rate_in=r_in,
                         n_total_out=n_out, n_lowconf_out=l_out, rate_out=r_out))
    by_class = pd.DataFrame(rows)
    return overall, by_class


def plot_figures(overall: pd.DataFrame, by_class: pd.DataFrame, out_path: Path, thr: float = 0.25):
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Left: overall in/out low-confidence rates
    ax = axes[0]
    groups = ["in_region", "out_region"]
    rates = [float(overall.loc[overall["group"] == g, "rate"].values[0]) if g in overall["group"].values else 0.0 for g in groups]
    ns = [int(overall.loc[overall["group"] == g, "n_total"].values[0]) if g in overall["group"].values else 0 for g in groups]
    lows = [int(overall.loc[overall["group"] == g, "n_lowconf"].values[0]) if g in overall["group"].values else 0 for g in groups]

    x = np.arange(len(groups))
    bars = ax.bar(x, rates, color=["#1f77b4", "#ff7f0e"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["In constrained regions", "Outside constrained regions"])
    ax.set_ylabel(f"Low-confidence rate (margin < {thr:.2f})")
    ax.set_ylim(0, max(rates) * 1.20 + 0.02 if rates else 1.0)
    ax.set_title("Low-confidence: constrained vs. unconstrained")
    for b, r in zip(bars, rates):
        ax.annotate(f"{(r*100):.1f}%", xy=(b.get_x() + b.get_width()/2, r),
                    xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    # summary box
    lines = ["Counts (n) and low-conf (n):"] + [f"{lab}: {n:,} | {l:,}" for lab, n, l in zip(["In", "Out"], ns, lows)]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85))

    # Right: per-class grouped rates (in/out)
    ax2 = axes[1]
    classes = [CLASS_ID_TO_NAME[c] for c in CLASS_ORDER]
    bc = by_class.set_index('class_name').reindex(classes)
    rin = bc['rate_in'].fillna(0.0).values
    rout = bc['rate_out'].fillna(0.0).values
    x = np.arange(len(classes))
    w = 0.42
    b1 = ax2.bar(x - w/2, rin, width=w, color="#1f77b4", alpha=0.9, label="In")
    b2 = ax2.bar(x + w/2, rout, width=w, color="#ff7f0e", alpha=0.9, label="Out")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylabel(f"Low-confidence rate (margin < {thr:.2f})")
    ax2.set_ylim(0, max(np.nanmax(rin), np.nanmax(rout)) * 1.20 + 0.02 if len(classes) else 1.0)
    ax2.set_title("Low-confidence by class: in vs out")
    ax2.legend(frameon=False, loc="upper right")
    ax2.yaxis.grid(True, linestyle='-', alpha=0.2)

    # annotate values above bars
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax2.annotate(f"{(v*100):.1f}%", xy=(b.get_x()+b.get_width()/2, v),
                         xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    crowns = read_crowns(CROWNS)
    regions = read_regions(REGIONS, layer="combined_classes")
    overall, by_class = summarize_in_out(crowns, regions, thr=0.25)

    # Save CSVs
    overall_csv = OUT_DIR / "supp_uncertainty_constraints_overall.csv"
    byclass_csv = OUT_DIR / "supp_uncertainty_constraints_by_class.csv"
    overall.to_csv(overall_csv, index=False)
    by_class.to_csv(byclass_csv, index=False)
    print(f"[OK] Saved {overall_csv}")
    print(f"[OK] Saved {byclass_csv}")

    # Plot
    fig_path = OUT_DIR / "supp_uncertainty_constraints.png"
    plot_figures(overall, by_class, fig_path, thr=0.25)


if __name__ == "__main__":
    main()

