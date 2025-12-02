#!/usr/bin/env python3
"""
Supplement — XGBoost prediction margin distribution

Reads root-level GeoPackage `classified.gpkg` (table `classified`) and produces:
- Histogram of `prediction_margin` (top-two probability gap), with a vertical line
  at 0.25 and a compact bin-count summary.

Outputs under manuscript/figures/output:
- supp_uncertainty_margin_distribution.png
- supp_uncertainty_margin_bins.csv (bin edges and counts)
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
GPKG_PATH = REPO_ROOT / "classified.gpkg"
OUT_DIR = ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_margins(gpkg_path: Path) -> np.ndarray:
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Not found: {gpkg_path}")
    with sqlite3.connect(str(gpkg_path)) as con:
        df = pd.read_sql_query("SELECT prediction_margin FROM classified;", con)
    x = pd.to_numeric(df["prediction_margin"], errors="coerce").dropna().values
    return x.astype(float)


def compute_bins(x: np.ndarray):
    # Focused bins near 0 for better readability + wider bins later
    # 0–0.05, 0.05–0.10, 0.10–0.15, 0.15–0.20, 0.20–0.25, 0.25–0.30,
    # 0.30–0.40, 0.40–0.50, 0.50–0.75, 0.75–1.00
    edges = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00])
    counts, _ = np.histogram(x, bins=edges)
    return edges, counts


def plot_hist(x: np.ndarray, edges: np.ndarray, counts: np.ndarray, out_path: Path, thr: float = 0.25):
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

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [2.2, 1.2]})

    # Left: full-range histogram with log y-scale (skewed near 1)
    ax = axes[0]
    ax.hist(x, bins=50, color="#1f77b4", alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel("Prediction margin (top-two probability gap)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Margin distribution (full range)")
    ax.axvline(thr, color="#d62728", linestyle="--", linewidth=1.2, label=f"threshold = {thr:.2f}")
    ax.legend(frameon=False, loc="lower left")

    # Right: zoom on [0, 0.5]
    ax2 = axes[1]
    ax2.hist(x[(x >= 0.0) & (x <= 0.5)], bins=np.linspace(0, 0.5, 26), color="#ff7f0e", alpha=0.85)
    ax2.set_xlabel("Margin (0–0.5)")
    ax2.set_ylabel("Count")
    ax2.set_title("Zoom near low margins")
    ax2.axvline(thr, color="#d62728", linestyle="--", linewidth=1.2)

    # Compact bin-counts box
    total = x.size
    low = int((x < thr).sum())
    share = (low / total * 100.0) if total else 0.0
    lines = [f"Crowns: {total:,}", f"< {thr:.2f}: {low:,} ({share:.2f}%)", "", "Bin counts:"]
    for i in range(len(edges) - 1):
        lines.append(f"[{edges[i]:.2f}, {edges[i+1]:.2f}): {int(counts[i]):,}")
    axes[0].text(0.02, 0.98, "\n".join(lines), transform=axes[0].transAxes, ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    x = load_margins(GPKG_PATH)
    edges, counts = compute_bins(x)

    # Save CSV of bin counts
    df_bins = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts.astype(int),
    })
    csv_path = OUT_DIR / "supp_uncertainty_margin_bins.csv"
    df_bins.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")

    # Plot
    fig_path = OUT_DIR / "supp_uncertainty_margin_distribution.png"
    plot_hist(x, edges, counts, fig_path, thr=0.25)


if __name__ == "__main__":
    main()

