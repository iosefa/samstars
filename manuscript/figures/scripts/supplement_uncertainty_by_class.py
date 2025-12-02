#!/usr/bin/env python3
"""
Supplement — XGBoost uncertainty by class (low-confidence share and totals)

Reads root-level GeoPackage `classified.gpkg` (table `classified`) and computes:
- Per-class totals and low-confidence counts (prediction_margin < 0.25)
- Low-confidence rate per class

Outputs under manuscript/figures/output:
- supp_uncertainty_by_class.csv  (class_id, class_name, n_total, n_lowconf, rate)
- supp_uncertainty_by_class.png  (bar plot of low-confidence rate per class)
"""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]  # manuscript
REPO_ROOT = ROOT.parent
GPKG_PATH = REPO_ROOT / "classified.gpkg"
OUT_DIR = ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical class mapping used across manuscript
CLASS_ID_TO_NAME = {1: "METPOL", 2: "GREROB", 3: "ACAKOA", 4: "DIOSAN", 5: "PROPAL", 6: "OTHER"}
CLASS_ORDER = [1, 2, 3, 4, 5, 6]


def load_counts(gpkg_path: Path) -> pd.DataFrame:
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Not found: {gpkg_path}")
    with sqlite3.connect(str(gpkg_path)) as con:
        # Ensure columns exist
        cur = con.cursor()
        cols = [r[1] for r in cur.execute("PRAGMA table_info(classified);").fetchall()]
        required = {"predicted_class", "prediction_margin"}
        if not required.issubset(set(cols)):
            raise RuntimeError(f"classified.gpkg: missing columns {sorted(required - set(cols))}")

        # Pull only the fields we need; cast to numeric
        df = pd.read_sql_query(
            "SELECT predicted_class, prediction_margin FROM classified;",
            con,
        )
    # Coerce class to Int; drop rows where class or margin is missing
    df["predicted_class"] = pd.to_numeric(df["predicted_class"], errors="coerce").astype("Int64")
    df["prediction_margin"] = pd.to_numeric(df["prediction_margin"], errors="coerce")
    df = df.dropna(subset=["predicted_class", "prediction_margin"]).copy()
    df["predicted_class"] = df["predicted_class"].astype(int)
    return df


def compute_by_class(df: pd.DataFrame, thr: float = 0.25) -> pd.DataFrame:
    agg = (
        df.assign(low_conf=lambda d: d["prediction_margin"] < thr)
          .groupby("predicted_class")["low_conf"].agg(["count", "sum"])  # count = total; sum = low_conf
          .rename(columns={"count": "n_total", "sum": "n_lowconf"})
          .reset_index()
    )
    # Ensure all classes appear, even if zero
    base = pd.DataFrame({"predicted_class": CLASS_ORDER})
    out = base.merge(agg, on="predicted_class", how="left").fillna({"n_total": 0, "n_lowconf": 0})
    out["n_total"] = out["n_total"].astype(int)
    out["n_lowconf"] = out["n_lowconf"].astype(int)
    out["rate"] = out.apply(lambda r: (r["n_lowconf"] / r["n_total"]) if r["n_total"] else 0.0, axis=1)
    out["class_name"] = out["predicted_class"].map(CLASS_ID_TO_NAME)
    return out[["predicted_class", "class_name", "n_total", "n_lowconf", "rate"]]


def plot_by_class(df: pd.DataFrame, out_path: Path, thr: float = 0.25):
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

    classes = [CLASS_ID_TO_NAME[c] for c in CLASS_ORDER]
    rates = df.set_index("class_name").loc[classes]["rate"].values
    totals = df.set_index("class_name").loc[classes]["n_total"].values
    lows = df.set_index("class_name").loc[classes]["n_lowconf"].values

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
    x = np.arange(len(classes))
    bars = ax.bar(x, rates, color="#1f77b4", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(f"Low-confidence rate (margin < {thr:.2f})")
    ax.set_ylim(0, max(0.01, (rates.max() if len(rates) else 0) * 1.20 + 0.02))
    ax.set_title("Low-confidence by class (XGBoost)")
    ax.yaxis.grid(True, linestyle='-', alpha=0.2)

    # Annotate values as percentages above bars
    for b, r in zip(bars, rates):
        ax.annotate(f"{(r*100):.1f}%", xy=(b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=10)

    # Compact info box: totals per class and overall
    lines = ["Totals (n) and low-conf (n):"]
    for cls, nt, nl in zip(classes, totals, lows):
        lines.append(f"{cls}: {nt:,} | {nl:,}")
    overall_n = int(df["n_total"].sum())
    overall_l = int(df["n_lowconf"].sum())
    overall_p = (overall_l / overall_n * 100.0) if overall_n else 0.0
    lines.append(f"Overall: {overall_n:,} | {overall_l:,} ({overall_p:.2f}%)")
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    df = load_counts(GPKG_PATH)
    byc = compute_by_class(df, thr=0.25)

    # Save CSV
    csv_path = OUT_DIR / "supp_uncertainty_by_class.csv"
    byc.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")

    # Plot
    fig_path = OUT_DIR / "supp_uncertainty_by_class.png"
    plot_by_class(byc, fig_path, thr=0.25)


if __name__ == "__main__":
    main()

