#!/usr/bin/env python3
"""
Make Fig 14: species composition within Jacobi forest classes.

Inputs:
  - pw_section34_species_composition_by_jacobi_class.csv  (from section34 script)

Outputs:
  - fig14_jacobi_species_heatmap.png
  - fig14_jacobi_species_stackedbars_top6.png  (optional, also saved)

Notes:
  - Values are % of canopy area within each Jacobi class (row-normalized).
  - Rows are sorted by total class area in the AOI (descending).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Config ----------------
IN_CSV = "pw_section34_species_composition_by_jacobi_class.csv"
OUT_HEAT = "fig14_jacobi_species_heatmap.png"
OUT_STACK = "fig14_jacobi_species_stackedbars_top6.png"

# Keep species order consistent with the manuscript
SPECIES_ORDER = ["METPOL","ACAKOA","GREROB","DIOSAN","PROPAL","OTHER"]

# Color palette (match your other figs)
PALETTE = {
    "METPOL": "#1f78b4",  # blue
    "ACAKOA": "#6a3d9a",  # purple
    "GREROB": "#33a02c",  # green
    "DIOSAN": "#e31a1c",  # red
    "PROPAL": "#ff7f00",  # orange
    "OTHER":  "#b2b2b2",  # grey
}

# Option: only show Jacobi classes that matter (by total canopy area in AOI)
MAX_ROWS_HEATMAP = 12   # show top N Jacobi classes in heatmap
TOPN_FOR_STACK   = 6    # and top N for stacked bars

plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

# ------------- Load & tidy --------------
comp = pd.read_csv(IN_CSV)
# Expect columns: jacobi_class, species, part_area_m2, class_area_m2, species_share_pct
req = {"jacobi_class","species","species_share_pct","class_area_m2"}
missing = req - set(comp.columns)
if missing:
    raise SystemExit(f"[Fig14] Missing columns in {IN_CSV}: {sorted(missing)}")

# Collapse to one row per (class,species) with share %
comp = comp.groupby(["jacobi_class","species"], as_index=False)["species_share_pct"].sum()

# Total AOI canopy area per Jacobi class (for sorting/importance)
tot = (pd.read_csv(IN_CSV)
         .groupby("jacobi_class", as_index=False)["class_area_m2"].max())
tot = tot.sort_values("class_area_m2", ascending=False)
keep_rows = tot["jacobi_class"].head(MAX_ROWS_HEATMAP).tolist()

# Pivot to matrix (rows=classes, cols=species in fixed order)
mat = (comp.pivot(index="jacobi_class", columns="species", values="species_share_pct")
          .reindex(index=keep_rows)
          .reindex(columns=SPECIES_ORDER)
          .fillna(0.0))

# ------------- Panel A: Heatmap ----------
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0, vmax=max(1, mat.values.max()))

# Y ticks = Jacobi class names
ax.set_yticks(range(mat.shape[0]))
ax.set_yticklabels(mat.index, ha="right")
# X ticks = species codes
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels(SPECIES_ORDER, rotation=30, ha="right")

ax.set_xlabel("Species")
ax.set_ylabel("Jacobi forest class")
ax.set_title("A) Species composition by Jacobi class (share of canopy area, %)")

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
cbar.set_label("% of canopy area within class")

plt.tight_layout()
fig.savefig(OUT_HEAT, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved {OUT_HEAT}")

# -------- Panel B: Stacked bars (top N classes) --------
top_classes = tot["jacobi_class"].head(TOPN_FOR_STACK).tolist()
mat_top = (comp.pivot(index="jacobi_class", columns="species", values="species_share_pct")
             .reindex(index=top_classes)
             .reindex(columns=SPECIES_ORDER)
             .fillna(0.0))

fig2, ax2 = plt.subplots(figsize=(10, 5))
x = np.arange(mat_top.shape[0])
bottom = np.zeros(mat_top.shape[0])

for sp in SPECIES_ORDER:
    vals = mat_top[sp].to_numpy()
    ax2.bar(x, vals, bottom=bottom, color=PALETTE[sp], label=sp, width=0.65)
    bottom += vals

ax2.set_xticks(x)
ax2.set_xticklabels(mat_top.index, rotation=20, ha="right")
ax2.set_ylim(0, max(100, (bottom.max() * 1.05)))
ax2.set_ylabel("Share of canopy area within class (%)")
ax2.set_title("B) Species composition (stacked), top classes")
ax2.legend(ncol=len(SPECIES_ORDER), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))

plt.tight_layout()
fig2.savefig(OUT_STACK, bbox_inches="tight")
plt.close(fig2)
print(f"[OK] Saved {OUT_STACK}")