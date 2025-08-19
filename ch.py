# plot_global_canopy_height.py
# Requires: geopandas, seaborn, matplotlib (already in your environment)

import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
GPKG_PATH = Path("objects.gpkg")  # change path if needed
CH_FIELD  = "ch"                  # name of the canopy-height column

# ──────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────
gdf = gpd.read_file(GPKG_PATH)

if CH_FIELD not in gdf.columns:
    raise ValueError(f"Column '{CH_FIELD}' not found in {GPKG_PATH}")

ch_values = gdf[CH_FIELD].dropna()

# ──────────────────────────────────────────────────────────────────────
# 2. Global median
# ──────────────────────────────────────────────────────────────────────
global_median = ch_values.median()
print(f"Global median canopy height (ch): {global_median:.3f}")

# ──────────────────────────────────────────────────────────────────────
# 3. Single violin + box plot
# ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(4, 6))

# Violin (distribution)
sns.violinplot(y=ch_values, color="#8ecae6", inner=None)

# Box overlay (summary)
sns.boxplot(
    y=ch_values,
    width=0.25,
    showcaps=True,
    boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1},
    medianprops={"color": "red", "linewidth": 2},
    whiskerprops={"linewidth": 1},
    showfliers=False,
)

plt.title("Distribution of Canopy Height (ch)")
plt.ylabel("Canopy Height (m)")
plt.xticks([0], ["All objects"])
plt.tight_layout()
plt.show()