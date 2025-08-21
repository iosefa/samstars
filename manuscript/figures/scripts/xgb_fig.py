#!/usr/bin/env python3
"""
Figure 4 — Feature engineering → XGBoost (schematic, publication-ready)

Layout:
- Left: Per-crown features (placeholders for spectral, textural, structural).
- Middle: Feature vector bar (48 spectral, 6 textural, 5 structural).
- Right: XGBoost ensemble → Species prediction.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import patheffects as pe

# ------------------------------
# helpers
# ------------------------------
def draw_empty_panel(ax, edgecolor="#444", face="#EEEEEE"):
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=face, edgecolor=edgecolor, linewidth=1.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

# ------------------------------
# layout
# ------------------------------
fig = plt.figure(figsize=(14, 6), dpi=300)

gs = fig.add_gridspec(
    nrows=3, ncols=8,
    width_ratios=[1.0, 1.0, 1.0, 0.18, 2.2, 0.08, 1.6, 1.0],
    height_ratios=[1, 1, 1],
    wspace=0.10, hspace=0.28
)

pe_white = [pe.withStroke(linewidth=2.0, foreground="white")]

# -------- Left column: Per-Crown Features --------
ax_wv3   = fig.add_subplot(gs[0, 0])
ax_tex   = fig.add_subplot(gs[1, 0])
ax_lidar = fig.add_subplot(gs[2, 0])

draw_empty_panel(ax_wv3)
draw_empty_panel(ax_tex)
draw_empty_panel(ax_lidar)

# Captions
ax_wv3.text(0.5, -0.14, "WV3 Multispectral",
            ha="center", va="top", fontsize=10, color="#333",
            transform=ax_wv3.transAxes, clip_on=False, path_effects=pe_white)
ax_tex.text(0.5, -0.14, "GLCM Texture",
            ha="center", va="top", fontsize=10, color="#333",
            transform=ax_tex.transAxes, clip_on=False, path_effects=pe_white)
ax_lidar.text(0.5, -0.14, "Lidar Structure",
              ha="center", va="top", fontsize=10, color="#333",
              transform=ax_lidar.transAxes, clip_on=False, path_effects=pe_white)

# Column header
fig.text(0.12, 0.96, "Per-Crown Features", ha="center", va="center",
         fontsize=13, weight="bold")

# -------- Middle-left: Feature Vector --------
ax_feats = fig.add_subplot(gs[:, 1:3]); ax_feats.set_axis_off()
fig.text(0.40, 0.96, "Feature Vector", ha="center", va="center",
         fontsize=13, weight="bold")

# Main feature bar
bar_x, bar_y, bar_w, bar_h = 0.06, 0.33, 0.88, 0.34
ax_feats.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h,
                             facecolor="#E8E8E8", edgecolor="#333", linewidth=2))

# Subsegments
total = 59
w_spec = bar_w * (48/total)
w_text = bar_w * ( 6/total)
w_strc = bar_w * ( 5/total)

ax_feats.add_patch(Rectangle((bar_x, bar_y), w_spec, bar_h,
                             facecolor="#d9ebff", edgecolor="#2c6aa6", linewidth=1.8))
ax_feats.text(bar_x + w_spec/2, bar_y + bar_h/2,
              "48 Spectral", ha="center", va="center",
              fontsize=12, color="#1e4e7a")

ax_feats.add_patch(Rectangle((bar_x + w_spec, bar_y), w_text, bar_h,
                             facecolor="#fff0d8", edgecolor="#ad7a00", linewidth=1.8))
ax_feats.text(bar_x + w_spec + w_text/2, bar_y + bar_h/2,
              "6 Textural", ha="center", va="center",
              fontsize=12, color="#7a5600", rotation=90)

ax_feats.add_patch(Rectangle((bar_x + w_spec + w_text, bar_y), w_strc, bar_h,
                             facecolor="#e6f5eb", edgecolor="#2c7a4b", linewidth=1.8))
ax_feats.text(bar_x + w_spec + w_text + w_strc/2, bar_y + bar_h/2,
              "5 Structural", ha="center", va="center",
              fontsize=12, color="#1f5a37", rotation=90)

# Feature vector caption
ax_feats.text(0.5, 0.05, "59 Dimensions per crown",
              ha="center", va="bottom", fontsize=11, color="#333",
              transform=ax_feats.transAxes)

# -------- Middle-right: XGBoost --------
ax_xgb = fig.add_subplot(gs[:, 4]); ax_xgb.set_axis_off()
fig.text(0.70, 0.96, "XGBoost (Ensemble)", ha="center", va="center",
         fontsize=13, weight="bold")

xgb_x, xgb_y, xgb_w, xgb_h = 0.15, 0.08, 0.70, 0.82
ax_xgb.add_patch(FancyBboxPatch((xgb_x, xgb_y), xgb_w, xgb_h,
                                boxstyle="round,pad=0.014,rounding_size=0.02",
                                facecolor="#f6d7aa", edgecolor="#333", linewidth=2))

# Trees inside
rows, cols = 3, 3
marg_x, marg_y = 0.06, 0.08
cell_w = (xgb_w - (cols + 1)*marg_x) / cols
cell_h = (xgb_h - (rows + 1)*marg_y) / rows
for r in range(rows):
    for c in range(cols):
        cx = xgb_x + marg_x + c*(cell_w + marg_x)
        cy = xgb_y + marg_y + r*(cell_h + marg_y)
        ax_xgb.add_patch(Rectangle((cx, cy), cell_w, cell_h,
                                   facecolor="#fbe6c9", edgecolor="#7b5a2e", linewidth=1.2))

# -------- Right: Species Prediction --------
ax_pred = fig.add_subplot(gs[:, 6]); ax_pred.set_axis_off()
fig.text(0.90, 0.96, "Species Prediction", ha="center", va="center",
         fontsize=13, weight="bold")

ax_pred.add_patch(FancyBboxPatch((0.12, 0.30), 0.76, 0.40,
                                 boxstyle="round,pad=0.02,rounding_size=0.03",
                                 facecolor="#d9efd9", edgecolor="#2b6b2b", linewidth=2))
ax_pred.text(0.50, 0.50, "Species\nlabel", ha="center", va="center",
             fontsize=15, weight="bold", color="#1f4e1f")

# Save
plt.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.12)
fig.savefig("figure4_pipeline.png", dpi=300)
print("[OK] Saved figure4_pipeline.png")