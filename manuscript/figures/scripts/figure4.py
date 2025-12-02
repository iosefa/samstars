#!/usr/bin/env python3
"""
Figure 4 — Feature engineering → XGBoost (schematic, publication-ready)

Layout:
- Left: Per-crown features (placeholders for spectral, textural, structural).
- Middle: Feature vector bar (48 spectral, 6 textural, 5 structural).
- Right: XGBoost ensemble → Species prediction.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import patheffects as pe
import geopandas as gpd

# ------------------------------
# helpers
# ------------------------------
def draw_empty_panel(ax, edgecolor="#444", face="#EEEEEE"):
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=face, edgecolor=edgecolor, linewidth=1.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

# ------------------------------
# layout — 4 gridded columns: per‑crown | features | XGBoost | species
# ------------------------------
fig = plt.figure(figsize=(12, 6), dpi=300)
pe_white = [pe.withStroke(linewidth=2.0, foreground="white")]

import matplotlib.gridspec as gridspec
gs_main = fig.add_gridspec(1, 4, width_ratios=[0.7, 0.8, 1.2, 1.5], wspace=0.20)

# -------- Column 1: Per-Crown Features (single axes with three boxes) --------
ax_left = fig.add_subplot(gs_main[0, 0]); ax_left.set_axis_off()
# Target square boxes in display: compute width to match height using axes aspect
box_h = 0.24
_pos = ax_left.get_position()
_aspect = (_pos.height / _pos.width) if _pos.width > 0 else 1.0
box_w = min(box_h * _aspect, 0.84)  # ensure it fits; cap to avoid overflow
left_x = 0.5 - box_w/2.0

# Top: Spectral (blue)
ax_left.add_patch(Rectangle((left_x, 0.70), box_w, box_h, transform=ax_left.transAxes,
                            facecolor="#EEEEEE", edgecolor="#2c6aa6", linewidth=1.8))
ax_left.text(0.5, 0.66, "WV3 Multispectral", transform=ax_left.transAxes,
             ha='center', va='top', fontsize=10, color="#333", path_effects=pe_white)

# Middle: Textural (yellow)
ax_left.add_patch(Rectangle((left_x, 0.38), box_w, box_h, transform=ax_left.transAxes,
                            facecolor="#EEEEEE", edgecolor="#ad7a00", linewidth=1.8))
ax_left.text(0.5, 0.34, "GLCM Texture", transform=ax_left.transAxes,
             ha='center', va='top', fontsize=10, color="#333", path_effects=pe_white)

# Bottom: Structural/Lidar (green)
ax_left.add_patch(Rectangle((left_x, 0.06), box_w, box_h, transform=ax_left.transAxes,
                            facecolor="#EEEEEE", edgecolor="#2c7a4b", linewidth=1.8))
ax_left.text(0.5, 0.02, "Lidar Structure", transform=ax_left.transAxes,
             ha='center', va='top', fontsize=10, color="#333", path_effects=pe_white)

# Column header (positioned dynamically above left column)
# Defer exact x placement until after axes exist; set later.

# -------- Column 2: Feature Vector --------
ax_feats = fig.add_subplot(gs_main[0, 1]); ax_feats.set_axis_off()
# Header will be placed dynamically later.

# Main feature bar (now vertical stack)
# Make the bar significantly wider within the column
bar_w = 0.70
bar_x, bar_y, bar_h = (0.5 - bar_w/2), 0.12, 0.76
ax_feats.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h,
                             facecolor="#E8E8E8", edgecolor="#333", linewidth=2))

# Subsegments (stacked vertically)
total = 59
h_spec = bar_h * (48/total)
h_text = bar_h * ( 6/total)
h_strc = bar_h * ( 5/total)

# Structural (bottom)
ax_feats.add_patch(Rectangle((bar_x, bar_y), bar_w, h_strc,
                             facecolor="#e6f5eb", edgecolor="#2c7a4b", linewidth=1.8))
ax_feats.text(bar_x + bar_w/2, bar_y + h_strc/2,
              "5 Structural", ha="center", va="center",
              fontsize=12, color="#1f5a37")

# Textural (middle)
ax_feats.add_patch(Rectangle((bar_x, bar_y + h_strc), bar_w, h_text,
                             facecolor="#fff0d8", edgecolor="#ad7a00", linewidth=1.8))
ax_feats.text(bar_x + bar_w/2, bar_y + h_strc + h_text/2,
              "6 Textural", ha="center", va="center",
              fontsize=12, color="#7a5600")

# Spectral (top)
ax_feats.add_patch(Rectangle((bar_x, bar_y + h_strc + h_text), bar_w, h_spec,
                             facecolor="#d9ebff", edgecolor="#2c6aa6", linewidth=1.8))
ax_feats.text(bar_x + bar_w/2, bar_y + h_strc + h_text + h_spec/2,
              "48 Spectral", ha="center", va="center",
              fontsize=12, color="#1e4e7a")

# Feature vector caption
ax_feats.text(0.5, 0.05, "59 Dimensions per crown",
              ha="center", va="bottom", fontsize=11, color="#333",
              transform=ax_feats.transAxes)

# -------- Column 3: XGBoost --------
ax_xgb = fig.add_subplot(gs_main[0, 2]); ax_xgb.set_axis_off()
# Header will be placed dynamically later.

# Define a logical drawing area within the axes (formerly the box bounds)
# Use nearly full width/height of the column for the tree
xgb_x, xgb_y, xgb_w, xgb_h = 0.02, 0.04, 0.96, 0.92

# Stylized decision tree inside the box (nodes + connectors)
def _within_xgb(fx, fy):
    """Map normalized coords (0..1 within the ensemble box) to axes fraction."""
    return xgb_x + fx * xgb_w, xgb_y + fy * xgb_h

# Layout (normalized within the drawing area), horizontal orientation
# Five columns (deeper by one):
# root -> 2 children -> 4 grandchildren -> 8 internal -> 8 leaves
# X positions spread to use almost full width; Y positions evenly spaced at the leaves.
levels = {}

root_x, l1_x, l2_x, l3_x, leaf_x = 0.06, 0.30, 0.56, 0.80, 0.94

# Evenly spaced leaves across full height (centered in 8 equal bands)
_y_leaves = [ (i + 0.5) / 8.0 for i in range(8) ]  # 0.0625 .. 0.9375

# Derive parent levels by averaging child positions
_y_l3 = _y_leaves
_y_l2 = [ (_y_l3[2*i] + _y_l3[2*i+1]) / 2.0 for i in range(4) ]
_y_l1 = [ (_y_l2[0] + _y_l2[1]) / 2.0, (_y_l2[2] + _y_l2[3]) / 2.0 ]
_y_root = 0.5

levels[0] = [(root_x, _y_root)]
levels[1] = [(l1_x, _y_l1[0]), (l1_x, _y_l1[1])]
levels[2] = [(l2_x, y) for y in _y_l2]
levels[3] = [(l3_x, y) for y in _y_l3]
levels[4] = [(leaf_x, y) for y in _y_leaves]

# Connector and node color/style (avoid green/orange to reduce confusion)
edge_color = "#4a4a4a"   # connectors (neutral dark gray)
node_edge  = "#6b59b3"   # internal node edge (purple)
leaf_edge  = "#b23a86"   # leaf edge (magenta)
node_face  = "#efe3ff"   # internal node fill (light lavender)
leaf_face  = "#fde2f2"   # leaf fill (light pink)

# Node sizes (relative to smallest side of the drawing area)
base = min(xgb_w, xgb_h)
R_int = base * 0.038  # internal node radius
R_leaf = base * 0.032 # leaf node radius

# Helper: draw connection between two normalized points
def _connect(p0, p1):
    (x0, y0) = _within_xgb(*p0)
    (x1, y1) = _within_xgb(*p1)
    ax_xgb.plot([x0, x1], [y0, y1], '-', color=edge_color, linewidth=1.3,
                transform=ax_xgb.transAxes, zorder=1)

# Build parent-child relationships (balanced fan-out 1->2->4->8, then 1:1 to leaves)
for i in range(2):
    _connect(levels[0][0], levels[1][i])
for i in range(2):
    _connect(levels[1][i], levels[2][2*i])
    _connect(levels[1][i], levels[2][2*i+1])
for i in range(4):
    _connect(levels[2][i], levels[3][2*i])
    _connect(levels[2][i], levels[3][2*i+1])
for i in range(8):
    _connect(levels[3][i], levels[4][i])

# Draw nodes (internal: tan; leaves: greenish)
from matplotlib.patches import Circle

def _circle(fx, fy, r, face, edge, lw=1.4, z=2):
    x, y = _within_xgb(fx, fy)
    circ = Circle((x, y), r, transform=ax_xgb.transAxes,
                  facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax_xgb.add_patch(circ)

# Internal nodes (levels 0..3)
for (fx, fy) in levels[0] + levels[1] + levels[2] + levels[3]:
    _circle(fx, fy, R_int, node_face, node_edge)

# Leaf nodes (level 4)
for (fx, fy) in levels[4]:
    _circle(fx, fy, R_leaf, leaf_face, leaf_edge)

# Optional: tiny split markers on internal nodes for recognizability
for (fx, fy) in levels[0] + levels[1] + levels[2] + levels[3]:
    x, y = _within_xgb(fx, fy)
    ax_xgb.plot([x - R_int*0.6, x + R_int*0.6], [y, y], color=node_edge,
                linewidth=1.0, transform=ax_xgb.transAxes, zorder=3)

# -------- Column 4: Species Prediction --------
ax_pred = fig.add_subplot(gs_main[0, 3]); ax_pred.set_axis_off()
# Header will be placed dynamically later.

# Container box (neutral styling; wider inside the axis)
pred_box = (0.06, 0.30, 0.88, 0.40)
ax_pred.add_patch(FancyBboxPatch((pred_box[0], pred_box[1]), pred_box[2], pred_box[3],
                                 boxstyle="round,pad=0.02,rounding_size=0.03",
                                 facecolor="#ffffff", edgecolor="#333333", linewidth=2))

# Plot segment_id=111325 polygon from objects.gpkg inside the box
try:
    OBJECTS_PATH = Path(__file__).resolve().parents[2].parent / "objects.gpkg"
    seg_id = 111325
    if OBJECTS_PATH.exists():
        objs = gpd.read_file(OBJECTS_PATH)
        if "segment_id" in objs.columns:
            row = objs[objs["segment_id"] == seg_id]
            if not row.empty and row.geometry.notna().any():
                geom = row.iloc[0].geometry
                # Create two inset axes within the rounded box: top for polygon, bottom for label
                pos_pred = ax_pred.get_position()
                ix = pos_pred.x0 + pred_box[0] * pos_pred.width
                iy = pos_pred.y0 + pred_box[1] * pos_pred.height
                iw = pred_box[2] * pos_pred.width
                ih = pred_box[3] * pos_pred.height

                # Make polygon much larger within the box; keep a small area for label
                label_frac = 0.0  # reclaim full height since bottom label is removed
                # Split the polygon area into left (polygon) and right (text) panes
                w_poly = 0.72
                ax_poly = fig.add_axes([ix, iy, iw * w_poly, ih * (1 - label_frac)], zorder=2)
                ax_txt  = fig.add_axes([ix + iw * w_poly, iy, iw * (1 - w_poly), ih * (1 - label_frac)], zorder=2)
                ax_poly.set_axis_off(); ax_txt.set_axis_off()

                # Determine polygon bounds and center-scale to fill axes while preserving aspect
                minx, miny, maxx, maxy = geom.bounds
                dx = float(maxx - minx)
                dy = float(maxy - miny)
                cx = float((minx + maxx) / 2.0)
                cy = float((miny + maxy) / 2.0)
                # Axes aspect ratio (width/height) in pixels/figure fraction
                ar_axes = (iw * w_poly) / (ih * (1 - label_frac) + 1e-9)
                # Base extents with zero padding so the bbox fills the box
                pad = 0.0
                dxp = dx * (1 + pad)
                dyp = dy * (1 + pad)
                # Expand shorter side to match axes aspect
                if dyp > 0 and dxp / dyp < ar_axes:
                    dxp = dyp * ar_axes
                elif dxp > 0 and dxp / dyp > ar_axes:
                    dyp = dxp / ar_axes
                # Set symmetric limits around center
                view_minx = cx - dxp/2.0
                view_maxx = cx + dxp/2.0
                view_miny = cy - dyp/2.0
                view_maxy = cy + dyp/2.0
                ax_poly.set_xlim(view_minx, view_maxx)
                ax_poly.set_ylim(view_miny, view_maxy)
                ax_poly.set_aspect('equal')

                # Color to match class 3 in Figure 10 palette (ACAKOA → #377eb8)
                class3_color = "#377eb8"
                gpd.GeoSeries([geom]).plot(ax=ax_poly, facecolor=class3_color, edgecolor="#1d3f6e", linewidth=0.7)

                # Draw the (expanded-to-fit) bbox as an outline that fills the entire box area
                from matplotlib.patches import Rectangle as BBoxRect
                # (Removed bbox outline per request; polygon fills pane area visually)

                # Right-side text column with composition
                # Add sub-heading centered at the top of the species label box
                ax_pred.text(pred_box[0] + pred_box[2]/2.0,
                             pred_box[1] + pred_box[3] - 0.010,
                             "Class probabilities",
                             ha='center', va='top', fontsize=12, weight='bold', color="#222222",
                             transform=ax_pred.transAxes)
                txt_lines = [
                    "ACAKOA: 87%",
                    "GREROB: 7%",
                    "DIOSAN: 5%",
                    "METPOL: 1%",
                ]
                ys = [0.84, 0.62, 0.40, 0.18]
                for y, s in zip(ys, txt_lines):
                    ax_txt.text(0.03, y, s, ha='left', va='center', fontsize=11, color="#222222",
                                transform=ax_txt.transAxes)

                # (Removed bottom species label text per request)
except Exception:
    pass

# Save to manuscript/figures/output like other figures
ROOT = Path(__file__).resolve().parents[2]  # manuscript
OUT_DIR = ROOT / "figures/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure4.png"

# Place headers as axes titles to guarantee centering within each column
ax_left.set_title("Per-Crown Features", fontsize=13, weight='bold', pad=10)
ax_feats.set_title("Feature Vector", fontsize=13, weight='bold', pad=10)
ax_xgb.set_title("XGBoost (Ensemble)", fontsize=13, weight='bold', pad=10)
ax_pred.set_title("Species Prediction", fontsize=13, weight='bold', pad=10)

plt.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.12)
# Debug borders removed for final rendering

# Add large thick arrows between columns (left -> features -> xgboost -> species)
try:
    from matplotlib.patches import FancyArrowPatch

    def _mid_y(ax):
        p = ax.get_position(); return p.y0 + 0.5 * p.height
    def _right_x(ax):
        p = ax.get_position(); return p.x0 + p.width
    def _left_x(ax):
        p = ax.get_position(); return p.x0

    fig.canvas.draw()
    pad = 0.006
    arrow_kwargs = dict(
        arrowstyle='Simple,head_width=12,head_length=16,tail_width=3.4',
        color='#444444',
        linewidth=0.0,
        alpha=0.95,
        zorder=50,
        mutation_scale=1.0,
    )

    pairs = [(ax_left, ax_feats), (ax_feats, ax_xgb), (ax_xgb, ax_pred)]
    for a, b in pairs:
        x0, y0 = _right_x(a) + pad, _mid_y(a)
        x1, y1 = _left_x(b)  - pad, _mid_y(b)
        # Horizontal alignment: use a straight arrow; if y differs slightly, tilt minimally
        arr = FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, **arrow_kwargs)
        fig.add_artist(arr)
except Exception:
    pass

# Ensure per‑crown boxes are truly square in display after layout
try:
    fig.canvas.draw()
    # Figure dimensions in pixels
    fw, fh = fig.get_size_inches()[0]*fig.dpi, fig.get_size_inches()[1]*fig.dpi
    pos = ax_left.get_position()
    ax_w_px, ax_h_px = pos.width*fw, pos.height*fh
    # Desired square size: based on vertical fraction of axes height
    box_h_frac = 0.24
    s_px = ax_h_px * box_h_frac
    # Cap to fit within axes width with some margin
    s_px = min(s_px, ax_w_px * 0.95)
    # Convert to figure-fraction units
    s_w_fig = s_px / fw
    s_h_fig = s_px / fh
    # Center horizontally within the left axes
    x_fig = pos.x0 + (pos.width - s_w_fig) / 2.0
    # Y positions in axes fraction for the three boxes
    y_fracs = [0.70, 0.38, 0.06]
    for idx, y_frac in enumerate(y_fracs):
        if idx < len(ax_left.patches):
            r = ax_left.patches[idx]
            y_fig = pos.y0 + y_frac * pos.height
            # Switch rectangle to figure coordinates and set exact square size
            r.set_transform(fig.transFigure)
            r.set_x(x_fig)
            r.set_y(y_fig)
            r.set_width(s_w_fig)
            r.set_height(s_h_fig)
    # After sizing boxes, overlay images within them (square PNGs)
    try:
        import matplotlib.image as mpimg
        img_paths = [
            ROOT / "data/ahupuaa/fig4_class3_rgb753.png",
            ROOT / "data/ahupuaa/fig4_class3_glcm_corr.png",
            ROOT / "data/ahupuaa/fig4_class3_pointcloud.png",
        ]
        for idx, y_frac in enumerate(y_fracs):
            if idx >= len(img_paths):
                continue
            y_fig = pos.y0 + y_frac * pos.height
            ax_img = fig.add_axes([x_fig, y_fig, s_w_fig, s_h_fig], zorder=2)
            ax_img.imshow(mpimg.imread(img_paths[idx]))
            ax_img.set_axis_off()
    except Exception:
        pass
except Exception:
    pass
fig.savefig(OUT_FIG, dpi=300)
print(f"[OK] Saved {OUT_FIG}")
