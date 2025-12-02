#!/usr/bin/env python3
"""
Figure 9 — Set C RGB basemap with SAM outlines and FP/FN overlays

What it does
- Renders Set C imagery using RGB bands 5,3,2 (1-indexed) from set_c.tif.
- Overlays SAM–StarDist predicted crown polygons as outlines.
- Colors False Positives (FP; predicted unmatched) and False Negatives (FN; reference unmatched)
  using class-agnostic IoU ≥ 0.5 greedy matching.

Inputs (relative to manuscript/)
- imagery: data/set_c/set_c.tif
- reference: data/set_c/set_c_validation.gpkg
- SAM preds: data/set_c/set_c_sam_predicted.gpkg

Output
- figures/output/figure9.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_raster

try:
    import fiona
except Exception:
    fiona = None


ROOT = Path(__file__).resolve().parents[2]  # manuscript
SETC = ROOT / "data" / "set_c"
IMG_PATH = SETC / "set_c.tif"
REF_PATH = SETC / "set_c_validation.gpkg"
SAM_PATH = SETC / "set_c_sam_predicted.gpkg"

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure9.png"

IOU_THR = 0.50


def _to_int_series(series: pd.Series) -> pd.Series:
    try:
        s = pd.to_numeric(series, errors='coerce').astype('Int64')
        return s
    except Exception:
        return pd.Series([None]*len(series), dtype='Int64')


def _standardize_str(series):
    s = series.astype(object)
    return s.astype(str).str.strip().str.lower()


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


def load_gpkg_polygons(path: Path):
    lyr = pick_polygon_layer(path)
    if lyr is None:
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(path, layer=lyr)
    return gdf


def ensure_metric_crs(gdf: gpd.GeoDataFrame, target):
    if (not gdf.crs) or (gdf.crs != target):
        return gdf.to_crs(target)
    return gdf


def greedy_match(df_pairs, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df_pairs is None or df_pairs.empty:
        return df_pairs
    out = df_pairs.sort_values(iou_col, ascending=False)
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out


def compute_fp_fn(preds: gpd.GeoDataFrame, refs: gpd.GeoDataFrame, iou_thr=0.5):
    preds = preds.reset_index(drop=True).copy(); preds["pid"] = preds.index
    refs = refs.reset_index(drop=True).copy(); refs["vid"] = refs.index

    # Intersections (class-agnostic)
    inter = gpd.overlay(preds[["pid","geometry"]], refs[["vid","geometry"]], how="intersection")
    if inter.empty:
        # no TP; all preds are FP, all refs are FN
        return preds.iloc[0:0], preds, refs, preds.iloc[0:0]

    # IoU
    ap = preds.set_index("pid").geometry.area
    av = refs.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(0.0, 1.0)
    pairs = greedy_match(inter.loc[inter["iou"] >= iou_thr, ["pid","vid","iou"]])

    tp_pids = set(pairs["pid"].astype(int).tolist()) if not pairs.empty else set()
    tp_vids = set(pairs["vid"].astype(int).tolist()) if not pairs.empty else set()

    fp_pred = preds.loc[~preds["pid"].astype(int).isin(tp_pids)]
    fn_ref  = refs.loc[~refs["vid"].astype(int).isin(tp_vids)]
    tp_pred = preds.loc[preds["pid"].astype(int).isin(tp_pids)]
    return pairs, fp_pred, fn_ref, tp_pred


def compute_fp_fn_classaware(
    preds: gpd.GeoDataFrame,
    refs: gpd.GeoDataFrame,
    pred_class_field: str = 'predicted_class',
    ref_class_field: str = 'class',
    iou_thr: float = 0.5,
):
    preds = preds.reset_index(drop=True).copy(); preds["pid"] = preds.index
    refs = refs.reset_index(drop=True).copy(); refs["vid"] = refs.index

    inter = gpd.overlay(preds[["pid","geometry"]], refs[["vid","geometry"]], how="intersection")
    if inter.empty:
        return preds.iloc[0:0], preds, refs, preds.iloc[0:0]

    # IoU
    ap = preds.set_index("pid").geometry.area
    av = refs.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    inter["iou"] = inter["iou"].clip(0.0, 1.0)

    # Class alignment/matching (mirror logic from fig8)
    p_raw = preds.get(pred_class_field)
    r_raw = refs.get(ref_class_field)
    p_num = _to_int_series(p_raw)
    r_num = _to_int_series(r_raw)
    if p_num.notna().any() and r_num.notna().any():
        inter = inter.merge(preds[["pid"]].assign(pred_cls=p_num), on="pid", how="left")
        inter = inter.merge(refs[["vid"]].assign(ref_cls=r_num), on="vid", how="left")
        tbl = inter.dropna(subset=["pred_cls", "ref_cls"])[["pred_cls","ref_cls"]]
        mapping = {}
        if not tbl.empty:
            p_vals = sorted(tbl["pred_cls"].astype(int).unique().tolist())
            r_vals = sorted(tbl["ref_cls"].astype(int).unique().tolist())
            if p_vals and r_vals and min(p_vals) == 0 and min(r_vals) >= 1 and len(p_vals) == len(r_vals):
                shift = min(r_vals) - min(p_vals)
                mapping = {pi: pi + shift for pi in p_vals}
            else:
                counts = {int(pi): {int(rj): 0 for rj in r_vals} for pi in p_vals}
                for a, b in tbl.values:
                    a = int(a); b = int(b)
                    counts[a][b] = counts[a].get(b, 0) + 1
                assigned = set()
                for pi in p_vals:
                    cand = sorted(counts[pi].items(), key=lambda kv: (-kv[1], kv[0]))
                    for rj, _cnt in cand:
                        if rj not in assigned:
                            mapping[pi] = rj; assigned.add(rj); break
                    if pi not in mapping and cand:
                        mapping[pi] = cand[0][0]
        inter["pred_cls_map"] = inter["pred_cls"].apply(lambda v: mapping.get(int(v), np.nan) if pd.notna(v) else np.nan)
        inter = inter.loc[(inter["pred_cls_map"].notna()) & (inter["ref_cls"].notna()) & (inter["pred_cls_map"] == inter["ref_cls"]), ["pid","vid","iou"]].copy()
    else:
        preds["cls_pred"] = _standardize_str(p_raw)
        refs["cls_ref"] = _standardize_str(r_raw)
        inter = inter.merge(preds[["pid","cls_pred"]], on="pid", how="left")
        inter = inter.merge(refs[["vid","cls_ref"]], on="vid", how="left")
        inter = inter.loc[inter["cls_pred"] == inter["cls_ref"], ["pid","vid","iou"]].copy()

    pairs = greedy_match(inter.loc[inter["iou"] >= iou_thr, ["pid","vid","iou"]])
    tp_pids = set(pairs["pid"].astype(int).tolist()) if not pairs.empty else set()
    tp_vids = set(pairs["vid"].astype(int).tolist()) if not pairs.empty else set()
    fp_pred = preds.loc[~preds["pid"].astype(int).isin(tp_pids)]
    fn_ref  = refs.loc[~refs["vid"].astype(int).isin(tp_vids)]
    tp_pred = preds.loc[preds["pid"].astype(int).isin(tp_pids)]
    return pairs, fp_pred, fn_ref, tp_pred


def read_rgb_bands(path: Path, bands_1idx=(5,3,2)):
    with rasterio.open(path) as src:
        rgb = src.read(list(bands_1idx))  # (3,H,W)
        bounds = src.bounds
        crs = src.crs
    # Per-channel 2–98% scaling to 0..1
    out = []
    for c in range(rgb.shape[0]):
        band = rgb[c].astype(np.float32)
        finite = np.isfinite(band)
        if not np.any(finite):
            out.append(np.zeros_like(band, dtype=np.float32))
        else:
            lo, hi = np.percentile(band[finite], [2, 98])
            if hi <= lo:
                lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0
            band = np.clip(band, lo, hi)
            band = (band - lo) / max(hi - lo, 1e-6)
            out.append(band)
    rgb01 = np.stack(out, axis=0)
    # to HxWx3
    img = np.transpose(rgb01, (1,2,0))
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return img, crs, extent


def main():
    # Read imagery
    img, img_crs, extent = read_rgb_bands(IMG_PATH, (5,3,2))

    # Load vectors and align CRS
    ref = load_gpkg_polygons(REF_PATH)
    sam = load_gpkg_polygons(SAM_PATH)
    ref = ensure_metric_crs(ref, img_crs)
    sam = ensure_metric_crs(sam, img_crs)

    # Compute TP/FP/FN using class-aware matching to align with Figure 8
    pairs, fp_pred, fn_ref, tp_pred = compute_fp_fn_classaware(sam, ref, pred_class_field='predicted_class', ref_class_field='class', iou_thr=IOU_THR)

    # Plot
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    ax.imshow(img, extent=extent, origin='upper', zorder=1)
    ax.set_axis_off()
    # Title
    try:
        ax.set_title("SAM–StarDist on Set C", fontsize=14)
    except Exception:
        pass

    # FP predicted (fill)
    if not fp_pred.empty:
        fp_pred.plot(ax=ax,
                     facecolor=(1.0, 0.0, 0.0, 0.50),
                     edgecolor=(0.8, 0.0, 0.0, 0.90),
                     linewidth=0.6,
                     zorder=2)

    # FN reference (fill)
    if not fn_ref.empty:
        fn_ref.plot(ax=ax,
                    facecolor=(1.0, 0.8, 0.0, 0.50),
                    edgecolor=(0.9, 0.6, 0.0, 0.90),
                    linewidth=0.6,
                    zorder=2)

    # True Positive predicted outlines on top
    try:
        if tp_pred is not None and not tp_pred.empty:
            tp_pred.plot(ax=ax, facecolor='none', edgecolor='#00ffff', linewidth=0.8, alpha=0.95, zorder=3)
    except Exception:
        pass

    # Ensure axes limits match the image extent so the map border aligns with axes transforms
    try:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    except Exception:
        pass

    # Simple legend proxies
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=(0,0,0,0), edgecolor='#00ffff', label='True Positive'),
        mpatches.Patch(facecolor=(1.0,0.0,0.0,0.50), edgecolor=(0.8,0.0,0.0,0.9), label='False Positive'),
        mpatches.Patch(facecolor=(1.0,0.8,0.0,0.50), edgecolor=(0.9,0.6,0.0,0.9), label='False Negative')
    ]
    leg = ax.legend(handles=handles, frameon=True, loc='upper right')
    try:
        leg.get_frame().set_alpha(0.95)
    except Exception:
        pass

    # Scale bar (meters) in bottom-left corner with opaque white background
    try:
        # Compute a nice round bar length up to ~1/4 of figure width
        x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
        width = float(x1 - x0)
        height = float(y1 - y0)
        candidates = [50, 100, 200, 250, 500, 1000, 2000]
        target = width * 0.25
        bar_len = max([c for c in candidates if c <= target] or [min(candidates)])
        pad_x = width * 0.03
        pad_y = height * 0.03
        bar_h = max(height * 0.006, bar_len * 0.01)
        bx = x0 + pad_x
        by = y0 + pad_y
        # Background box (opaque white) behind bar and label
        bg_pad = pad_y * 0.4
        bg_h = bar_h + pad_y  # room for label above bar
        bg = mpatches.Rectangle((bx - bg_pad, by - bg_pad/2), bar_len + 2*bg_pad, bg_h + bg_pad,
                                facecolor='white', edgecolor='black', linewidth=0.6, zorder=3.8)
        ax.add_patch(bg)
        # Bar on top of background
        rect = mpatches.Rectangle((bx, by), bar_len, bar_h, facecolor='black', edgecolor='white', linewidth=0.8, zorder=4)
        ax.add_patch(rect)
        # Label in black over white background
        ax.text(bx + bar_len/2, by + bar_h + pad_y*0.2, f"{int(bar_len)} m", ha='center', va='bottom', color='black', fontsize=9,
                path_effects=None, zorder=4)
    except Exception:
        pass

    # Donut chart with TP/FP/FN counts in top-left
    try:
        tp_n = int(len(pairs))
        fp_n = int(len(fp_pred))
        fn_n = int(len(fn_ref))
        vals = [tp_n, fp_n, fn_n]
        colors = ['#00ffff', '#ff0000', '#ffcc00']
        # Place inset using DATA coordinates so it hugs the TIFF border exactly
        # Slightly smaller size, anchored at the same top-left
        width_data  = (extent[1] - extent[0]) * 0.24
        height_data = width_data
        eps_x = (extent[1] - extent[0]) * 0.001  # hairline inset from left
        eps_y = (extent[3] - extent[2]) * 0.001  # hairline inset from top
        left_data = extent[0] + eps_x
        bottom_data = extent[3] - height_data - eps_y
        # Fallback clamp if extent is smaller than chosen size
        if bottom_data < extent[2] + eps_y:
            height_data = max((extent[3] - extent[2]) * 0.20, (extent[3] - extent[2]) - 2*eps_y)
            width_data = height_data
            bottom_data = extent[3] - height_data - eps_y
        ax_in = ax.inset_axes([left_data, bottom_data, width_data, height_data], transform=ax.transData)
        # Explicit semi-transparent background rectangle under inset content
        from matplotlib.patches import Rectangle
        bg = Rectangle((0, 0), 1, 1, transform=ax_in.transAxes,
                       facecolor='white', edgecolor='black', linewidth=0.6, alpha=0.78, zorder=0)
        ax_in.add_patch(bg)

        # Draw donut (no autopct); we'll place counts ourselves to avoid overlap
        ring_width = 0.40
        wedges, _ = ax_in.pie(
            vals,
            colors=colors,
            startangle=90,
            radius=1.0,
            labels=None,
            wedgeprops=dict(width=ring_width, edgecolor='white'),
        )

        # Place counts inside each wedge and short labels near the counts (no leader lines)
        import math
        import matplotlib.patheffects as pe
        labels = ['TP', 'FP', 'FN']
        inner_r = 1.0 - ring_width/2  # centerline of ring
        for w, val, lab in zip(wedges, vals, labels):
            ang = 0.5 * (w.theta1 + w.theta2) * math.pi / 180.0
            # Count inside ring (slightly toward center)
            cx, cy = (inner_r - 0.08) * math.cos(ang), (inner_r - 0.08) * math.sin(ang)
            if val > 0:
                t = ax_in.text(cx, cy, f"{val}", ha='center', va='center', fontsize=9, color='black')
                t.set_path_effects([pe.withStroke(linewidth=1.2, foreground='white')])
            # Label near count using a fixed point offset to avoid overlap
            ax_in.annotate(
                lab,
                xy=(cx, cy), xycoords='data',
                xytext=(0, -10), textcoords='offset points',
                ha='center', va='top', fontsize=8,
            )

        # Center total
        total_n = tp_n + fp_n + fn_n
        ct = ax_in.text(0.0, 0.0, f"Total\n{total_n}", ha='center', va='center', fontsize=9, fontweight='bold')
        ct.set_path_effects([pe.withStroke(linewidth=1.2, foreground='white')])
        ax_in.set_aspect('equal')
        ax_in.set_xticks([]); ax_in.set_yticks([])
        # Thin border around inset
        for spine in ax_in.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.6); spine.set_edgecolor('black')
    except Exception:
        pass

    fig.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
