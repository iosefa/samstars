#!/usr/bin/env python3
"""
Figure 6 — Segmentation metrics (Set A)

Inputs (hard-coded):
  - validation:   manuscript/data/set_a/validation_crowns.gpkg
  - SAM–StarDist: manuscript/data/set_a/sam_crowns.gpkg
  - CNN (Mask R-CNN/Detectron2): manuscript/data/set_a/detectron_crowns.gpkg
  - DeepForest:   manuscript/data/set_a/deepforest_crowns.gpkg

Outputs (under manuscript/figures/output):
  - figure6.png
  - figure6_overall_metrics.csv
  - figure6_iou_height.csv
  - figure6_iou_density.csv
  - figure6_pvalues_height.csv
  - figure6_pvalues_density.csv
  - TP/FP/FN GPKGs for SAM, CNN, DeepForest in output/diags/ (e.g., figure6_sam_tp_pred.gpkg, ...)

Notes:
  - IoU threshold is 0.50 for PQ/SQ/RQ and recall.
  - For DeepForest only, IoU is computed using validation bboxes (envelopes) vs. DF predictions (polygons).
  - Height bins: if a prediction file contains a numeric 'ch' field, panel (b) uses it. Otherwise those groups may be empty.
  - Density bins: median split of local canopy cover (reference crowns within 15 m neighborhood).
  - Validation crowns are used as-is (no clipping to prediction extent).
"""

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
try:
    import fiona
except Exception:
    fiona = None

# ---------------- Config (hard-coded paths) ----------------
IOU_THR = 0.50
HEIGHT_FIELD_NAME = "ch"
HEIGHT_SPLIT_M = 10.0
DENSITY_BUFFER_M = 15.0

FIG_DIR = Path(__file__).resolve().parents[1]  # manuscript/figures
MANUSCRIPT_DIR = FIG_DIR.parent               # manuscript
DATA_SET_A = MANUSCRIPT_DIR / "data" / "set_a"
REF_PATH = DATA_SET_A / "validation_crowns.gpkg"
SAM_PATH = DATA_SET_A / "sam_crowns.gpkg"
CNN_PATH = DATA_SET_A / "detectron_crowns.gpkg"
DF_PATH  = DATA_SET_A / "deepforest_crowns.gpkg"

OUT_DIR = FIG_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = OUT_DIR / "diags"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "figure6.png"


# ---------------- Helpers ----------------
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


def ensure_metric_crs(gdf: gpd.GeoDataFrame, target=None):
    if target is None:
        if (not gdf.crs) or gdf.crs.is_geographic:
            return gdf.to_crs(6635)
        return gdf
    else:
        if (not gdf.crs) or (gdf.crs != target):
            return gdf.to_crs(target)
        return gdf


def greedy_match(df_pairs, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df_pairs.empty:
        return df_pairs
    out = df_pairs.sort_values(iou_col, ascending=False)
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out


def compute_counts_and_pq(n_pred, n_val, pairs_df, iou_col="iou"):
    TP = len(pairs_df)
    FP = n_pred - TP
    FN = n_val - TP
    SQ = pairs_df[iou_col].mean() if TP else 0.0
    RQ = TP / (TP + 0.5 * FP + 0.5 * FN) if (TP + FP + FN) else 0.0
    PQ = SQ * RQ
    return PQ, SQ, RQ


def load_gpkg_polygons(path: Path):
    lyr = pick_polygon_layer(path)
    try:
        if lyr is None:
            # Fallback: try default layer; prefer Fiona engine for robust GPKG handling
            gdf = gpd.read_file(path, engine="fiona")
        else:
            gdf = gpd.read_file(path, layer=lyr, engine="fiona")
    except Exception as e:
        raise FileNotFoundError(f"Unable to read {path}: {e}")
    return gdf


def intersections_and_metrics(preds: gpd.GeoDataFrame, validation: gpd.GeoDataFrame, ref_use_bbox=False):
    preds = preds.reset_index(drop=True).copy(); preds["pid"] = preds.index
    validation = validation.reset_index(drop=True).copy(); validation["vid"] = validation.index
    validation = ensure_metric_crs(validation, target=preds.crs)
    # cleanup invalids
    try:
        if not preds.geometry.is_valid.all():
            preds["geometry"] = preds.geometry.buffer(0)
        if not validation.geometry.is_valid.all():
            validation["geometry"] = validation.geometry.buffer(0)
    except Exception:
        pass
    vgeo = validation[["vid","geometry"]]
    if ref_use_bbox:
        vgeo = vgeo.copy(); vgeo["geometry"] = vgeo.geometry.envelope
    inter = gpd.overlay(preds[["pid","geometry"]], vgeo, how="intersection")
    if inter.empty:
        preds["iou_best"] = 0.0
        validation["iou_best_from_pred"] = 0.0
        pairs_all = inter
        pairs_50 = inter
        PQ = SQ = RQ = 0.0
        return inter, preds, validation, PQ, SQ, RQ, pairs_all, pairs_50
    # IoU fields
    ap = preds.set_index("pid").geometry.area
    # Use the same geometry used in intersections for the validation side area
    av = vgeo.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(ap.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(av.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    # Numerical safety: clip to [0, 1]
    inter["iou"] = inter["iou"].clip(lower=0.0, upper=1.0)

    preds["iou_best"] = inter.groupby("pid")["iou"].max().reindex(preds["pid"], fill_value=0).values
    validation["iou_best_from_pred"] = inter.groupby("vid")["iou"].max().reindex(validation["vid"], fill_value=0).values

    pairs_all = greedy_match(inter.loc[inter["iou"] > 0, ["pid","vid","iou"]])
    pairs_50  = greedy_match(inter.loc[inter["iou"] >= IOU_THR, ["pid","vid","iou"]])
    PQ, SQ, RQ = compute_counts_and_pq(preds.shape[0], validation.shape[0], pairs_50)
    return inter, preds, validation, PQ, SQ, RQ, pairs_all, pairs_50


def _pairwise_pvals_mwu(groups):
    k = len(groups)
    p = np.ones((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            gi, gj = np.asarray(groups[i]), np.asarray(groups[j])
            if len(gi) == 0 or len(gj) == 0:
                pv = 1.0
            else:
                try:
                    _, pv = stats.mannwhitneyu(gi, gj, alternative='two-sided')
                except Exception:
                    pv = 1.0
            p[i, j] = p[j, i] = pv
    # Bonferroni
    m = k * (k - 1) / 2
    p = np.minimum(p * m, 1.0)
    return p


def _cld_for_groups_simple(groups, alpha=0.05):
    non_empty = [(i, np.asarray(g)) for i, g in enumerate(groups) if len(g) > 0]
    if len(non_empty) == 0:
        return [''] * len(groups)
    if len(non_empty) == 1:
        out = [''] * len(groups)
        out[non_empty[0][0]] = 'a'
        return out
    order = sorted(non_empty, key=lambda t: np.nanmedian(t[1]), reverse=True)
    p = _pairwise_pvals_mwu([t[1] for t in order])
    letters_map = {}
    used = []
    for oi, (orig_idx, _) in enumerate(order):
        assigned = ''
        for letter in used:
            ok = True
            for oj, (o2, _) in enumerate(order[:oi]):
                if letters_map.get(o2, '') == letter:
                    if p[oi, oj] < alpha:
                        ok = False; break
            if ok:
                assigned = letter; break
        if not assigned:
            nxt = chr(ord('a') + len(used))
            if nxt > 'c': nxt = 'c'
            used.append(nxt)
            assigned = nxt
        letters_map[orig_idx] = assigned
    out = [''] * len(groups)
    for idx, letter in letters_map.items():
        out[idx] = letter
    return out


# Simple Mann–Whitney U p-value helper available before plotting
def mwu_p(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    try:
        _, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
        return float(pv)
    except Exception:
        return np.nan


def parse_args():
    p = argparse.ArgumentParser(description="Figure 6 — Segmentation metrics (Set A)")
    p.add_argument(
        "--save-thresholded-figure",
        action="store_true",
        help="Also save a second figure where panels (a–c) use IoU ≥ 0.5 matched pairs.",
    )
    p.add_argument(
        "--thresholded-out-name",
        type=str,
        default="figure6_iou50.png",
        help="Filename for the thresholded-panels figure (under figures/output/).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # 1) Load reference and predictions
    ref = ensure_metric_crs(load_gpkg_polygons(REF_PATH))
    sam = ensure_metric_crs(load_gpkg_polygons(SAM_PATH), target=ref.crs)
    cnn = ensure_metric_crs(load_gpkg_polygons(CNN_PATH), target=ref.crs)
    dfo = ensure_metric_crs(load_gpkg_polygons(DF_PATH),  target=ref.crs)

    # 2) Prepare reference (no clipping)
    ref_use = ref.reset_index(drop=True)
    ref_use["vid"] = ref_use.index

    # 3) Intersections & metrics per method (DF uses bbox on ref side)
    out = {}
    for label, preds, use_bbox in [
        ("SAM–StarDist", sam, False),
        ("CNN",          cnn, False),
        ("DeepForest",   dfo, True),
    ]:
        inter, preds2, ref2, PQ, SQ, RQ, pairs_all, pairs_50 = intersections_and_metrics(preds, ref_use, ref_use_bbox=use_bbox)
        out[label] = dict(preds=preds2, ref=ref2, inter=inter, PQ=PQ, SQ=SQ, RQ=RQ, pairs_all=pairs_all, pairs_50=pairs_50)

    # 3b) Console summary: mean IoU for greedy matches (IoU > 0)
    labels = ["SAM–StarDist", "CNN", "DeepForest"]
    print("[IoU Means] Greedy matched pairs (IoU > 0):")
    all_vals = []
    for lab in labels:
        vals = out[lab]["pairs_all"]["iou"].values if (out[lab]["pairs_all"] is not None and not out[lab]["pairs_all"].empty) else np.array([])
        if len(vals) > 0:
            all_vals.append(vals)
            print(f"- {lab}: mean IoU = {np.nanmean(vals):.3f} over n={len(vals)} pairs")
        else:
            print(f"- {lab}: mean IoU = n/a over n=0 pairs")
    if len(all_vals) > 0:
        cat = np.concatenate(all_vals)
        print(f"- Overall (all methods): mean IoU = {np.nanmean(cat):.3f} over n={len(cat)} pairs")

    # 4) Height bins on REFERENCE side (use reference 'ch' when available); IoU of matched pairs (IoU > 0)
    hlbl = [f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"]
    def method_height_groups_ref(label, pair_key="pairs_all"):
        pa = out[label].get(pair_key)
        ref_df = out[label]["ref"]
        preds_df = out[label]["preds"]
        if pa is None or pa.empty:
            return [np.array([]), np.array([])]
        # Prefer reference-side CH if present; else fall back to prediction-side
        if HEIGHT_FIELD_NAME in ref_df.columns:
            df = pa.merge(ref_df[["vid", HEIGHT_FIELD_NAME]], on="vid", how="left")
        elif HEIGHT_FIELD_NAME in preds_df.columns:
            df = pa.merge(preds_df[["pid", HEIGHT_FIELD_NAME]], on="pid", how="left")
        else:
            return [np.array([]), np.array([])]
        df["bin"] = pd.cut(
            df[HEIGHT_FIELD_NAME],
            bins=[-float("inf"), HEIGHT_SPLIT_M, float("inf")],
            labels=hlbl,
            right=True,
        )
        g1 = df.loc[df["bin"] == hlbl[0], "iou"].dropna().values
        g2 = df.loc[df["bin"] == hlbl[1], "iou"].dropna().values
        return [np.asarray(g1), np.asarray(g2)]

    GA = method_height_groups_ref("SAM–StarDist", pair_key="pairs_all")
    GB = method_height_groups_ref("CNN",          pair_key="pairs_all")
    GC = method_height_groups_ref("DeepForest",   pair_key="pairs_all")

    # 5) Density bins on validation side (15 m neighborhood cover of reference crowns)
    buffer_area = np.pi * (DENSITY_BUFFER_M ** 2)
    val_union_geom = ref_use.geometry.unary_union
    def local_cover_fraction(geom):
        rp = geom.representative_point(); buf = rp.buffer(DENSITY_BUFFER_M)
        return float(buf.intersection(val_union_geom).area) / buffer_area
    ref_use = ref_use.copy()
    ref_use["lcc_15m"] = ref_use.geometry.apply(local_cover_fraction)
    med_thr = float(ref_use["lcc_15m"].dropna().median()) if ref_use["lcc_15m"].notna().any() else None
    if med_thr is not None:
        ref_use["dens_bin"] = pd.cut(ref_use["lcc_15m"], bins=[-np.inf, med_thr, np.inf], labels=["low","high"])
    else:
        ref_use["dens_bin"] = pd.Series([None]*len(ref_use))

    def method_density_groups(label, pair_key="pairs_all"):
        # build vid->iou for this method (pairs_all matched pairs)
        pa = out[label].get(pair_key)
        if pa is None or pa.empty:
            return [np.array([]), np.array([])]
        m = dict(zip(pa["vid"].values, pa["iou"].values))
        low = ref_use.loc[ref_use["dens_bin"] == "low", "vid"].values
        high = ref_use.loc[ref_use["dens_bin"] == "high", "vid"].values
        g1 = np.array([m[v] for v in low if v in m], dtype=float)
        g2 = np.array([m[v] for v in high if v in m], dtype=float)
        return [g1, g2]

    DA = method_density_groups("SAM–StarDist", pair_key="pairs_all")
    DB = method_density_groups("CNN",          pair_key="pairs_all")
    DC = method_density_groups("DeepForest",   pair_key="pairs_all")

    # 6) Build figure (2x2)
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) IoU distribution per method (matched pairs; violins with box overlays)
    series = [
        (1, out["SAM–StarDist"]["pairs_all"]["iou"].values if not out["SAM–StarDist"]["pairs_all"].empty else np.array([]), "#1f77b4", "SAM–StarDist"),
        (2, out["CNN"]["pairs_all"]["iou"].values if not out["CNN"]["pairs_all"].empty else np.array([]), "#ff7f0e", "CNN"),
        (3, out["DeepForest"]["pairs_all"]["iou"].values if not out["DeepForest"]["pairs_all"].empty else np.array([]), "#2ca02c", "DeepForest")
    ]
    for x, vals, col, lab in series:
        if len(vals) > 0:
            vp = ax_a.violinplot([vals], positions=[x], widths=0.6, showmeans=False, showmedians=False, showextrema=False)
            for b in vp['bodies']:
                b.set_alpha(0.3); b.set_facecolor(col)
            ax_a.boxplot([vals], positions=[x], widths=0.35, patch_artist=True,
                         boxprops=dict(facecolor=col, alpha=0.45, linewidth=1.0), medianprops=dict(color="black", linewidth=1.4),
                         whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=True)
        else:
            ax_a.text(x, 0.5, "No data", ha='center', fontsize=8)
    ax_a.set_xticks([1,2,3]); ax_a.set_xticklabels([s[3] for s in series])
    ax_a.set_ylabel("IoU (matched pairs)")
    ax_a.set_title("(a) IoU distribution")
    ax_a.set_ylim(0.0, 1.25)
    ax_a.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    # Add between-method brackets + stars for overall distributions (no CLD in (a))
    try:
        groups_a = [
            out["SAM–StarDist"]["pairs_all"]["iou"].values if not out["SAM–StarDist"]["pairs_all"].empty else np.array([]),
            out["CNN"]["pairs_all"]["iou"].values if not out["CNN"]["pairs_all"].empty else np.array([]),
            out["DeepForest"]["pairs_all"]["iou"].values if not out["DeepForest"]["pairs_all"].empty else np.array([]),
        ]
        pvals_a = _pairwise_pvals_mwu(groups_a)
        xs = [1, 2, 3]
        top = 1.25
        base = 0.93
        step = 0.065
        lev = 0
        for a in range(3):
            for b in range(a+1, 3):
                if len(groups_a[a]) == 0 or len(groups_a[b]) == 0:
                    continue
                stars = '***' if pvals_a[a,b] < 1e-3 else ('**' if pvals_a[a,b] < 1e-2 else ('*' if pvals_a[a,b] < 5e-2 else 'ns'))
                y0 = base + lev*step
                y1 = min(y0 + 0.020, top - 0.03)
                ax_a.plot([xs[a], xs[a], xs[b], xs[b]], [y0, y1, y1, y0], color='black', linewidth=1.0)
                ax_a.text((xs[a]+xs[b])/2, min(y1 + 0.012, top - 0.01), stars, ha='center', va='bottom', fontsize=9)
                lev += 1
        ax_a.set_ylim(0.0, 1.25)
    except Exception:
        pass

    # (b) IoU by height (two bins on x; methods side-by-side)
    hlbl = [f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"]
    hb = [GA, GB, GC]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    pos = [1, 2]
    for i, pbin in enumerate(pos):
        groups = [hb[0][i] if len(hb[0])>i else np.array([]), hb[1][i] if len(hb[1])>i else np.array([]), hb[2][i] if len(hb[2])>i else np.array([])]
        offs = [-0.2, 0.0, 0.2]
        for g, off, col in zip(groups, offs, cols):
            if len(g) > 0:
                vp = ax_b.violinplot([g], positions=[pbin + off], widths=0.22, showmeans=False, showmedians=False, showextrema=False)
                for b in vp['bodies']:
                    b.set_alpha(0.35); b.set_facecolor(col)
                ax_b.boxplot([g], positions=[pbin + off], widths=0.10, patch_artist=True,
                             boxprops=dict(facecolor=col, alpha=0.35, linewidth=1.0), medianprops=dict(color="black", linewidth=1.2),
                             whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=False)
        # y-lims for annotation placement
        ax_b.set_ylim(0.0, 1.25)
        top = 1.25
        # Between-method brackets within this bin
        try:
            k = len(groups);
            pvals = _pairwise_pvals_mwu(groups)
            # start a bit lower so annotations sit below CLDs
            base = 0.93
            step = 0.065
            lev = 0
            xs = [pbin + off for off in offs]
            for a in range(k):
                for b in range(a+1, k):
                    if len(groups[a]) == 0 or len(groups[b]) == 0:
                        continue
                    stars = '***' if pvals[a,b] < 1e-3 else ('**' if pvals[a,b] < 1e-2 else ('*' if pvals[a,b] < 5e-2 else 'ns'))
                    y0 = base + lev*step
                    y1 = min(y0 + 0.020, top - 0.03)
                    ax_b.plot([xs[a], xs[a], xs[b], xs[b]], [y0, y1, y1, y0], color='black', linewidth=1.0)
                    ax_b.text((xs[a]+xs[b])/2, min(y1 + 0.012, top - 0.01), stars, ha='center', va='bottom', fontsize=9)
                    lev += 1
            ax_b.set_ylim(0.0, 1.25)
        except Exception:
            pass
    # CLD: within-method across bins (two groups per method), mirroring panel (c)
    # Use Dunn/MWU-style CLD assignment via helper, and render above violins.
    try:
        offs_for_methods = [-0.2, 0.0, 0.2]
        hb_methods = [GA, GB, GC]
        letter_y = 1.25 - 0.03
        for m_idx, groups2 in enumerate(hb_methods):
            if len(groups2) == 2:
                cld2 = _cld_for_groups_simple(groups2, alpha=0.05)
                # Debug: report bin sizes and letters to console to diagnose visibility
                try:
                    mname = ["SAM–StarDist", "CNN", "DeepForest"][m_idx]
                    print(f"[CLD-B] {mname}: n_left={len(groups2[0])}, n_right={len(groups2[1])}, letters={cld2}")
                except Exception:
                    pass
                for i, pbin in enumerate(pos):
                    gvals = groups2[i]
                    if len(gvals) > 0 and cld2[i]:
                        ax_b.text(
                            pbin + offs_for_methods[m_idx],
                            letter_y,
                            cld2[i],
                            ha='center',
                            va='top',
                            fontsize=9,
                            color=cols[m_idx],
                            zorder=30,
                            clip_on=False,
                        )
    except Exception:
        pass
    ax_b.set_xticks(pos); ax_b.set_xticklabels(hlbl)
    ax_b.set_xlabel(f"Canopy-height class (split at {int(HEIGHT_SPLIT_M)} m)")
    ax_b.set_ylabel("IoU (matched pairs)")
    ax_b.set_title("(b) IoU by height")
    ax_b.set_ylim(0.0, 1.25)
    ax_b.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    # Note: CLD already drawn above with high z-order; no second pass needed.

    # (c) IoU by density (two bins on x; methods side-by-side)
    dlbl = ["low", "high"]
    db = [DA, DB, DC]
    for i, pbin in enumerate(pos):
        groups = [db[0][i], db[1][i], db[2][i]]
        offs = [-0.2, 0.0, 0.2]
        for g, off, col in zip(groups, offs, cols):
            if len(g) > 0:
                vp = ax_c.violinplot([g], positions=[pbin + off], widths=0.22, showmeans=False, showmedians=False, showextrema=False)
                for b in vp['bodies']:
                    b.set_alpha(0.35); b.set_facecolor(col)
                ax_c.boxplot([g], positions=[pbin + off], widths=0.10, patch_artist=True,
                             boxprops=dict(facecolor=col, alpha=0.35, linewidth=1.0), medianprops=dict(color="black", linewidth=1.2),
                             whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=False)
        # y-lims for annotation placement
        ax_c.set_ylim(0.0, 1.25)
        top = 1.25
        # Between-method brackets within this bin
        try:
            k = len(groups); pvals = _pairwise_pvals_mwu(groups)
            base = 0.93; step = 0.065
            lev = 0; xs = [pbin + off for off in offs]
            for a in range(k):
                for b in range(a+1, k):
                    if len(groups[a]) == 0 or len(groups[b]) == 0:
                        continue
                    stars = '***' if pvals[a,b] < 1e-3 else ('**' if pvals[a,b] < 1e-2 else ('*' if pvals[a,b] < 5e-2 else 'ns'))
                    y0 = base + lev*step
                    y1 = min(y0 + 0.020, top - 0.03)
                    ax_c.plot([xs[a], xs[a], xs[b], xs[b]], [y0, y1, y1, y0], color='black', linewidth=1.0)
                    ax_c.text((xs[a]+xs[b])/2, min(y1 + 0.012, top - 0.01), stars, ha='center', va='bottom', fontsize=9)
                    lev += 1
            ax_c.set_ylim(0.0, 1.25)
        except Exception:
            pass
    # CLD: within-method across bins (two groups per method)
    try:
        offs_for_methods = [-0.2, 0.0, 0.2]
        db_methods = [DA, DB, DC]
        letter_y = 1.25 - 0.03
        for m_idx, groups2 in enumerate(db_methods):
            if len(groups2) == 2:
                cld2 = _cld_for_groups_simple(groups2, alpha=0.05)
                # Debug: report bin sizes and letters to console to diagnose visibility
                try:
                    mname = ["SAM–StarDist", "CNN", "DeepForest"][m_idx]
                    print(f"[CLD-C] {mname}: n_low={len(groups2[0])}, n_high={len(groups2[1])}, letters={cld2}")
                except Exception:
                    pass
                for i, pbin in enumerate(pos):
                    gvals = groups2[i]
                    if len(gvals) > 0 and cld2[i]:
                        ax_c.text(
                            pbin + offs_for_methods[m_idx],
                            letter_y,
                            cld2[i],
                            ha='center',
                            va='top',
                            fontsize=9,
                            color=cols[m_idx],
                        )
    except Exception:
        pass
    ax_c.set_xticks(pos); ax_c.set_xticklabels(dlbl)
    ax_c.set_xlabel("Local canopy density (15 m ref. cover)")
    ax_c.set_ylabel("IoU (matched pairs)")
    ax_c.set_title("(c) IoU by local canopy density")
    ax_c.set_ylim(0.0, 1.25)
    ax_c.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)

    # (d) Overall PQ/SQ/RQ grouped bars (order SQ, RQ, PQ) with value labels
    met = ["SQ", "RQ", "PQ"]
    x = np.arange(len(met))
    labels = ["SAM–StarDist", "CNN", "DeepForest"]
    cols_bar = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    m = len(labels)
    bar_w = 0.75 / m
    for i, (lab, col) in enumerate(zip(labels, cols_bar)):
        vals = [out[lab][k] for k in met]
        centers = x + (i - (m - 1) / 2) * bar_w
        ax_d.bar(centers, vals, width=bar_w, label=lab, color=col, alpha=0.85)
        # Annotate values above bars (match style used in Fig 8)
        for cx, v in zip(centers, vals):
            ax_d.annotate(f"{v:.2f}", xy=(cx, v), xytext=(0, 4), textcoords='offset points',
                          ha='center', va='bottom', fontsize=9, clip_on=False)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(met)
    ax_d.set_ylim(0, 1.08)
    ax_d.set_ylabel("Score")
    ax_d.set_title("(d) Class-Agnostic SQ, RQ, PQ")
    ax_d.legend(frameon=False)

    # Add a single compact top-left box with TP/FP/FN per method
    n_ref = int(len(ref_use))
    lines = ["TP/FP/FN"]
    for lab in labels:
        TP = int(len(out[lab]["pairs_50"]))
        FP = int(len(out[lab]["preds"]) - TP)
        FN = int(n_ref - TP)
        lines.append(f"{lab}: {TP}/{FP}/{FN}")
    txt = "\n".join(lines)
    props = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, linewidth=0.4)
    ax_d.text(0.02, 0.98, txt, transform=ax_d.transAxes, fontsize=8.5, va="top", ha="left", bbox=props)

    # Save figure (default: greedy matched pairs for panels a–c)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {OUT_FIG}")

    # Save CSV summaries
    rows = []
    n_ref = int(len(ref_use))
    for lab in labels:
        TP = int(len(out[lab]["pairs_50"]))
        FP = int(len(out[lab]["preds"]) - TP)
        FN = int(n_ref - TP)
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        rows.append(dict(method=lab, SQ=out[lab]["SQ"], RQ=out[lab]["RQ"], PQ=out[lab]["PQ"], TP=TP, FP=FP, FN=FN, recall50=recall, iou_thr=IOU_THR))
    pd.DataFrame(rows).to_csv(OUT_DIR / "figure6_overall_metrics.csv", index=False)

    # IoU by height (per method)
    rows_h = []
    for lab, h in zip(labels, [GA, GB, GC]):
        if len(h) == 2:
            def _summ(a):
                return dict(n=int(len(a)), mean=float(np.nanmean(a)) if len(a) else float('nan'), sd=float(np.nanstd(a, ddof=1)) if len(a)>1 else float('nan'))
            rows_h.append(dict(method=lab, bin=hlbl[0], **_summ(h[0])))
            rows_h.append(dict(method=lab, bin=hlbl[1], **_summ(h[1])))
    if rows_h:
        pd.DataFrame(rows_h).to_csv(OUT_DIR / "figure6_iou_height.csv", index=False)

    # IoU by density (per method)
    rows_d = []
    for lab, d in zip(labels, [DA, DB, DC]):
        def _summ(a):
            return dict(n=int(len(a)), mean=float(np.nanmean(a)) if len(a) else float('nan'), sd=float(np.nanstd(a, ddof=1)) if len(a)>1 else float('nan'))
        rows_d.append(dict(method=lab, bin="low",  **_summ(d[0])))
        rows_d.append(dict(method=lab, bin="high", **_summ(d[1])))
    pd.DataFrame(rows_d).to_csv(OUT_DIR / "figure6_iou_density.csv", index=False)

    # Export TP/FP/FN GPKGs for all methods (to diags/)
    for lab, prefix in zip(labels, ["figure6_sam", "figure6_cnn", "figure6_df"]):
        preds2 = out[lab]["preds"].copy()
        pairs50 = out[lab]["pairs_50"]
        ref2 = out[lab]["ref"].copy()
        tp_pids = set(pairs50["pid"].astype(int).tolist()) if pairs50 is not None and not pairs50.empty else set()
        tp_vids = set(pairs50["vid"].astype(int).tolist()) if pairs50 is not None and not pairs50.empty else set()
        tp_pred = preds2.loc[preds2["pid"].astype(int).isin(tp_pids)] if len(preds2) else preds2.iloc[0:0]
        fp_pred = preds2.loc[~preds2["pid"].astype(int).isin(tp_pids)] if len(preds2) else preds2.iloc[0:0]
        fn_ref = ref2.loc[~ref2["vid"].astype(int).isin(tp_vids)] if len(ref2) else ref2.iloc[0:0]
        # Save to diags folder regardless of emptiness
        tp_pred.to_file(DIAG_DIR / f"{prefix}_tp_pred.gpkg", driver="GPKG")
        fp_pred.to_file(DIAG_DIR / f"{prefix}_fp_pred.gpkg", driver="GPKG")
        fn_ref.to_file(DIAG_DIR / f"{prefix}_fn_ref.gpkg", driver="GPKG")
    print(f"[OK] Exported TP/FP/FN GPKGs to {DIAG_DIR}")

    # P-values: within-method (bin A vs B) and between-method per bin, for height and density
    def mwu_p(a, b):
        a = np.asarray(a); b = np.asarray(b)
        if len(a) == 0 or len(b) == 0:
            return np.nan
        try:
            _, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
            return float(pv)
        except Exception:
            return np.nan

    # Height panel p-values
    labels3 = ["SAM–StarDist", "CNN", "DeepForest"]
    hlbl = [f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"]
    height_within_rows = []
    for lab, hg in zip(labels3, [GA, GB, GC]):
        if len(hg) == 2:
            p_raw = mwu_p(hg[0], hg[1])
            # Bonferroni across 3 methods in this panel
            p_adj = float(min(p_raw * 3, 1.0)) if not np.isnan(p_raw) else np.nan
            height_within_rows.append(dict(kind="within", method=lab, group_a=hlbl[0], n_a=int(len(hg[0])), group_b=hlbl[1], n_b=int(len(hg[1])), p_raw=p_raw, p_adj=p_adj))
    # Between-method per height bin
    height_between_rows = []
    hb_all = [GA, GB, GC]
    for i, bin_label in enumerate(hlbl):
        groups = [hb_all[0][i] if len(hb_all[0])>i else np.array([]), hb_all[1][i] if len(hb_all[1])>i else np.array([]), hb_all[2][i] if len(hb_all[2])>i else np.array([])]
        # pairwise tests
        for a in range(3):
            for b in range(a+1, 3):
                p_raw = mwu_p(groups[a], groups[b])
                # Bonferroni across 3 pairs within this bin
                p_adj = float(min(p_raw * 3, 1.0)) if not np.isnan(p_raw) else np.nan
                height_between_rows.append(dict(kind="between", bin=bin_label, method_a=labels3[a], n_a=int(len(groups[a])), method_b=labels3[b], n_b=int(len(groups[b])), p_raw=p_raw, p_adj=p_adj))
    df_height_p = pd.DataFrame(height_within_rows + height_between_rows)
    df_height_p.to_csv(OUT_DIR / "figure6_pvalues_height.csv", index=False)
    if not df_height_p.empty:
        print("[P-VALUES] Height panel:")
        print(df_height_p.to_string(index=False))

    # Density panel p-values
    dlbl = ["low", "high"]
    density_within_rows = []
    for lab, dg in zip(labels3, [DA, DB, DC]):
        p_raw = mwu_p(dg[0], dg[1])
        p_adj = float(min(p_raw * 3, 1.0)) if not np.isnan(p_raw) else np.nan
        density_within_rows.append(dict(kind="within", method=lab, group_a=dlbl[0], n_a=int(len(dg[0])), group_b=dlbl[1], n_b=int(len(dg[1])), p_raw=p_raw, p_adj=p_adj))
    density_between_rows = []
    db_all = [DA, DB, DC]
    for i, bin_label in enumerate(dlbl):
        groups = [db_all[0][i], db_all[1][i], db_all[2][i]]
        for a in range(3):
            for b in range(a+1, 3):
                p_raw = mwu_p(groups[a], groups[b])
                p_adj = float(min(p_raw * 3, 1.0)) if not np.isnan(p_raw) else np.nan
                density_between_rows.append(dict(kind="between", bin=bin_label, method_a=labels3[a], n_a=int(len(groups[a])), method_b=labels3[b], n_b=int(len(groups[b])), p_raw=p_raw, p_adj=p_adj))
    df_density_p = pd.DataFrame(density_within_rows + density_between_rows)
    df_density_p.to_csv(OUT_DIR / "figure6_pvalues_density.csv", index=False)
    if not df_density_p.empty:
        print("[P-VALUES] Density panel:")
        print(df_density_p.to_string(index=False))

    # Optionally render a second figure where panels (a–c) use IoU ≥ 0.5 matches
    if args.save_thresholded_figure:
        # Console summary: mean IoU for thresholded matches (IoU ≥ IOU_THR)
        print(f"[IoU Means] Thresholded matched pairs (IoU ≥ {IOU_THR:g}):")
        all_vals_t = []
        for lab in labels:
            vals = out[lab]["pairs_50"]["iou"].values if (out[lab]["pairs_50"] is not None and not out[lab]["pairs_50"].empty) else np.array([])
            if len(vals) > 0:
                all_vals_t.append(vals)
                print(f"- {lab}: mean IoU = {np.nanmean(vals):.3f} over n={len(vals)} pairs")
            else:
                print(f"- {lab}: mean IoU = n/a over n=0 pairs")
        if len(all_vals_t) > 0:
            cat_t = np.concatenate(all_vals_t)
            print(f"- Overall (all methods): mean IoU = {np.nanmean(cat_t):.3f} over n={len(cat_t)} pairs")

        # Recompute height/density groups from thresholded matches
        GA_t = method_height_groups_ref("SAM–StarDist", pair_key="pairs_50")
        GB_t = method_height_groups_ref("CNN",          pair_key="pairs_50")
        GC_t = method_height_groups_ref("DeepForest",   pair_key="pairs_50")
        DA_t = method_density_groups("SAM–StarDist", pair_key="pairs_50")
        DB_t = method_density_groups("CNN",          pair_key="pairs_50")
        DC_t = method_density_groups("DeepForest",   pair_key="pairs_50")

        fig2 = plt.figure(figsize=(11, 8), constrained_layout=True)
        gs2 = fig2.add_gridspec(2, 2)
        ax2_a = fig2.add_subplot(gs2[0, 0])
        ax2_b = fig2.add_subplot(gs2[0, 1])
        ax2_c = fig2.add_subplot(gs2[1, 0])
        ax2_d = fig2.add_subplot(gs2[1, 1])

        # (a) IoU distribution per method (thresholded matched pairs)
        series_t = [
            (1, out["SAM–StarDist"]["pairs_50"]["iou"].values if not out["SAM–StarDist"]["pairs_50"].empty else np.array([]), "#1f77b4", "SAM–StarDist"),
            (2, out["CNN"]["pairs_50"]["iou"].values          if not out["CNN"]["pairs_50"].empty          else np.array([]), "#ff7f0e", "CNN"),
            (3, out["DeepForest"]["pairs_50"]["iou"].values    if not out["DeepForest"]["pairs_50"].empty    else np.array([]), "#2ca02c", "DeepForest"),
        ]
        for x, vals, col, lab in series_t:
            if len(vals) > 0:
                vp = ax2_a.violinplot([vals], positions=[x], widths=0.6, showmeans=False, showmedians=False, showextrema=False)
                for b in vp['bodies']:
                    b.set_alpha(0.3); b.set_facecolor(col)
                ax2_a.boxplot([vals], positions=[x], widths=0.35, patch_artist=True,
                              boxprops=dict(facecolor=col, alpha=0.45, linewidth=1.0), medianprops=dict(color="black", linewidth=1.4),
                              whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=True)
            else:
                ax2_a.text(x, 0.5, "No data", ha='center', fontsize=8)
        ax2_a.set_xticks([1,2,3]); ax2_a.set_xticklabels([s[3] for s in series_t])
        ax2_a.set_ylabel("IoU (matched pairs, IoU ≥ 0.5)")
        ax2_a.set_title("(a) IoU distribution (thresholded)")
        ax2_a.set_ylim(0.0, 1.25)
        ax2_a.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)

        # (b) IoU by height (thresholded)
        hlbl = [f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"]
        hb_t = [GA_t, GB_t, GC_t]
        cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        pos = [1, 2]
        for i, pbin in enumerate(pos):
            groups = [hb_t[0][i] if len(hb_t[0])>i else np.array([]), hb_t[1][i] if len(hb_t[1])>i else np.array([]), hb_t[2][i] if len(hb_t[2])>i else np.array([])]
            offs = [-0.2, 0.0, 0.2]
            for g, off, col in zip(groups, offs, cols):
                if len(g) > 0:
                    vp = ax2_b.violinplot([g], positions=[pbin + off], widths=0.22, showmeans=False, showmedians=False, showextrema=False)
                    for b in vp['bodies']:
                        b.set_alpha(0.35); b.set_facecolor(col)
                    ax2_b.boxplot([g], positions=[pbin + off], widths=0.10, patch_artist=True,
                                   boxprops=dict(facecolor=col, alpha=0.35, linewidth=1.0), medianprops=dict(color="black", linewidth=1.2),
                                   whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=False)
            ax2_b.set_ylim(0.0, 1.25)
        ax2_b.set_xticks(pos); ax2_b.set_xticklabels(hlbl)
        ax2_b.set_xlabel(f"Canopy-height class (split at {int(HEIGHT_SPLIT_M)} m)")
        ax2_b.set_ylabel("IoU (matched pairs, IoU ≥ 0.5)")
        ax2_b.set_title("(b) IoU by height (thresholded)")

        # (c) IoU by density (thresholded)
        dlbl = ["low", "high"]
        db_t = [DA_t, DB_t, DC_t]
        pos_c = [1, 2]
        for i, pbin in enumerate(pos_c):
            groups = [db_t[0][i], db_t[1][i], db_t[2][i]]
            offs = [-0.2, 0.0, 0.2]
            for g, off, col in zip(groups, offs, cols):
                if len(g) > 0:
                    vp = ax2_c.violinplot([g], positions=[pbin + off], widths=0.22, showmeans=False, showmedians=False, showextrema=False)
                    for b in vp['bodies']:
                        b.set_alpha(0.35); b.set_facecolor(col)
                    ax2_c.boxplot([g], positions=[pbin + off], widths=0.10, patch_artist=True,
                                   boxprops=dict(facecolor=col, alpha=0.35, linewidth=1.0), medianprops=dict(color="black", linewidth=1.2),
                                   whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0), showfliers=False)
            ax2_c.set_ylim(0.0, 1.25)
        ax2_c.set_xticks(pos_c); ax2_c.set_xticklabels(dlbl)
        ax2_c.set_xlabel("Local canopy density (15 m radius)")
        ax2_c.set_ylabel("IoU (matched pairs, IoU ≥ 0.5)")
        ax2_c.set_title("(c) IoU by density (thresholded)")

        # (d) Overall SQ/RQ/PQ — same as default (based on IoU ≥ 0.5)
        labels = ["SAM–StarDist", "CNN", "DeepForest"]
        met = ["SQ", "RQ", "PQ"]
        x = np.arange(len(met)); w = 0.25
        for i, lab in enumerate(labels):
            vals = [out[lab][k] for k in met]
            ax2_d.bar(x + (i - 1) * w, vals, width=w, label=lab, alpha=0.85)
        ax2_d.set_xticks(x); ax2_d.set_xticklabels(met)
        ax2_d.set_ylim(0, 1.0)
        ax2_d.set_ylabel("Score")
        ax2_d.set_title("(d) Class-Agnostic SQ, RQ, PQ")
        ax2_d.legend(frameon=False)

        alt_path = OUT_DIR / args.thresholded_out_name
        fig2.savefig(alt_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"[OK] Saved {alt_path}")


if __name__ == "__main__":
    main()
