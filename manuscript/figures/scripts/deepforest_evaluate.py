#!/usr/bin/env python3
"""
DeepForest-based segmentation evaluation (class-agnostic), aligned with Fig. 6 metrics.

Inputs:
  --ref   Path to reference crowns (GeoPackage, polygon layer). Example: ../area51_labelled_validated_subset1.gpkg
  --pred  Path to DeepForest predicted crowns (GeoPackage, polygon layer). If omitted,
          the script will attempt to import DeepForest and requires --image to run inference.
  --image Optional orthorectified image used for inference if running DeepForest.
  --chm   Optional CHM raster to derive canopy height per reference crown (median), default ../chm.tif

Outputs (under manuscript/figures/output by default):
  - <out_prefix>_metrics_overall.csv         # class-agnostic PQ/SQ/RQ at IoU ≥ 0.50
  - <out_prefix>_iou_height.csv              # matched-only IoU by height bins (≤/> split)
  - <out_prefix>_iou_density.csv             # matched-only IoU by density bins (low/high within 15 m)
  - <out_prefix>_predicted.gpkg (optional)   # saved predictions if run via DeepForest (bbox→polygon)
  - <out_prefix>_figure6_style.png (optional) # single-method figure (like Fig.6 but for one method)

Notes:
  - IoU distributions are computed on matched one-to-one pairs only (IoU > 0) to reflect shape quality.
  - Coverage (recall@0.50, TP/FP/FN) is reported in the overall CSV.
  - Height bins are computed from CHM per reference crown (median); density uses reference crowns’ 15 m local cover.
"""

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt

try:
    import fiona
except Exception:
    fiona = None

IOU_THR = 0.50
HEIGHT_SPLIT_M = 10.0
DENSITY_BUFFER_M = 15.0

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def ensure_same_crs(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if a.crs is None or b.crs is None:
        raise RuntimeError("Both GeoDataFrames must have a valid CRS.")
    return b if a.crs == b.crs else b.to_crs(a.crs)


def greedy_match(df_pairs, pid_col="pid", vid_col="vid", iou_col="iou"):
    if df_pairs.empty:
        return df_pairs
    out = df_pairs.sort_values(iou_col, ascending=False).copy()
    out = out.drop_duplicates(subset=pid_col)
    out = out.drop_duplicates(subset=vid_col)
    return out


def intersections_and_pairs(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, ref_use_bbox: bool = False):
    pred = pred.reset_index(drop=True).copy(); pred["pid"] = pred.index
    ref = ref.reset_index(drop=True).copy(); ref["vid"] = ref.index
    ref = ensure_same_crs(pred, ref)
    # Attempt to fix invalid geometries
    try:
        if not pred.geometry.is_valid.all():
            pred["geometry"] = pred.geometry.buffer(0)
        if not ref.geometry.is_valid.all():
            ref["geometry"] = ref.geometry.buffer(0)
    except Exception:
        pass
    ref_geo = ref[["vid","geometry"]]
    if ref_use_bbox:
        ref_geo = ref_geo.copy()
        ref_geo["geometry"] = ref_geo.geometry.envelope
    inter = gpd.overlay(pred[["pid","geometry"]], ref_geo, how="intersection")
    if inter.empty:
        inter["iou"] = []
        pairs_all = inter
        pairs_50 = inter
        return inter, pairs_all, pairs_50
    # IoU
    pred_area = pred.set_index("pid").geometry.area
    ref_area = ref.set_index("vid").geometry.area
    inter["ai"] = inter.geometry.area
    inter = inter.merge(pred_area.rename("ap"), left_on="pid", right_index=True, how="left")
    inter = inter.merge(ref_area.rename("av"), left_on="vid", right_index=True, how="left")
    inter["iou"] = inter["ai"] / (inter["ap"] + inter["av"] - inter["ai"] + 1e-12)
    pairs_all = greedy_match(inter.loc[inter["iou"] > 0, ["pid","vid","iou"]])
    pairs_50 = greedy_match(inter.loc[inter["iou"] >= IOU_THR, ["pid","vid","iou"]])
    return inter, pairs_all, pairs_50


def compute_pq(n_pred, n_ref, pairs_50, iou_col="iou"):
    TP = int(len(pairs_50))
    FP = int(n_pred - TP)
    FN = int(n_ref - TP)
    SQ = float(pairs_50[iou_col].mean()) if TP else 0.0
    RQ = TP / (TP + 0.5*FP + 0.5*FN) if (TP + FP + FN) else 0.0
    PQ = SQ * RQ
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    return dict(PQ=PQ, SQ=SQ, RQ=RQ, TP=TP, FP=FP, FN=FN, recall50=recall)


def zonal_median_chm(ref: gpd.GeoDataFrame, chm_path: Path, band=1):
    """Return per-reference CHM height.
    Strategy: polygon median; if NaN, fallback to centroid sample.
    Robust to CRS mismatch (reprojects refs to CHM CRS for sampling).
    """
    vals = np.full(len(ref), np.nan, dtype=float)
    with rasterio.open(chm_path) as src:
        # Reproject reference geometries to CHM CRS if needed
        if ref.crs and src.crs and ref.crs != src.crs:
            ref_samp = ref.to_crs(src.crs)
        else:
            ref_samp = ref
        # Prepare nodata-aware function
        def to_nan(a):
            a = a.astype("float32")
            if src.nodata is not None:
                a = np.where(a == src.nodata, np.nan, a)
            return a
        # Compute polygon medians with fallback to centroid sample
        for i, geom in zip(ref.index, ref_samp.geometry):
            med = np.nan
            try:
                out, _ = rio_mask(src, [geom.__geo_interface__], crop=True, filled=True, indexes=band)
                a = to_nan(out)
                if np.isfinite(a).any():
                    med = float(np.nanmedian(a))
            except Exception:
                med = np.nan
            if not np.isfinite(med):
                # Fallback: centroid sample
                try:
                    cx, cy = geom.representative_point().x, geom.representative_point().y
                    for v in src.sample([(cx, cy)], indexes=band):
                        v = float(v[0])
                        if src.nodata is not None and v == src.nodata:
                            v = np.nan
                        med = v
                        break
                except Exception:
                    med = np.nan
            vals[int(i)] = med
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference crowns GPKG (polygon layer)")
    ap.add_argument("--pred", help="Predicted crowns GPKG (polygon layer)")
    ap.add_argument("--image", help="Orthorectified image for DeepForest inference (used if --pred is omitted)")
    ap.add_argument("--rgb-bands", default=None,
                    help="Comma-separated 1-based band indices to build a 3-band RGB image for DeepForest (e.g., '5,4,3'). If omitted and the image has >3 bands, the first three are used.")
    ap.add_argument("--patch-size", type=int, default=800, help="DeepForest tile patch size (pixels)")
    ap.add_argument("--overlap", type=float, default=0.2, help="DeepForest tile overlap fraction [0,1)")
    ap.add_argument("--score-thr", type=float, default=0.2, help="Minimum detection score")
    ap.add_argument("--chm", default=str(Path(__file__).resolve().parents[2] / "area51_subset1_chm.tif"),
                    help="CHM raster used for height bins (default: ../area51_subset1_chm.tif)")
    ap.add_argument("--chm-band", type=int, default=1, help="CHM band index (1-based), default 1")
    ap.add_argument("--out-prefix", default=str(OUT_DIR / "deepforest"))
    ap.add_argument("--figure", action="store_true", help="Save a single-method figure (Fig.6 style)")
    ap.add_argument("--ref-bbox", action="store_true", help="Compare against reference bounding boxes instead of polygons")
    args = ap.parse_args()

    ref_path = Path(args.ref)
    pred_path = Path(args.pred) if args.pred else None
    chm_path = Path(args.chm)

    # Load reference polygons
    lyr = pick_polygon_layer(ref_path)
    if lyr is None:
        raise FileNotFoundError(f"No polygon layers found in {ref_path}")
    ref = gpd.read_file(ref_path, layer=lyr)

    # Load predictions if provided; otherwise attempt DeepForest inference on --image
    if pred_path and pred_path.exists():
        lyrp = pick_polygon_layer(pred_path)
        if lyrp is None:
            raise FileNotFoundError(f"No polygon layers in {pred_path}")
        pred = gpd.read_file(pred_path, layer=lyrp)
    else:
        if not args.image:
            raise RuntimeError("Either provide --pred (GPKG) or --image to run DeepForest inference.")
        try:
            from deepforest import main as df_main
        except Exception as e:
            raise RuntimeError("DeepForest not available in this environment. Install and rerun with --image, or supply --pred.") from e
        # If the source image has more than 3 bands, build an RGB 3-band temp file
        image_path = Path(args.image)
        rgb_image_path = image_path
        with rasterio.open(image_path) as src:
            if src.count != 3:
                # pick bands
                if args.rgb_bands:
                    try:
                        bands = [int(x.strip()) for x in args.rgb_bands.split(',')]
                    except Exception:
                        raise ValueError("--rgb-bands must be a comma-separated list of 3 integers (1-based).")
                    if len(bands) != 3:
                        raise ValueError("--rgb-bands must specify exactly 3 bands, e.g., '5,4,3'.")
                else:
                    bands = [1, 2, 3]
                # Read and write RGB image (uint8, 2–98% stretch per band)
                arr = src.read(bands).astype("float32")  # (3, H, W)
                # Stretch each band robustly to [0,255]
                for i in range(arr.shape[0]):
                    band = arr[i]
                    finite = np.isfinite(band)
                    if finite.any():
                        lo, hi = np.percentile(band[finite], [2, 98])
                        if hi > lo:
                            band = np.clip((band - lo) / (hi - lo), 0, 1)
                        else:
                            band = np.zeros_like(band, dtype="float32")
                    else:
                        band = np.zeros_like(band, dtype="float32")
                    arr[i] = (band * 255.0).astype("uint8")
                arr = arr.astype("uint8")
                profile = src.profile.copy()
                profile.update(count=3, dtype="uint8", nodata=None)
                rgb_image_path = OUT_DIR / f"{Path(args.out_prefix).stem}_rgb.tif"
                with rasterio.open(rgb_image_path, 'w', **profile) as dst:
                    dst.write(arr)
        # Run DeepForest on the (possibly RGB-converted) image with tiling
        model = df_main.deepforest()
        try:
            model.use_release()
        except Exception:
            pass
        # predict_tile returns a GeoDataFrame (georeferenced boxes)
        # Try multiple signatures (versions differ). Avoid unsupported kwargs like use_soft_nms.
        gdf = None
        attempts = [
            dict(raster_path=args.image, return_plot=False, patch_size=args.patch_size,
                 patch_overlap=args.overlap, iou_threshold=0.1, thresh=args.score_thr),
            dict(raster_path=args.image, return_plot=False, patch_size=args.patch_size,
                 patch_overlap=args.overlap, iou_threshold=0.1, score_threshold=args.score_thr),
            dict(raster_path=args.image, return_plot=False, patch_size=args.patch_size,
                 patch_overlap=args.overlap, iou_threshold=0.1),
        ]
        # Replace raster_path with rgb_image_path
        for kw in attempts:
            kw["raster_path"] = str(rgb_image_path)
        last_err = None
        for kw in attempts:
            try:
                gdf = model.predict_tile(**kw)
                break
            except TypeError as e:
                last_err = e
                continue
        if gdf is None:
            raise RuntimeError(f"DeepForest predict_tile failed with all known signatures: {last_err}")
        if gdf is None or gdf.empty:
            raise RuntimeError("DeepForest produced no detections on the given image.")
        # Keep polygons (boxes) and save
        pred = gdf
        # Ensure CRS is set and transform from pixel to map coordinates if needed
        try:
            with rasterio.open(rgb_image_path) as src:
                img_crs = src.crs
                img_transform = src.transform
                img_w, img_h = src.width, src.height
        except Exception:
            img_crs = None
            img_transform = None
            img_w = img_h = None
        if getattr(pred, 'crs', None) is None and img_crs is not None:
            pred.set_crs(img_crs, inplace=True)
        # If bounds look like pixel space, convert geometry using the image affine transform
        try:
            bx = pred.total_bounds
            if img_transform is not None and img_w is not None and img_h is not None:
                within_pixels = (0.0 <= bx[0] <= img_w) and (0.0 <= bx[2] <= img_w) and \
                                (0.0 <= bx[1] <= img_h) and (0.0 <= bx[3] <= img_h)
                if within_pixels:
                    from shapely.ops import transform as shp_transform
                    a = img_transform.a; b = img_transform.b; c = img_transform.c
                    d = img_transform.d; e = img_transform.e; f = img_transform.f
                    def px_to_map(x, y, z=None):
                        x = np.asarray(x); y = np.asarray(y)
                        X = a * x + b * y + c
                        Y = d * x + e * y + f
                        return (X, Y) if z is None else (X, Y, z)
                    pred = pred.copy()
                    pred["geometry"] = pred.geometry.apply(lambda g: shp_transform(px_to_map, g))
                    if img_crs is not None:
                        pred.set_crs(img_crs, inplace=True)
        except Exception:
            pass
        out_pred = f"{args.out_prefix}_predicted.gpkg"
        try:
            Path(out_pred).unlink(missing_ok=True)
        except Exception:
            pass
        pred.to_file(out_pred, driver="GPKG")

    # Align CRS
    img_crs_dbg = None
    try:
        with rasterio.open(rgb_image_path if 'rgb_image_path' in locals() else args.image) as _src:
            img_crs_dbg = _src.crs
            img_bounds_dbg = _src.bounds
    except Exception:
        img_bounds_dbg = None
    pred_before = len(pred)
    pred = ensure_same_crs(ref, pred)
    # Debug info
    try:
        print(f"REF CRS={ref.crs} bounds={ref.total_bounds}")
        print(f"IMG CRS={img_crs_dbg} bounds={img_bounds_dbg}")
        print(f"PRED CRS(before align) unknown or {img_crs_dbg}; after align={pred.crs} bounds={pred.total_bounds}; count(before clip)={pred_before}")
    except Exception:
        pass
    # Optionally clip to AOI; if it empties predictions, skip clipping
    try:
        ref_env = ref.geometry.union_all() if hasattr(ref.geometry, "union_all") else ref.geometry.unary_union
        pred_clipped = gpd.clip(pred, ref_env)
        if len(pred_clipped) > 0:
            pred = pred_clipped
        else:
            print("[WARN] Clipping to reference AOI removed all predictions; proceeding without clip.")
    except Exception as e:
        print(f"[WARN] Clip failed: {e}; proceeding without clip.")

    # Intersections and matched pairs
    inter, pairs_all, pairs_50 = intersections_and_pairs(pred, ref, ref_use_bbox=args.ref_bbox)
    print(f"Pred polygons: {len(pred)} | Ref polygons: {len(ref)} | Intersections: {len(inter)} | Matched (IoU>0): {len(pairs_all)} | Matched@0.50: {len(pairs_50)}")

    # Overall class-agnostic PQ/SQ/RQ and coverage
    overall = compute_pq(len(pred), len(ref), pairs_50)
    pd.DataFrame([overall]).to_csv(f"{args.out_prefix}_metrics_overall.csv", index=False)

    # IoU (matched-only) vector per reference crown
    iou_map = {} if pairs_all.empty else dict(zip(pairs_all["vid"].values, pairs_all["iou"].values))
    ref = ref.reset_index(drop=True)
    ref["iou_matched"] = [float(iou_map.get(i, np.nan)) for i in range(len(ref))]

    # Height bins from CHM median per reference crown
    if chm_path.exists():
        # Debug CHM info
        try:
            with rasterio.open(chm_path) as _chm:
                print(f"CHM CRS={_chm.crs} bounds={_chm.bounds} band_count={_chm.count}")
        except Exception:
            pass
        ch = zonal_median_chm(ref, chm_path, band=args.chm_band)
        ref["ch_med"] = ch
        ref["height_bin"] = pd.cut(ref["ch_med"], bins=[-np.inf, HEIGHT_SPLIT_M, np.inf], labels=[f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"])
        try:
            n_valid = int(np.isfinite(ch).sum())
            print(f"CHM medians computed: {n_valid}/{len(ch)} valid")
        except Exception:
            pass
    else:
        warnings.warn(f"CHM not found at {chm_path}; skipping height stratification.")
        ref["height_bin"] = pd.Series([np.nan]*len(ref))

    # Density bins from reference crowns (within 15 m)
    buffer_area = np.pi * (DENSITY_BUFFER_M ** 2)
    val_union = ref.geometry.union_all() if hasattr(ref.geometry, "union_all") else ref.geometry.unary_union
    def local_cover_fraction(g):
        rp = g.representative_point(); buf = rp.buffer(DENSITY_BUFFER_M)
        return float(buf.intersection(val_union).area) / buffer_area
    ref["lcc_15m"] = ref.geometry.apply(local_cover_fraction)
    thr = float(ref["lcc_15m"].dropna().median()) if ref["lcc_15m"].notna().any() else None
    if thr is not None:
        ref["dens_bin"] = pd.cut(ref["lcc_15m"], bins=[-np.inf, thr, np.inf], labels=["low","high"])
    else:
        ref["dens_bin"] = pd.Series([np.nan]*len(ref))

    # Collect matched-only IoUs per bin
    def bin_iou(series):
        idx = series.dropna().index
        return ref.loc[idx, "iou_matched"].dropna().values

    out_height = []
    for lbl in [f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"]:
        vals = bin_iou(ref.loc[ref["height_bin"] == lbl, "height_bin"]) if lbl in ref["height_bin"].unique().tolist() else np.array([])
        out_height.append({"bin": lbl, "n": int(len(vals)), "iou_mean": float(np.nanmean(vals)) if len(vals) else np.nan})
    pd.DataFrame(out_height).to_csv(f"{args.out_prefix}_iou_height.csv", index=False)

    out_dens = []
    for lbl in ["low", "high"]:
        vals = bin_iou(ref.loc[ref["dens_bin"] == lbl, "dens_bin"]) if lbl in ref["dens_bin"].unique().tolist() else np.array([])
        out_dens.append({"bin": lbl, "n": int(len(vals)), "iou_mean": float(np.nanmean(vals)) if len(vals) else np.nan})
    pd.DataFrame(out_dens).to_csv(f"{args.out_prefix}_iou_density.csv", index=False)

    # Optional single-method figure (Fig.6 style)
    if args.figure:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        # (a) IoU distribution (matched-only)
        vals = [ref["iou_matched"].dropna().values]
        if len(vals[0]) > 0:
            ax_a.violinplot(vals, positions=[1], widths=0.8, showmeans=False, showmedians=False, showextrema=False)
            ax_a.boxplot(vals, positions=[1], widths=0.25, patch_artist=True,
                         boxprops=dict(alpha=0.5, linewidth=1.0), medianprops=dict(color="black", linewidth=1.2))
        else:
            ax_a.text(0.5, 0.5, "No matched pairs", transform=ax_a.transAxes, ha="center")
        ax_a.set_xticks([1]); ax_a.set_xticklabels(["IoU (matched only)"])
        ax_a.set_ylim(0, 1.0)

        # (b) IoU by height (skip if empty)
        hb = [ref.loc[ref["height_bin"] == f"≤{int(HEIGHT_SPLIT_M)} m", "iou_matched"].dropna().values,
              ref.loc[ref["height_bin"] == f">{int(HEIGHT_SPLIT_M)} m", "iou_matched"].dropna().values]
        if any(len(v) for v in hb):
            ax_b.violinplot(hb, positions=[1,2], widths=0.8, showmeans=False, showmedians=False, showextrema=False)
            ax_b.boxplot(hb, positions=[1,2], widths=0.25, patch_artist=True,
                         boxprops=dict(alpha=0.5, linewidth=1.0), medianprops=dict(color="black", linewidth=1.2))
        else:
            ax_b.text(0.5, 0.5, "No matched pairs with height", transform=ax_b.transAxes, ha="center")
        ax_b.set_xticks([1,2]); ax_b.set_xticklabels([f"≤{int(HEIGHT_SPLIT_M)} m", f">{int(HEIGHT_SPLIT_M)} m"])
        ax_b.set_ylim(0, 1.0)

        # (c) IoU by density (skip if empty)
        db = [ref.loc[ref["dens_bin"] == "low", "iou_matched"].dropna().values,
              ref.loc[ref["dens_bin"] == "high", "iou_matched"].dropna().values]
        if any(len(v) for v in db):
            ax_c.violinplot(db, positions=[1,2], widths=0.8, showmeans=False, showmedians=False, showextrema=False)
            ax_c.boxplot(db, positions=[1,2], widths=0.25, patch_artist=True,
                         boxprops=dict(alpha=0.5, linewidth=1.0), medianprops=dict(color="black", linewidth=1.2))
        else:
            ax_c.text(0.5, 0.5, "No matched pairs for density bins", transform=ax_c.transAxes, ha="center")
        ax_c.set_xticks([1,2]); ax_c.set_xticklabels(["Low", "High"])
        ax_c.set_ylim(0, 1.0)

        # (d) PQ/SQ/RQ bars
        x = np.arange(3); vals = [overall["PQ"], overall["SQ"], overall["RQ"]]
        ax_d.bar(x, vals, color="#1f77b4", alpha=0.85)
        ax_d.set_xticks(x); ax_d.set_xticklabels(["PQ","SQ","RQ"])
        ax_d.set_ylim(0, 1.0)
        ax_d.set_ylabel("Score")
        fig.savefig(f"{args.out_prefix}_figure6_style.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Wrote: {args.out_prefix}_metrics_overall.csv; height/density CSVs under same prefix")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
