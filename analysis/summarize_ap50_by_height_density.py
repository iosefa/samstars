#!/usr/bin/env python3
# STATUS: active
"""
Summarize AP50 by height and density bins for a given chip size.

Requires per-chip prediction GPKGs under:
  {preds_root}/{size}/{per_chip_subdir}/{chip_name}.gpkg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid
from shapely.geometry import box
from geopandas.tools import sjoin


def _valid_polys(geoms):
    polys = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        try:
            gv = make_valid(g)
        except Exception:
            try:
                gv = g.buffer(0)
            except Exception:
                continue
        if gv.is_empty:
            continue
        if gv.geom_type == "Polygon":
            polys.append(gv)
        elif gv.geom_type == "MultiPolygon":
            polys.extend([p for p in gv.geoms if p.area > 0])
    return polys


def _geom_iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def compute_ap50(gt_polys, pred_polys, pred_scores, iou_thr=0.5):
    if not gt_polys:
        return float("nan")
    if not pred_polys:
        return 0.0
    order = np.argsort(-np.array(pred_scores))
    preds = [pred_polys[i] for i in order]
    matched = [False] * len(gt_polys)
    tp = []
    fp = []
    for p in preds:
        best_iou = 0.0
        best_idx = -1
        for j, g in enumerate(gt_polys):
            if matched[j]:
                continue
            iou = _geom_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_thr and best_idx >= 0:
            matched[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / max(len(gt_polys), 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    return float(np.trapz(mpre, mrec))


def _load_val_manifest(val_manifest: Path):
    pairs = json.load(open(val_manifest))
    return pairs


def _load_gt_gdf(rec, target_crs=None):
    gdf = gpd.read_file(rec["gpkg_path"])
    if target_crs is not None and gdf.crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def _load_preds_gdf(path: Path, target_crs=None):
    if not path.exists():
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry")
    gdf = gpd.read_file(path)
    if target_crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        elif gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
    return gdf


def _chip_name(rec):
    return rec.get("name", Path(rec["tif_path"]).stem)


def _ap50_for_bin(gt_by_chip, preds_by_chip, chips_in_bin):
    gt_polys = []
    pred_polys = []
    pred_scores = []
    n_pred = 0
    for chip in chips_in_bin:
        gt_polys.extend(gt_by_chip.get(chip, []))
        preds = preds_by_chip.get(chip, [])
        for poly, score in preds:
            pred_polys.append(poly)
            pred_scores.append(score)
        n_pred += len(preds)
    ap50 = compute_ap50(gt_polys, pred_polys, pred_scores, iou_thr=0.5)
    return ap50, len(gt_polys), n_pred


def _aggregate_bin_from_per_chip(rows, agg):
    if not rows:
        return float("nan"), 0, 0
    n_gt = sum(r["n_gt"] for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    ap_vals = np.array([r["ap50"] for r in rows], dtype=float)
    if agg == "mean":
        return float(np.nanmean(ap_vals)), n_gt, n_pred
    if agg == "weighted-gt":
        weights = np.array([r["n_gt"] for r in rows], dtype=float)
    elif agg == "weighted-pred":
        weights = np.array([r["n_pred"] for r in rows], dtype=float)
    else:
        raise ValueError(f"Unknown bin aggregation: {agg}")
    if weights.sum() <= 0:
        return float("nan"), n_gt, n_pred
    return float(np.average(ap_vals, weights=weights)), n_gt, n_pred


def _build_chip_gdf(pairs, target_crs):
    rows = []
    for rec in pairs:
        chip = _chip_name(rec)
        b = rec.get("bounds")
        if b is None:
            continue
        geom = box(b[0], b[1], b[2], b[3])
        rows.append({"chip": chip, "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)
    return gdf


def _preds_by_chip_from_merged(preds_gpkg: Path, chip_gdf: gpd.GeoDataFrame):
    if not preds_gpkg.exists():
        return {}
    preds = gpd.read_file(preds_gpkg)
    if preds.crs is None:
        preds = preds.set_crs(chip_gdf.crs)
    elif preds.crs != chip_gdf.crs:
        preds = preds.to_crs(chip_gdf.crs)
    preds["centroid"] = preds.geometry.centroid
    cent = gpd.GeoDataFrame(preds.drop(columns="geometry"), geometry="centroid", crs=preds.crs)
    joined = sjoin(cent, chip_gdf[["chip", "geometry"]], how="left", predicate="within")
    preds = preds.loc[joined.index]
    preds["chip"] = joined["chip"].values
    preds_by_chip = {}
    scores = preds["score"].astype(float).values if "score" in preds.columns else np.ones(len(preds))
    for chip, geom, score in zip(preds["chip"].values, preds.geometry, scores):
        if chip is None or pd.isna(chip):
            continue
        vals = _valid_polys([geom])
        for v in vals:
            preds_by_chip.setdefault(chip, []).append((v, float(score)))
    return preds_by_chip


def _summarize_model(
    label: str,
    preds_root: Path,
    sizes: list[str],
    pairs,
    val_all: gpd.GeoDataFrame,
    dens_med: float | None,
    chip_gdf: gpd.GeoDataFrame,
    args,
):
    per_chip_height_rows = []
    per_chip_density_rows = []
    bins_height_rows = []
    bins_density_rows = []

    for sz in sizes:
        preds_dir = preds_root / sz / args.per_chip_subdir
        merged_gpkg = preds_root / sz / args.gpkg_name
        preds_by_chip = {}
        if preds_dir.exists() and not args.use_merged:
            for rec in pairs:
                chip = _chip_name(rec)
                pred_path = preds_dir / f"{chip}.gpkg"
                pred_gdf = _load_preds_gdf(pred_path, target_crs=val_all.crs)
                scores = pred_gdf["score"].astype(float).values if "score" in pred_gdf.columns else np.ones(len(pred_gdf))
                if args.min_score is not None:
                    keep = scores >= args.min_score
                    pred_gdf = pred_gdf.loc[keep]
                    scores = scores[keep]
                pred_polys = []
                pred_scores = []
                for geom, score in zip(pred_gdf.geometry, scores):
                    vals = _valid_polys([geom])
                    for v in vals:
                        pred_polys.append(v)
                        pred_scores.append(float(score))
                preds_by_chip[chip] = list(zip(pred_polys, pred_scores))
        else:
            preds_by_chip = _preds_by_chip_from_merged(merged_gpkg, chip_gdf)
            if args.min_score is not None:
                preds_by_chip = {
                    k: [(g, s) for g, s in v if s >= args.min_score] for k, v in preds_by_chip.items()
                }

        gt_by_chip = {}
        gt_rows = []
        for rec in pairs:
            chip = _chip_name(rec)
            gt_gdf = _load_gt_gdf(rec, target_crs=val_all.crs)
            gt_polys = _valid_polys(gt_gdf.geometry)
            gt_by_chip[chip] = gt_polys
            if args.height_field in gt_gdf.columns:
                gt_rows.append(
                    pd.DataFrame(
                        {
                            "chip": chip,
                            "geometry": gt_gdf.geometry.values,
                            "height": gt_gdf[args.height_field].values,
                        }
                    )
                )

        # Height bins
        if gt_rows:
            gt_h = gpd.GeoDataFrame(pd.concat(gt_rows, ignore_index=True), geometry="geometry", crs=val_all.crs)
            gt_h["bin"] = pd.cut(
                gt_h["height"],
                bins=[-np.inf, args.height_split, np.inf],
                labels=[f"≤{int(args.height_split)} m", f">{int(args.height_split)} m"],
            )
            for bin_label in gt_h["bin"].dropna().unique():
                gt_bin = gt_h.loc[gt_h["bin"] == bin_label]
                chips_in_bin = sorted(gt_bin["chip"].unique())
                per_chip_rows = []
                for chip in chips_in_bin:
                    gt_chip = _valid_polys(gt_bin.loc[gt_bin["chip"] == chip, "geometry"])
                    ap50_c, n_gt_c, n_pred_c = _ap50_for_bin({chip: gt_chip}, preds_by_chip, [chip])
                    row = dict(
                        model=label,
                        size=sz,
                        chip=chip,
                        bin=str(bin_label),
                        ap50=ap50_c,
                        n_gt=n_gt_c,
                        n_pred=n_pred_c,
                    )
                    per_chip_rows.append(row)
                    per_chip_height_rows.append(row)
                if args.bin_agg == "pooled":
                    ap50, n_gt, n_pred = _ap50_for_bin(
                        {c: _valid_polys(gt_bin.loc[gt_bin["chip"] == c, "geometry"]) for c in chips_in_bin},
                        preds_by_chip,
                        chips_in_bin,
                    )
                else:
                    ap50, n_gt, n_pred = _aggregate_bin_from_per_chip(per_chip_rows, args.bin_agg)
                bins_height_rows.append(
                    dict(
                        model=label,
                        size=sz,
                        bin=str(bin_label),
                        ap50=ap50,
                        n_gt=n_gt,
                        n_pred=n_pred,
                        n_chips=len(chips_in_bin),
                        height_split=args.height_split,
                        bin_agg=args.bin_agg,
                    )
                )

        # Density bins
        if dens_med is not None:
            for bin_label in ["low", "high"]:
                gt_bin = val_all.loc[val_all["dens_bin"] == bin_label]
                chips_in_bin = sorted(gt_bin["chip"].unique())
                per_chip_rows = []
                for chip in chips_in_bin:
                    gt_chip = _valid_polys(gt_bin.loc[gt_bin["chip"] == chip, "geometry"])
                    ap50_c, n_gt_c, n_pred_c = _ap50_for_bin({chip: gt_chip}, preds_by_chip, [chip])
                    row = dict(
                        model=label,
                        size=sz,
                        chip=chip,
                        bin=bin_label,
                        ap50=ap50_c,
                        n_gt=n_gt_c,
                        n_pred=n_pred_c,
                    )
                    per_chip_rows.append(row)
                    per_chip_density_rows.append(row)
                if args.bin_agg == "pooled":
                    ap50, n_gt, n_pred = _ap50_for_bin(
                        {c: _valid_polys(gt_bin.loc[gt_bin["chip"] == c, "geometry"]) for c in chips_in_bin},
                        preds_by_chip,
                        chips_in_bin,
                    )
                else:
                    ap50, n_gt, n_pred = _aggregate_bin_from_per_chip(per_chip_rows, args.bin_agg)
                bins_density_rows.append(
                    dict(
                        model=label,
                        size=sz,
                        bin=bin_label,
                        ap50=ap50,
                        n_gt=n_gt,
                        n_pred=n_pred,
                        n_chips=len(chips_in_bin),
                        density_buffer_m=args.density_buffer_m,
                        density_median=dens_med,
                        bin_agg=args.bin_agg,
                    )
                )

    return (
        pd.DataFrame(per_chip_height_rows),
        pd.DataFrame(per_chip_density_rows),
        pd.DataFrame(bins_height_rows),
        pd.DataFrame(bins_density_rows),
    )


def main() -> None:
    pa = argparse.ArgumentParser(description="Summarize AP50 by height and density bins for a chip size.")
    pa.add_argument("--val-manifest", type=Path, required=True)
    pa.add_argument("--preds-root", type=Path, required=True)
    pa.add_argument("--preds-root-b", type=Path, default=None)
    pa.add_argument("--label", type=str, default="model_a")
    pa.add_argument("--label-b", type=str, default="model_b")
    pa.add_argument("--size", type=str, default=None)
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes (optional)")
    pa.add_argument("--per-chip-subdir", type=str, default="per_chip_preds")
    pa.add_argument("--height-field", type=str, default="ch")
    pa.add_argument("--height-split", type=float, default=10.0)
    pa.add_argument("--density-buffer-m", type=float, default=15.0)
    pa.add_argument("--min-score", type=float, default=None)
    pa.add_argument("--gpkg-name", type=str, default="val_preds.gpkg")
    pa.add_argument("--use-merged", action="store_true", help="Use merged val_preds.gpkg even if per-chip exists.")
    pa.add_argument(
        "--bin-agg",
        type=str,
        default="pooled",
        choices=["pooled", "mean", "weighted-gt", "weighted-pred"],
        help="How to aggregate per-chip AP50 into a bin summary.",
    )
    pa.add_argument("--out-dir", type=Path, default=None)
    args = pa.parse_args()

    sizes = []
    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    elif args.size:
        sizes = [args.size]
    else:
        raise SystemExit("Provide --size or --sizes.")

    pairs = _load_val_manifest(args.val_manifest)
    if not pairs:
        raise SystemExit("val-manifest is empty.")

    # Load full validation once for density bins
    gdfs = []
    for rec in pairs:
        gdfs.append(_load_gt_gdf(rec))
    val_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    if val_all.crs is None or val_all.crs.is_geographic:
        val_all = val_all.to_crs(6635)
    buffer_area = np.pi * (args.density_buffer_m ** 2)
    try:
        val_union_geom = val_all.geometry.union_all()
    except Exception:
        val_union_geom = val_all.geometry.unary_union

    def local_cover_fraction(geom):
        rp = geom.representative_point()
        buf = rp.buffer(args.density_buffer_m)
        return float(buf.intersection(val_union_geom).area) / buffer_area

    val_all = val_all.copy()
    val_all["lcc"] = val_all.geometry.apply(local_cover_fraction)
    dens_med = float(val_all["lcc"].median()) if val_all["lcc"].notna().any() else None
    if dens_med is not None:
        val_all["dens_bin"] = pd.cut(val_all["lcc"], bins=[-np.inf, dens_med, np.inf], labels=["low", "high"])
    else:
        val_all["dens_bin"] = pd.Series([None] * len(val_all))

    # assign chip names to val_all (order matches concatenated gdfs above)
    chip_list = []
    for rec in pairs:
        chip = _chip_name(rec)
        gdf = _load_gt_gdf(rec, target_crs=val_all.crs)
        chip_list.extend([chip] * len(gdf))
    val_all = val_all.reset_index(drop=True)
    val_all["chip"] = chip_list

    chip_gdf = _build_chip_gdf(pairs, val_all.crs)

    per_chip_h_a, per_chip_d_a, bins_h_a, bins_d_a = _summarize_model(
        args.label, args.preds_root, sizes, pairs, val_all, dens_med, chip_gdf, args
    )

    if args.preds_root_b is not None:
        per_chip_h_b, per_chip_d_b, bins_h_b, bins_d_b = _summarize_model(
            args.label_b, args.preds_root_b, sizes, pairs, val_all, dens_med, chip_gdf, args
        )
        per_chip_h = pd.concat([per_chip_h_a, per_chip_h_b], ignore_index=True)
        per_chip_d = pd.concat([per_chip_d_a, per_chip_d_b], ignore_index=True)
        bins_h = pd.concat([bins_h_a, bins_h_b], ignore_index=True)
        bins_d = pd.concat([bins_d_a, bins_d_b], ignore_index=True)
    else:
        per_chip_h, per_chip_d = per_chip_h_a, per_chip_d_a
        bins_h, bins_d = bins_h_a, bins_d_a

    out_dir = args.out_dir if args.out_dir else (args.preds_root / sizes[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not per_chip_h.empty:
        per_chip_h.to_csv(out_dir / "ap50_height_per_chip.csv", index=False)
        print(f"wrote {out_dir / 'ap50_height_per_chip.csv'}")
    else:
        print("[warn] no height data")
    if not per_chip_d.empty:
        per_chip_d.to_csv(out_dir / "ap50_density_per_chip.csv", index=False)
        print(f"wrote {out_dir / 'ap50_density_per_chip.csv'}")
    else:
        print("[warn] no density data")
    if not bins_h.empty:
        bins_h.to_csv(out_dir / "ap50_height_bins.csv", index=False)
        print(f"wrote {out_dir / 'ap50_height_bins.csv'}")
    if not bins_d.empty:
        bins_d.to_csv(out_dir / "ap50_density_bins.csv", index=False)
        print(f"wrote {out_dir / 'ap50_density_bins.csv'}")


if __name__ == "__main__":
    main()
