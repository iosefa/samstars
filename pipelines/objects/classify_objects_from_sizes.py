#!/usr/bin/env python3
"""
Train an XGBoost classifier on labeled reference objects and classify per-size objects.

Typical use:
  - training objects = handcrafted (reference) segments with object features
  - training points  = labeled points with a 'class' field
  - objects_root     = per-size objects (detectron or sam-stardist)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from obia.classify import classify
from utils.utils import label_segments


LEGACY_FEATURES = [
    "b0_mean",
    "b1_mean",
    "b2_mean",
    "b3_mean",
    "b4_mean",
    "b5_mean",
    "b6_mean",
    "b7_mean",
    "b0_min",
    "b1_min",
    "b2_min",
    "b3_min",
    "b4_min",
    "b5_min",
    "b6_min",
    "b7_min",
    "b0_max",
    "b1_max",
    "b2_max",
    "b3_max",
    "b4_max",
    "b5_max",
    "b6_max",
    "b7_max",
    "b0_energy",
    "b0_contrast",
    "b0_dissimilarity",
    "b0_ASM",
    "b0_correlation",
    "b0_homogeneity",
    "pai",
    "fhd",
    "ch",
    "mean_intensity",
    "variance_intensity",
]


def _parse_sizes(sizes_arg: str | None, objects_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(objects_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "objects.gpkg").exists():
            sizes.append(p.name)
    return sizes


def _load_training_points(path: Path, class_field: str) -> gpd.GeoDataFrame:
    pts = gpd.read_file(path)
    pts[class_field] = pd.to_numeric(pts[class_field], errors="coerce").astype("Int64")
    n_before = len(pts)
    pts = pts.dropna(subset=[class_field]).copy()
    dropped = n_before - len(pts)
    if dropped:
        print(f"[train] dropped {dropped} rows with non-numeric {class_field}")
    return pts[[class_field, "geometry"]].copy()


def _load_regions(path: Path | None, target_crs) -> gpd.GeoDataFrame | None:
    if path is None:
        return None
    regions = gpd.read_file(path)
    if target_crs is not None and regions.crs != target_crs:
        regions = regions.to_crs(target_crs)
    return regions


def _select_columns(df: gpd.GeoDataFrame, cols: list[str], label: str) -> gpd.GeoDataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{label}] missing columns: {missing}")
    return df[cols].copy()


def main() -> None:
    pa = argparse.ArgumentParser(description="Classify per-size objects with XGBoost.")
    pa.add_argument("--train-objects", type=Path, required=True, help="Training objects.gpkg")
    pa.add_argument(
        "--training-points",
        type=Path,
        default=None,
        help="Optional labeled points GPKG (if provided, labels are assigned via label_segments)",
    )
    pa.add_argument("--objects-root", type=Path, required=True, help="Root with per-size objects.gpkg")
    pa.add_argument("--out-root", type=Path, required=True, help="Root for classified outputs")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect")
    pa.add_argument("--regions", type=Path, default=None, help="Optional regions.gpkg with acceptable_classes")
    pa.add_argument("--class-field", type=str, default="class", help="Class field in training points")
    pa.add_argument("--method", type=str, default="xgb", choices=["xgb", "rf", "mlp"])
    pa.add_argument("--test-size", type=float, default=0.3)
    pa.add_argument("--compute-reports", action="store_true")
    pa.add_argument("--compute-shap", action="store_true")
    pa.add_argument("--sample-shap", action="store_true")
    pa.add_argument("--margin-threshold", type=float, default=0.25)
    pa.add_argument("--use-all-features", action="store_true", help="Use all numeric features (except geometry)")
    pa.add_argument("--feature-list", type=str, default=None, help="Comma-separated feature columns override")
    args = pa.parse_args()

    sizes = _parse_sizes(args.sizes, args.objects_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure objects.gpkg exists under --objects-root.")

    train_objects = gpd.read_file(args.train_objects)
    if train_objects.empty:
        raise RuntimeError("Training objects are empty.")

    if args.training_points:
        training_points = _load_training_points(args.training_points, args.class_field)
        if train_objects.crs is not None and training_points.crs != train_objects.crs:
            training_points = training_points.to_crs(train_objects.crs)

        training_points = training_points.copy()
        training_points["feature_class"] = training_points[args.class_field].astype(int)
        training_points = training_points[["feature_class", "geometry"]]

        training_labeled, mixed = label_segments(train_objects, training_points)
        print(f"[train] labeled={len(training_labeled)} mixed={len(mixed)}")
    else:
        if args.class_field not in train_objects.columns:
            raise RuntimeError(f"Training objects missing '{args.class_field}' column.")
        training_labeled = train_objects.copy()
        training_labeled["feature_class"] = (
            pd.to_numeric(training_labeled[args.class_field], errors="coerce")
              .astype("Int64")
        )
        n_before = len(training_labeled)
        training_labeled = training_labeled.dropna(subset=["feature_class"]).copy()
        dropped = n_before - len(training_labeled)
        if dropped:
            print(f"[train] dropped {dropped} rows with non-numeric {args.class_field}")
        training_labeled["feature_class"] = training_labeled["feature_class"].astype(int)

    if args.feature_list:
        feature_cols = [c.strip() for c in args.feature_list.split(",") if c.strip()]
    elif args.use_all_features:
        feature_cols = [
            c for c in training_labeled.columns
            if c not in ("feature_class", "geometry", "segment_id")
        ]
    else:
        feature_cols = LEGACY_FEATURES

    train_cols = ["segment_id"] + feature_cols + ["feature_class", "geometry"]
    training_labeled = _select_columns(training_labeled, train_cols, "training")

    regions = _load_regions(args.regions, train_objects.crs)

    for sz in sizes:
        obj_path = args.objects_root / sz / "objects.gpkg"
        if not obj_path.exists():
            print(f"[skip] missing objects: {obj_path}")
            continue

        objects = gpd.read_file(obj_path)
        if objects.empty:
            print(f"[skip] empty objects: {obj_path}")
            continue

        if train_objects.crs is not None and objects.crs != train_objects.crs:
            objects = objects.to_crs(train_objects.crs)

        obj_cols = ["segment_id"] + feature_cols + ["geometry"]
        objects = _select_columns(objects, obj_cols, f"objects {sz}")

        classified = classify(
            objects,
            training_labeled,
            acceptable_classes_gdf=regions,
            method=args.method,
            test_size=args.test_size,
            compute_reports=args.compute_reports,
            compute_shap=args.compute_shap,
            sample_shap=args.sample_shap,
            margin_threshold=args.margin_threshold,
        )

        out_dir = args.out_root / sz
        out_dir.mkdir(parents=True, exist_ok=True)
        out_gpkg = out_dir / "classified_objects.gpkg"
        classified.classified.to_file(out_gpkg, driver="GPKG")
        print(f"✓ wrote {out_gpkg}")

        if args.compute_reports:
            if classified.report is not None:
                (out_dir / "classification_report.txt").write_text(str(classified.report))
            if classified.confusion_matrix is not None:
                pd.DataFrame(classified.confusion_matrix).to_csv(out_dir / "confusion_matrix.csv", index=False)


if __name__ == "__main__":
    main()
