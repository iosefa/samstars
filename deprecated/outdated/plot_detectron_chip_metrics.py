#!/usr/bin/env python3
"""
Plot per-chip AP50 and IoU vs number of training instances across chip sizes.
Uses per-chip metrics written by eval_detectron_on_chip_tifs.py and counts training
instances from chips_train/detectron_manifest.json.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


def read_per_chip_metrics(csv_path: Path):
    ap_vals = []
    iou_vals = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iou = float(row.get("iou_union", "nan"))
            except ValueError:
                iou = float("nan")
            try:
                ap = float(row.get("ap50", "nan"))
            except ValueError:
                ap = float("nan")
            iou_vals.append(iou)
            ap_vals.append(ap)
    mean_iou = float(np.nanmean(iou_vals)) if iou_vals else float("nan")
    mean_ap = float(np.nanmean(ap_vals)) if ap_vals else float("nan")
    return mean_iou, mean_ap


def count_train_instances(manifest_path: Path):
    import json
    data = json.load(open(manifest_path))
    total = 0
    for rec in data:
        gpkg = Path(rec["gpkg_path"])
        if not gpkg.exists():
            continue
        gdf = gpd.read_file(gpkg)
        total += len(gdf)
    return total


def main():
    pa = argparse.ArgumentParser(description="Plot Detectron per-chip AP/IoU vs training instances.")
    pa.add_argument("--detectron-root", type=Path, default=Path("v2/detectron"), help="Root with per-size eval outputs")
    pa.add_argument("--chips-root", type=Path, default=Path("v2/chips_125_50"), help="Root with per-size chips_train manifests")
    pa.add_argument("--metrics-name", default="metrics_per_chip.csv", help="Per-chip metrics CSV name under size dir")
    pa.add_argument("--output", type=Path, default=None, help="Output PNG path")
    args = pa.parse_args()

    sizes = []
    train_instances = []
    mean_ious = []
    mean_aps = []

    for size_dir in sorted(args.detectron_root.iterdir()):
        if not size_dir.is_dir():
            continue
        try:
            n = int(size_dir.name)
        except ValueError:
            continue
        per_chip_csv = size_dir / args.metrics_name
        if not per_chip_csv.exists():
            continue
        manifest_path = args.chips_root / size_dir.name / "chips_train" / "detectron_manifest.json"
        if not manifest_path.exists():
            continue
        iou_mean, ap_mean = read_per_chip_metrics(per_chip_csv)
        inst = count_train_instances(manifest_path)
        sizes.append(n)
        train_instances.append(inst)
        mean_ious.append(iou_mean)
        mean_aps.append(ap_mean)

    if not sizes:
        raise SystemExit("No per-chip metrics found; check --detectron-root and --metrics-name.")

    order = np.argsort(train_instances)
    train_instances = np.array(train_instances)[order]
    mean_ious = np.array(mean_ious)[order]
    mean_aps = np.array(mean_aps)[order]

    plt.figure(figsize=(8, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(train_instances, mean_aps, marker="o")
    ax1.set_ylabel("AP50 (per-chip mean)")
    ax1.set_xlabel("Training instances (crowns)")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(train_instances, mean_ious, marker="o")
    ax2.set_ylabel("IoU (per-chip mean)")
    ax2.set_xlabel("Training instances (crowns)")
    ax2.grid(True, alpha=0.3)

    out_path = args.output or args.detectron_root / "detectron_ap_iou_vs_train_instances.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"✓ wrote plot to {out_path}")


if __name__ == "__main__":
    main()
