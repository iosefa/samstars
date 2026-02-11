#!/usr/bin/env python3
"""
Summarize eval-time for Detectron2 and SAM+StarDist using existing logs.

Outputs a CSV with seconds, seconds per km^2, and seconds per crown for a size.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union


def _parse_hms_to_seconds(hms: str) -> float:
    parts = hms.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        return float(hms)
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_time_from_lines(lines: Iterable[str]) -> Optional[float]:
    seconds = None
    for line in lines:
        if "Total inference time:" in line:
            m = re.search(r"Total inference time:\s*([0-9:.]+)", line)
            if m:
                seconds = _parse_hms_to_seconds(m.group(1))
        if "TOTAL_TIME_SECONDS=" in line:
            m = re.search(r"TOTAL_TIME_SECONDS=([0-9.]+)", line)
            if m:
                seconds = float(m.group(1))
        m = re.search(r"real\s+([0-9.]+)m([0-9.]+)s", line)
        if m:
            seconds = float(m.group(1)) * 60 + float(m.group(2))
        m = re.search(r"\breal\s+([0-9.]+)\s*$", line)
        if m:
            seconds = float(m.group(1))
        m = re.search(r"\b([0-9.]+)\s*total\b", line)
        if m:
            seconds = float(m.group(1))
        m = re.search(r"Elapsed time:\s*([0-9.]+)s", line)
        if m:
            seconds = float(m.group(1))
    return seconds


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if path_str.startswith("v2/"):
        alt = Path("data") / path_str[len("v2/") :]
        if alt.exists():
            return alt
    return path


def _load_fixed_val_stats(manifest_path: Path) -> Dict[str, float]:
    recs = json.load(open(manifest_path))
    all_boxes = []
    crowns_groups: Dict[str, Dict[str, object]] = {}
    for rec in recs:
        bounds = rec.get("bounds")
        if not bounds:
            continue
        geom = box(*bounds)
        all_boxes.append(geom)
        crowns_path = rec.get("crowns_path")
        if crowns_path:
            entry = crowns_groups.setdefault(
                crowns_path, {"boxes": [], "crs": rec.get("crs")}
            )
            entry["boxes"].append(geom)

    area_m2 = unary_union(all_boxes).area if all_boxes else 0.0

    crown_count = 0
    for crowns_path, entry in crowns_groups.items():
        crowns_path = _resolve_path(crowns_path)
        if not crowns_path.exists():
            raise FileNotFoundError(f"Missing crowns file: {crowns_path}")
        gdf = gpd.read_file(crowns_path)
        if gdf.empty:
            continue
        if gdf.crs is None and entry.get("crs"):
            gdf = gdf.set_crs(entry["crs"])
        gdf = gdf[gdf.geometry.notnull()]
        if gdf.empty:
            continue
        union_geom = unary_union(entry["boxes"])
        crown_count += int(gdf.intersects(union_geom).sum())

    return {
        "n_chips": len(recs),
        "area_m2": area_m2,
        "area_km2": area_m2 / 1e6,
        "n_crowns": crown_count,
    }


def _load_seconds(log_path: Path) -> Optional[float]:
    if not log_path or not log_path.exists():
        return None
    lines = log_path.read_text(errors="ignore").splitlines()
    return _parse_time_from_lines(lines)


def main() -> None:
    pa = argparse.ArgumentParser(description="Summarize eval-time for Detectron2 and SAM+StarDist.")
    pa.add_argument("--size", type=str, required=True)
    pa.add_argument("--fixed-val-manifest", type=Path, required=True)
    pa.add_argument("--detectron-log", type=Path, default=None, help="Detectron2 eval log file")
    pa.add_argument("--sam-log", type=Path, default=None, help="SAM inference log file")
    pa.add_argument("--stardist-log", type=Path, default=None, help="StarDist eval log file")
    pa.add_argument("--out", type=Path, required=True)
    args = pa.parse_args()

    stats = _load_fixed_val_stats(args.fixed_val_manifest)
    area_km2 = stats["area_km2"]
    n_crowns = stats["n_crowns"]
    n_chips = stats["n_chips"]

    rows = []

    det_sec = _load_seconds(args.detectron_log) if args.detectron_log else None
    if det_sec is not None:
        rows.append(
            {
                "size": args.size,
                "method": "detectron2",
                "stage": "eval",
                "seconds": det_sec,
                "sec_per_km2": det_sec / area_km2 if area_km2 else None,
                "sec_per_crown": det_sec / n_crowns if n_crowns else None,
                "area_km2": area_km2,
                "n_crowns": n_crowns,
                "n_chips": n_chips,
            }
        )

    sam_sec = _load_seconds(args.sam_log) if args.sam_log else None
    if sam_sec is not None:
        rows.append(
            {
                "size": args.size,
                "method": "sam-star",
                "stage": "sam_eval",
                "seconds": sam_sec,
                "sec_per_km2": sam_sec / area_km2 if area_km2 else None,
                "sec_per_crown": sam_sec / n_crowns if n_crowns else None,
                "area_km2": area_km2,
                "n_crowns": n_crowns,
                "n_chips": n_chips,
            }
        )

    sd_sec = _load_seconds(args.stardist_log) if args.stardist_log else None
    if sd_sec is not None:
        rows.append(
            {
                "size": args.size,
                "method": "sam-star",
                "stage": "stardist_eval",
                "seconds": sd_sec,
                "sec_per_km2": sd_sec / area_km2 if area_km2 else None,
                "sec_per_crown": sd_sec / n_crowns if n_crowns else None,
                "area_km2": area_km2,
                "n_crowns": n_crowns,
                "n_chips": n_chips,
            }
        )

    if sam_sec is not None and sd_sec is not None:
        total = sam_sec + sd_sec
        rows.append(
            {
                "size": args.size,
                "method": "sam-star",
                "stage": "combined_eval",
                "seconds": total,
                "sec_per_km2": total / area_km2 if area_km2 else None,
                "sec_per_crown": total / n_crowns if n_crowns else None,
                "area_km2": area_km2,
                "n_crowns": n_crowns,
                "n_chips": n_chips,
            }
        )

    if not rows:
        raise SystemExit("No timings parsed. Provide log files with parseable timing lines.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
