#!/usr/bin/env python3
"""
Create Table 3 (Set C semantic panoptic summary) from existing outputs.

Reads manuscript/figures/output/figure8_overall_metrics.csv and writes:
  - manuscript/figures/output/table3_setc_semantic_panoptic_summary.csv

Columns in the output:
  method, SQ, RQ, PQ, TP, FP, FN, precision, recall, f1, iou_thr

Notes:
  - Renames recall50 -> recall for clarity/consistency with other tables.
  - Passes through all methods present in the source (e.g., SAM–StarDist, CNN baseline,
    optionally DeepForest if included during figure generation).
"""

from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / 'figures' / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    src = out_dir / 'figure8_overall_metrics.csv'
    dst = out_dir / 'table3_setc_semantic_panoptic_summary.csv'

    if not src.exists():
        raise FileNotFoundError(f"Expected source not found: {src}")

    df = pd.read_csv(src)
    # Normalize recall naming
    if 'recall50' in df.columns and 'recall' not in df.columns:
        df = df.rename(columns={'recall50': 'recall'})

    # Ensure expected columns exist; fill NA for any missing ones
    keep = ['method', 'SQ', 'RQ', 'PQ', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1', 'iou_thr']
    out = pd.DataFrame()
    for k in keep:
        out[k] = df[k] if k in df.columns else pd.NA

    out.to_csv(dst, index=False)
    print(f"[OK] Wrote {dst}")


if __name__ == '__main__':
    main()

