#!/usr/bin/env python3
"""
Create Table 1 (Figure 6 panoptic summary) from existing outputs.

Reads manuscript/figures/output/figure6_overall_metrics.csv and writes:
  - manuscript/figures/output/table1_figure6_panoptic_summary.csv

Columns in the output:
  method, SQ, RQ, PQ, TP, FP, FN, precision, recall, iou_thr
"""

from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / 'figures' / 'output'
    src = out_dir / 'figure6_overall_metrics.csv'
    dst = out_dir / 'table1_figure6_panoptic_summary.csv'

    df = pd.read_csv(src)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Compute precision if missing
    if 'precision' not in [c.lower() for c in df.columns]:
        tp = df[cols.get('tp','TP')].astype(float)
        fp = df[cols.get('fp','FP')].astype(float)
        df['precision'] = tp / (tp + fp).replace({0: float('nan')})
    # Map recall50 -> recall for clarity
    if 'recall50' in df.columns and 'recall' not in df.columns:
        df = df.rename(columns={'recall50': 'recall'})

    keep = ['method', 'SQ', 'RQ', 'PQ', 'TP', 'FP', 'FN', 'precision', 'recall', 'iou_thr']
    # Ensure ordering and availability
    out = pd.DataFrame()
    for k in keep:
        if k in df.columns:
            out[k] = df[k]
        else:
            out[k] = pd.NA

    out.to_csv(dst, index=False)
    print(f"[OK] Wrote {dst}")


if __name__ == '__main__':
    main()
