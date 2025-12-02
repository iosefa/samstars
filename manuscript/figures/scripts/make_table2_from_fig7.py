#!/usr/bin/env python3
"""
Create Table 2 for Set B species classification metrics.

Reads:
- XGBoost per-class metrics: classification_metrics_by_class.csv
- CNN per-class report: manuscript/data/set_b/cnn_report_set_b.csv (sklearn-style)

Writes a consolidated CSV with columns:
  class_name, XGB_precision, XGB_recall, XGB_f1, XGB_support,
              CNN_precision, CNN_recall, CNN_f1, CNN_support,
  plus macro/accuracy rows if present.

Output path:
- manuscript/figures/output/table2_setb_classification_metrics.csv
"""

from pathlib import Path
import pandas as pd

CLASS_CODE_TO_NAME = {0:"METPOL",1:"GREROB",2:"ACAKOA",3:"DIOSAN",4:"PROPAL",5:"OTHER"}


def load_xgb(path: Path):
    df = pd.read_csv(path)
    # Standardize names
    df = df.rename(columns={'f1-score': 'f1'})
    # Per-class table
    per_class = df[df['class_name'].str.lower().isin(['metpol','grerob','acakoa','diosan','propal','other'])][
        ['class_name','precision','recall','f1','support']
    ].copy()
    # Macro + accuracy
    macro = df[df['class_name'].str.lower().eq('macro avg')].copy()
    acc = df[df['class_name'].str.lower().eq('accuracy')].copy()
    xgb_macro = dict(precision=float(macro['precision'].iloc[0]) if not macro.empty else None,
                     recall=float(macro['recall'].iloc[0]) if not macro.empty else None,
                     f1=float(macro['f1'].iloc[0]) if not macro.empty else None,
                     support=int(macro['support'].iloc[0]) if (not macro.empty and pd.notna(macro['support'].iloc[0])) else None)
    xgb_acc = float(acc['f1'].iloc[0]) if not acc.empty else None  # accuracy often in f1 field
    return per_class, xgb_macro, xgb_acc


def load_cnn(path: Path):
    df = pd.read_csv(path)
    # Macro + accuracy rows
    macro = df[df['label'].astype(str).str.lower().eq('macro avg')].copy()
    acc = df[df['label'].astype(str).str.lower().eq('accuracy')].copy()
    cnn_macro = dict(precision=float(macro['precision'].iloc[0]) if not macro.empty else None,
                     recall=float(macro['recall'].iloc[0]) if not macro.empty else None,
                     f1=float(macro['f1-score'].iloc[0]) if not macro.empty else None,
                     support=int(macro['support'].iloc[0]) if (not macro.empty and pd.notna(macro['support'].iloc[0])) else None)
    cnn_acc = float(acc['precision'].iloc[0]) if (not acc.empty and pd.notna(acc['precision'].iloc[0])) else (
               float(acc['recall'].iloc[0]) if (not acc.empty and pd.notna(acc['recall'].iloc[0])) else (
               float(acc['f1-score'].iloc[0]) if not acc.empty else None))
    # Drop aggregate rows to get per-class
    agg = {'accuracy','macro avg','weighted avg','micro avg'}
    df2 = df[~df['label'].astype(str).str.lower().isin(agg)].copy()
    # Map numeric labels -> names
    def to_name(x):
        s = str(x).strip()
        if s.isdigit():
            return CLASS_CODE_TO_NAME.get(int(s), s)
        return s
    df2['class_name'] = df2['label'].apply(to_name)
    df2 = df2.rename(columns={'f1-score':'f1'})
    per_class = df2[['class_name','precision','recall','f1','support']]
    return per_class, cnn_macro, cnn_acc


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / 'figures' / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb_csv = root.parent / 'classification_metrics_by_class.csv'
    cnn_csv = root / 'data' / 'set_b' / 'cnn_report_set_b.csv'

    xgb_per, xgb_macro, xgb_acc = load_xgb(xgb_csv)
    cnn_per, cnn_macro, cnn_acc = load_cnn(cnn_csv)

    # Merge on class_name
    m = xgb_per.merge(cnn_per, on='class_name', how='outer', suffixes=('_XGB','_CNN'))
    # Order rows by canonical class order
    order = [CLASS_CODE_TO_NAME[i] for i in sorted(CLASS_CODE_TO_NAME.keys())]
    m['__ord'] = m['class_name'].apply(lambda s: order.index(s) if s in order else 999)
    m = m.sort_values('__ord').drop(columns='__ord')

    # Append summary rows
    # Ensure accuracy columns exist
    for col in ['XGB_accuracy','CNN_accuracy']:
        if col not in m.columns:
            m[col] = pd.NA
    # Summary (macro)
    summ = {
        'class_name': 'Summary (macro)',
        'precision_XGB': xgb_macro.get('precision'),
        'recall_XGB': xgb_macro.get('recall'),
        'f1_XGB': xgb_macro.get('f1'),
        'support_XGB': xgb_macro.get('support'),
        'precision_CNN': cnn_macro.get('precision'),
        'recall_CNN': cnn_macro.get('recall'),
        'f1_CNN': cnn_macro.get('f1'),
        'support_CNN': cnn_macro.get('support'),
        'XGB_accuracy': pd.NA,
        'CNN_accuracy': pd.NA,
    }
    # Accuracy row
    accrow = {
        'class_name': 'Accuracy',
        'precision_XGB': pd.NA, 'recall_XGB': pd.NA, 'f1_XGB': pd.NA, 'support_XGB': pd.NA,
        'precision_CNN': pd.NA, 'recall_CNN': pd.NA, 'f1_CNN': pd.NA, 'support_CNN': pd.NA,
        'XGB_accuracy': xgb_acc,
        'CNN_accuracy': cnn_acc,
    }
    out_df = pd.concat([m, pd.DataFrame([summ, accrow])], ignore_index=True)

    dst = out_dir / 'table2_setb_classification_metrics.csv'
    out_df.to_csv(dst, index=False)
    print(f"[OK] Wrote {dst}")


if __name__ == '__main__':
    main()
