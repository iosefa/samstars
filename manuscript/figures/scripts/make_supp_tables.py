#!/usr/bin/env python3
"""
Supplementary tables

Creates:
- supp_table_classification_full.csv (per-class precision/recall/F1 for XGB and CNN)
- supp_table_shap_features.csv (mean |SHAP| per feature overall and per class)
- supp_table_shap_modalities.csv (mean |SHAP| aggregated by modality)

Inputs:
- classification_metrics_by_class.csv (root) for XGB
- manuscript/data/set_b/cnn_report_set_b.csv (sklearn report) for CNN
- manuscript/results/shap_values_sampled.npz (S: n_samples x n_features x n_classes)
- manuscript/results/shap_feature_names.json (raw and pretty names)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2].parent
XGB_METRICS = ROOT / "classification_metrics_by_class.csv"
CNN_REPORT = Path("manuscript/data/set_b/cnn_report_set_b.csv")

SHAP_NPZ = Path("manuscript/results/shap_values_sampled.npz")
SHAP_NAMES = Path("manuscript/results/shap_feature_names.json")

CLASS_ORDER = ["METPOL","GREROB","ACAKOA","DIOSAN","PROPAL","OTHER"]


def load_metrics_any(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # handle sklearn style
    if 'label' in df.columns and 'f1-score' in df.columns:
        df = df[~df['label'].astype(str).str.contains('accuracy|macro|weighted', case=False, na=False)].copy()
        def to_name(x):
            sx = str(x).strip()
            if sx.isdigit():
                idx = int(sx)
                return ["METPOL","GREROB","ACAKOA","DIOSAN","PROPAL","OTHER"][idx] if 0 <= idx < 6 else sx
            return sx
        df['class_name'] = df['label'].apply(to_name)
        df = df.rename(columns={'precision':'precision','recall':'recall','f1-score':'f1'})
        keep = ['class_name'] + [c for c in ['precision','recall','f1','support'] if c in df.columns]
        return df[keep]
    # otherwise assume class_name columns exist
    cname = None
    for cand in ['class_name','class','label','Class','ClassName','class_index']:
        if cand in df.columns: cname = cand; break
    if cname is None:
        raise ValueError(f"Cannot find class column in {path}")
    df['class_name'] = df[cname]
    # pick precision/recall/f1 columns
    out = pd.DataFrame({'class_name': df['class_name']})
    for m in ['precision','recall','f1','support']:
        for cand in [m, m.capitalize(), m.upper(), 'f1-score','F1','F1-score']:
            if cand in df.columns:
                out[m] = df[cand]
                break
    return out


def save_classification_table():
    if not XGB_METRICS.exists():
        print(f"[WARN] Missing {XGB_METRICS}")
        return
    xgb = load_metrics_any(XGB_METRICS)
    if set(CLASS_ORDER).issubset(set(xgb['class_name'])):
        xgb = xgb.set_index('class_name').loc[CLASS_ORDER].reset_index()
    xgb['method'] = 'XGBoost'

    rows = [xgb]
    if CNN_REPORT.exists():
        cnn = load_metrics_any(CNN_REPORT)
        if set(CLASS_ORDER).issubset(set(cnn['class_name'])):
            cnn = cnn.set_index('class_name').loc[CLASS_ORDER].reset_index()
        cnn['method'] = 'CNN'
        rows.append(cnn)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_DIR/"supp_table_classification_full.csv", index=False)
    print(f"[OK] Saved {OUT_DIR/'supp_table_classification_full.csv'}")


def save_shap_tables():
    if not (SHAP_NPZ.exists() and SHAP_NAMES.exists()):
        print("[WARN] SHAP files missing — skipping SHAP tables")
        return
    with open(SHAP_NAMES, 'r') as f:
        names = json.load(f)
    raw, pretty = names.get('raw', []), names.get('pretty', [])
    if not raw:
        print('[WARN] No feature names')
        return
    S = np.load(SHAP_NPZ)['S']  # shape (n, n_features, n_classes)
    n, F, C = S.shape
    assert F == len(raw), "Feature name length mismatch"

    # Overall mean |SHAP| per feature
    mean_abs = np.abs(S).mean(axis=(0,2))  # (F,)
    total_abs = float(mean_abs.sum()) if np.isfinite(mean_abs).all() else float(np.nansum(mean_abs))
    df_feat = (
        pd.DataFrame({
            'feature_raw': raw,
            'feature': pretty if pretty else raw,
            'mean_abs_shap': mean_abs,
        })
        .assign(percent_of_total=lambda d: 100.0 * d['mean_abs_shap'] / max(total_abs, 1e-12))
        .sort_values('mean_abs_shap', ascending=False)
    )

    # Group by modality
    def modality_of(name: str) -> str:
        if name.startswith('b0_') or name.startswith('b1_') or name.startswith('b2_') or \
           name.startswith('b3_') or name.startswith('b4_') or name.startswith('b5_') or \
           name.startswith('b6_') or name.startswith('b7_'):
            # texture features start with b0_* but specific metric names (energy, contrast, etc.)
            if any(name.startswith(f"b0_{t}") for t in ['energy','contrast','dissimilarity','ASM','correlation','homogeneity']):
                return 'Texture'
            return 'Spectral'
        if name in ['pai','fhd','ch','mean_intensity','variance_intensity']:
            return 'Structure'
        return 'Other'

    df_feat['modality'] = [modality_of(nm) for nm in raw]
    df_mod = (df_feat.groupby('modality', as_index=False)
                    ['mean_abs_shap'].sum()
                    .sort_values('mean_abs_shap', ascending=False))

    df_feat.to_csv(OUT_DIR/"supp_table_shap_features.csv", index=False)
    df_mod.to_csv(OUT_DIR/"supp_table_shap_modalities.csv", index=False)
    print(f"[OK] Saved {OUT_DIR/'supp_table_shap_features.csv'} and {OUT_DIR/'supp_table_shap_modalities.csv'}")


def main():
    save_classification_table()
    save_shap_tables()


if __name__ == '__main__':
    main()
