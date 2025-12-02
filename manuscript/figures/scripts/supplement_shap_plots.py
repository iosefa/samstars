#!/usr/bin/env python3
"""
Supplement — SHAP plots

Generates:
- supp_shap_modalities.png — bar chart of mean |SHAP| by modality
- supp_shap_top_features.png — bar chart of top-N features by mean |SHAP|

Inputs:
- manuscript/results/shap_values_sampled.npz (key 'S': n x F x K)
- manuscript/results/shap_feature_names.json (raw + pretty names)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_JSON = Path("manuscript/results/shap_feature_names.json")
SHAP_NPZ  = Path("manuscript/results/shap_values_sampled.npz")


def load_shap():
    with open(FEAT_JSON, 'r') as f:
        names = json.load(f)
    raw, pretty = names.get('raw', []), names.get('pretty', [])
    S = np.load(SHAP_NPZ)['S']  # (n, F, K)
    return raw, pretty if pretty else raw, S


def modality_of(name: str) -> str:
    if name.startswith(('b0_','b1_','b2_','b3_','b4_','b5_','b6_','b7_')):
        # texture on coastal band b0_ with specific metric names
        if any(name.startswith(f"b0_{t}") for t in ['energy','contrast','dissimilarity','ASM','correlation','homogeneity']):
            return 'Texture'
        return 'Spectral'
    if name in ['pai','fhd','ch','mean_intensity','variance_intensity']:
        return 'Structure'
    return 'Other'


def plot_modalities(raw, S):
    mean_abs = np.abs(S).mean(axis=(0,2))  # F
    df = pd.DataFrame({'name': raw, 'mean_abs': mean_abs})
    df['modality'] = df['name'].apply(modality_of)
    g = df.groupby('modality', as_index=False)['mean_abs'].sum()
    g = g.sort_values('mean_abs', ascending=False)
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=300)
    ax.bar(g['modality'], g['mean_abs'], color=['#6c8ae2','#f2b5d9','#a6d9a7'])
    ax.set_ylabel('Sum mean |SHAP|')
    ax.set_title('SHAP by modality')
    for i, v in enumerate(g['mean_abs']):
        ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    out = OUT_DIR/"supp_shap_modalities.png"
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"[OK] Saved {out}")


def plot_top_features(raw, pretty, S, topn=15):
    mean_abs = np.abs(S).mean(axis=(0,2))  # F
    df = pd.DataFrame({'raw': raw, 'feature': pretty, 'mean_abs': mean_abs})
    df = df.sort_values('mean_abs', ascending=False).head(topn)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    ax.barh(df['feature'], df['mean_abs'], color='#6c8ae2')
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(f'Top {topn} features by mean |SHAP|')
    fig.tight_layout()
    out = OUT_DIR/"supp_shap_top_features.png"
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"[OK] Saved {out}")


def main():
    raw, pretty, S = load_shap()
    plot_modalities(raw, S)
    plot_top_features(raw, pretty, S, topn=15)


if __name__ == '__main__':
    main()

