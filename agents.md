# Project Status and Handoff Notes (RSE Manuscript)

This document summarizes what is complete, what remains, and exactly where to run things so a future agent (or you) can pick up seamlessly.

## What’s Complete

### Figure 6 (Set A, segmentation)
- Script: `manuscript/figures/scripts/figure6.py`
- Status: Complete; height stratification now uses reference‑side CH; density uses reference‑side local cover.
- Outputs in `manuscript/figures/output/`:
  - `figure6.png`, `figure6_overall_metrics.csv`, `figure6_iou_height.csv`, `figure6_iou_density.csv`
  - P‑values: `figure6_pvalues_height.csv`, `figure6_pvalues_density.csv`
  - Diagnostics (TP/FP/FN GPKGs): `output/diags/figure6_*_{tp_pred,fp_pred,fn_ref}.gpkg`

### Figure 7 (Set B, classification + SHAP)
- Script: `manuscript/figures/scripts/figure7.py`
- Status: Complete; panel (a) per‑class F1 (CNN vs XGB), panel (b) SHAP modality summary.
- Output: `manuscript/figures/output/figure7.png`
- Table 2 script: `manuscript/figures/scripts/make_table2_from_fig7.py`
  - Output: `manuscript/figures/output/table2_setb_classification_metrics.csv` (XGB+CNN per‑class + macro + accuracy)

### Figure 8 (Set C, panoptic bars; formerly Fig 9)
- Script: `manuscript/figures/scripts/figure8.py`
- Status: Complete; “Panoptic SQ, RQ, PQ” bars; compact TP/FP/FN box; optional `--include-df`.
- Outputs: `figure8.png`, `figure8_overall_metrics.csv` + diagnostics in `output/diags/`.

### Figure 9 (Set C, RGB + SAM–StarDist overlay with FP/FN)
- Script: `manuscript/figures/scripts/figure9.py`
- Status: Complete; RGB (5/3/2), FP/FN fills with higher opacity, SAM–StarDist outlines on top, legend, scale bar.
- Output: `manuscript/figures/output/figure9.png`

### Figure 10 (Landscape AOI map + species composition)
- Script: `manuscript/figures/scripts/figure10.py`
- Status: Complete; thin polygon outlines by species, legend to the right, scale bar; composition by canopy area only.
- Output: `manuscript/figures/output/figure10.png`

### Figure 11 (30 m canopy cover vs CAH)
- Script: `manuscript/figures/scripts/figure11.py`
- Status: Complete; crowns cover, CAH cover, and difference maps.
- Output: `manuscript/figures/output/figure11.png`

### Figure 12 (Set C, four-panel overlays + diagnostics)
- Script: `manuscript/figures/scripts/figure12.py`
- Status: Complete; 2×2 RGB panels clipped to test extents; SAM (blue) and reference (orange) outlines; FP (red, dashed) and FN (purple, dotted) overlays with white halos; per-crown annotations ("Label\nmismatch" vs "IoU<0.5"); scale bars in each subplot.
- Output: `manuscript/figures/output/figure12.png`

### Figure 13 (CAH land cover vs classified crowns — detail)
- Script: `manuscript/figures/scripts/figure13.py`
- Status: Complete; compares CAH forest classes (left) and classified crowns (right) within `manuscript/data/landcover_detail.gpkg`; legends show CAH class names and species names with codes (e.g., "Metrosideros polymorpha (METPOL)"); 1 km scalebars bottom-left; reduced subplot spacing.
- Output: `manuscript/figures/output/figure13.png`
- Notes: CAH source `CAH_LandCover.gpkg` (root, EPSG:26904) is reprojected to EPSG:6635; crowns from root `classified_xgb.gpkg`.

### Tables
- Table 1 (Figure 6 panoptic summary): `manuscript/figures/scripts/make_table1_from_fig6.py` → `manuscript/figures/output/table1_figure6_panoptic_summary.csv`
- Table 2 (Set B classification): `manuscript/figures/scripts/make_table2_from_fig7.py` → `manuscript/figures/output/table2_setb_classification_metrics.csv`

## Remaining Work (Supplement only)

- S7 features: align 59‑feature definition with main text (6 stats/band), list features explicitly, and fix modality labels (Structure vs Texture vs Spectral).
- S9 metrics/statistics: add IoU, SQ, RQ, PQ equations; matching procedure; statistical tests and corrections.
- S10 CNN baseline details: document Detectron2 + ResNet‑34/PointNet setup, training data (5,480 pseudo‑labels), splits, hyperparameters, thresholds.
- S11 sensitivity/per‑plot: summarize IoU/PQ at 0.3 vs 0.5 and Set‑A per‑plot metrics; reference existing CSVs.
- S12 reproducibility: commands, env versions, seeds, and repository tag/commit to regenerate supplemental assets.
- S13 data package: enumerate derived artifacts for deposition (prompts, cost surfaces, crowns, evaluation pairs/metrics, figures/tables) with paths.
- Citations pass (Supplement): add any missing references (e.g., CSF/FLAASH/PDAL), ensure all Supplement citations resolve; standardize Fig/Tab S‑labels.

## Quick Commands (from `manuscript/`)

- Figure 6 (Set A): `python3 figures/scripts/figure6.py`
- Figure 7: `python3 figures/scripts/figure7.py`
- Figure 8 (Set C panoptic): `python3 figures/scripts/figure8.py`
- Figure 9 (Set C overlay): `python3 figures/scripts/figure9.py`
- Figure 10: `python3 figures/scripts/figure10.py`
- Figure 11: `python3 figures/scripts/figure11.py`
- Figure 12: `python3 figures/scripts/figure12.py`
- Figure 13: `python3 figures/scripts/figure13.py`
- Table 1: `python3 figures/scripts/make_table1_from_fig6.py`
- Table 2: `python3 figures/scripts/make_table2_from_fig7.py`
 - Supplement SHAP figs: `python3 figures/scripts/make_supp_shap_plots.py`

## Data & Models Summary

- Figure 6 (Set A): inputs in `manuscript/data/set_a/` — `validation_crowns.gpkg` (with CH), `sam_crowns.gpkg`, `detectron_crowns.gpkg`, `deepforest_crowns.gpkg`.
- Set B classification: XGBoost metrics at repo root `classification_metrics_by_class.csv`; CNN report/confusion under `manuscript/data/set_b/`.
- Set C panoptic (Figure 8) and overlay (Figure 9): uses `manuscript/data/set_c/` — `set_c_validation.gpkg`, `set_c_sam_predicted.gpkg`, `set_c_cnn_predicted_classified.gpkg`, `set_c.tif`.
