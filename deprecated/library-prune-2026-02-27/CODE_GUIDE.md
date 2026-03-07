# Code Guide (Active vs Deprecated)

This repo has a mix of current workflows and legacy code. This guide lists what is
actively used and what is deprecated. If a script is not listed here, treat it as
"unknown/one-off" until reviewed.

## Active (current workflows)

### Data prep / splits / chips
- `data_prep/build_chip_splits.py`: create train-size splits + fixed validation manifest.
- `data_prep/add_chip_split_size.py`: add a missing size to an existing split set.
- `data_prep/ensure_val_has_all_classes.py`: enforce validation coverage across classes.
- `data_prep/build_chips_from_manifest.py`: build chips from split manifests.
- `data_prep/export_val_chips.py`: export fixed validation chips (used when needed).
- `data_prep/align_chm_to_img.py`: align CHM to image grid (utility used in pipeline).
- `data_prep/compute_unique_crown_counts.py`: unique crown counts for splits.
- `data_prep/make_reference_data_card.py`: data card generation.

### Scaling (SAM input / aux bands)
- `pipelines/sam/compute_sam_global_scaling.py`: compute global scaling stats.
- `pipelines/shared/scale_stats.py`: helpers to load/scale inputs.

### SAM → StarDist pipeline
- `pipelines/sam/train_sam_decoder_from_chips.py`: train SAM decoder per size.
- `pipelines/sam/predict_sam_from_manifests.py`: SAM inference -> prob/logit outputs per chip.
- `pipelines/sam/build_fixed_val_manifests.py`: create fixed val manifests (if needed).
- `pipelines/stardist/build_prob_manifests.py`: create train/val prob manifests for StarDist.
- `pipelines/stardist/train_stardist_on_probs.py`: train StarDist on SAM outputs.
- `pipelines/stardist/eval_stardist_on_probs.py`: evaluate StarDist and write val_preds.gpkg.
- `pipelines/stardist/predict_stardist_from_probs.py`: inference-only StarDist -> predicted polygons (no GT masks).
- `pipelines/stardist/eval_stardist_panoptic.py`: panoptic eval helper (when doing panoptic).
- `pipelines/stardist/train_stardist_panoptic.py`: panoptic training helper.

### Detectron2 (segmentation)
- `pipelines/detectron/train_detectron_on_chip_tifs.py`: instance segmentation training.
- `pipelines/detectron/eval_detectron_on_chip_tifs.py`: per-size eval on chips (writes gpkg/per-chip).
- `pipelines/detectron/eval_detectron_fixed_val.py`: eval multiple sizes against fixed val.

### Detectron2 (multiclass segmentation)
- `pipelines/detectron_mc/train_detectron_on_chip_tifs_multiclass.py`: multiclass training.
- `pipelines/detectron_mc/eval_detectron_on_chip_tifs_multiclass.py`: per-size eval (gpkg/per-chip).
- `pipelines/detectron_mc/eval_detectron_fixed_val_multiclass.py`: fixed-val eval across sizes.

### Objects + XGB classification
- `pipelines/objects/create_objects_from_detectron_sizes.py`: reference objects with class field.
- `pipelines/objects/create_objects_from_detectron_eval_sizes.py`: objects from detectron outputs.
- `pipelines/objects/create_objects_from_stardist_eval_sizes.py`: objects from stardist outputs.
- `pipelines/objects/classify_objects_from_sizes.py`: XGB classifier for object GPKGs.
- `pipelines/objects/make_pred_gpkgs_classified.py`: build classified prediction GPKGs.

### Metrics / summaries
- `analysis/summarize_fixed_val_iou_classaware.py`: class-aware PQ/RQ/SQ (classified objects).
- `analysis/summarize_fixed_val_iou_classaware_predpolys.py`: class-aware PQ/RQ/SQ on pred polys.
- `analysis/summarize_fixed_val_iou_multiclass_overall.py`: overall PQ/RQ/SQ for Detectron2-mc.
- `analysis/summarize_fixed_val_iou_multiclass.py`: per-class PQ/RQ/SQ for Detectron2-mc.
- `analysis/summarize_fixed_val_ap.py`: AP/AP50/AP75 from GPKGs (both methods).
- `analysis/summarize_ap50_by_height_density.py`: AP50 by height/density bins.
- `analysis/summarize_class_metrics_from_classified_objects.py`: class-aware metrics (objects).
- `analysis/summarize_multiclass_per_class_metrics.py`: per-class F1 for Detectron2-mc.
- `analysis/run_nonparametric_bin_tests.py`: non-parametric stats on per-chip AP50.
- `analysis/summarize_eval_time.py`: eval-time summary (Detectron2 vs SAM+StarDist).

### Plots
- `plots/plot_ap_compare.py`: AP line comparisons (AP/AP50/AP75).
- `plots/plot_ap50_pq_line_2panel.py`: AP50 + class-aware PQ lines vs training size.
- `plots/plot_ap50_density_height_boxplots_2panel.py`: AP50 bin boxplots (+ significance).
- `plots/plot_ap50_bin_significance.py`: significance overlays for bin plots.
- `plots/plot_panoptic_metrics_bar.py`: SQ/RQ/PQ bar chart at a single size.
- `plots/plot_class_f1_by_species.py`: 2-panel F1 (model-level + per-class).
- `plots/plot_eval_time.py`: normalized eval-time comparisons.

### Documentation
- `docs/reference_data_card.md`: latest data card output.
- `docs/detectron_instructions.md`: detectron/SAM/StarDist workflow notes.

## Deprecated / legacy

### Explicitly deprecated
- `deprecated/summarize_fixed_val_iou.py`: class-agnostic PQ/RQ/SQ.
- `deprecated/plot_pq_rq_sq_vs_size.py`
- `deprecated/plot_pq_rq_sq_compare.py`
- `deprecated/plot_pq_rq_sq_compare_panels.py`
- `deprecated/plot_pq_rq_sq_three_panel.py`
- `deprecated/plot_ap50_height_density_boxplots.py`
- `deprecated/plot_ap50_density_height_classaware_pq_4panel.py`

### Outdated folder (do not use)
- `deprecated/outdated/*` (legacy scripts).

## Notes / conventions

- **Fixed validation** is defined by `data/chip_splits_*/fixed_val_manifest.json`
  and chips under `data/chips_*/fixed_val/chips_val`.
- **Legacy data** previously stored at repo root now lives under `data/legacy_root/`.
- **Per-chip evaluation** uses `{preds_root}/{size}/per_chip_preds/*.gpkg`
  and should be used consistently when comparing metrics.
- **Class-aware PQ** should come from the class-aware summary scripts above.
- **Class-agnostic scripts** are deprecated to avoid confusion.

If you need a script added to "Active" or reclassified, update this guide along with the code.
