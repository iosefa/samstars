# Code Guide (Active vs Deprecated)

This repo has a lot of legacy code. This guide lists what is actively used in the current
v2 workflows and what is deprecated. If a script is not listed here, treat it as "unknown/one-off"
until reviewed.

## Active (current workflows)

### Data prep / splits / chips
- `v2/build_chip_splits.py`: create train-size splits + fixed validation manifest.
- `v2/add_chip_split_size.py`: add a missing size to an existing split set.
- `v2/ensure_val_has_all_classes.py`: enforce validation coverage across classes.
- `v2/build_chips_from_manifest.py`: build chips from split manifests.
- `v2/export_val_chips.py`: export fixed validation chips (used when needed).
- `v2/align_chm_to_img.py`: align CHM to image grid (utility used in pipeline).

### Scaling (SAM input / aux bands)
- `v2/compute_sam_global_scaling.py`: compute global scaling stats.
- `v2/scale_stats.py`: helpers to load/scale inputs.

### SAM → StarDist pipeline
- `v2/train_sam_decoder_from_chips.py`: train SAM decoder per size.
- `v2/predict_sam_from_manifests.py`: SAM inference -> prob/logit outputs per chip.
- `v2/build_fixed_val_manifests.py`: create fixed val manifests (if needed).
- `v2/build_prob_manifests.py`: create train/val prob manifests for StarDist.
- `v2/train_stardist_on_probs.py`: train StarDist on SAM outputs.
- `v2/eval_stardist_on_probs.py`: evaluate StarDist and write val_preds.gpkg.
- `v2/eval_stardist_panoptic.py`: panoptic eval helper (use when doing panoptic).
- `v2/train_stardist_panoptic.py`: panoptic training helper (experimental but in use).

### Detectron2 (segmentation)
- `v2/train_detectron_on_chip_tifs.py`: instance segmentation training.
- `v2/eval_detectron_on_chip_tifs.py`: per-size eval on chips (writes gpkg/per-chip).
- `v2/eval_detectron_fixed_val.py`: eval multiple sizes against fixed val.

### Detectron2 (multiclass segmentation)
- `v2/train_detectron_on_chip_tifs_multiclass.py`: multiclass training.
- `v2/eval_detectron_on_chip_tifs_multiclass.py`: per-size eval (gpkg/per-chip).
- `v2/eval_detectron_fixed_val_multiclass.py`: fixed-val eval across sizes.

### Objects + XGB classification
- `v2/create_objects_from_detectron_sizes.py`: reference objects with class field.
- `v2/create_objects_from_detectron_eval_sizes.py`: objects from detectron outputs.
- `v2/create_objects_from_stardist_eval_sizes.py`: objects from stardist outputs.
- `v2/classify_objects_from_sizes.py`: XGB classifier for object GPKGs.
- `v2/summarize_class_metrics_from_classified_objects.py`: class-aware metrics (objects).
- `v2/summarize_multiclass_per_class_metrics.py`: per-class F1 for Detectron2-mc.
- `v2/make_pred_gpkgs_classified.py`: utility to build classified prediction GPKGs.

### Metrics / summaries
- `v2/summarize_fixed_val_iou_classaware.py`: class-aware PQ/RQ/SQ (classified objects).
- `v2/summarize_fixed_val_iou_classaware_predpolys.py`: class-aware PQ/RQ/SQ on pred polys.
- `v2/summarize_fixed_val_iou_multiclass_overall.py`: overall PQ/RQ/SQ for Detectron2-mc.
- `v2/summarize_fixed_val_iou_multiclass.py`: per-class PQ/RQ/SQ for Detectron2-mc.
- `v2/summarize_fixed_val_ap.py`: AP/AP50/AP75 from GPKGs (both methods).
- `v2/summarize_ap50_by_height_density.py`: AP50 by height/density bins.
- `v2/run_nonparametric_bin_tests.py`: non-parametric stats on per-chip AP50.

### Plots
- `v2/plot_ap_compare.py`: AP line comparisons (AP/AP50/AP75).
- `v2/plot_ap50_pq_line_2panel.py`: AP50 + class-aware PQ lines vs training size.
- `v2/plot_ap50_density_height_boxplots_2panel.py`: AP50 bin boxplots (+ significance).
- `v2/plot_ap50_bin_significance.py`: significance overlays for bin plots.
- `v2/plot_panoptic_metrics_bar.py`: SQ/RQ/PQ bar chart at a single size.
- `v2/plot_class_f1_by_species.py`: 2-panel F1 (model-level + per-class).

### Documentation
- `v2/make_reference_data_card.py`: produces the data card (counts by class/size).

## Deprecated / legacy

### Explicitly marked deprecated
- `v2/summarize_fixed_val_iou.py`: class-agnostic PQ/RQ/SQ (deprecated).
- `v2/plot_pq_rq_sq_vs_size.py`
- `v2/plot_pq_rq_sq_compare.py`
- `v2/plot_pq_rq_sq_compare_panels.py`
- `v2/plot_pq_rq_sq_three_panel.py`
- `v2/plot_ap50_height_density_boxplots.py`
- `v2/plot_ap50_density_height_classaware_pq_4panel.py`

### Outdated folder (do not use)
- `v2/outdated/*` (all scripts in this folder are legacy and superseded).

## Notes / conventions

- **Fixed validation** is defined by `v2/chip_splits_*/fixed_val_manifest.json`
  and chips under `v2/chips_*/fixed_val/chips_val`.
- **Per-chip evaluation** uses `{preds_root}/{size}/per_chip_preds/*.gpkg`
  and should be used consistently when comparing metrics.
- **Class-aware PQ** should come from the class-aware summary scripts above.
- **Class-agnostic scripts** are deprecated to avoid confusion.

If you need a script added to "Active" or reclassified, update this guide along with the code.
