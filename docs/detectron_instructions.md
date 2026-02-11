the original implementation ran multiple detectron related files. 

detectron runs with X env, so load with `conda activate X`

1. `manuscript/experiments/detectron/detectron.py`
   - this file trained detectron on the crowns in set a and output the model file
   - requires a root dir where all files are kept (input img etc)
   - set img and mask globs for tif and gpkg files
   - uses bands 7,5,3.
   - no holdout fraction since we are using a pre-split train/validation mix
   - outputs model pth and config yml files.
2. manuscript/experiments/detectron/run_detectron_set_a.py
   - runs the detectron script with weights on:
     - input_dir (root input dir)
     - output_gpkg
3. similar to the above but as used with the set c originally. I think fine to just use 2. above.


Current pipelines (data root: `data/`):
## Build Chips

1. build splits with a fixed validation set:
```bash
  python data_prep/build_chip_splits.py \
    --tiles-dir data/train-sam \
    --out-root data/chip_splits_128_100 \
    --size 512 --stride 128 \
    --split-seed 7 \
    --start-chips 1 --step 1 --max-chips 100 \
    --fixed-val-fraction 0.3 \
    --order-by instances-asc \
    --count-instances
```

```bash
  python data_prep/build_chips_from_manifest.py \
    --splits-root data/chip_splits_128_100 \
    --out-dir data/chips_128_100 \
    --size 512 \
    --fixed-val-manifest data/chip_splits_128_100/fixed_val_manifest.json
```

need to add another chip to validation:

```bash
python data_prep/ensure_val_has_all_classes.py \
    --splits-root data/chip_splits_128_100 \
    --tiles-dir data/train-sam \
    --avoid-train-manifest data/chip_splits_128_100/0200/train_manifest.json \
    --skip-train-update
    
python data_prep/build_chips_from_manifest.py \
    --splits-root data/chip_splits_128_100 \
    --out-dir data/chips_128_100 \
    --size 512 \
    --fixed-val-manifest data/chip_splits_128_100/fixed_val_manifest.json    
```

note, can check training leakage:

```bash
python - <<'PY'
import json
from pathlib import Path

chip = ("tile-006", 6)
sizes = ["0002","0004","0006","0008","0012","0016","0024","0032","0048","0064","0075","0080","0084","0100","0150","0200","0250"]
root = Path("data/chip_splits_128_100")

for sz in sizes:
    recs = json.load(open(root/sz/"train_manifest.json"))
    hit = any(r["tile"]==chip[0] and int(r["chip_id"])==chip[1] for r in recs)
    print(sz, "overlap:", hit)
PY
0002 overlap: False
0004 overlap: False
0006 overlap: False
0008 overlap: False
0012 overlap: False
0016 overlap: False
0024 overlap: False
0032 overlap: False
0048 overlap: False
0064 overlap: False
0075 overlap: False
0080 overlap: False
0084 overlap: False
0100 overlap: False
0150 overlap: False
0200 overlap: False
0250 overlap: True

# added an update to add more class 3 validation
python data_prep/ensure_val_has_all_classes.py \
    --splits-root data/chip_splits_128_100 \
    --tiles-dir data/train-sam \
    --avoid-train-manifest data/chip_splits_128_100/0200/train_manifest.json \
    --add-classes 3 \
    --min-class-count 20 \
    --skip-train-update
    
python - <<'PY'
import json
from pathlib import Path

chip = ("tile-008", 7)
sizes = ["0002","0004","0006","0008","0012","0016","0024","0032","0048","0064","0075","0080","0084","0100","0150","0200","0250"]
root = Path("data/chip_splits_128_100")

for sz in sizes:
    recs = json.load(open(root/sz/"train_manifest.json"))
    hit = any(r["tile"]==chip[0] and int(r["chip_id"])==chip[1] for r in recs)
    print(sz, "overlap:", hit)
PY
0002 overlap: False
0004 overlap: False
0006 overlap: False
0008 overlap: False
0012 overlap: False
0016 overlap: False
0024 overlap: False
0032 overlap: False
0048 overlap: False
0064 overlap: False
0075 overlap: False
0080 overlap: False
0084 overlap: False
0100 overlap: False
0150 overlap: False
0200 overlap: False
0250 overlap: True
```

# Size comparisons
These commands show how to do the comparisons across training sizes

## Detectron      

1. run train_detectron_on_chip_tifs.py

0004 0012 0041 0048 0051 0064 0084 0100 0150

```bash
for sz in 0004 0012 0041 0048 0051 0064 0084 0100 0150; do
    python pipelines/detectron/train_detectron_on_chip_tifs.py \
        --train-manifest data/chips_128_100/${sz}/chips_train/detectron_manifest.json \
        --output-dir     data/detectron/${sz}/train_logs \
        --device cpu \
        --skip-val-eval \
        --min-iters 200 \
        --auto-epochs-ref-size 0100 \
        --auto-epochs-ref-epochs 50 \
        --auto-epochs-chips-root data/chips_128_100 --use-chm
  done
```

2. validate: pipelines/detectron/eval_detectron_fixed_val.py

```bash
for sz in 0004 0012 0041 0048 0051 0064 0084 0100 0150; do
    python pipelines/detectron/eval_detectron_fixed_val.py \
        --val-manifest data/chips_128_100/fixed_val/chips_val/detectron_manifest.json \
        --weights-root data/detectron \
        --out-root data/detectron_fixed_val \
        --sizes ${sz} \
        --device cpu \
        --score-thresh 0.3 \
        --min-mask-px 10 \
        --max-dets 500 \
        --rpn-post-nms-topk 2000 \
        --use-chm --write-per-chip
  done
  ```

3. summarize


```bash
python analysis/summarize_fixed_val_ap.py \
    --preds-root data/detectron_fixed_val \
    --sizes 0004,0048,0064,0120,0150,0200 \
    --train-manifests-root data/chips_128_100 \
    --val-manifest data/chips_128_100/fixed_val/chips_val/detectron_manifest.json

python analysis/summarize_fixed_val_ap.py \
    --preds-root data/stardist_eval \
    --sizes 0004,0048,0064,0120,0150,0200 \
    --train-manifests-root data/chips_128_100 \
    --val-manifest data/chips_128_100/fixed_val/chips_val/detectron_manifest.json \
    --scale-100
```

```bash
python plots/plot_ap_compare.py \
    --summary-a data/stardist_eval/summary_ap.csv \
    --summary-b data/detectron_fixed_val/summary_ap.csv \
    --label-a sam-star \
    --label-b detectron \
    --out data/figures/ap_compare_ap50.png \
    --x n_train_instances \
    --metric segm_AP50 \
    --scale-100 \
    --show-points --x-max 5008 --title "Average Precision vs training size"  --y-label "Average Precision (AP)"
```


now creating objects and class aware:

conda activate pyforestscan-312
```bash
PYTHONPATH=/Users/iosefa/repos/sam python pipelines/objects/create_objects_from_detectron_sizes.py \
    --chips-root data/chips_128_100 \
    --out-root data/chips_128_100 \
    --sizes 0004,0048,0064,0120,0150,0200 \
    --ept /Users/iosefa/repos/obia/docs/example_data/full/ept/ept.json \
    --ept-srs EPSG:32605 \
    --ept-geom-srs EPSG:32605 \
    --class-field class \
    --overwrite

PYTHONPATH=/Users/iosefa/repos/sam python pipelines/objects/create_objects_from_detectron_eval_sizes.py \
    --chips-root data/chips_128_100 \
    --preds-root data/detectron_fixed_val \
    --out-root data/detectron_objects \
    --val-size fixed_val \
    --sizes 0004,0048,0064,0120,0150,0200 \
    --ept /Users/iosefa/repos/obia/docs/example_data/full/ept/ept.json \
    --ept-srs EPSG:32605 \
    --ept-geom-srs EPSG:32605
```

```bash
PYTHONPATH=/Users/iosefa/repos/sam python pipelines/objects/create_objects_from_stardist_eval_sizes.py \
    --chips-root data/chips_128_100 \
    --stardist-root data/stardist_eval \
    --probs-root data/sam_probs_fixed_val \
    --out-root data/stardist_objects \
    --val-size fixed_val \
    --sizes 0250 \
    --ept /Users/iosefa/repos/obia/docs/example_data/full/ept/ept.json \
    --ept-srs EPSG:32605 \
    --ept-geom-srs EPSG:32605 --overwrite
```

then for each size:
```bash
for sz in 0004 0048 0064 0120 0150 0200; do
    PYTHONPATH=/Users/iosefa/repos/sam python pipelines/objects/classify_objects_from_sizes.py \
      --train-objects data/chips_128_100/${sz}/objects.gpkg \
      --objects-root data/detectron_objects \
      --out-root data/detectron_classified \
      --sizes ${sz} \
      --regions /Users/iosefa/repos/sam/data/regions.gpkg \
      --test-size 0
  done
```

```bash 
  for sz in 0004 0048 0064 0120 0150 0200; do
    PYTHONPATH=/Users/iosefa/repos/sam python pipelines/objects/classify_objects_from_sizes.py \
      --train-objects data/chips_128_100/${sz}/objects.gpkg \
      --objects-root data/stardist_objects \
      --out-root data/stardist_classified \
      --sizes ${sz} \
      --regions /Users/iosefa/repos/sam/data/regions.gpkg \
      --test-size 0
  done
```

python analysis/summarize_fixed_val_iou_classaware.py \
    --val-manifest data/chips_128_100/fixed_val/chips_val/detectron_manifest.json --iou-thr 0.5 \
    --classified-root data/detectron_classified \
    --sizes 0048,0064,0080,0120,0150,0200

python analysis/summarize_fixed_val_iou_classaware.py \
    --val-manifest data/chips_128_100/fixed_val/chips_val/detectron_manifest.json --iou-thr 0.5 \
    --classified-root data/stardist_classified \
    --sizes 0048,0064,0080,0120,0150,0200

python plots/plot_ap50_pq_line_2panel.py \
    --summary-a data/stardist_classified/summary_iou_classaware.csv \
    --summary-b data/detectron_classified/summary_iou_classaware.csv \
    --label-a sam-star \
    --label-b detectron \
    --out data/figures/pq_rq_sq_combined_aware.png \
    --x n_train_instances \
    --combined-only \
    --x-max 5008
---

sam:

conda activate sam_star

first compute scaling stats so we can scale the chips. no need to do this with detectron since it does its own scaling

```bash
python pipelines/sam/compute_sam_global_scaling.py \
    --chips-root data/chips_128_100 \
    --sizes 0100 \
    --percentiles 2,98 \
    --out data/chips_128_100/sam_scale_2_98.json
```

then we need to copy val files to match the call. this can be refactored to eliminate this need later:

```bash
sizes=($(ls data/chip_splits_128_100_full | rg '^[0-9]{4}$' | sort))
fixed_val=data/chip_splits_128_100_full/fixed_val_manifest.json
out_root=data/sam_manifests_fixed_val

mkdir -p ${out_root}

for sz in "${sizes[@]}"; do
    src=data/chip_splits_128_100_full/${sz}/train_manifest.json
    if [[ ! -f "${src}" ]]; then
      echo "skip missing train_manifest: ${sz}"
      continue
    fi
    mkdir -p ${out_root}/${sz}
    if [[ -f ${out_root}/${sz}/train_manifest.json ]]; then chmod u+w ${out_root}/${sz}/train_manifest.json 2>/dev/null || true
    fi
    cp "${src}" ${out_root}/${sz}/train_manifest.json
    cp "${fixed_val}" ${out_root}/${sz}/val_manifest.json
    echo "✓ ${sz} train_manifest + fixed val"
done
```

then train the decoder:

```bash
for sz in 0004 0048 0064 0080 0120 0150 0200; do
    python pipelines/sam/train_sam_decoder_from_chips.py \
      --chips-root data/chips_128_100 \
      --ckpt sam/sam_vit_l_0b3195.pth \
      --device mps \
      --sizes ${sz} \
      --min-iters 200 \
      --auto-epochs-ref-size 0100 \
      --auto-epochs-ref-epochs 50 \
      --scale-stats data/chips_128_100/sam_scale_2_98.json

    python pipelines/sam/predict_sam_from_manifests.py \
      --manifests-root data/sam_manifests_fixed_val \
      --ckpt-root data/chips_128_100 \
      --out-root data/sam_probs_fixed_val \
      --device mps \
      --sizes ${sz} \
      --splits train,val \
      --scale-stats data/chips_128_100/sam_scale_2_98.json
  done
```

for sz in 0474; do
    python pipelines/sam/predict_sam_from_manifests.py \
      --manifests-root data/sam_manifests_fixed_val \
      --ckpt-root data/chips_128_100 \
      --out-root data/sam_probs_fixed_val \
      --device mps \
      --sizes ${sz} \
      --splits train,val \
      --scale-stats data/chips_128_100/sam_scale_2_98.json
done

then, need to build the prob manifest json:

```bash
python pipelines/stardist/build_prob_manifests.py \
    --probs-root data/sam_probs_fixed_val \
    --chips-root data/chips_128_100 \
    --fixed-val-size fixed_val \
    --sizes 0004,0012,0041,0048,0051,0064,0084,0100,0150
```

## stardist:

conda activate stardist-env

```bash
for sz in 0004 0048 0064 0080 0120 0150 0200; do  
        python pipelines/stardist/train_stardist_on_probs.py \
        --train-probs data/sam_probs_fixed_val/${sz}/train_probs.json \
        --val-probs data/sam_probs_fixed_val/${sz}/val_probs.json \
        --out-dir data/stardist/${sz} \
        --input-key file_name \
        --use-aux --scale-stats data/chips_128_100/sam_scale_2_98.json --epochs 50
        
    python pipelines/stardist/eval_stardist_on_probs.py \
        --val-probs data/sam_probs_fixed_val/${sz}/val_probs.json \
        --model-dir data/stardist/${sz} \
        --out-dir data/stardist_eval/${sz} \
        --input-key file_name  --use-aux --scale-stats data/chips_128_100/sam_scale_2_98.json     
done
```

for sz in 0004 0012 0041 0048 0051 0064 0084 0100 0150; do
    python pipelines/stardist/eval_stardist_on_probs.py \
        --val-probs data/sam_probs_fixed_val/${sz}/val_probs.json \
        --model-dir data/stardist/${sz} \
        --out-dir data/stardist_eval/${sz} \
        --input-key file_name  --use-aux --scale-stats data/chips_128_100/sam_scale_2_98.json     
done


Added analysis/summarize_fixed_val_ap.py to summarize COCO AP
  metrics per size from each val_metrics.json, with
  optional training/validation counts so it lines up with
  your IoU summary workflow. It reads data/
  detectron_fixed_val/<size>/val_metrics.json (COCO segm
  by default) and writes a summary_ap.csv next to the
  preds root.

  Example (segm AP only, with train+val counts):
```bash
  python analysis/summarize_fixed_val_ap.py \
    --preds-root data/detectron_fixed_val \
    --sizes 0004,0012,0041,0048,0051,0064,0084,0100,0150 \
    --train-manifests-root data/chips_128_100 \
    --val-manifest data/chips_128_100/fixed_val/chips_val/detectron_manifest.json
```
  If you want bbox metrics too:
```bash
  python analysis/summarize_fixed_val_ap.py \
    --preds-root data/detectron_fixed_val \
    --sizes 0004,0012,0041,0048,0051,0064,0084,0100,0150 \
    --train-manifests-root data/chips_128_100 \
    --val-manifest data/chips_128_100/fixed_val/chips_val/detectron_manifest.json \
    --include-bbox

  Output: data/detectron_fixed_val/summary_ap.csv.
```


and to plot:
```bash
 python plots/plot_ap_compare.py \
    --summary-a data/stardist_eval/summary_ap.csv \
    --summary-b data/detectron_fixed_val/summary_ap.csv \
    --label-a sam-star \
    --label-b detectron \
    --out data/figures/ap_compare.png \
    --x n_train_instances \
    --metric segm_AP \
    --include-ap50 \
    --include-ap75 \
    --scale-100 \
    --x-max 3008
```
  - If you only want AP: drop --include-ap50/--include-
    ap75.
  - If you want bbox metrics instead: --metric bbox_AP
    (and same optional flags).


for sz in 0004 0048 0064 0080 0120 0150 0200 0250; do
    python pipelines/detectron_mc/train_detectron_on_chip_tifs_multiclass.py \
      --train-manifest data/chips_128_100/${sz}/chips_train/
  detectron_manifest.json \
      --output-dir data/detectron_mc/${sz}/train_logs \
      --device cpu \
      --epochs 50 \
      --min-iters 200 \
      --class-field class
  done

  python pipelines/detectron_mc/eval_detectron_fixed_val_multiclass.py \
    --val-manifest data/chips_128_100/fixed_val/chips_val/
  detectron_manifest.json \
    --weights-root data/detectron_mc \
    --out-root data/detectron_mc_fixed_val \
    --sizes 0004,0048,0064,0120,0150,0200 \
    --device cpu \
    --score-thresh 0.3 \
    --min-mask-px 10 \
    --max-dets 500 \
    --rpn-post-nms-topk 2000 \
    --write-per-chip
