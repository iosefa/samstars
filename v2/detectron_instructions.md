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


For v2:
## Build Chips

1. build splits with a fixed validation set:
- ```
  python v2/build_chip_splits.py \
    --tiles-dir v2/train-sam \
    --out-root v2/chip_splits_128_100 \
    --size 512 --stride 128 \
    --split-seed 7 \
    --start-chips 1 --step 1 --max-chips 100 \
    --fixed-val-crowns 1000 \
    --count-instances
  ```
2. split chips      
   - ```
       python v2/build_chips_from_manifest.py \
        --splits-root v2/chip_splits_128_100 \
        --out-dir v2/chips_128_100 \
        --size 512 \
        --fixed-val-manifest v2/chip_splits_128_100/fixed_val_manifest.json
       ```

# Size comparisons
These commands show how to do the comparisons across training sizes

## Detectron      

1. run train_detectron_on_chip_tifs.py

```bash
for sz in 0001 0010 0020 0030 0040 0050 0060 0070 0080 0090 0100; do
    python v2/train_detectron_on_chip_tifs.py \
      --train-manifest v2/chips_128_100/${sz}/chips_train/detectron_manifest.json \
      --output-dir     v2/detectron/${sz}/train_logs \
      --device cpu \
      --skip-val-eval \
      --min-iters 200 \
      --auto-epochs-ref-size 0100 \
      --auto-epochs-ref-epochs 50 \
      --auto-epochs-chips-root v2/chips_128_100
  done
```

2. validate: v2/eval_detectron_fixed_val.py

```bash
for sz in 0001 0010 0020 0030 0040 0050 0060 0070 0080 0090 0100; do
    python v2/eval_detectron_fixed_val.py \
      --val-manifest v2/chips_128_100/fixed_val/chips_val/detectron_manifest.json \
      --weights-root v2/detectron \
      --out-root v2/detectron_fixed_val \
      --sizes ${sz} \
      --device cpu \
      --score-thresh 0.3 \
      --min-mask-px 10 \
      --max-dets 500 \
      --rpn-post-nms-topk 2000 \
      --write-per-chip
  done
  ```

3. summarize

```bash
python v2/summarize_fixed_val_iou.py \
    --val-manifest v2/chips_128_100/fixed_val/chips_val/detectron_manifest.json \
    --preds-root v2/detectron_fixed_val \
    --train-manifests-root v2/chips_128_100 \
    --use-per-chip --sizes 0001,0010,0020,0030,0040,0050,0060,0070,0080,0090,0100
```

Then the plot:

```bash
python v2/plot_pq_rq_sq_vs_size.py \
    --summary-csv v2/detectron_fixed_val/summary_iou.csv \
    --out v2/detectron_fixed_val/pq_rq_sq_vs_size.png \
    --x n_train_instances
```

---

sam:

conda activate sam_star

first compute scaling stats so we can scale the chips. no need to do this with detectron since it does its own scaling

```bash
python v2/compute_sam_global_scaling.py \
    --chips-root v2/chips_128_100 \
    --sizes 0100 \
    --percentiles 2,98 \
    --out v2/chips_128_100/sam_scale_2_98.json
```

then we need to copy val files to match the call. this can be refactored to eliminate this need later:

```bash
sizes=($(ls v2/chip_splits_128_100 | rg '^[0-9]{4}$' | sort))
fixed_val=v2/chip_splits_128_100/fixed_val_manifest.json
out_root=v2/sam_manifests_fixed_val

mkdir -p ${out_root}

for sz in "${sizes[@]}"; do
    src=v2/chip_splits_128_100/${sz}/train_manifest.json
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
sizes=($(ls v2/chips_128_100 | rg '^[0-9]{4}$' | sort))

for sz in "${sizes[@]}"; do
    python v2/train_sam_decoder_from_chips.py \
      --chips-root v2/chips_128_100 \
      --ckpt sam/sam_vit_l_0b3195.pth \
      --device mps \
      --sizes ${sz} \
      --min-iters 15 \
      --auto-epochs-ref-size 0100 \
      --auto-epochs-ref-epochs 50 \
      --scale-stats v2/chips_128_100/sam_scale_2_98.json

    python v2/predict_sam_from_manifests.py \
      --manifests-root v2/sam_manifests_fixed_val \
      --ckpt-root v2/chips_128_100 \
      --out-root v2/sam_probs_fixed_val \
      --device mps \
      --sizes ${sz} \  # can also be size like 0001 (or maybe a list?)
      --splits train \  # can also be train,val
      --scale-stats v2/chips_128_100/sam_scale_2_98.json
  done
```

then, need to build the prob manifest json:

```bash
python v2/build_prob_manifests.py \
    --probs-root v2/sam_probs_fixed_val \
    --chips-root v2/chips_128_100 \
    --fixed-val-size fixed_val \
    --sizes 0001 # use whatever sizes or in loop
```

## stardist:

conda activate stardist-env

```bash
sizes=($(ls v2/chips_128_100 | rg '^[0-9]{4}$' | sort))

for sz in "${sizes[@]}"; do
    
    python v2/train_stardist_on_probs.py \
    --train-probs v2/sam_probs_fixed_val/${sz}/train_probs.json \
    --val-probs v2/sam_probs_fixed_val/${sz}/val_probs.json \
    --out-dir v2/stardist/${sz} \
    --input-key file_name \  # used to change what sam output to train on
    --use-aux --scale-stats v2/chips_128_100/scale_2_98.json \
    --epochs 50 \
    --min-iters 200
      
  python v2/eval_stardist_on_probs.py \
    --val-probs v2/sam_probs_fixed_val/${sz}/val_probs.json \
    --model-dir v2/stardist/${sz} \
    --out-dir v2/stardist_eval/${sz} \
    --input-key file_name  --use-aux --scale-stats v2/chips_128_100/scale_2_98.json
    # can also add: --prob-thr 0.2 --nms-thr 0.2 --normalize or --no-normalize (must match training
done
```
