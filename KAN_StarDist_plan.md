# SAM → StarDist → KAN Integration Plan

## Goal
End-to-end architecture: train SAM for crown probability, train StarDist for instance segmentation, then attach a Kolmogorov–Arnold Network (KAN) to classify each crown polygon (using class labels already present in `crowns.gpkg`). Keep segmentation fixed; add a cooperative KAN head that uses StarDist encoder features + crown descriptors.

## Text Diagram
```
Tiles (img.tif, chm.tif, cost.tif, seeds.gpkg, crowns.gpkg with class)
  │
  ├─ SAM training (chips) → fine-tuned SAM decoder
  │
  ├─ SAM inference per tile → crown_probability.tif
  │                              │
  ├─ Rasterize crowns.gpkg → instance mask (for StarDist train)
  │                              │
  ├─ StarDist training on (prob raster, mask) → StarDist model
  │
  ├─ StarDist inference → crown label raster/vector
  │                              │
  ├─ Pool StarDist encoder features per crown + descriptors (img/chm/prob/shape)
  │                              │
  └─ KAN classifier → per-crown class label → write to vector output
```

## Components
### SAM
- Fine-tune decoder on chips from tiles (multi-tile support). Produces `sam_crowns_decoder.pth`.
- Inference via `predict_sam.py` needs `img.tif`, `seeds.gpkg`, `chm.tif`, `cost.tif` and outputs `crown_probability.tif` per tile.

### StarDist
- Training data: (`crown_probability.tif`, rasterized `crowns.gpkg` with unique IDs) per tile.
- Outputs instance label raster and polygons. Remains unchanged in architecture.

### KAN Classifier (cooperative head)
- Inputs: pooled StarDist encoder features per crown + per-crown descriptors.
- Pooled features: for each crown (GT during training; predicted at inference), mask-average or ROI Align the StarDist encoder feature map(s) to get an embedding.
- Descriptors: from `img.tif`, `chm.tif`, `crown_probability.tif` (spectral stats, CHM stats, SAM prob stats) + geometry (area/perimeter/circularity/elongation/convexity).
- KAN layer (one hidden unit):
  ```
  h_j(x) = Σ_k a_{jk} B_k(x⋅w_j + b_j) + c_j
  p = softmax(W h + b)
  L_cls = -Σ y_i log p_i
  ```
  `B_k` are spline bases; `a_{jk}` are spline coefficients; `w_j` mixes inputs; `p` is class probability; `L_cls` is cross-entropy.
- Cooperative but shallow: StarDist is the segmenter; KAN sits on pooled encoder features. Optional light fine-tuning of the last encoder block with a small CE weight; no changes to StarDist decoder logic.

### CoNN Clarification
- We keep SAM class-agnostic. Cooperation is between StarDist (encoder features) and the KAN head. If joint tuning is desired, only unfreeze the last encoder layers; otherwise staged training keeps modules decoupled.

## Training Pipeline
1) **SAM training:** fine-tune decoder on tiles (run once).  
2) **SAM inference:** run on tiles → `crown_probability.tif`.  
3) **StarDist training:** rasterize crowns.gpkg to masks; train StarDist on (prob raster, mask).  
4) **KAN dataset:** for tiles with class labels, pool StarDist encoder features per crown (GT masks) + descriptors from `img.tif`, `chm.tif`, `crown_probability.tif`; pair with class label. Split train/val by tile or polygon.  
5) **KAN training (staged):**  
   - Stage 1: Freeze StarDist; train KAN head on pooled features + descriptors.  
   - Stage 2 (optional): Unfreeze last encoder block; joint fine-tune with StarDist loss + weighted KAN cross-entropy (small CE weight).  
6) **Inference:** run SAM + StarDist; pool encoder features on predicted crowns; build descriptors; run KAN; write class field into output vector layer.

## Risks / Mitigations
- Label completeness: use only tiles with full crown + class labels.
- Class imbalance: class weights or focal loss.
- Consistency: identical feature extraction for train/infer; standardize features.
- Domain shift: ensure band order/CRS/resolution match across tiles.
- Speed: parallelize feature extraction across tiles/polygons.

## Implementation Steps
1. Feature/embedding builder: given polygons + (`img.tif`, `chm.tif`, `crown_probability.tif`) and StarDist encoder outputs, pool encoder features per crown and compute descriptors → X, y, scaler.  
2. KAN trainer: load X/y, split train/val, train KAN (staged, with frozen StarDist), save scaler/weights/config.  
3. Optional joint fine-tune: unfreeze last StarDist encoder block; train with StarDist loss + weighted KAN CE.  
4. Inference hook: after `predict_stars.py`, pool encoder features for predicted crowns, compute descriptors, run KAN, add class field to vectors.  
5. Integrate into the end-to-end driver with flags to train/use KAN after StarDist.

## Deliverables
- Models: `sam_crowns_decoder.pth`, `models/stardist/*`, `kan_weights.pth`, `kan_scaler.pkl`, `kan_config.json`.  
- Data products: per-tile `crown_probability.tif`; StarDist outputs with class attribute added by KAN.
