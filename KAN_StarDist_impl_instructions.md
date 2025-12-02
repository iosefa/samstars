# Implementation Instructions for a Coding Agent

## Scope
Add a cooperative KAN classification head that uses pooled StarDist encoder features plus per-crown descriptors. Keep SAM class-agnostic. Staged training (freeze StarDist, train KAN), with optional light joint fine-tuning of the last StarDist encoder block.

## Prereqs
- StarDist training/inference already working on `crown_probability.tif`.
- Access to StarDist encoder feature maps in code (may need to expose encoder outputs).
- Crowns have class labels in `crowns.gpkg`.

## Tasks
1) **Expose StarDist encoder features**
   - Identify the StarDist 2D model definition in use.
   - Add a method or hook to return encoder feature maps alongside logits during forward.

2) **Feature/embedding builder**
   - Inputs per crown: polygon mask (GT for train), `img.tif`, `chm.tif`, `crown_probability.tif`, StarDist encoder feature map(s).
   - Pool encoder features per crown (mask-average or ROI Align on box/mask) → embedding.
   - Compute descriptors: spectral stats (mean/std/p10/p90 per band), indices if available; CHM stats; SAM prob stats; geometry (area/perimeter/circularity/elongation/convexity).
   - Concatenate embedding + descriptors → feature vector. Save scaler (fit on train).
   - Output tensors X (features), y (class labels), plus scaler and any config.

3) **KAN classifier head**
   - Implement a small KAN (1–2 layers, width ~32–64) on the feature vector.
   - Loss: cross-entropy. Optimizer: AdamW. Early stop on val F1.
   - Save weights, scaler, and config.

4) **Training loop (staged)**
   - Stage 1: freeze StarDist; load pooled features/descriptors; train only KAN.
   - Stage 2 (optional): unfreeze the last StarDist encoder block; joint train with StarDist loss (on masks) + weighted KAN CE (small weight); keep StarDist decoder unchanged.
   - Data loader should provide GT masks + class labels to pool encoder features and build descriptors.

5) **Inference hook**
   - After StarDist inference: get predicted masks/polygons and the corresponding encoder feature map(s).
   - Pool encoder features per predicted crown; compute descriptors; apply scaler; run KAN; write class field into output vectors.

6) **Integration**
   - Add CLI flags to the end-to-end driver to train KAN and to apply KAN at inference.
   - Keep StarDist inference behavior unchanged aside from emitting encoder features for the classifier path.

## Validation
- Train/val split by tile or polygon; report val F1 and confusion matrix.
- Check feature/label alignment (stems, IDs).
- Confirm inference adds a class field to vector outputs.

## Deliverables
- Code changes to expose encoder features and pooling.
- Feature builder script/module.
- KAN trainer script/module + saved artifacts (`kan_weights.pth`, `kan_scaler.pkl`, `kan_config.json`).
- Inference path that tags predicted crowns with classes.***
