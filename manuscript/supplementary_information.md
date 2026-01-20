# Supplementary Information — Multimodal, prompt‑guided crown delineation and species mapping using SAM–StarDist with VHR multispectral imagery and lidar

## S1. Lidar preprocessing
Airborne lidar tiles were obtained from the NOAA/USGS Hawaiʻi QL‑1 project (2018–2020; ≥ 8 pts m⁻²; Leica ALS80) via NOAA Digital Coast. We applied a standard workflow using PDAL v2.9: (i) de‑noising with a 3σ statistical outlier filter; (ii) ground classification with the Cloth Simulation Filter (CSF; Zhang et al., 2016) configured with 0.5 m cloth resolution, rigidness 3, and 500 iterations; (iii) height normalization to height‑above‑ground (HAG) by subtracting a triangulated ground surface; and (iv) tiling to Entwine Point Tile (EPT) format. These steps produce consistent, well‑behaved normalized point clouds for canopy structure modeling.

## S2. WorldView‑3 preprocessing
Two cloud‑free WV3 scenes (December 2020) were pansharpened to 0.31 m. Imagery was orthorectified using vendor RPCs and co‑registered to the lidar DEM with automated tie‑points (sub‑pixel RMSE < 0.25 m). Atmospheric correction employed FLAASH (tropical profile; rural aerosol), with water vapour estimated from the 940 nm band. Scenes were relatively normalized (spectralmatch) to reduce residual gain/offset differences. All rasters were reprojected to EPSG:6635 and aligned on the common 0.31 m grid. Bands NIR‑1 (7), Red (5), and Green (3) were used for SAM fine‑tuning and inference.

## S3. Cost‑surface construction
To encode canopy boundary likelihoods on the 0.31 m grid, we combined three percentile‑stretched (2–98%), min–max scaled layers:

$$ H = \left\lVert\nabla\,\mathrm{CHM}\right\rVert \tag{S3.1} $$

$$ N = 1 - \frac{\mathrm{NIR1} - \mathrm{Red}}{\mathrm{NIR1} + \mathrm{Red}} \tag{S3.2} $$

$$ E = -\sum_i p_i\,\log_2 p_i \tag{S3.3} $$

representing (i) CHM gradient magnitude (Sobel), (ii) inverse NDVI from WV3 NIR‑1 and Red, and (iii) Haralick entropy (Coastal band; 3‑px window). The composite cost surface is

$$ C = 0.4\,H + 0.3\,N + 0.3\,E. \tag{S3.4} $$

## S4. Seed generation and cost‑aware clustering
Seed points were derived from both CHM and first‑return density rasters. After Gaussian smoothing

$$ R_\sigma = R * \mathcal{G}_\sigma,\qquad \sigma = 2\ \text{px}, \tag{S4.1} $$

a pixel $(i,j)$ was marked as a candidate treetop if

$$ R_\sigma(i,j) = \max\limits_{(u,v)\in \mathcal{N}_{i,j}} R_\sigma(u,v)\quad \text{and}\quad R_\sigma(i,j) \ge 4.5\,\text{m}. \tag{S4.2} $$

Nearby peaks were grouped using a height‑aware radius

$$ \varepsilon_i = \mathrm{clip}(\alpha h_i, \varepsilon_{\min}, \varepsilon_{\max})\quad (\alpha=0.15,\ \varepsilon_{\min}=1.5\,\text{m},\ \varepsilon_{\max}=4\,\text{m}). \tag{S4.3} $$

Cost‑aware pruning penalized edges between seeds based on the mean cost sampled along straight‑line paths:

$$
d_{ij} = \begin{cases}
 d^{xy}_{ij}, & d^{xy}_{ij} \le d_{\mathrm{thr}}^{xy} \\
 d^{xy}_{ij}\,\bigl(1 + w_c\,\bar{C}_{ij}\bigr), & \text{otherwise,}
\end{cases}\qquad d_{\mathrm{thr}}^{xy}=3\,\text{m},\; w_c=1.
\tag{S4.4}
$$
Seeds connected by low‑cost paths were merged; duplicates within 0.8 m were removed. Where intra‑cluster height ranges were large, clusters were split at the median height. The canonical seed is the cluster centroid with representative height at the cluster maximum.

## S5. SAM fine‑tuning and inference
We used SAM‑ViT‑L with frozen image and prompt encoders, fine‑tuning only the mask decoder on 254 manually delineated crowns from two 500 m × 500 m tiles (inputs: WV3 bands 7/5/3). Chip extraction: 512×512 px with 32 px stride; resized to 1024 px; normalized to [0,1]. Training: 15 epochs, batch size = 1, AdamW (LR $=5\times10^{-5}$), BCE + soft Dice. Inference: tiled 1024×1024 px crops with overlap. Prompts: 8×8 m box per seed; positive points = tallest CHM pixel + up to 80 high‑CHM pixels; negative points = up to 150 ground pixels + up to 300 neighbouring‑crown pixels. Decoder logits were scaled with $\mathrm{LOGIT\_SCALE}=1.4$, CHM‑weighted (×3), tapered with a Gaussian of $\sigma = \mathrm{box\_width}/3$, and mosaicked via pixelwise maximum to obtain a canopy‑likelihood surface $P$.

## S6. StarDist segmentation and post‑processing
We fused $P$ with CHM and the inverse cost to emphasize plausible boundaries, masking pixels with CHM < 3 m. The fused intensity was

$$ I = P\,\bigl(0.75\,\mathrm{stretch}(\mathrm{CHM}) + 0.25\,\mathrm{stretch}(1 - C)\bigr). \tag{S6.1} $$

The map was stretched to 1–99.8% and segmented with StarDist‑2D (Schmidt et al., 2018) using $\texttt{prob\_thresh}=0.15$ and $\texttt{nms\_thresh}=0.20$. Overlaps (IoU ≥ 0.5) were merged and smaller duplicates removed. Very small/low objects (median HAG < 2 m or max HAG < 3 m) were filtered to exclude shrubs and noise.

## S7. Crown features for classification
From each crown, we derived a compact 59‑feature descriptor comprising:
- Spectral (48): six statistics per WorldView‑3 band — mean, variance, skewness, kurtosis, minimum, maximum — across eight bands (Coastal, Blue, Green, Yellow, Red, Red‑edge, NIR‑1, NIR‑2). Feature names follow `b{0..7}_{stat}` where `b0..b7` map to [Coastal, Blue, Green, Yellow, Red, Red‑edge, NIR‑1, NIR‑2].
- Texture (6): Haralick GLCM metrics on the Coastal band at 0.31 m, using a 2‑pixel offset averaged over four directions (0°, 45°, 90°, 135°): contrast, dissimilarity, homogeneity, energy, correlation, entropy. Feature names follow `b0_{glcm_metric}`.
- Structure (5): lidar‑derived canopy structure: plant area index (PAI), foliage‑height diversity (FHD), maximum canopy height (CH), and lidar intensity mean and variance.

These features capture complementary pigment/texture (spectral/GLCM) and structural (lidar) information and form the input to the tabular classifier.

## S8. Classifier settings and SHAP analyses
We trained XGBoost (primary), Random Forest, and a shallow MLP on a 70/30 species‑stratified split of labelled crowns across six classes (METPOL, GREROB, ACAKOA, DIOSAN, PROPAL, OTHER). Low‑confidence predictions (top‑two probability gap < 0.25) were flagged; where appropriate, we verified plausibility against species range references (Little & Skolmen, 1989; Wagner et al., 1999; Price et al., 2012). For interpretability, SHAP values were computed and summarized as (i) overall mean |SHAP| per feature and (ii) sums by modality (Spectral, Texture, Structure).

Figure S1. Confusion matrices for Set B classification (XGBoost and CNN). The left panel shows the XGBoost confusion matrix computed on the 30% species‑stratified hold‑out set; the right panel shows the CNN baseline using the same hold‑out. Labels follow the six‑class scheme in Section 2.3 of the manuscript (METPOL, GREROB, ACAKOA, DIOSAN, PROPAL, OTHER).

<img src="figures/output/supp_confusion_side_by_side.png" alt="Confusion matrices (XGBoost vs CNN)" width="100%" />

Table S1. Full per‑class precision, recall, and F1 for XGBoost and the CNN baseline on Set B (30% species‑stratified hold‑out).

| class_name | precision | recall | f1    | support | method  |
| ---------- | --------- | ------ | ----- | ------- | ------- |
| METPOL     | 0.93      | 0.93   | 0.93  | 363.0   | XGBoost |
| GREROB     | 0.87      | 0.9    | 0.89  | 324.0   | XGBoost |
| ACAKOA     | 0.97      | 0.96   | 0.97  | 188.0   | XGBoost |
| DIOSAN     | 0.89      | 0.94   | 0.91  | 299.0   | XGBoost |
| PROPAL     | 0.99      | 0.99   | 0.99  | 280.0   | XGBoost |
| OTHER      | 0.8       | 0.69   | 0.74  | 190.0   | XGBoost |
| METPOL     | 0.793     | 0.609  | 0.689 | 384.0   | CNN     |
| GREROB     | 0.904     | 0.318  | 0.471 | 327.0   | CNN     |
| ACAKOA     | 0.27      | 0.995  | 0.424 | 187.0   | CNN     |
| DIOSAN     | 0.93      | 0.139  | 0.242 | 288.0   | CNN     |
| PROPAL     | 0.973     | 0.761  | 0.854 | 280.0   | CNN     |
| OTHER      | 0.447     | 0.708  | 0.548 | 178.0   | CNN     |

Figure S2. SHAP modality summary and top features. Top: contributions summed within each modality (Spectral, Texture, Structure) using mean |SHAP| across crowns and classes. Bottom: the top fifteen features ranked by mean |SHAP| with human‑readable names corresponding to WV3 bands and lidar descriptors.

<img src="figures/output/supp_shap_modalities.png" alt="SHAP by modality" width="55%" />

<img src="figures/output/supp_shap_top_features.png" alt="Top features by mean |SHAP|" width="75%" />

Table S2. SHAP contributions aggregated by modality (sum of mean |SHAP| across features within each modality).

| modality  | mean_abs_shap |
| --------- | ------------- |
| Spectral  | 5.131         |
| Texture   | 0.335         |
| Structure | 0.121         |

Figure S3. Global SHAP feature importance (top‑20). Bar chart of the 20 highest mean |SHAP| features across all classes.

<img src="figures/output/supp_shap_global_top20.png" alt="Global top‑20 SHAP features" width="70%" />

Figure S4. Per‑class SHAP beeswarm plots (top‑20 features). Each panel shows the distribution of SHAP values for the top features within a class (METPOL, GREROB, ACAKOA, DIOSAN, PROPAL, OTHER) using a deterministic 1,000‑row sample to balance readability and fidelity.

<img src="figures/output/supp_shap_beeswarm_class_0.png" alt="Beeswarm METPOL" width="49%" />
<img src="figures/output/supp_shap_beeswarm_class_1.png" alt="Beeswarm GREROB" width="49%" />
<img src="figures/output/supp_shap_beeswarm_class_2.png" alt="Beeswarm ACAKOA" width="49%" />
<img src="figures/output/supp_shap_beeswarm_class_3.png" alt="Beeswarm DIOSAN" width="49%" />
<img src="figures/output/supp_shap_beeswarm_class_4.png" alt="Beeswarm PROPAL" width="49%" />
<img src="figures/output/supp_shap_beeswarm_class_5.png" alt="Beeswarm OTHER" width="49%" />

Figure S5. Global SHAP beeswarm (top‑20). Beeswarm plot using class‑summed SHAP values across all samples and features.

<img src="figures/output/supp_shap_beeswarm_global.png" alt="Global SHAP beeswarm" width="70%" />

Figure S6. Per‑class SHAP dependence (top‑1 feature per class). For each class, dependence plot shows how the most influential feature varies with its SHAP values.

<img src="figures/output/supp_shap_dependence_class_0_feat_0.png" alt="Dependence METPOL top feature" width="49%" />
<img src="figures/output/supp_shap_dependence_class_1_feat_0.png" alt="Dependence GREROB top feature" width="49%" />
<img src="figures/output/supp_shap_dependence_class_2_feat_0.png" alt="Dependence ACAKOA top feature" width="49%" />
<img src="figures/output/supp_shap_dependence_class_3_feat_0.png" alt="Dependence DIOSAN top feature" width="49%" />
<img src="figures/output/supp_shap_dependence_class_4_feat_0.png" alt="Dependence PROPAL top feature" width="49%" />
<img src="figures/output/supp_shap_dependence_class_5_feat_0.png" alt="Dependence OTHER top feature" width="49%" />

Table S3. SHAP contributions per feature (mean |SHAP| across crowns and classes). The complete listing is provided in `figures/output/supp_table_shap_features.csv`. Modality labels are assigned as: Spectral (all per‑band statistics), Texture (GLCM metrics), and Structure (CH, FHD, PAI, intensity mean/variance).

| feature_raw      | feature                      | mean_abs_shap | modality |
| ---------------- | ---------------------------- | ------------- | -------- |
| b0_mean          | Coastal (mean)               | 1.037         | Spectral |
| ch               | Canopy Height (max)          | 0.464         | Structure|
| b0_max           | Coastal (max)                | 0.390         | Spectral |
| b0_min           | Coastal (min)                | 0.289         | Spectral |
| mean_intensity   | Intensity (mean)             | 0.253         | Structure|
| b7_max           | NIR‑2 (max)                  | 0.249         | Spectral |
| b6_max           | NIR‑1 (max)                  | 0.225         | Spectral |
| b5_mean          | Red‑edge (mean)              | 0.182         | Spectral |
| fhd              | Foliage Height Diversity     | 0.131         | Structure|
| variance_intensity | Intensity (variance)       | 0.120         | Structure|
| b0_dissimilarity | Coastal (GLCM dissimilarity) | 0.054         | Texture  |
| b0_correlation   | Coastal (GLCM correlation)   | 0.050         | Texture  |

Note: The table above shows an illustrative subset; see the CSV for the complete, modality‑corrected feature list.

## S9. Metrics and statistics
Let $\mathcal{P}$ and $\mathcal{R}$ be the sets of predicted and reference crowns. For a predicted–reference pair $(p, r)$, the intersection‑over‑union is $\mathrm{IoU}(p,r) = \tfrac{|p \cap r|}{|p \cup r|}$. Matching used a greedy assignment over pairs with IoU > 0, and a true positive (TP) required $\mathrm{IoU} \ge 0.5$; unmatched predictions are false positives (FP) and unmatched references are false negatives (FN). Panoptic metrics (Kirillov et al., 2019) were computed as $\mathrm{SQ} = \tfrac{1}{|\mathrm{TP}|} \sum\_{(p,r)\in \mathrm{TP}} \mathrm{IoU}(p,r)$, $\mathrm{RQ} = \tfrac{|\mathrm{TP}|}{|\mathrm{TP}| + 0.5\,|\mathrm{FP}| + 0.5\,|\mathrm{FN}|}$, and $\mathrm{PQ} = \mathrm{SQ}\,\times\,\mathrm{RQ}$. We also report precision $P = \tfrac{|\mathrm{TP}|}{|\mathrm{TP}|+|\mathrm{FP}|}$, recall $R = \tfrac{|\mathrm{TP}|}{|\mathrm{TP}|+|\mathrm{FN}|}$, and $F1 = \tfrac{2PR}{P+R}$.\
Statistics: For Set A, IoU distributions were compared within methods across strata (height and local density) using two‑sided Mann–Whitney U tests with Bonferroni corrections; between‑method differences within strata used Dunn tests with Bonferroni adjustment. Software: scipy v1.11 (stats), numpy v1.26, pandas v2.2, geopandas v0.14.

Optional effect sizes: For the height and density contrasts, Cliff’s delta (δ) can complement p‑values. We compute δ with bias‑corrected confidence intervals using the orddom method; large |δ| indicates stronger separation. We summarize δ alongside p‑values in the per‑stratum outputs when requested.

Table S4. Set A per‑plot metrics (PQ, SQ, RQ) at IoU ≥ 0.5, with TP/FP/FN, precision, recall, and F1 for each method.

| plot_id | method       | PQ    | SQ    | RQ    | TP  | FP | FN  | precision | recall50 | f1    |
| ------- | ------------ | ----- | ----- | ----- | --- | -- | --- | --------- | -------- | ----- |
| 1       | SAM–StarDist | 0.382 | 0.664 | 0.576 | 78  | 47 | 68  | 0.624     | 0.534    | 0.576 |
| 1       | CNN          | 0.333 | 0.675 | 0.494 | 65  | 52 | 81  | 0.556     | 0.445    | 0.494 |
| 1       | DeepForest   | 0.052 | 0.594 | 0.087 | 8   | 30 | 138 | 0.211     | 0.055    | 0.087 |
| 2       | SAM–StarDist | 0.403 | 0.67  | 0.601 | 83  | 31 | 79  | 0.728     | 0.512    | 0.601 |
| 2       | CNN          | 0.305 | 0.69  | 0.442 | 59  | 46 | 103 | 0.562     | 0.364    | 0.442 |
| 2       | DeepForest   | 0.006 | 0.604 | 0.011 | 1   | 26 | 161 | 0.037     | 0.006    | 0.011 |
| 3       | SAM–StarDist | 0.33  | 0.654 | 0.504 | 57  | 37 | 75  | 0.606     | 0.432    | 0.504 |
| 3       | CNN          | 0.286 | 0.672 | 0.425 | 44  | 31 | 88  | 0.587     | 0.333    | 0.425 |
| 3       | DeepForest   | 0.05  | 0.645 | 0.078 | 7   | 40 | 125 | 0.149     | 0.053    | 0.078 |
| 4       | SAM–StarDist | 0.56  | 0.711 | 0.788 | 78  | 17 | 25  | 0.821     | 0.757    | 0.788 |
| 4       | CNN          | 0.16  | 0.612 | 0.261 | 29  | 90 | 74  | 0.244     | 0.282    | 0.261 |
| 4       | DeepForest   | 0.083 | 0.589 | 0.141 | 10  | 29 | 93  | 0.256     | 0.097    | 0.141 |
| 5       | SAM–StarDist | 0.53  | 0.7   | 0.757 | 140 | 42 | 48  | 0.769     | 0.745    | 0.757 |
| 5       | CNN          | 0.299 | 0.682 | 0.439 | 77  | 86 | 111 | 0.472     | 0.41     | 0.439 |
| 5       | DeepForest   | 0.047 | 0.597 | 0.079 | 10  | 55 | 178 | 0.154     | 0.053    | 0.079 |
| 6       | SAM–StarDist | 0.444 | 0.713 | 0.622 | 88  | 33 | 74  | 0.727     | 0.543    | 0.622 |
| 6       | CNN          | 0.235 | 0.643 | 0.366 | 49  | 57 | 113 | 0.462     | 0.302    | 0.366 |
| 6       | DeepForest   | 0.031 | 0.613 | 0.051 | 5   | 28 | 157 | 0.152     | 0.031    | 0.051 |

Table S5. IoU threshold sensitivity (0.30 vs 0.50) for Set A (class‑agnostic) and Set C (class‑aware). Entries report PQ, SQ, RQ, and TP/FP/FN alongside precision, recall, and F1.

Set A (class‑agnostic):

| set  | method       | iou_thr | PQ    | SQ    | RQ    | TP  | FP  | FN  | precision | recall | f1    |
| ---- | ------------ | ------- | ----- | ----- | ----- | --- | --- | --- | --------- | ------ | ----- |
| SetA | SAM–StarDist | 0.3     | 0.497 | 0.638 | 0.78  | 633 | 98  | 260 | 0.866     | 0.709  | 0.78  |
| SetA | CNN          | 0.3     | 0.341 | 0.594 | 0.574 | 453 | 232 | 440 | 0.661     | 0.507  | 0.574 |
| SetA | DeepForest   | 0.3     | 0.083 | 0.47  | 0.177 | 101 | 148 | 792 | 0.406     | 0.113  | 0.177 |
| SetA | SAM–StarDist | 0.5     | 0.443 | 0.688 | 0.643 | 522 | 209 | 371 | 0.714     | 0.585  | 0.643 |
| SetA | CNN          | 0.5     | 0.273 | 0.668 | 0.408 | 322 | 363 | 571 | 0.47      | 0.361  | 0.408 |
| SetA | DeepForest   | 0.5     | 0.043 | 0.605 | 0.072 | 41  | 208 | 852 | 0.165     | 0.046  | 0.072 |

Set C (class‑aware):

| set  | method       | iou_thr | PQ    | SQ    | RQ    | TP   | FP  | FN   | precision | recall | f1    |
| ---- | ------------ | ------- | ----- | ----- | ----- | ---- | --- | ---- | --------- | ------ | ----- |
| SetC | SAM–StarDist | 0.3     | 0.484 | 0.673 | 0.719 | 1623 | 226 | 1044 | 0.878     | 0.609  | 0.719 |
| SetC | CNN          | 0.3     | 0.0   | 0.0   | 0.0   | 0    | 595 | 2667 | 0.0       | 0.0    | 0.0   |
| SetC | SAM–StarDist | 0.5     | 0.442 | 0.716 | 0.618 | 1395 | 454 | 1272 | 0.754     | 0.523  | 0.618 |
| SetC | CNN          | 0.5     | 0.0   | 0.0   | 0.0   | 0    | 595 | 2667 | 0.0       | 0.0    | 0.0   |
## S10. CNN baseline models
Segmentation baseline (Detectron2): class‑agnostic Mask R‑CNN with ResNet‑50–FPN backbone (config `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`). Inputs are 3‑band WV3 chips (NIR‑1, Red, Green; robust 2–98% per‑band normalization) mapped to RGB order. Masks are polygons rasterized from GeoPackage crowns (single category “crown”), sourced from SAM–StarDist pseudo‑labels (5,480 polygons). Key settings: `IMS_PER_BATCH=1–2`, `BASE_LR=2.5e-4`, `MAX_ITERS=5000`, `WARMUP_ITERS=500`, `ROI_HEADS.BATCH_SIZE_PER_IMAGE=128`, `ROI_HEADS.NUM_CLASSES=1`, `SCORE_THRESH_TEST=0.5`. Training saves final weights (`maskrcnn_crowns_model_final.pth`) and config YAML; inference uses the same config and score threshold.

Classification baseline (ResNet‑34 dual‑pool + point MLP): a dual‑encoder network combining an image branch (ImageNet‑pretrained ResNet‑34 truncated before the classification head; global average pooled to a 512‑D vector) and a lidar point branch (per‑point MLP 3→64→128→256 with ReLU). Point features are aggregated via dual pooling (max and mean) to 512‑D, then projected 512→128→64 and concatenated with the 512‑D image vector; a final linear layer outputs class logits. Training uses 3‑band image chips resized to 224 px (dataset‑specific mean/std normalization), up to 2048 lidar points per crown (XY normalized to zero‑mean, unit‑variance; Z standardized), batch size 32, 40 epochs, Adam optimizer with separate learning rates for backbone and head (`LR_BACKBONE=5e-5`, `LR_HEAD=1.5e-4`), class‑balanced loss (focal loss with γ=1.5 and label smoothing 0.02, or class‑weighted cross‑entropy), simple flips for augmentation, and a 10% stratified validation split. Band selections follow metadata (for 9‑band vs 8‑band inputs). Best weights and metadata are saved (`*.best.pth`, `*.meta.json`) and used for inference on Set C.

Detectron2 versioning: COCO model URL identifies the R50‑FPN 3× config; we ran training/inference with Detectron2 [commit/tag: FILL_ME] and CUDA/cuDNN [FILL_ME CUDA/cuDNN], on GPU [FILL_ME model, VRAM] (CPU fallback for macOS/MPS in config). The full YAML is saved at `manuscript/experiments/cnn/01_mask_rcnn/maskrcnn_crowns_config.yaml`.

SAM checkpoint and seeds: Fine‑tuning used SAM‑ViT‑L checkpoint [FILL_ME e.g., sam_vit_l_0b3195.pth] with random seed(s) [FILL_ME] for data tiling and optimizer initialization.

StarDist configuration: StarDist‑2D v0.9.1 with `prob_thresh=0.15`, `nms_thresh=0.20`, `n_rays=32`, and `grid=[2,2]` (from `models/stardist/config.json`). No fixed morphological footprint is applied at inference; star‑convex radii are regressed per pixel by the model. The prediction script (`helpers/predict_stars.py`) loads the pretrained model and uses these settings.

Classifier hyperparameters:
- XGBoost (XGBClassifier): `n_estimators=500`, `learning_rate=0.03`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.8`, `objective='multi:softprob'`, `eval_metric='mlogloss'`, `n_jobs=-1`, `random_state=42`.
- Random Forest (placeholder): `n_estimators=[FILL_ME]`, `max_depth=[FILL_ME or None]`, `min_samples_split=[FILL_ME]`, `min_samples_leaf=[FILL_ME]`, `max_features=[FILL_ME]`, `random_state=42`.
- MLP (placeholder): `hidden_layer_sizes=[FILL_ME e.g., (256,128)]`, `activation=[relu]`, `alpha=[FILL_ME]`, `batch_size=[FILL_ME]`, `learning_rate_init=[FILL_ME]`, `max_iter=[FILL_ME]`, `random_state=42`.

Determinism & environment: Python [FILL_ME e.g., 3.12]; xgboost [FILL_ME], stardist 0.9.1, detectron2 [FILL_ME], torch [FILL_ME], rasterio [FILL_ME], shap [FILL_ME], numpy 1.26, pandas 2.2, geopandas 0.14, scipy 1.11. OS [FILL_ME], GPU [FILL_ME model/VRAM].

Stretch and spectralmatch: `stretch(·)` rescales each raster to [0,1] using per‑layer robust percentiles (p2,p98): `y = clip((x − p2)/(p98 − p2), 0, 1)`. Spectral normalization (“spectralmatch”) fits per‑band affine transforms (gain/offset) aligning a secondary scene to a reference using pseudo‑invariant pixels; gains/offsets are estimated by robust least squares on matched pixels, then applied per band.

SAM training tiles: Two 500×500 m tiles used for SAM fine‑tuning were [FILL_ME tile IDs / UTM bounds], with chip extraction seed [FILL_ME]; these tiles do not overlap Sets A–C.

Packaging (SI PDF): To produce a single SI PDF with embedded figures (RSE preference), export with Pandoc + LaTeX engine or wkhtmltopdf. Example:
`pandoc -s supplementary_information.md -o supplementary_information.pdf --pdf-engine=xelatex`.
Alternatively: `wkhtmltopdf supplementary_information.html supplementary_information.pdf`.
