"""SAM probabilities with a shared prompt policy and one optional blur."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
import geopandas as gpd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from samstars._preprocessing import rgb_to_u8
from samstars._prompts import sample_seed_prompts


def smooth_probabilities(probabilities, sigma, valid):
    """Normalize only for true NoData; valid zero probabilities remain background."""
    valid = np.asarray(valid, dtype=bool)
    values = np.where(valid, probabilities, 0).astype(np.float32)
    if sigma > 0:
        numerator = gaussian_filter(values, sigma, mode='nearest')
        denominator = gaussian_filter(valid.astype(np.float32), sigma, mode='nearest')
        values = np.divide(numerator, denominator, out=np.zeros_like(values), where=denominator > 0)
    return np.where(valid, np.clip(values, 0, 1), 0).astype(np.float32)


def process_simple_window(rec, predictor, device, *, fg_extra=80, random_seed=0,
                          smooth_sigma=0, fast=True, fast_batch=32, seeds=None):
    image_path = Path(rec['img_path'])
    x0, y0, x1, y1 = rec['window']
    window = Window(x0, y0, x1-x0, y1-y0)
    with rasterio.open(image_path) as src:
        image = src.read((1, 2, 3), window=window, boundless=True, masked=True).astype('float32')
        transform, crs = src.window_transform(window), src.crs
    with rasterio.open(image_path.parent/'chm.tif') as src:
        chm = src.read(1, window=window, boundless=True, masked=True).astype('float32')
    with rasterio.open(image_path.parent/'cost.tif') as src:
        cost = src.read(1, window=window, boundless=True, masked=True).astype('float32').filled(np.nan)
    valid = (~np.ma.getmaskarray(image).any(axis=0) & ~np.ma.getmaskarray(chm)
             & np.isfinite(image.filled(np.nan)).all(axis=0) & np.isfinite(chm.filled(np.nan)))
    chm = chm.filled(np.nan)
    chm[~valid] = np.nan
    image = rgb_to_u8(image.filled(0).transpose(1, 2, 0), valid)
    height, width = chm.shape
    seeds = gpd.read_file(image_path.parent/'seeds.gpkg') if seeds is None else seeds
    seeds = seeds.to_crs(crs)
    prompts = []
    for geometry in seeds.geometry:
        r, c = rowcol(transform, geometry.x, geometry.y)
        if 0 <= r < height and 0 <= c < width and valid[r, c]:
            prompts.append(sample_seed_prompts(chm, (c, r), abs(transform.a),
                                              fg_extra=fg_extra, random_seed=random_seed))
    probabilities = np.zeros((height, width), dtype=np.float32)
    if prompts:
        predictor.set_image(image)
    batch_size = max(1, fast_batch) if fast else 1
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start+batch_size]
        count = max(len(points) for points, _, _ in batch)
        points = np.full((len(batch), count, 2), -10, dtype=np.float32)
        labels = np.full((len(batch), count), -1, dtype=np.int64)
        for i, (coords, kinds, _) in enumerate(batch):
            points[i, :len(coords)] = coords
            labels[i, :len(coords)] = kinds
        points = predictor.transform.apply_coords_torch(torch.as_tensor(points, device=device), (height, width))
        boxes = predictor.transform.apply_boxes_torch(
            torch.as_tensor(np.stack([p[2] for p in batch]), device=device), (height, width))
        with torch.inference_mode():
            _, _, logits = predictor.predict_torch(point_coords=points,
                point_labels=torch.as_tensor(labels, device=device), boxes=boxes,
                multimask_output=False, return_logits=True)
            logits = F.interpolate(logits, (height, width), mode='bilinear', align_corners=False)
            response = torch.sigmoid(logits).amax(dim=0)[0].cpu().numpy()
        probabilities = np.maximum(probabilities, response)
    probabilities[~valid] = 0
    smoothed = smooth_probabilities(probabilities, smooth_sigma, valid)
    return probabilities, smoothed, smoothed.copy(), chm, cost, crs, transform
