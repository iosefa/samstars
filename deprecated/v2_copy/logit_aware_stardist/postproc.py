from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon


def ray_directions(n_rays: int):
    """Ray directions."""
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def decode_rays(prob: np.ndarray, rays: np.ndarray, prob_thresh=0.5, nms_thresh=0.3, topk=200):
    """
    Decode StarDist rays into polygons and apply simple IoU-based NMS.
    """
    H, W, n_rays = rays.shape
    dirs = ray_directions(n_rays)
    coords = np.argwhere(prob > prob_thresh)
    scores = prob[prob > prob_thresh]
    order = scores.argsort()[::-1]
    coords = coords[order][:topk]
    scores = scores[order][:topk]

    polys = []
    for (y, x), s in zip(coords, scores):
        r = rays[y, x]
        pts = np.stack([x + r * dirs[:, 0], y + r * dirs[:, 1]], axis=1)
        poly = Polygon(pts)
        if poly.is_valid and poly.area > 0:
            polys.append((s, poly))

    keep = []
    for s, p in polys:
        drop = False
        for _, kp in keep:
            inter = p.intersection(kp).area
            union = p.union(kp).area
            iou = inter / union if union > 0 else 0
            if iou > nms_thresh:
                drop = True
                break
        if not drop:
            keep.append((s, p))
    return keep
