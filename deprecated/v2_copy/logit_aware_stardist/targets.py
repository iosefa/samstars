from __future__ import annotations
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_erosion


def ray_directions(n_rays: int):
    """Ray directions."""
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _ray_distances_for_instance(inst_mask: np.ndarray, n_rays: int, erosion: int = 0):
    """Internal helper function."""
    H, W = inst_mask.shape
    if erosion > 0:
        inst_mask = binary_erosion(inst_mask, iterations=erosion)
    coords = np.argwhere(inst_mask)
    if coords.size == 0:
        return np.zeros((H, W, n_rays), np.float32), np.zeros((H, W), np.float32)

    dirs = ray_directions(n_rays)
    rays = np.zeros((H, W, n_rays), dtype=np.float32)
    obj = inst_mask.astype(np.float32)

    for y, x in coords:
        for r in range(n_rays):
            dx, dy = dirs[r]
            dist = 0.0
            while True:
                xx = x + dist * dx
                yy = y + dist * dy
                xi, yi = int(round(xx)), int(round(yy))
                if xi < 0 or xi >= W or yi < 0 or yi >= H or not inst_mask[yi, xi]:
                    break
                dist += 1.0
            rays[y, x, r] = dist
    return rays, obj


def compute_ray_targets(mask: np.ndarray, n_rays: int = 32, erosion: int = 0):
    """Compute ray targets."""
    labeled = label(mask > 0)
    H, W = mask.shape
    rays = np.zeros((H, W, n_rays), dtype=np.float32)
    obj = np.zeros((H, W), dtype=np.float32)
    for inst_id in np.unique(labeled):
        if inst_id == 0:
            continue
        inst_mask = labeled == inst_id
        r_inst, o_inst = _ray_distances_for_instance(inst_mask, n_rays, erosion=erosion)
        rays += r_inst
        obj = np.maximum(obj, o_inst)
    return rays, obj
