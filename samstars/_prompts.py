"""Shared prompt defaults and validation without scientific dependencies."""

from numbers import Integral

DEFAULT_FG_EXTRA = 80


def validate_fg_extra(value: int) -> int:
    """Return a nonnegative integer count, preserving zero as seed-only prompting."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("fg_extra must be a nonnegative integer (0 uses only the seed as foreground).")
    return int(value)


def sample_seed_prompts(chm, seed_xy, resolution, *, fg_extra=80, random_seed=0):
    """One deterministic prompt policy for training and simple prediction."""
    import numpy as np

    fg_extra = validate_fg_extra(fg_extra)
    chm = np.asarray(chm)
    height, width = chm.shape
    sx, sy = map(int, seed_xy)
    if not (0 <= sx < width and 0 <= sy < height and np.isfinite(chm[sy, sx])):
        raise ValueError('The foreground seed must lie on a valid CHM pixel.')
    half_box = 4.0 / resolution
    context = int(round(12.0 / resolution))
    box = np.array([max(0, sx-half_box), max(0, sy-half_box),
                    min(width-1, sx+half_box), min(height-1, sy+half_box)], dtype=np.float32)
    yy, xx = np.indices(chm.shape)
    inside = (xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])
    valid = np.isfinite(chm)
    foreground = valid & inside & (chm > 3)
    foreground[sy, sx] = False
    fy, fx = np.where(foreground)
    order = np.lexsort((fx, fy, -chm[fy, fx]))[:fg_extra]
    fg = np.column_stack((fx[order], fy[order])).astype(np.float32)
    nearby = (np.abs(xx-sx) < context) & (np.abs(yy-sy) < context)
    background = valid & nearby & (chm < 3)
    background[sy, sx] = False
    gy, gx = np.where(background)
    keep = min(50, len(gy))
    rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), sx, sy]))
    selected = rng.choice(len(gy), keep, replace=False) if keep else np.array([], dtype=int)
    ground = np.column_stack((gx[selected], gy[selected])).astype(np.float32)
    points = np.concatenate((np.array([[sx, sy]], dtype=np.float32), fg, ground))
    labels = np.r_[np.ones(1+len(fg), dtype=np.int64), np.zeros(len(ground), dtype=np.int64)]
    return points, labels, box
