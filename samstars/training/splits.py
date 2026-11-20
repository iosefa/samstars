"""Split spatially disjoint tile groups before fitting either neural model."""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path


def split_training_tiles(tiles_dir, out_dir, *, val_fraction=.2, seed=0, validation_tiles_dir=None):
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box

    if not 0 <= val_fraction < 1:
        raise ValueError('val_fraction must be in [0, 1).')
    tiles = sorted(p for p in Path(tiles_dir).iterdir() if p.is_dir() and (p/'img.tif').is_file())
    if not tiles:
        raise ValueError('No prepared training tiles found.')
    explicit = [] if validation_tiles_dir is None else sorted(
        p for p in Path(validation_tiles_dir).iterdir() if p.is_dir() and (p/'img.tif').is_file())
    if validation_tiles_dir is not None and not explicit:
        raise ValueError('No prepared validation tiles found.')
    reference_crs = None
    extents = {}
    for tile in tiles + explicit:
        with rasterio.open(tile/'img.tif') as src:
            reference_crs = reference_crs or src.crs
            geometry = box(*src.bounds)
            if src.crs != reference_crs:
                geometry = gpd.GeoSeries([geometry], crs=src.crs).to_crs(reference_crs).iloc[0]
            extents[tile] = geometry
    if explicit:
        if any(extents[a].intersection(extents[b]).area > 0 for a in tiles for b in explicit):
            raise ValueError('Training and validation raster footprints overlap. Supply disjoint areas.')
        training, validation = tiles, explicit
    else:
        groups = []
        remaining = set(tiles)
        while remaining:
            group = {min(remaining)}
            remaining -= group
            while True:
                touching = {p for p in remaining if any(extents[p].intersection(extents[q]).area > 0 for q in group)}
                if not touching:
                    break
                group |= touching
                remaining -= touching
            groups.append(sorted(group))
        random.Random(seed).shuffle(groups)
        n_val = min(len(groups)-1, max(1, round(len(groups)*val_fraction))) if val_fraction > 0 and len(groups)>1 else 0
        validation = [p for group in groups[:n_val] for p in group]
        training = [p for group in groups[n_val:] for p in group]
        if val_fraction > 0 and not validation:
            warnings.warn('Only one spatial group is available; training uses all labels with no held-out validation. '
                          'Supply validation_tiles_dir for an independent evaluation.', UserWarning, stacklevel=2)
    root = Path(out_dir)
    for label, paths in (('train', training), ('validation', validation)):
        folder = root/label
        folder.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(paths):
            destination = folder/path.name
            destination.symlink_to(path.resolve(), target_is_directory=True)
    summary = {'seed': seed, 'validation': 'held_out' if validation else 'none',
               'training_tiles': [str(p.resolve()) for p in training],
               'validation_tiles': [str(p.resolve()) for p in validation]}
    (root/'split.json').write_text(json.dumps(summary, indent=2))
    return root/'train', (root/'validation' if validation else None), summary
