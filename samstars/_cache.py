"""Content-based identities for prepared inputs and reusable proposals."""
from __future__ import annotations

import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=256)
def _digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size,
            'sha256': _digest(str(path), stat.st_size, stat.st_mtime_ns)}


def proposal_identity(records, checkpoint, preprocessing, options):
    rows = json.loads(Path(records).read_text())
    paths = {Path(checkpoint).resolve()}
    for row in rows:
        image = Path(row['img_path']).resolve()
        paths.add(image)
        paths.update(image.parent / name for name in ('chm.tif', 'cost.tif', 'seeds.gpkg'))
    return {'version': 1, 'preprocessing': preprocessing, 'options': options,
            'records': rows, 'inputs': [file_identity(p) for p in sorted(paths)]}


def check_cache(root, identity, *, overwrite=False, stamp='identity.json'):
    root = Path(root)
    marker = root / stamp
    previous = json.loads(marker.read_text()) if marker.is_file() else None
    populated = root.exists() and any(p.name != stamp for p in root.iterdir())
    if populated and previous != identity:
        if not overwrite:
            raise ValueError('Cached outputs do not match the inputs, model, or processing settings. '
                             'Choose a new output folder or set skip_existing_sam=False.')
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(identity, sort_keys=True, indent=2))
    temporary.replace(marker)
