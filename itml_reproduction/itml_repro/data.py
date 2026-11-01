"""One manifest, one identity split, and shared persisted pair indices."""
import csv
import importlib
import itertools
from pathlib import Path
import numpy as np


def read_manifest(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'sample_id', 'identity', 'rgb_path', 'depth_path'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest requires {sorted(required)}')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample_id')
    if any(not r['identity'] or not r['sample_id'] for r in rows):
        raise ValueError('Empty sample_id or identity')
    return rows


def split_identities(rows, counts, seed):
    ids = np.array(sorted({r['identity'] for r in rows}))
    if sum(counts) != len(ids) or min(counts) < 1:
        raise ValueError('Split counts must exhaust identities with no overlap')
    ids = np.random.default_rng(seed).permutation(ids)
    groups = np.split(ids, np.cumsum(counts)[:-1])
    return {name: np.array([i for i,r in enumerate(rows) if r['identity'] in group], dtype=int)
            for name, group in zip(('train','val','test'), groups)}


def make_pairs(labels, positive, negative, seed, positive_seed=None):
    """Uniform sampling WITHOUT replacement, canonical unordered indices.

    Enumerates O(n^2) candidate pairs; adequate for the paper's face datasets.
    """
    rng = np.random.default_rng(seed)
    candidates = [[], []]
    for i,j in itertools.combinations(range(len(labels)), 2):
        candidates[int(labels[i] != labels[j])].append((i,j))
    output = []
    for c,n,y in zip(candidates, (positive, negative), (1,-1)):
        if n < 1 or n > len(c):
            raise ValueError(f'Requested {n} pairs for label {y}, available {len(c)}')
        sampler = np.random.default_rng(positive_seed) if y == 1 and positive_seed is not None else rng
        chosen = np.arange(len(c)) if n == len(c) else sampler.choice(len(c), n, replace=False)
        output.extend((*c[k], y) for k in chosen)
    return np.array(output, dtype=np.int64)[rng.permutation(len(output))]


def load_features(rows, manifest, archive=None, extractor=None):
    """Archive: sample_ids, X, Z (NaN allowed ONLY for unused val/test depth).

    Or --extractor module:function(path, modality)->1D vector. This is an
    explicit interface, not an unverified GLBP implementation.
    """
    if bool(archive) == bool(extractor):
        raise ValueError('Supply exactly one of features archive / extractor')
    if archive:
        with np.load(archive, allow_pickle=False) as f:
            ids = f['sample_ids'].astype(str)
            if len(set(ids)) != len(ids) or set(ids) != {r['sample_id'] for r in rows}:
                raise ValueError('Feature sample IDs must exactly match manifest')
            order = {s:i for i,s in enumerate(ids)}
            indices = [order[r['sample_id']] for r in rows]
            X, Z = f['X'][indices], f['Z'][indices]
    else:
        mod, func = extractor.split(':')
        extract = getattr(importlib.import_module(mod), func)
        root = Path(manifest).resolve().parent
        X = np.stack([extract(root/r['rgb_path'], 'rgb') for r in rows])
        depth = [None if not r['depth_path'] else extract(root/r['depth_path'], 'depth') for r in rows]
        shape = next((np.asarray(z).shape for z in depth if z is not None), None)
        if shape is None:
            raise ValueError('Training depth features are required')
        Z = np.stack([np.full(shape, np.nan) if z is None else z for z in depth])
    X, Z = np.asarray(X, dtype=float), np.asarray(Z, dtype=float)
    if X.ndim != 2 or Z.ndim != 2 or len(X) != len(rows) or len(Z) != len(rows) or not np.isfinite(X).all():
        raise ValueError('Invalid feature shapes or RGB values')
    return X, Z
