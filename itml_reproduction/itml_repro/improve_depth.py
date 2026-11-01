"""Predefined depth-only ablations; preserve RGB, identities and pair indices."""
import argparse
import csv
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .data import read_manifest
from .pca import fit_pca, transform
from .run import save_json, sha
from .audit_washington import feasibility
from .metrics import distances


def augmented_witness(X, Z, pairs, certificate, factor, folder):
    with np.load(certificate) as data:
        M, old_p = data['M'], data['P']
    P = np.zeros((Z.shape[1], Z.shape[1]))
    P[:8, :8] = old_p/factor**2
    c = pairs[:, 2]*(distances(X, pairs, M)-distances(Z, pairs, P))
    extra = np.sum((Z[pairs[:, 0], 8:]-Z[pairs[:, 1], 8:])**2, axis=1)
    negative = (pairs[:, 2] == -1) & (extra > 0)
    if c.max() >= 0:
        raise ValueError('Original witness is not strictly feasible')
    epsilon = min(1., float(np.min(-c[negative]/extra[negative]))/4) if negative.any() else 1.
    P[8:, 8:] = epsilon*np.eye(Z.shape[1]-8)
    normalizer = np.trace(M)+np.trace(P)
    M, P = M/normalizer, P/normalizer
    violation = float(np.max(pairs[:, 2]*(distances(X, pairs, M)-distances(Z, pairs, P))))
    eigenvalue = float(min(np.linalg.eigvalsh(M).min(), np.linalg.eigvalsh(P).min()))
    if violation >= 0 or eigenvalue <= 0:
        raise ValueError('Augmented witness failed independent check')
    np.savez(folder/'feasibility_certificate.npz', M=M, P=P)
    return dict(classification='numerically_spd_feasible', method='embedded original strict witness',
                max_original_inequality=violation, min_eigenvalue=eigenvalue)


def depth_shape(depth, mask):
    if depth.dtype != np.uint16 or depth.shape != mask.shape:
        raise ValueError('Expected aligned uint16 depth and mask')
    valid = mask & (depth > 0)
    if not valid.any():
        raise ValueError('No valid object depth')
    # Fixed per-image robust scale retains local shape, removes camera distance.
    center = np.median(depth[valid])
    scale = max(float(np.quantile(depth[valid], .9)-np.quantile(depth[valid], .1)), 1.)
    relative = (depth.astype(float)-center)/scale
    features = []
    for yy in np.array_split(np.arange(mask.shape[0]), 4):
        for xx in np.array_split(np.arange(mask.shape[1]), 4):
            ix = np.ix_(yy, xx)
            v = relative[ix][valid[ix]]
            features.extend([float(v.mean()), float(v.std()), *np.quantile(v, [.1, .5, .9])]
                            if len(v) else [0.]*5)
            features.extend([float(mask[ix].mean()), float(valid[ix].mean())])
    features.extend([mask.shape[1]/mask.shape[0], float(mask.mean()), float(valid.sum()/mask.sum())])
    return np.asarray(features)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = read_manifest(a.input/'manifest.csv')
    with np.load(a.input/'features_pca.npz', allow_pickle=False) as data:
        saved = {k: data[k] for k in data.files}
    with np.load(a.input/'split.npz', allow_pickle=False) as data:
        splits = {k: data[k] for k in data.files}
    with np.load(a.input/'pca.npz', allow_pickle=False) as data:
        original_pca = {k: data[k] for k in data.files}
    raw = []
    for row in rows:
        with Image.open(a.input/row['depth_path']) as im:
            depth = np.asarray(im)
        with Image.open(a.input/row['mask_path']) as im:
            mask = np.asarray(im, dtype=bool)
        raw.append(depth_shape(depth, mask))
    raw = np.asarray(raw)
    mean, components = fit_pca(raw[splits['train']], 8)
    projected = transform(raw, (mean, components))
    a.output.mkdir(parents=True, exist_ok=False)
    for name in ('scale_only', 'shape', 'shape_augmented'):
        out = a.output/name
        out.mkdir()
        for file in ('split.npz',):
            shutil.copyfile(a.input/file, out/file)
        updated = []
        for row in rows:
            row = dict(row)
            for k in ('rgb_path', 'depth_path', 'mask_path', 'loc_path'):
                row[k] = str((a.input/row[k]).resolve())
            updated.append(row)
        with (out/'manifest.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(updated[0]))
            writer.writeheader(); writer.writerows(updated)
        arrays = {k: v for k, v in saved.items() if k.startswith('X_')}
        zz = ({s: saved['Z_'+s] for s in splits} if name == 'scale_only'
              else {s: projected[idx] for s, idx in splits.items()})
        shape_factor = 1.
        if name == 'shape_augmented':
            shape_factor = float(np.sqrt(np.sum(saved['Z_train']**2)/np.sum(zz['train']**2)))
            zz = {s: np.column_stack((saved['Z_'+s], values*shape_factor)) for s, values in zz.items()}
        # Fit one scalar on training only; no per-test fitting or label use.
        factor = float(np.sqrt(np.sum(arrays['X_train']**2)/np.sum(zz['train']**2)))
        arrays.update({'Z_'+s: z*factor for s, z in zz.items()})
        np.savez(out/'features_pca.npz', **arrays)
        zmean = original_pca['Z_mean'] if name=='scale_only' else mean
        zcomponents = original_pca['Z_components'] if name=='scale_only' else components
        if name == 'shape_augmented':
            zmean = np.r_[original_pca['Z_mean'], mean]
            zcomponents = np.zeros((16, len(zmean)))
            zcomponents[:8, :len(original_pca['Z_mean'])] = original_pca['Z_components']
            zcomponents[8:, len(original_pca['Z_mean']):] = components*shape_factor
        np.savez(out/'pca.npz', X_mean=original_pca['X_mean'], X_components=original_pca['X_components'],
                 Z_mean=zmean, Z_components=zcomponents*factor)
        if name != 'scale_only':
            np.savez(out/'depth_pca.npz', mean=mean, components=components)
            np.savez(out/'depth_features.npz', sample_ids=np.array([r['sample_id'] for r in rows]), Z=raw)
        save_json(out/'feature_definition.json', dict(variant=name, training_rms_scale=factor,
            rgb='exactly unchanged from source', source=str(a.input.resolve()),
            source_features_sha256=sha(a.input/'features_pca.npz'),
            shape_block_scale=shape_factor,
            depth=('original features with training RMS scale matching' if name=='scale_only' else
            '4x4 robust relative-depth mean/std/q10/q50/q90, mask and valid occupancy; aspect/fill/coverage; 115 raw dimensions; train-only PCA 8D'),
            augmentation='original 8D plus shape 8D, each block same training RMS' if name=='shape_augmented' else None,
            missing='zero depth excluded; empty cell statistics zero; no sample removed',
            not_glbp=True))
        checks = []
        for pair_file in sorted(a.input.glob('round_*/pairs.npz')):
            folder = out/pair_file.parent.name
            folder.mkdir()
            for file in ('pairs.npz', 'xi0.npy'):
                shutil.copyfile(pair_file.parent/file, folder/file)
            with np.load(folder/'pairs.npz') as data:
                pairs = data['train']
            if name == 'shape_augmented':
                check = augmented_witness(arrays['X_train'], arrays['Z_train'], pairs,
                    pair_file.parent/'feasibility_certificate.npz', factor, folder)
            else:
                check = feasibility(arrays['X_train'], arrays['Z_train'], pairs, folder)
            checks.append(dict(round=folder.name, result=check))
        save_json(out/'feasibility.json', checks)
        print(name, checks, flush=True)


if __name__ == '__main__':
    main()
