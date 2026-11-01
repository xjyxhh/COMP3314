"""Recheck saved ORL experiments without training or selecting new models.

Run from itml_reproduction with /opt/miniconda3/bin/python <this file>.
AP and AUC below are computed independently of itml_repro.evaluation.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('runs/olivetti_clean_privileged_20261031')
records = json.loads((root/'results.json').read_text())


def scores(X, pairs, M):
    delta = X[pairs[:, 0]]-X[pairs[:, 1]]
    distance = np.array([v @ M @ v for v in delta])
    positive = pairs[:, 2] == 1
    # Descending confidence = ascending distance; threshold ties as a group.
    ap, previous = 0., 0
    for threshold in np.unique(distance):
        retrieved = distance <= threshold
        tp = int(np.sum(positive & retrieved))
        ap += (tp-previous)/positive.sum() * tp/retrieved.sum()
        previous = tp
    pos, neg = distance[positive], distance[~positive]
    auc = np.mean((pos[:, None] < neg).astype(float) + .5*(pos[:, None] == neg))
    return dict(AP=float(ap), AUC=float(auc))


def quad(X, pairs, M):
    v = X[pairs[:, 0]]-X[pairs[:, 1]]
    return np.array([t @ M @ t for t in v])


checks, recomputed, convergence = [], [], []
for split in range(3):
    inputs = root/f'split_{split:02d}'/'input'
    search = root/f'split_{split:02d}'/'search'
    manifest = list(csv.DictReader((inputs/'manifest.csv').open()))
    with np.load(inputs/'split.npz') as groups:
        ids = [{manifest[i]['identity'] for i in groups[s]} for s in ('train', 'val', 'test')]
        assert all(not ids[a] & ids[b] for a,b in ((0,1),(0,2),(1,2)))
        assert len(set.union(*ids)) == 40
        with np.load(root/'image_features.npz') as raw, np.load(inputs/'pca.npz') as pca:
            train_idx = groups['train']
            for modality in ('X', 'Z'):
                assert np.allclose(pca[modality+'_mean'], raw[modality][train_idx].mean(0))
                # Check projection subspace against training-only covariance.
                centered = raw[modality][train_idx]-pca[modality+'_mean']
                components = pca[modality+'_components']
                basis = components/np.linalg.norm(components, axis=1, keepdims=True)
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                assert np.allclose(basis @ vt[:16].T @ vt[:16], basis, atol=1e-9)
        with np.load(inputs/'round_00/pairs.npz') as pp:
            for name in ('train','val','test'):
                p = pp[name]
                labels = np.array([manifest[i]['identity'] for i in groups[name]])
                assert len(p) == 270 and np.sum(p[:,2]==1) == 90
                assert np.all(p[:,0]<p[:,1]) and len(np.unique(p[:,:2],axis=0)) == 270
                assert np.array_equal(labels[p[:,0]]==labels[p[:,1]], p[:,2]==1)
    protocol = json.loads((search/'protocol.json').read_text())
    assert protocol['test_deferred']
    assert all(hashlib.sha256((inputs/n).read_bytes()).hexdigest()==h for n,h in protocol['sha256'].items())
    with np.load(inputs/'features_pca.npz') as features, np.load(inputs/'round_00/pairs.npz') as pp:
        X, Z, p = features['X_train'], features['Z_train'], pp['train']
        for record in [r for r in records if r['split']==split]:
            method = record['method']
            candidates = json.loads((search/'round_00'/f'{method}_search.json').read_text())
            chosen = max((c for c in candidates if c['status']=='completed'), key=lambda c:c['validation']['AP'])
            assert (chosen['gamma'], chosen['lam']) == (record['selected']['gamma'], record['selected']['lam'])
            with np.load(search/'round_00'/f'{method}_model.npz') as model:
                result = scores(features['X_test'], pp['test'], model['M'])
                assert all(abs(result[m]-record['test'][m]) < 1e-12 for m in result)
            recomputed.append(dict(split=split, method=method, **result))
            if 'plus' in method:
                with np.load(root/f'split_{split:02d}'/f'{method}_shuffled.npz') as model:
                    control = scores(features['X_test'], pp['test'], model['M'])
                    assert all(abs(control[m]-record['shuffle']['test'][m])<1e-12 for m in control)
                checkpoint = Path('runs/olivetti_steps12_20261031')/f'split_{split:02d}'/f'{method}_1000.npz'
                with np.load(checkpoint) as model:
                    r, s = quad(X,p,model['M']), quad(Z,p,model['P'])
                    row = dict(split=split,method=method,
                        relative_violation=float(np.max(np.maximum(p[:,2]*(r-s),0)/np.maximum(r,s))),
                        relative_equality=float(np.max(np.abs(model['xi']-s)/np.maximum(model['xi'],s))))
                    if method.endswith('corrected'):
                        gamma, lam = chosen['gamma'], chosen['lam']
                        xi0 = np.load(inputs/'round_00/xi0.npy')
                        v,w = X[p[:,0]]-X[p[:,1]], Z[p[:,0]]-Z[p[:,1]]
                        a,b = model['dual'],model['beta']
                        left = [np.linalg.inv(model['M']),lam*np.linalg.inv(model['P']),gamma/model['xi']]
                        right = [np.eye(X.shape[1])+sum(y*t*np.outer(q,q) for y,t,q in zip(p[:,2],a,v)),
                            lam*np.eye(Z.shape[1])-sum(t*np.outer(q,q) for t,q in zip(b,w)), gamma/xi0-p[:,2]*a+b]
                        row['stationarity_max'] = float(max(np.linalg.norm(l-r)/max(1.,np.linalg.norm(l),np.linalg.norm(r)) for l,r in zip(left,right)))
                        row['complementarity'] = float(np.max(np.abs(a*p[:,2]*(r-s))))
                    convergence.append(row)
    checks.append(dict(split=split,identity_disjoint=True,training_only_pca=True,pairs_valid=True,
        input_hashes_match=True,validation_selection_matches=True,test_and_shuffle_metrics_match=True))

summary = {method: {m:float(np.mean([r[m] for r in recomputed if r['method']==method])) for m in ('AP','AUC')}
           for method in sorted({r['method'] for r in recomputed})}
output = dict(checks=checks,mean_test_scores=summary,checkpoints_1000=convergence)
Path(__file__).with_name('artifact_recheck.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
