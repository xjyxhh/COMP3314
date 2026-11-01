"""Select a depth ablation by validation AP, then evaluate RGB-only test and shuffle control."""
import argparse
import json
from pathlib import Path
import numpy as np
from .evaluation import verification
from .metrics import fit
from .corrected import fit_corrected
from .run import save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', nargs=3, action='append', metavar=('NAME', 'INPUT', 'SEARCH'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    candidates = []
    for name, source, search in a.candidate:
        source, search = Path(source), Path(search)
        if not (search/'integrity.json').exists():
            raise ValueError('Search is incomplete: '+str(search))
        protocol = json.loads((search/'protocol.json').read_text())
        if any(sha(source/file) != digest for file, digest in protocol['sha256'].items()):
            raise ValueError('Source changed after search')
        candidates.append((name, source, search, json.loads((search/'results.json').read_text())))
    a.output.mkdir(parents=True, exist_ok=False)
    save_json(a.output/'protocol.json', dict(candidates=a.candidate,
        selection='per round and method, maximal validation AP; first on ties',
        shuffle='training depth rows permuted with seed 3329+round; same chosen hyperparameters and 100 sweeps; no shuffle retuning',
        limitation='exploratory reuse of previously inspected Washington test split; requires new holdout for confirmation'))
    results = []
    for rd in ('round_00', 'round_01'):
        for method in ('euclidean', 'itml', 'itml_plus', 'itml_plus_corrected'):
            choices = []
            for name, source, search, rows in candidates:
                matches = [r for r in rows if r['round']==rd and r['method']==method and r['status']=='completed']
                if matches:
                    choices.append((name, source, search, matches[0]))
            if not choices:
                raise ValueError('No successful model')
            name, source, search, selected = max(choices, key=lambda c: c[3]['selected']['validation']['AP'])
            with np.load(source/'features_pca.npz') as data:
                X, Z, test = data['X_train'], data['Z_train'], data['X_test']
            with np.load(source/rd/'pairs.npz') as data:
                train_pairs, test_pairs = data['train'], data['test']
            xi0 = np.load(source/rd/'xi0.npy')
            with np.load(search/rd/f'{method}_model.npz') as data:
                model = {k: data[k] for k in data.files}
            record = dict(round=rd, method=method, variant=name, selected=selected['selected'],
                test=verification(test, test_pairs, model['M']))
            np.savez(a.output/f'{rd}_{method}.npz', **model)
            if 'plus' in method:
                permutation = np.random.default_rng(3329+int(rd[-2:])).permutation(len(Z))
                np.save(a.output/f'{rd}_{method}_shuffle.npy', permutation)
                solver = fit_corrected if method.endswith('corrected') else fit
                config = selected['selected']
                try:
                    shuffled = solver(X, train_pairs, xi0, config['gamma'], Z=Z[permutation],
                                      lam=config['lam'], max_sweeps=100, tol=0.)
                    record['shuffle_control'] = dict(status='completed',
                        test=verification(test, test_pairs, shuffled['M']), final=shuffled['history'][-1])
                    np.savez(a.output/f'{rd}_{method}_shuffled.npz', **{k:v for k,v in shuffled.items() if isinstance(v,np.ndarray)})
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    record['shuffle_control'] = dict(status='failed', error=str(error))
            results.append(record)
            save_json(a.output/'results.json', results)
            print(record, flush=True)
    summary = {}
    for method in ('euclidean', 'itml', 'itml_plus', 'itml_plus_corrected'):
        rows = [r for r in results if r['method']==method]
        summary[method] = {m:float(np.mean([r['test'][m] for r in rows])) for m in ('AP','AUC')}
    save_json(a.output/'summary.json', summary)


if __name__ == '__main__':
    main()
