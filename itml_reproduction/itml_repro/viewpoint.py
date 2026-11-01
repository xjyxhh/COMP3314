"""Validation-selected privileged-information experiment on fixed audited inputs."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from .metrics import fit
from .corrected import fit_corrected
from .evaluation import verification
from .run import save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--defer-test', action='store_true', help='Save selected models without evaluating test pairs')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with np.load(a.input/'features_pca.npz', allow_pickle=False) as data:
        xs = {s: data['X_'+s] for s in ('train', 'val', 'test')}
        z = data['Z_train']
    files = [a.input/'features_pca.npz', a.input/'split.npz']
    files += sorted(a.input.glob('round_*/*np*'))
    hashes = {str(f.relative_to(a.input)): sha(f) for f in files}
    save_json(a.output/'protocol.json', dict(input=str(a.input.resolve()), sha256=hashes,
        gamma_grid=[0.01, 0.1, 1.], lambda_grid=[0.1, 1., 10.], max_sweeps=100,
        selection='validation AP, first candidate on ties; test only selected model',
        test_deferred=a.defer_test,
        scope='Privileged-information diagnostic; see input feature_definition.json; not original paper benchmark'))
    results = []
    for pair_file in sorted(a.input.glob('round_*/pairs.npz')):
        folder = a.output/pair_file.parent.name
        folder.mkdir()
        with np.load(pair_file, allow_pickle=False) as data:
            pairs = {s: data[s] for s in xs}
        xi0 = np.load(pair_file.parent/'xi0.npy', allow_pickle=False)
        for method in ('euclidean', 'itml', 'itml_plus', 'itml_plus_corrected'):
            search, best = [], None
            for gamma in ([None] if method == 'euclidean' else [0.01, 0.1, 1.]):
                for lam in ([0.1, 1., 10.] if 'plus' in method else [1.]):
                    start = perf_counter()
                    row = dict(gamma=gamma, lam=lam)
                    try:
                        solver = fit_corrected if method.endswith('corrected') else fit
                        model = dict(M=np.eye(xs['train'].shape[1]), history=[], stop='fixed_metric') if gamma is None else solver(
                            xs['train'], pairs['train'], xi0, gamma,
                            Z=z if 'plus' in method else None, lam=lam, max_sweeps=100, tol=0.)
                        row.update(status='completed', validation=verification(xs['val'], pairs['val'], model['M']),
                                   final=model['history'][-1] if model['history'] else {}, stop=model['stop'])
                        if best is None or row['validation']['AP'] > best[0]:
                            best = (row['validation']['AP'], model, row)
                    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        row.update(status='failed', error=str(error))
                    row['seconds'] = perf_counter()-start
                    search.append(row)
                    save_json(folder/f'{method}_search.json', search)
            if best is None:
                record = dict(round=folder.name, method=method, status='all_candidates_failed')
            else:
                _, model, selected = best
                record = dict(round=folder.name, method=method, status='completed', selected=selected)
                if not a.defer_test:
                    record['test'] = verification(xs['test'], pairs['test'], model['M'])
                np.savez(folder/f'{method}_model.npz', **{k: v for k, v in model.items() if isinstance(v, np.ndarray)})
            results.append(record)
            save_json(a.output/'results.json', results)
            print(record, flush=True)
    assert all(sha(a.input/name) == digest for name, digest in hashes.items())
    save_json(a.output/'integrity.json', dict(source_unchanged=True))


if __name__ == '__main__':
    main()
