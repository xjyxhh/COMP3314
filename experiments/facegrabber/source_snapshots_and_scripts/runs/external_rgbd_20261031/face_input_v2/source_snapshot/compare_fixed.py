"""Compare solvers on saved, audited inputs without resampling or refitting PCA."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from .metrics import fit, distances
from .corrected import fit_corrected
from .evaluation import verification
from .run import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[100, 500, 1000])
    args = parser.parse_args()
    if any(b < 1 for b in args.budgets):
        parser.error('budgets must be positive')
    source, out = args.input, args.output
    with np.load(source/'features_pca.npz', allow_pickle=False) as data:
        xs = {s: data['X_'+s] for s in ('train', 'val', 'test')}
        z = data['Z_train']
    rounds = sorted(source.glob('round_*/pairs.npz'))
    if not rounds:
        raise ValueError('No saved pairs found')
    files = [source/name for name in ('manifest.csv', 'split.npz', 'pca.npz', 'features_pca.npz')]
    files += rounds + [p.parent/'xi0.npy' for p in rounds]
    hashes = {str(p.relative_to(source)): sha(p) for p in files}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'protocol.json', dict(input=str(source.resolve()), sha256=hashes,
        gamma=0.1, lam=1., budgets=args.budgets, tolerance=0.,
        selection='No parameter or budget selection; all predefined budgets reported.',
        scope='Washington diagnostic subset; basic features, not paper GLBP.',
        numpy_version=np.__version__))
    records = []
    for pair_path in rounds:
        folder = out/pair_path.parent.name
        folder.mkdir()
        with np.load(pair_path, allow_pickle=False) as data:
            pairs = {s: data[s] for s in xs}
        xi0 = np.load(pair_path.parent/'xi0.npy', allow_pickle=False)
        baseline = {s: verification(xs[s], pairs[s], np.eye(xs[s].shape[1])) for s in ('val', 'test')}
        save_json(folder/'euclidean.json', baseline)
        rgb_model = fit(xs['train'], pairs['train'], xi0, 0.1,
                        max_sweeps=max(args.budgets), tol=0.)
        save_json(folder/'itml.json', dict(
            history=rgb_model['history'], stop=rgb_model['stop'],
            validation=verification(xs['val'], pairs['val'], rgb_model['M']),
            test=verification(xs['test'], pairs['test'], rgb_model['M'])))
        np.savez(folder/'itml_model.npz', M=rgb_model['M'], xi=rgb_model['xi'])
        for method, solver in [('itml_plus', fit), ('itml_plus_corrected', fit_corrected)]:
            for budget in args.budgets:
                start = perf_counter()
                record = dict(round=pair_path.parent.name, method=method, budget=budget)
                try:
                    model = solver(xs['train'], pairs['train'], xi0, 0.1,
                                   Z=z, lam=1., max_sweeps=budget, tol=0.)
                    r = distances(xs['train'], pairs['train'], model['M'])
                    s = distances(z, pairs['train'], model['P'])
                    record.update(status='completed', stop=model['stop'],
                        sweeps=len(model['history']), final=model['history'][-1],
                        relative_violation=float(np.max(np.maximum(pairs['train'][:, 2]*(r-s), 0)/np.maximum(r, s))),
                        relative_equality=float(np.max(np.abs(model['xi']-s)/np.maximum(model['xi'], s))),
                        validation=verification(xs['val'], pairs['val'], model['M']),
                        test=verification(xs['test'], pairs['test'], model['M']))
                    np.savez(folder/f'{method}_{budget}.npz', **{k: model[k] for k in ('M', 'P', 'xi', 'dual', 'beta')})
                    save_json(folder/f'{method}_{budget}_history.json', model['history'])
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    record.update(status='failed', error=str(error))
                record['seconds'] = perf_counter()-start
                records.append(record)
                save_json(out/'results.json', records)
                print(record, flush=True)
    if any(sha(source/name) != digest for name, digest in hashes.items()):
        raise RuntimeError('Source inputs changed during comparison')
    save_json(out/'integrity.json', dict(source_unchanged=True, completed=len(records)))


if __name__ == '__main__':
    main()
