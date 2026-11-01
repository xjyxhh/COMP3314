"""Section IV-E / V-D1: partial ITML+ on fixed inputs, validation only.

No test features or test pairs are loaded. Original full ITML+ and ordinary
ITML updates are reused; this is not the cumulative corrected solver.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .metrics import fit
from .evaluation import verification
from .run import save_json, sha


def privileged_prefix(pairs, fraction):
    """V-D1: first fraction of positive AND negative pairs, in saved order.

    Floor each label count when the percentage is not integral. Masks are
    nested; do not reshuffle or drop constraints as the percentage changes.
    """
    pairs = np.asarray(pairs)
    if (not np.isfinite(fraction) or not 0 <= fraction <= 1 or pairs.ndim != 2
            or pairs.shape[1] != 3 or not np.isin(pairs[:, 2], [-1, 1]).all()):
        raise ValueError('Expected valid pairs and a fraction in [0,1]')
    mask = np.zeros(len(pairs), dtype=bool)
    for label in (1, -1):
        indices = np.flatnonzero(pairs[:, 2] == label)
        mask[indices[:int(np.floor(fraction*len(indices)))]] = True
    return mask


def run(args):
    config = json.loads(args.config.read_text())
    fractions = args.fractions
    # Validate before creating output; preserve original pair order at all ratios.
    for fraction in fractions:
        privileged_prefix(np.array([[0, 1, 1], [0, 2, -1]]), fraction)
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError('Fractions must be nonempty and unique')
    pair_files = sorted(args.input.glob('round_*/pairs.npz'))
    if not pair_files:
        raise ValueError('No saved round_*/pairs.npz inputs')
    with np.load(args.input/'features_pca.npz', allow_pickle=False) as data:
        X, Z, validation = data['X_train'], data['Z_train'], data['X_val']
    files = [args.input/'features_pca.npz']
    files += [f for p in pair_files for f in (p, p.parent/'xi0.npy')]
    hashes = {str(f.relative_to(args.input)): sha(f) for f in files}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'protocol.json', dict(
        input=str(args.input.resolve()), config=config, fractions=fractions,
        input_sha256=hashes, config_sha256=sha(args.config),
        code_sha256={n: sha(Path(__file__).parent/n) for n in ('metrics.py', 'partial.py', 'evaluation.py')},
        scope='Partial ITML+ formula/protocol extension on supplied fixed features; not original benchmark',
        mask='first floor(fraction*N) pairs within each label, saved order, nested masks',
        selection='validation AP only; first candidate on ties; no test evaluation',
        zero_percent='ordinary ITML; lambda irrelevant, P implicitly I',
        stopping='norm check per sweep, original implemented convention; budget from config',
        test_evaluated=False))
    results = []
    for pair_file in pair_files:
        with np.load(pair_file, allow_pickle=False) as data:
            pairs, val_pairs = data['train'], data['val']
        xi0 = np.load(pair_file.parent/'xi0.npy', allow_pickle=False)
        for index, fraction in enumerate(fractions):
            folder = args.output/pair_file.parent.name/f'ratio_{index:02d}'
            folder.mkdir(parents=True)
            mask = privileged_prefix(pairs, fraction)
            np.save(folder/'privileged_mask.npy', mask)
            search, best = [], None
            for gamma in config['gamma_grid']:
                for lam in (config['lambda_grid'] if mask.any() else [1.]):
                    start = perf_counter()
                    row = dict(gamma=gamma, lam=lam)
                    try:
                        model = fit(X, pairs, xi0, gamma, Z=Z, lam=lam,
                            max_sweeps=config['max_sweeps'], tol=config['tolerance'], privileged_mask=mask)
                        row.update(status='completed', validation=verification(validation, val_pairs, model['M']),
                            final=model['history'][-1], stop=model['stop'])
                        if best is None or row['validation']['AP'] > best[0]:
                            best = (row['validation']['AP'], model, row)
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                        row.update(status='failed', error=str(error))
                    row['seconds'] = perf_counter()-start
                    search.append(row)
            save_json(folder/'search.json', search)
            record = dict(round=pair_file.parent.name, fraction=fraction,
                privileged_positive=int(np.sum(mask & (pairs[:, 2] == 1))),
                privileged_negative=int(np.sum(mask & (pairs[:, 2] == -1))),
                total_pairs=len(pairs), failed_candidates=sum(r['status']=='failed' for r in search))
            if best is None:
                record['status'] = 'all_candidates_failed'
            else:
                _, model, selected = best
                record.update(status='completed', selected=selected)
                np.savez(folder/'model.npz', **{k: v for k,v in model.items() if isinstance(v, np.ndarray)})
                save_json(folder/'history.json', model['history'])
            results.append(record)
            save_json(args.output/'results.json', results)
            print(record['round'], fraction, record['status'], flush=True)
    unchanged = all(sha(args.input/n) == digest for n,digest in hashes.items())
    save_json(args.output/'integrity.json', dict(source_unchanged=unchanged, test_evaluated=False))
    if not unchanged:
        raise RuntimeError('Input changed during experiment')
    if any(r['status'] != 'completed' for r in results):
        raise RuntimeError('Some ratios have no completed candidate; see results.json')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/eurecom.json')
    parser.add_argument('--fractions', type=float, nargs='+', default=[0., .25, .5, .75, 1.])
    run(parser.parse_args())
