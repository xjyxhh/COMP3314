"""python -m itml_repro.run --config configs/smoke.json ..."""
import argparse
import csv
from time import perf_counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import read_manifest, split_identities, make_pairs, load_features
from .pca import fit_pca, transform
from .metrics import fit, distances
from .evaluation import verification


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite experiment artifacts
    rows = read_manifest(args.manifest)
    X, Z = load_features(rows, args.manifest, args.features, args.extractor)
    split = split_identities(rows, config['split_counts'], config['seed'])
    for name, idx in split.items():
        with (out/f'{name}_manifest.csv').open('w', newline='') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows([rows[i] for i in idx])
    np.savez(out/'split.npz', **split)
    save_json(out/'split.json', {k:[rows[i]['sample_id'] for i in v] for k,v in split.items()})
    models = [fit_pca(A[split['train']],config['pca_dimension']) for A in (X,Z)]
    xs = {k:transform(X[v],models[0]) for k,v in split.items()}
    ztrain = transform(Z[split['train']],models[1])  # no val/test depth use
    np.savez(out/'pca.npz', rgb_mean=models[0][0], rgb_components=models[0][1],
             depth_mean=models[1][0], depth_components=models[1][1])
    np.savez(out/'features_pca.npz', X_train=xs['train'], X_val=xs['val'], X_test=xs['test'], Z_train=ztrain)
    # Exact all-pair training percentiles; L2 or squared distances is explicit.
    train = xs['train']
    i,j = np.triu_indices(len(train),k=1)
    ds = np.sum((train[i]-train[j])**2,axis=1)
    if config['bound_distance'] == 'euclidean':
        ds = np.sqrt(ds)
    elif config['bound_distance'] != 'squared':
        raise ValueError('bound_distance must be euclidean or squared')
    low,high = np.percentile(ds,[3,97])
    if config['bound_assignment'] == 'constraint_consistent':
        u,l = low,high
    elif config['bound_assignment'] == 'literal_v_b3':
        l,u = low,high
    else:
        raise ValueError('Unknown bound_assignment')
    if min(u,l) <= 0:
        raise ValueError('Nonpositive percentile bound')
    save_json(out/'bounds.json', dict(u=float(u),l=float(l),distance=config['bound_distance'], assignment=config['bound_assignment']))
    save_json(out/'config.json', config)
    save_json(out/'provenance.json', dict(manifest_sha256=sha(args.manifest),
        features_sha256=sha(args.features) if args.features else None,extractor=args.extractor,
        numpy_version=np.__version__, sample_ids=[r['sample_id'] for r in rows]))
    methods=config.get('methods', ['euclidean','itml','itml_plus','itml_plus_corrected'])
    if not methods or len(set(methods)) != len(methods) or set(methods)-{'euclidean','itml','itml_plus','itml_plus_corrected'}:
        raise ValueError('Unknown or duplicate methods')
    if 'itml_plus_corrected' in methods:
        from .corrected import fit_corrected
    results=[]
    for round_id in range(config['rounds']):
        folder=out/f'round_{round_id:02d}'; folder.mkdir()
        pairs={}
        for offset,(name,idx) in enumerate(split.items()):
            labels=[rows[i]['identity'] for i in idx]
            positive,negative=config['pair_counts'][name]
            pairs[name]=make_pairs(labels,positive,negative,config['seed']+1000*round_id+offset+1,
                                   positive_seed=config['seed']+offset+100)
        np.savez(folder/'pairs.npz', **pairs)
        xi0=np.where(pairs['train'][:,2]==1,u,l)
        np.save(folder/'xi0.npy',xi0)
        for method in methods:
            best=None; search=[]
            method_start=perf_counter()
            is_plus=method in ('itml_plus','itml_plus_corrected')
            gammas=[None] if method=='euclidean' else config['gamma_grid']
            lambdas=config['lambda_grid'] if is_plus else [None]
            for gamma in gammas:
                for lam in lambdas:
                    fit_start=perf_counter()
                    solver=fit_corrected if method=='itml_plus_corrected' else fit
                    model=(dict(M=np.eye(train.shape[1]),P=None,history=[],stop='fixed_metric')
                           if method=='euclidean' else solver(train,pairs['train'],xi0,gamma,
                               Z=ztrain if is_plus else None,
                               lam=lam if lam is not None else 1.,
                               max_sweeps=config['max_sweeps'],tol=config['tolerance']))
                    fit_seconds=perf_counter()-fit_start
                    diagnostics={}
                    if is_plus:
                        r=distances(train,pairs['train'],model['M'])
                        t=distances(ztrain,pairs['train'],model['P'])
                        diagnostics=dict(relative_inequality_violation=float(np.max(
                            np.maximum(pairs['train'][:,2]*(r-t),0)/np.maximum(r,t))),
                            relative_equality_residual=float(np.max(np.abs(model['xi']-t)/np.maximum(model['xi'],t))))
                        diagnostics['residuals_within_tolerance']=max(diagnostics.values()) < config['tolerance']
                    val=verification(xs['val'],pairs['val'],model['M'])
                    candidate=dict(gamma=gamma,lam=lam,validation=val,fit_seconds=fit_seconds,
                                   sweeps=len(model['history']),stop=model['stop'],diagnostics=diagnostics)
                    search.append(candidate)
                    if best is None or val[config['selection_metric']]>best[0]:
                        best=(val[config['selection_metric']],model,candidate)
            _,model,selected=best
            test=verification(xs['test'],pairs['test'],model['M'])
            arrays=dict(M=model['M'])
            if method!='euclidean':
                arrays.update(xi=model['xi'],dual=model['dual'],beta=model['beta'])
            if model['P'] is not None:
                arrays['P']=model['P']
            np.savez(folder/f'{method}_model.npz',**arrays)
            record=dict(round=round_id,method=method,selected=selected,test=test,stop=model['stop'],
                        method_seconds=perf_counter()-method_start)
            results.append(record)
            save_json(folder/f'{method}.json',dict(result=record,search=search,history=model['history']))
            print(f'round {round_id} {method}: {test}', flush=True)
    summary={}
    for method in methods:
        records=[r for r in results if r['method']==method]
        summary[method]={metric:dict(mean=float(np.mean([r['test'][metric] for r in records])),
                                    std=float(np.std([r['test'][metric] for r in records],ddof=config['std_ddof'])))
                         for metric in ('AP','AUC')}
    with (out/'comparison.csv').open('w',newline='') as f:
        fields=['round','method','gamma','lam','AP','AUC','stop','sweeps','fit_seconds','method_seconds',
                'relative_inequality_violation','relative_equality_residual','residuals_within_tolerance']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in results:
            selected=r['selected']
            writer.writerow(dict(round=r['round'],method=r['method'],gamma=selected['gamma'],lam=selected['lam'],
                **r['test'],stop=r['stop'],sweeps=selected['sweeps'],fit_seconds=selected['fit_seconds'],
                method_seconds=r['method_seconds'],**selected['diagnostics']))
    save_json(out/'summary.json',dict(summary=summary,rounds=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--manifest',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--features')
    group.add_argument('--extractor')
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
