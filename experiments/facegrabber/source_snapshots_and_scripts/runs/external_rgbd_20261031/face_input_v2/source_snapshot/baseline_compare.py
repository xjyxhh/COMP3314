"""All seven Xu methods on fixed inputs: validation selection, no test reads."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import cvxpy as cp
from .baselines import fit_svm, fit_lmnn, fit_lmnn_projected, fit_itml_s, decision_scores
from .metrics import fit
from .evaluation import verification_scores
from .run import save_json, sha

METHODS = ['euclidean','itml','lmnn','svm','svm_plus','itml_s','itml_plus']


def training_labels(root, size):
    if (root/'train_manifest.csv').is_file():
        with (root/'train_manifest.csv').open(newline='') as file:
            labels = np.array([r['identity'] for r in csv.DictReader(file)])
    else:
        with (root/'manifest.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        with np.load(root/'split.npz',allow_pickle=False) as split:
            labels = np.array([rows[i]['identity'] for i in split['train']])
    if len(labels) != size:
        raise ValueError('Training identities do not align with saved features')
    return labels


def candidates(method, config):
    if method=='euclidean':
        return [{}]
    if method=='lmnn':
        return [dict(mu=mu) for mu in config['lmnn_mu_grid']]
    if method=='svm':
        return [dict(C=c) for c in config['C_grid']]
    if method=='svm_plus':
        return [dict(C=c,gamma=g) for c,g in itertools.product(config['C_grid'],config['svm_plus_gamma_grid'])]
    if method=='itml_plus':
        return [dict(gamma=g,lam=l) for g,l in itertools.product(config['gamma_grid'],config['lambda_grid'])]
    return [dict(gamma=g) for g in config['gamma_grid']]


def train(method, X, Z, labels, pairs, xi0, params, config):
    budget = dict(max_sweeps=config['max_sweeps'],tol=config['tolerance'])
    if method=='euclidean':
        return dict(M=np.eye(X.shape[1]),stop='fixed_metric')
    if method=='lmnn':
        if config.get('lmnn_backend','sdp')=='projected':
            return fit_lmnn_projected(X,labels,**params,k=config['lmnn_k'],
                max_iterations=config.get('lmnn_max_iterations',1000),tol=config.get('lmnn_tolerance',1e-7))
        if config.get('lmnn_backend','sdp')!='sdp':
            raise ValueError('Unknown LMNN backend')
        return fit_lmnn(X,labels,**params,k=config['lmnn_k'],max_triplets=config.get('lmnn_max_triplets',1000000))
    if method in ('svm','svm_plus'):
        return fit_svm(X,pairs,**params,Z=Z if method=='svm_plus' else None)
    if method=='itml_s':
        return fit_itml_s(X,pairs,xi0,**params,Z=Z,filter_rule=config['itml_s_filter_rule'],**budget)
    return fit(X,pairs,xi0,**params,Z=Z if method=='itml_plus' else None,**budget)


def run(args):
    config = json.loads(args.config.read_text())
    methods = config.get('methods',METHODS)
    if not methods or len(set(methods))!=len(methods) or set(methods)-set(METHODS):
        raise ValueError('Unknown or duplicate baseline methods')
    if config.get('svm_kernel') != 'linear' and set(methods)&{'svm','svm_plus'}:
        raise ValueError('Explicit svm_kernel=linear required; paper kernel unspecified')
    metric = config.get('selection_metric','AP')
    if metric not in ('AP','AUC'):
        raise ValueError('Unknown validation selection metric')
    for method in methods:
        if not candidates(method,config):
            raise ValueError('Empty parameter grid: '+method)
    with np.load(args.input/'features_pca.npz',allow_pickle=False) as data:
        X,Z,val = data['X_train'],data['Z_train'],data['X_val']
    labels = training_labels(args.input,len(X)) if 'lmnn' in methods else None
    pair_files = sorted(args.input.glob('round_*/pairs.npz'))
    requested=getattr(args,'rounds',None)
    if requested:
        if len(set(requested))!=len(requested) or set(requested)-{p.parent.name for p in pair_files}:
            raise ValueError('Unknown or duplicate requested rounds')
        pair_files=[p for p in pair_files if p.parent.name in requested]
    if not pair_files:
        raise ValueError('Missing fixed round inputs')
    files = [args.input/'features_pca.npz']+[f for p in pair_files for f in (p,p.parent/'xi0.npy')]
    files += [args.input/n for n in ('train_manifest.csv','manifest.csv','split.npz') if (args.input/n).is_file()]
    hashes = {str(p.relative_to(args.input)):sha(p) for p in files}
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/'protocol.json',dict(input=str(args.input.resolve()),config=config,methods=methods,
        input_sha256=hashes,config_sha256=sha(args.config),numpy_version=np.__version__,cvxpy_version=cp.__version__,
        code_sha256={n:sha(Path(__file__).parent/n) for n in ('baselines.py','baseline_compare.py','metrics.py','evaluation.py')},
        scope='Fixed-input formula/backend comparison; not original numerical reproduction',
        conventions=dict(svm_kernel='linear',svm_pair_gate='standard normal PDF, unscaled pair features',
            lmnn='fixed same-identity Euclidean k neighbors, sum hinge SDP, no triplet sampling',
            itml_s='explicit threshold convention from config; gamma shared between stages'),
        selection=metric+' on validation only; first candidate on ties',test_evaluated=False))
    results = []
    for path in pair_files:
        with np.load(path,allow_pickle=False) as data:
            pairs,val_pairs = data['train'],data['val']
        if labels is not None and not np.array_equal(labels[pairs[:,0]]==labels[pairs[:,1]],pairs[:,2]==1):
            raise ValueError('Training identity labels disagree with saved pairs')
        xi0 = np.load(path.parent/'xi0.npy',allow_pickle=False)
        folder = args.output/path.parent.name
        folder.mkdir()
        for method in methods:
            search,best = [],None
            for params in candidates(method,config):
                start = perf_counter()
                row = dict(parameters=params)
                try:
                    model = train(method,X,Z,labels,pairs,xi0,params,config)
                    scores = decision_scores(val,val_pairs,model)
                    row.update(status='completed',validation=verification_scores(scores,val_pairs[:,2]),
                        diagnostics=model.get('diagnostics',{}),stop=model['stop'],
                        final=model['history'][-1] if model.get('history') else {},
                        depth_final=model['depth_history'][-1] if model.get('depth_history') else {})
                    if best is None or row['validation'][metric]>best[0]:
                        best = (row['validation'][metric],model,row)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,cp.error.SolverError) as error:
                    row.update(status='failed',error=str(error))
                row['seconds'] = perf_counter()-start
                search.append(row)
                save_json(folder/f'{method}_search.json',search)
            record = dict(round=path.parent.name,method=method,failed_candidates=sum(r['status']=='failed' for r in search))
            if best is None:
                record['status']='all_candidates_failed'
            else:
                _,model,selected=best
                record.update(status='completed',selected=selected)
                np.savez(folder/f'{method}_model.npz',**{k:v for k,v in model.items() if isinstance(v,np.ndarray)})
                save_json(folder/f'{method}_history.json',dict(history=model.get('history',[]),depth_history=model.get('depth_history',[])))
            results.append(record)
            save_json(args.output/'results.json',results)
            print(record['round'],method,record['status'],flush=True)
    unchanged = all(sha(args.input/n)==h for n,h in hashes.items())
    save_json(args.output/'integrity.json',dict(source_unchanged=unchanged,test_evaluated=False))
    if not unchanged:
        raise RuntimeError('Fixed inputs changed')
    if any(r['status']!='completed' for r in results):
        raise RuntimeError('Incomplete baseline comparison; failures preserved in results.json')
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/baselines_paper.json')
    parser.add_argument('--rounds',nargs='+',help='Optional fixed round names, e.g. round_01')
    run(parser.parse_args())
