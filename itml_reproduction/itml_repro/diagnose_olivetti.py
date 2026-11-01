"""Train/validation-only convergence and clean-feature diagnostics; never score test pairs."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .metrics import fit, distances
from .corrected import fit_corrected
from .evaluation import verification
from .pca import fit_pca, transform
from .run import save_json, sha


def feature_diagnostic(source, out):
    with np.load(source/'image_features.npz', allow_pickle=False) as data:
        raw = {k:data[k] for k in ('X','Z')}
    records=[]
    for index in range(3):
        inputs=source/f'split_{index:02d}'/'input'
        folder=out/f'split_{index:02d}';folder.mkdir()
        with np.load(inputs/'split.npz', allow_pickle=False) as data:
            train_idx,val_idx=data['train'],data['val']
        with np.load(inputs/'pca.npz', allow_pickle=False) as data:
            pca={k:data[k] for k in data.files}
        with np.load(inputs/'round_00/pairs.npz', allow_pickle=False) as data:
            train_pairs,val_pairs=data['train'],data['val']
        with np.load(inputs/'features_pca.npz', allow_pickle=False) as data:
            saved_xtrain,saved_xval,saved_ztrain=data['X_train'],data['X_val'],data['Z_train']
        model=(pca['Z_mean'],pca['Z_components'])
        ztrain,zval=transform(raw['Z'][train_idx],model),transform(raw['Z'][val_idx],model)
        np.testing.assert_allclose(ztrain,saved_ztrain,atol=1e-12)
        # Independently fit train-only PCA; a global positive scale cannot change AP/AUC.
        independent=fit_pca(raw['Z'][train_idx],16)
        clean_val=transform(raw['Z'][val_idx],independent)
        clean_score=verification(zval,val_pairs,np.eye(16))
        np.testing.assert_allclose(list(clean_score.values()),list(verification(clean_val,val_pairs,np.eye(16)).values()))
        np.savez(folder/'validation_feature_inputs.npz', X_train=saved_xtrain,X_val=saved_xval,
                 Z_train=ztrain,Z_val=zval,train_pairs=train_pairs,val_pairs=val_pairs)
        for name,train,val in [('degraded',saved_xtrain,saved_xval),('clean',ztrain,zval)]:
            row=dict(split=index,feature=name,euclidean=verification(val,val_pairs,np.eye(train.shape[1])))
            d=distances(val,val_pairs,np.eye(train.shape[1]))
            row['validation_distance_medians']={label:float(np.median(d[val_pairs[:,2]==y])) for label,y in [('same_person',1),('different_person',-1)]}
            i,j=np.triu_indices(len(train),1)
            low,high=np.quantile(np.linalg.norm(train[i]-train[j],axis=1),[.03,.97])
            xi0=np.where(train_pairs[:,2]==1,low,high)
            search=[];best=None
            for gamma in (.01,.1,1.):
                learned=fit(train,train_pairs,xi0,gamma,max_sweeps=100,tol=0.)
                candidate=dict(gamma=gamma,validation=verification(val,val_pairs,learned['M']),final=learned['history'][-1])
                search.append(candidate)
                if best is None or candidate['validation']['AP']>best[0]:
                    best=(candidate['validation']['AP'],candidate,learned)
            row.update(itml_validation_selected=best[1],search=search,training_bounds=dict(low=float(low),high=float(high)))
            np.savez(folder/f'{name}_feature_itml.npz',M=best[2]['M'])
            records.append(row)
            save_json(out/'feature_quality.json',records)
            print('features',index,name,row['euclidean'],best[1]['validation'],flush=True)
    return records


def convergence_diagnostic(source,out):
    records=[];runs=[]
    budgets=(100,500,1000)
    for index in range(3):
        base=source/f'split_{index:02d}'
        inputs=base/'input';folder=out/f'split_{index:02d}'
        with np.load(inputs/'features_pca.npz',allow_pickle=False) as data:
            X,Z,val=data['X_train'],data['Z_train'],data['X_val']
        with np.load(inputs/'round_00/pairs.npz',allow_pickle=False) as data:
            pairs,val_pairs=data['train'],data['val']
        xi0=np.load(inputs/'round_00/xi0.npy',allow_pickle=False)
        selected={r['method']:r['selected'] for r in json.loads((base/'search/results.json').read_text()) if r['status']=='completed'}
        for method in ('itml','itml_plus','itml_plus_corrected'):
            config=selected[method];is_plus='plus' in method
            solver=fit_corrected if method.endswith('corrected') else fit
            start=perf_counter();last_history=[]
            def capture(sweep,state):
                nonlocal last_history
                last_history=state['history']
                if sweep not in budgets:
                    return
                r=distances(X,pairs,state['M'])
                target=distances(Z,pairs,state['P']) if is_plus else state['xi']
                row=dict(split=index,method=method,sweeps=sweep,gamma=config['gamma'],lam=config['lam'],
                    seconds=perf_counter()-start,validation=verification(val,val_pairs,state['M']),
                    relative_violation=float(np.max(np.maximum(pairs[:,2]*(r-target),0)/np.maximum(r,target))),
                    relative_equality=(float(np.max(np.abs(state['xi']-target)/np.maximum(state['xi'],target))) if is_plus else 0.),
                    final=state['history'][-1])
                row['primal_residuals_below_1e_6']=max(row['relative_violation'],row['relative_equality'])<1e-6
                if method.endswith('corrected'):
                    row['kkt_below_1e_6']=max(row['relative_violation'],row['relative_equality'],
                        *[row['final'][k] for k in ('complementarity','stationarity_M','stationarity_P','stationarity_xi','dual_nonnegative_violation')])<1e-6
                if sweep==100:
                    with np.load(base/'search/round_00'/f'{method}_model.npz',allow_pickle=False) as old:
                        for key in ('M','xi','P') if is_plus else ('M','xi'):
                            np.testing.assert_array_equal(state[key],old[key])
                    np.testing.assert_allclose(list(row['validation'].values()),list(config['validation'].values()),atol=1e-14)
                    row['matches_original_100_sweep_model']=True
                np.savez(folder/f'{method}_{sweep}.npz',**{k:state[k] for k in ('M','P','xi','dual','beta') if state[k] is not None})
                records.append(row);save_json(out/'convergence.json',records)
                print('checkpoint',index,method,sweep,row['relative_violation'],row['relative_equality'],row['validation'],flush=True)
            run=dict(split=index,method=method,gamma=config['gamma'],lam=config['lam'],requested_sweeps=1000)
            try:
                model=solver(X,pairs,xi0,config['gamma'],Z=Z if is_plus else None,lam=config['lam'],
                             max_sweeps=1000,tol=0.,on_sweep=capture)
                run.update(status='completed',stop=model['stop'],completed_sweeps=len(model['history']))
            except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                run.update(status='failed',error=str(error),completed_sweeps=len(last_history))
            run['seconds']=perf_counter()-start
            save_json(folder/f'{method}_history.json',last_history)
            runs.append(run);save_json(out/'solver_runs.json',runs)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    files=[args.source/'image_features.npz']
    for index in range(3):
        base=args.source/f'split_{index:02d}'
        files.extend(base/'input'/name for name in ('split.npz','pca.npz','features_pca.npz','round_00/pairs.npz','round_00/xi0.npy'))
        files.append(base/'search/results.json')
        files.extend((base/'search/round_00').glob('*_model.npz'))
    hashes={str(f.relative_to(args.source)):sha(f) for f in files}
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/'protocol.json',dict(source=str(args.source.resolve()),source_sha256=hashes,
        scope='steps 1 and 2 only; no test metrics read or computed',budgets=[100,500,1000],tolerance=0.,
        hyperparameters='fixed to each method and split original validation-selected values',
        checkpoint='one continuous trajectory from identity initialization; 100-sweep state must exactly match original',
        primary_quality_check='clean vs degraded Euclidean on same validation pairs, same 16 PCA dimensions',
        secondary_quality_check='independent ITML gamma grid {.01,.1,1}, 100 sweeps, validation selected; optimistic diagnostic, not held-out estimate',
        decisions='do not select test settings or automatically replace previous models/results'))
    feature_diagnostic(args.source,args.output)
    convergence_diagnostic(args.source,args.output)
    if any(sha(args.source/name)!=digest for name,digest in hashes.items()):
        raise RuntimeError('Source experiment changed')
    save_json(args.output/'integrity.json',dict(source_unchanged=True,test_metrics_used=False))


if __name__=='__main__':
    main()
