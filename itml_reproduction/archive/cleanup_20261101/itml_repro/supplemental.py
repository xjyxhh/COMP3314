"""Predeclared matched-parameter controls and finite-iteration diagnostics."""
import json
from pathlib import Path
import numpy as np
from .metrics import fit
from .evaluation import verification
from .reid import rank_scores
from .run import save_json,sha


def controls(inputs,baseline,out,retrieval=False):
    if out.exists():raise FileExistsError(out)
    main=json.loads((baseline/'results.json').read_text());protocol=json.loads((baseline/'protocol.json').read_text())
    if any(sha(inputs/n)!=h for n,h in protocol['input_sha256'].items()):raise ValueError('Main input integrity mismatch')
    if len(main)!=35 or any(r['status']!='completed' for r in main):raise ValueError('All main model selections must finish')
    config=protocol['config'];out.mkdir(parents=True)
    with np.load(inputs/'features_pca.npz') as data:X,Z=data['X_train'],data['Z_train']
    names=['no_PI_matched_gamma']+[f'{kind}_PI_{seed}' for kind in ('shuffle','random') for seed in (17,29,43)]+['duplicate_RGB_PI','extended_100_sweeps']
    save_json(out/'protocol.json',dict(variants=names,parameters='fixed main validation-selected ITML+ gamma/lambda; no control tuning',
        seeds=[17,29,43],main_protocol_sha256=sha(baseline/'protocol.json'),test_evaluated=False,
        extended_iterations='100 sweeps, same formula and norm tolerance; additional diagnostic, not replacement main results'))
    records=[]
    for record in main:
        if record['method']!='itml_plus':continue
        r=record['round'];params=record['selected']['parameters'];folder=out/r;folder.mkdir()
        with np.load(inputs/r/'pairs.npz') as f:pairs=f['train']
        xi0=np.load(inputs/r/'xi0.npy')
        for name in names:
            seed=int(name.rsplit('_',1)[1]) if name.startswith(('shuffle_','random_')) else None
            privileged=Z;budget=config['max_sweeps']
            if name=='no_PI_matched_gamma':privileged=None
            elif name.startswith('shuffle_'):privileged=Z[np.random.default_rng(seed).permutation(len(Z))]
            elif name.startswith('random_'):privileged=Z.mean(0)+Z.std(0)*np.random.default_rng(seed).normal(size=Z.shape)
            elif name=='duplicate_RGB_PI':privileged=X
            elif name=='extended_100_sweeps':budget=100
            row=dict(round=r,variant=name,parameters=params,max_sweeps=budget)
            try:
                model=fit(X,pairs,xi0,params['gamma'],Z=privileged,lam=params.get('lam',1.),max_sweeps=budget,tol=config['tolerance'])
                path=folder/(name+'_model.npz');np.savez(path,**{k:v for k,v in model.items() if isinstance(v,np.ndarray)})
                save_json(folder/(name+'_history.json'),model['history'])
                row.update(status='completed',stop=model['stop'],final=model['history'][-1],model_sha256=sha(path))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:row.update(status='failed',error=str(error))
            records.append(row);save_json(out/'training_results.json',records)
            print('control',r,name,row['status'],flush=True)
    # No test features or test pairs are read until every predeclared fit finishes.
    evaluated=[]
    if retrieval:
        c=json.loads((inputs/'protocol.json').read_text())['config'];settings=c.get('test_subsets',['walking','still'])
        with np.load(inputs/'features_pca.npz') as f:test={n:(f['X_'+n],f['labels_'+n]) for n in settings}
    else:
        with np.load(inputs/'features_pca.npz') as f:test=f['X_test']
    for row in records:
        if row['status']!='completed':continue
        path=out/row['round']/(row['variant']+'_model.npz')
        with np.load(path) as model:
            if retrieval:
                for setting,(features,labels) in test.items():
                    evaluated.append(dict(round=row['round'],variant=row['variant'],setting=setting,**rank_scores(features,labels,model,c['shots'])))
            else:
                with np.load(inputs/row['round']/'pairs.npz') as f:pairs=f['test']
                evaluated.append(dict(round=row['round'],variant=row['variant'],**verification(test,pairs,model['M'])))
    save_json(out/'test_results.json',evaluated)
    save_json(out/'completion.json',dict(predeclared_fits=len(records),completed_fits=sum(r['status']=='completed' for r in records),
        failures=[r for r in records if r['status']!='completed'],original_paper_reproduction=False))
    return records
