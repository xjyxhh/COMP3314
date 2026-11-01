"""Xu V-D1/2 validation searches on fixed input; final evaluation is separate."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .metrics import fit
from .partial import privileged_prefix
from .evaluation import verification
from .run import save_json,sha


def variants():
    return ([dict(kind='partial',fraction=f,method='itml_plus') for f in (0.,.25,.5,.75,1.)]
            +[dict(kind='sample_count',fraction=f,method=m) for f in (.2,.4,.6,.8,1.) for m in ('itml','itml_plus')])


def variant_name(v):
    return f"{v['kind']}_{int(round(v['fraction']*100)):03d}_{v['method']}"


def train_analyses(inputs,baseline,out,config_path,round_names=None):
    config=json.loads(config_path.read_text())
    integrity=json.loads((baseline/'integrity.json').read_text())
    if not integrity['source_unchanged']:
        raise ValueError('Baseline input integrity failed')
    with np.load(inputs/'features_pca.npz',allow_pickle=False) as f:
        X,Z,val=f['X_train'],f['Z_train'],f['X_val']
    pair_files=sorted(inputs.glob('round_*/pairs.npz'))
    if round_names:
        if set(round_names)-{p.parent.name for p in pair_files}:raise ValueError('Unknown analysis rounds')
        pair_files=[p for p in pair_files if p.parent.name in round_names]
    files=[inputs/'features_pca.npz']+[f for p in pair_files for f in (p,p.parent/'xi0.npy')]
    hashes={str(p.relative_to(inputs)):sha(p) for p in files}
    base_protocol=json.loads((baseline/'protocol.json').read_text())
    if any(base_protocol['input_sha256'].get(n)!=h for n,h in hashes.items()):
        raise ValueError('Analyses must reuse baseline inputs exactly')
    if config != base_protocol['config']:
        raise ValueError('Analysis settings must match the baseline protocol')
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',dict(config=config,input_sha256=hashes,baseline=str(baseline.resolve()),
        variants=variants(),selection='validation AP per variant; no test evaluation',
        sample_subset='nested first floor(f*N) within each label; all images/PCA/bounds retained',
        partial_subset='all pairs retained, nested first fraction per label has privileged information',
        cached_endpoints='0% partial=ITML; 100% partial/all-pairs=original baseline selected models',
        source_sha256=sha(Path(__file__)),test_evaluated=False))
    base_records=json.loads((baseline/'results.json').read_text())
    records=[]
    for pair_file in pair_files:
        with np.load(pair_file,allow_pickle=False) as f:pairs,vpairs=f['train'],f['val']
        xi0=np.load(pair_file.parent/'xi0.npy',allow_pickle=False)
        for v in variants():
            folder=out/pair_file.parent.name/variant_name(v)
            folder.mkdir(parents=True)
            subset=privileged_prefix(pairs,v['fraction'])
            cache=(v['fraction']==1.) or (v['kind']=='partial' and v['fraction']==0.)
            if cache:
                method='itml' if v['fraction']==0. else v['method']
                original=next(r for r in base_records if r['round']==pair_file.parent.name and r['method']==method)
                if original['status']!='completed':
                    raise ValueError('Missing endpoint baseline: '+method)
                with np.load(baseline/pair_file.parent.name/f'{method}_model.npz') as f:model={k:f[k] for k in f.files}
                np.savez(folder/'model.npz',**model)
                selected=original['selected']
                record=dict(**v,round=pair_file.parent.name,status='completed',cached_from=method,selected=selected)
            else:
                p=pairs if v['kind']=='partial' else pairs[subset]
                x0=xi0 if v['kind']=='partial' else xi0[subset]
                mask=subset if v['kind']=='partial' else None
                np.save(folder/'subset_mask.npy',subset)
                search=[];best=None
                for gamma in config['gamma_grid']:
                    for lam in (config['lambda_grid'] if v['method']=='itml_plus' else [1.]):
                        start=perf_counter();row=dict(parameters=dict(gamma=gamma,lam=lam))
                        try:
                            model=fit(X,p,x0,gamma,Z=Z if v['method']=='itml_plus' else None,lam=lam,
                                max_sweeps=config['max_sweeps'],tol=config['tolerance'],privileged_mask=mask)
                            row.update(status='completed',validation=verification(val,vpairs,model['M']),
                                       final=model['history'][-1],stop=model['stop'])
                            if best is None or row['validation']['AP']>best[0]:
                                best=(row['validation']['AP'],model,row)
                        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                            row.update(status='failed',error=str(error))
                        row['seconds']=perf_counter()-start
                        search.append(row)
                        save_json(folder/'search.json',search)
                record=dict(**v,round=pair_file.parent.name,status='all_candidates_failed')
                if best is not None:
                    _,model,selected=best
                    record.update(status='completed',selected=selected)
                    np.savez(folder/'model.npz',**{k:a for k,a in model.items() if isinstance(a,np.ndarray)})
                    save_json(folder/'history.json',model['history'])
            record['folder']=str(folder.relative_to(out))
            record['selected_fraction_positive']=int(np.sum(subset & (pairs[:,2]==1)))
            record['selected_fraction_negative']=int(np.sum(subset & (pairs[:,2]==-1)))
            records.append(record);save_json(out/'results.json',records)
            print(record['round'],variant_name(v),record['status'],flush=True)
    unchanged=all(sha(inputs/n)==h for n,h in hashes.items())
    save_json(out/'integrity.json',dict(source_unchanged=unchanged,test_evaluated=False))
    if not unchanged or any(r['status']!='completed' for r in records):
        raise RuntimeError('Incomplete analyses, see preserved results')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();train_analyses(a.input,a.baseline,a.output,a.config)
