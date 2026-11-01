"""Explicit BIWI-style 20-shot protocol for supplied precomputed features.

Manifest: sample_id,identity,subset,frame; subset=train/walking/still.
Dataset access and feature provenance are separate requirements.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cvxpy as cp
from .data import make_pairs
from .pca import fit_pca,transform
from .baselines import decision_scores
from .baseline_compare import METHODS,candidates,train
from .run import save_json,sha
from .kdes import validate_kdes_archive


def probe_gallery(labels,shots):
    labels=np.asarray(labels)
    queries=[];gallery=[]
    for identity in dict.fromkeys(labels.tolist()):
        indices=np.flatnonzero(labels==identity)
        if len(indices)!=shots or shots<2:raise ValueError('Every identity must have exactly shots samples')
        queries.append(indices[0]);gallery.extend(indices[1:])
    return np.asarray(queries),np.asarray(gallery)


def rank_scores(X,labels,model,shots=20):
    labels=np.asarray(labels)
    q,g=probe_gallery(labels,shots)
    pairs=np.column_stack((np.repeat(q,len(g)),np.tile(g,len(q)),
        np.where(labels[np.repeat(q,len(g))]==labels[np.tile(g,len(q))],1,-1)))
    score=decision_scores(X,pairs,model).reshape(len(q),len(g))
    if not np.isfinite(score).all():raise ValueError('Nonfinite retrieval scores')
    match=labels[q,None]==labels[g][None,:]
    order=np.argsort(-score,axis=1,kind='stable')
    cmc=np.maximum.accumulate(np.take_along_axis(match,order,axis=1),axis=1).mean(axis=0)
    return dict(rank1=float(cmc[0]),CMC=cmc.tolist())


def prepare(manifest,features,out,config):
    with manifest.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows or not {'sample_id','identity','subset','frame'}<=rows[0].keys():raise ValueError('Invalid re-ID manifest')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs')
    if {r['subset'] for r in rows}!={'train','walking','still'}:raise ValueError('Expected train/walking/still')
    with np.load(features,allow_pickle=False) as f:
        ids=f['sample_ids'].astype(str);rawX,rawZ=f['X'],f['Z']
    if len(set(ids))!=len(ids) or set(ids)!={r['sample_id'] for r in rows}:raise ValueError('Archive/manifest sample mismatch')
    lookup={s:i for i,s in enumerate(ids)}
    X,Z=rawX[[lookup[r['sample_id']] for r in rows]],rawZ[[lookup[r['sample_id']] for r in rows]]
    ordinary_ids=list(dict.fromkeys(r['identity'] for r in rows if r['subset']=='train'))
    order=config.get('identity_order',ordinary_ids)
    ntrain,nval=config['identity_split'];shots=config['shots']
    if len(set(order))!=len(order) or set(order)!=set(ordinary_ids) or len(order)!=ntrain+nval:
        raise ValueError('Identity order must exhaust raw training identities')
    groups={}
    for name,subset,wanted in [('train','train',order[:ntrain]),('val','train',order[ntrain:]),
                              ('walking','walking',None),('still','still',None)]:
        present=list(dict.fromkeys(r['identity'] for r in rows if r['subset']==subset))
        wanted=present if wanted is None else wanted
        if name in ('walking','still') and (len(wanted)!=config['test_identities'] or not set(wanted)<=set(order)):
            raise ValueError('Test identities must be from raw training identities and match specified count')
        chosen=[]
        for identity in wanted:
            matches=sorted((i for i,r in enumerate(rows) if r['identity']==identity and r['subset']==subset),key=lambda i:int(rows[i]['frame']))
            frames=[rows[i]['frame'] for i in matches]
            if len(matches)<shots or len(set(frames))!=len(frames):raise ValueError('Insufficient or duplicate frames')
            chosen.extend(np.asarray(matches)[np.linspace(0,len(matches)-1,shots,dtype=int)])
        groups[name]=np.array(chosen)
    feature_metadata=None
    if config.get('require_kdes',True):
        feature_metadata=validate_kdes_archive(features,features.with_suffix('.json'),
                                               [rows[i]['sample_id'] for i in groups['train']])
    px,pz=fit_pca(X[groups['train']],config['pca_dimension']),fit_pca(Z[groups['train']],config['pca_dimension'])
    xs={name:transform(X[idx],px) for name,idx in groups.items()}
    z=transform(Z[groups['train']],pz)
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',dict(config=config,manifest_sha256=sha(manifest),features_sha256=sha(features),
        identity_order=order,frame_sampling='sorted numeric frame, inclusive floor-linspace, without replacement',
        gallery='first selected sample per identity is probe; remaining samples are gallery',
        scope='BIWI-style protocol on supplied features; dataset/feature identity not inferred',
        feature_metadata=feature_metadata,test_evaluated=False))
    np.savez(out/'features_pca.npz',**{'X_'+name:x for name,x in xs.items()},Z_train=z,
        **{'labels_'+name:np.array([rows[i]['identity'] for i in idx]) for name,idx in groups.items()})
    np.savez(out/'pca.npz',rgb_mean=px[0],rgb_components=px[1],depth_mean=pz[0],depth_components=pz[1])
    save_json(out/'selected_samples.json',{name:[rows[i]['sample_id'] for i in idx] for name,idx in groups.items()})
    i,j=np.triu_indices(len(xs['train']),1)
    u,l=np.quantile(np.linalg.norm(xs['train'][i]-xs['train'][j],axis=1),[.03,.97])
    for n in range(config['rounds']):
        folder=out/f'round_{n:02d}';folder.mkdir()
        pairs=make_pairs([rows[i]['identity'] for i in groups['train']],*config['pair_counts'],
                         config['seed']+n,positive_seed=config['seed'])
        np.savez(folder/'pairs.npz',train=pairs);np.save(folder/'xi0.npy',np.where(pairs[:,2]==1,u,l))


def fit_reid(inputs,out,config_path):
    config=json.loads(config_path.read_text())
    protocol=json.loads((inputs/'protocol.json').read_text());shots=protocol['config']['shots']
    if config.get('svm_kernel')!='linear':raise ValueError('Explicit linear SVM convention required')
    with np.load(inputs/'features_pca.npz') as f:
        X,Z,val,labels,vl=f['X_train'],f['Z_train'],f['X_val'],f['labels_train'],f['labels_val']
    files=[inputs/'features_pca.npz',inputs/'protocol.json']+sorted(inputs.glob('round_*/*np*'))
    hashes={str(p.relative_to(inputs)):sha(p) for p in files}
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',dict(config=config,input_sha256=hashes,selection='validation Rank-1; first on ties',
                                     test_evaluated=False,shots=shots))
    records=[]
    for folder in sorted(inputs.glob('round_*')):
        with np.load(folder/'pairs.npz') as f:pairs=f['train']
        xi0=np.load(folder/'xi0.npy');dest=out/folder.name;dest.mkdir()
        for method in METHODS:
            search=[];best=None
            for params in candidates(method,config):
                row=dict(parameters=params)
                try:
                    model=train(method,X,Z,labels,pairs,xi0,params,config)
                    row.update(status='completed',validation=rank_scores(val,vl,model,shots),
                               diagnostics=model.get('diagnostics',{}),stop=model['stop'])
                    if best is None or row['validation']['rank1']>best[0]:best=(row['validation']['rank1'],model,row)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,cp.error.SolverError) as error:
                    row.update(status='failed',error=str(error))
                search.append(row);save_json(dest/f'{method}_search.json',search)
            record=dict(round=folder.name,method=method,status='all_candidates_failed')
            if best is not None:
                _,model,selected=best;record.update(status='completed',selected=selected)
                np.savez(dest/f'{method}_model.npz',**{k:v for k,v in model.items() if isinstance(v,np.ndarray)})
            records.append(record);save_json(out/'results.json',records)
    unchanged=all(sha(inputs/n)==h for n,h in hashes.items())
    save_json(out/'integrity.json',dict(source_unchanged=unchanged,test_evaluated=False))
    if not unchanged or any(r['status']!='completed' for r in records):raise RuntimeError('Incomplete re-ID selection')


def evaluate_reid(inputs,search,out):
    records=json.loads((search/'results.json').read_text())
    if len(records)!=7*len(list(inputs.glob('round_*'))) or any(r['status']!='completed' for r in records):
        raise ValueError('All re-ID searches must finish first')
    protocol=json.loads((search/'protocol.json').read_text())
    if not json.loads((search/'integrity.json').read_text())['source_unchanged']:
        raise ValueError('Input integrity failed')
    if any(sha(inputs/n)!=h for n,h in protocol['input_sha256'].items()):raise ValueError('Input changed after selection')
    out.mkdir(parents=True,exist_ok=False);results=[]
    with np.load(inputs/'features_pca.npz') as data:
        for record in records:
            with np.load(search/record['round']/f"{record['method']}_model.npz") as model:
                for setting in ('walking','still'):
                    result=rank_scores(data['X_'+setting],data['labels_'+setting],model,protocol['shots'])
                    results.append(dict(round=record['round'],method=record['method'],setting=setting,**result))
    save_json(out/'results.json',results)
    save_json(out/'summary.json',[dict(method=m,setting=s,
        mean_rank1=float(np.mean([r['rank1'] for r in results if r['method']==m and r['setting']==s])),
        std_rank1=float(np.std([r['rank1'] for r in results if r['method']==m and r['setting']==s])))
        for m in METHODS for s in ('walking','still')])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','train','evaluate'],required=True)
    for n in ('manifest','features','input','search','config'):p.add_argument('--'+n,type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.manifest,a.features,a.output,json.loads(a.config.read_text()))
    elif a.stage=='train':fit_reid(a.input,a.output,a.config)
    else:evaluate_reid(a.input,a.search,a.output)
