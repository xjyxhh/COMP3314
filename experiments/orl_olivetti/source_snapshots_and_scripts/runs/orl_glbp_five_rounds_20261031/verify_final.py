"""Read-only independent score/metric checks of the frozen 110-model result."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    base=read(ROOT/'baselines_final/results.json')
    analyses=read(ROOT/'analyses/results.json')
    tests=read(ROOT/'final/test_results.json')
    assert len(base)==35 and len(analyses)==75 and len(tests)==110
    assert len({(r['round'],r['method']) for r in base})==35
    assert len({(r['round'],r['kind'],r['method'],r['fraction']) for r in analyses})==75
    assert read(ROOT/'analyses/protocol.json')['parameter_policy']=='fixed_main'
    for folder in ('baselines_final','analyses'):
        protocol=read(ROOT/folder/'protocol.json')
        assert all(digest(ROOT/n)==h for n,h in protocol['input_sha256'].items())
    protocol=read(ROOT/'baselines_final/protocol.json')
    for name,h in protocol['code_sha256'].items():
        current=ROOT.parent.parent/'itml_repro'/name
        if name=='baseline_compare.py':
            # Round selection was added after serial round_00 started. Strip
            # only that CLI/filter addition and recover the exact recorded hash.
            source=current.read_text()
            start=source.index("    requested=getattr(args,'rounds',None)\n")
            end=source.index('    if not pair_files:',start)
            initial=source[:start]+source[end:]
            initial=initial.replace("    parser.add_argument('--rounds',nargs='+',help='Optional fixed round names, e.g. round_01')\n",'')
            assert hashlib.sha256(initial.encode()).hexdigest()==h
        else:assert digest(current)==h
    candidates=[r for p in (ROOT/'baselines_final').glob('round_*/*search.json') for r in read(p)]
    assert len(candidates)==700 and all(r['status']=='completed' for r in candidates)
    with np.load(ROOT/'features_pca.npz') as f:X=f['X_test']
    max_error=0.;round_effects=[]
    for row in tests:
        if row['kind']=='main':
            path=ROOT/'baselines_final'/row['round']/f"{row['method']}_model.npz"
        else:
            selected=next(a for a in analyses if all(a[k]==row[k] for k in ('round','kind','method','fraction')))
            path=ROOT/'analyses'/selected['folder']/'model.npz'
            if not selected.get('cached_from'):
                search=read(path.parent/'search.json');assert len(search)==1
                original=next(a for a in base if a['round']==row['round'] and a['method']==row['method'])
                assert all(original['selected']['parameters'][k]==v for k,v in search[0]['parameters'].items()
                           if k in original['selected']['parameters'])
        assert digest(path)==row['model_sha256']
        with np.load(ROOT/row['round']/'pairs.npz') as f:p=f['test']
        a,b=X[p[:,0]],X[p[:,1]]
        with np.load(path) as model:
            if 'M' in model:
                M=model['M'];assert np.linalg.eigvalsh(M).min()>-1e-8
                delta=a-b;score=-np.sum((delta@M)*delta,axis=1)
            else:
                gate=np.exp(-((a+b)/2)**2/2)/np.sqrt(2*np.pi)
                score=(np.concatenate([np.abs(a-b)*gate,a*b*gate],axis=1)@model['w']+model['b'])
        truth=p[:,2]==1
        pos,neg=score[truth],score[~truth]
        auc=np.mean((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg))
        _,inverse=np.unique(score,return_inverse=True)
        positives=np.bincount(inverse,weights=truth)[::-1]
        total=np.bincount(inverse)[::-1]
        ap=np.sum(positives*np.cumsum(positives)/np.cumsum(total))/truth.sum()
        error=max(abs(auc-row['test']['AUC']),abs(ap-row['test']['AP']))
        assert error<1e-12,(row,error)
        max_error=max(max_error,error)
    for n in range(5):
        r=[x for x in tests if x['round']==f'round_{n:02d}' and x['kind']=='main']
        scores={x['method']:x['test']['AP'] for x in r}
        round_effects.append(100*(scores['itml_plus']-scores['itml']))
    plus=[r['selected']['final'] for r in base if r['method']=='itml_plus']
    timing=read(ROOT/'final/training_times.json')['records']
    assert len(timing)==20 and all(r['model_matches'] for r in timing)
    report=dict(checked_models=110,completed_baseline_candidates=700,max_metric_error=max_error,
        frozen_hashes_match=True,training_core_code_unchanged=True,all_metrics_match=True,
        fixed_analysis_parameters_verified=True,all_20_serial_timing_models_match=True,
        itml_plus_minus_itml_AP_percentage_points_by_round=round_effects,
        itml_plus_stop_counts={s:sum(r['selected']['stop']==s for r in base if r['method']=='itml_plus')
                              for s in {r['selected']['stop'] for r in base if r['method']=='itml_plus'}},
        itml_plus_equality_residual_range=[min(r['max_equality_residual'] for r in plus),max(r['max_equality_residual'] for r in plus)],
        itml_plus_relative_equality_residual_range=[min(r['relative_equality_residual'] for r in plus),max(r['relative_equality_residual'] for r in plus)],
        original_paper_reproduction=False)
    (ROOT/'final/independent_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':verify()
