"""Ten frozen repeated identity partitions, two methods, full validation grids."""
import csv
import json
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from itml_repro.facegrabber import complete_depth_features
from itml_repro.data import split_identities,make_pairs
from itml_repro.pca import fit_pca,transform
from itml_repro.baseline_compare import run
from itml_repro.evaluation import verification
from itml_repro.run import save_json,sha

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'data/external_rgbd_20261031/facegrabber'

def partition(seed):
    primary=HERE/'face_input_v2';root=HERE/'robustness'/f'seed_{seed}';root.mkdir(exist_ok=False)
    rows=json.loads((SOURCE/'selection.json').read_text())['rows']
    with np.load(primary/'raw_glbp.npz') as f:X=f['X']
    with np.load(HERE/'face_secondary_depth.npz') as f:Z=f['Z']
    split=split_identities(rows,[20,10,10],seed)
    px,pz=fit_pca(X[split['train']],150),fit_pca(Z[split['train']],150)
    xs={n:transform(X[idx],px) for n,idx in split.items()}
    np.savez(root/'features_pca.npz',**{'X_'+n:a for n,a in xs.items()},Z_train=transform(Z[split['train']],pz))
    np.savez(root/'split.npz',**split)
    save_json(root/'split.json',{n:[rows[i]['sample_id'] for i in idx] for n,idx in split.items()})
    np.savez(root/'pca.npz',rgb_mean=px[0],rgb_components=px[1],depth_mean=pz[0],depth_components=pz[1])
    a,b=np.triu_indices(len(xs['train']),1);u,l=np.quantile(np.linalg.norm(xs['train'][a]-xs['train'][b],axis=1),[.03,.97])
    folder=root/'round_00';folder.mkdir();pairs={}
    for k,n in enumerate(split):pairs[n]=make_pairs([rows[i]['identity'] for i in split[n]],*( (1820,5872) if n=='train' else (910,2936)),seed+k,positive_seed=seed)
    np.savez(folder/'pairs.npz',**pairs);np.save(folder/'xi0.npy',np.where(pairs['train'][:,2]==1,u,l))
    config=json.loads((primary/'baseline_config.json').read_text());config['methods']=['itml','itml_plus'];save_json(root/'config.json',config)
    save_json(root/'protocol.json',dict(seed=seed,raw_RGB_sha256=sha(primary/'raw_glbp.npz'),raw_depth_sha256=sha(HERE/'face_secondary_depth.npz'),
        scope='Repeated overlapping identity partition; full validation tuning, one pair redraw per partition; exploratory robustness'))
    run(Namespace(input=root,output=root/'search',config=root/'config.json'))
    return seed

def main():
    plan=json.loads((HERE/'robustness_plan.json').read_text())
    while not (SOURCE/'secondary_acquisition_complete.json').exists():time.sleep(10)
    complete_depth_features(SOURCE,HERE/'face_input_v2',HERE/'face_secondary_depth.npz')
    (HERE/'robustness').mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for seed in pool.map(partition,plan['partition_seeds']):print('Robustness selected seed',seed,flush=True)
    # All twenty selections finish before any secondary test score is read.
    results=[]
    for seed in plan['partition_seeds']:
        root=HERE/'robustness'/f'seed_{seed}';search=root/'search'
        protocol=json.loads((search/'protocol.json').read_text())
        if any(sha(root/n)!=h for n,h in protocol['input_sha256'].items()):raise ValueError('Robustness input changed')
        records=json.loads((search/'results.json').read_text())
        if len(records)!=2 or any(r['status']!='completed' for r in records):raise ValueError('Incomplete robustness selection')
        with np.load(root/'features_pca.npz') as f:X=f['X_test']
        with np.load(root/'round_00/pairs.npz') as f:pairs=f['test']
        scores={}
        for r in records:
            model=search/'round_00'/f"{r['method']}_model.npz"
            with np.load(model) as f:scores[r['method']]=verification(X,pairs,f['M'])
        results.append(dict(seed=seed,scores=scores,AP_difference_percentage_points=100*(scores['itml_plus']['AP']-scores['itml']['AP'])))
    save_json(HERE/'robustness/results.json',results)
    values=[r['AP_difference_percentage_points'] for r in results]
    save_json(HERE/'robustness/completion.json',dict(partitions=10,models=20,AP_difference_mean=float(np.mean(values)),
        AP_difference_median=float(np.median(values)),AP_difference_min=float(min(values)),AP_difference_max=float(max(values)),
        positive_partitions=sum(v>0 for v in values),scope=plan['dataset'],independent_confirmation=False))
    print('TEN PARTITION ROBUSTNESS COMPLETED',flush=True)

if __name__=='__main__':main()
