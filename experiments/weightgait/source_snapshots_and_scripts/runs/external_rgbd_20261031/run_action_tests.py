"""Frozen normal-trained models on actual single-/both-leg load conditions.

No new training or hyperparameter search. All extra test features use the
existing NORMAL-training RGB KDES book and PCA; depth is never read.
"""
import csv
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from itml_repro.weightgait import frame_box,extract
from itml_repro.kdes_native import load_basis,dense,encode
from itml_repro.reid import rank_scores
from itml_repro.baselines import decision_scores
from itml_repro.run import save_json,sha

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'data/external_rgbd_20261031/weightgait'
SETTINGS={1:'single_leg_load',2:'both_leg_load'}

def main():
    selection=HERE/'action_selection';selection.mkdir(exist_ok=False)
    base=json.loads((HERE/'weight_selection/selection.json').read_text())
    with zipfile.ZipFile(SOURCE/'All Joints.zip') as z:raw=z.read('All Joints/Absolute_Data.csv').decode()
    clips={}
    for r in csv.reader(io.StringIO(raw)):
        condition=list(map(int,r[2:5]))
        if condition[0] not in SETTINGS or condition[1:]!=[0,0]:continue
        n=int(r[0]);clip=clips.setdefault(n,dict(person=int(r[5]),condition=condition[0],frames=[],max_frame=0))
        clip['max_frame']=max(clip['max_frame'],int(r[1]))
        joints=np.array([[float(x) for x in s.strip('[]').split(',')] for s in r[6:]])
        box=frame_box(joints)
        if box is not None:clip['frames'].append(dict(frame=int(r[1]),box=box))
    chosen=[];counts={}
    for p in sorted(map(int,base['identity_order'])):
        for c,setting in SETTINGS.items():
            pool=[(n,f) for n,v in sorted(clips.items()) if v['person']==p and v['condition']==c for f in v['frames']]
            counts[str(p)+'_'+setting]=len(pool)
            if len(pool)<20:raise ValueError('Extra condition has insufficient RGB-visible frames; no test retuning')
            samples=[pool[i] for i in np.linspace(0,len(pool)-1,20,dtype=int)]
            for n in sorted({n for n,_ in samples}):
                chosen.append(dict(identity=p,subset=setting,clip=n,frames=[f for clip,f in samples if clip==n],expected_frames=clips[n]['max_frame']+1,training_depth=False))
    plan=dict(selected_clips=chosen,scope='Additional frozen-model cross-action retrieval; not BIWI Still or cross-day clothing',
        depth='No extra-condition depth is read',pose_reference=base['pose_reference'],grouping='20 visible RGB frames pooled from ten plain clips of each load condition',
        eligible_frame_counts=counts,recorded_before_extra_test_evaluation=True)
    save_json(selection/'selection.json',plan)
    crops=HERE/'action_crops';extract(SOURCE,selection,crops,write_reid_config=False)
    while not (HERE/'weight_input/pca.npz').exists():time.sleep(10)
    basis=load_basis(HERE/'weight_kdes_basis.npz')
    with np.load(HERE/'weight_kdes_codebooks.npz') as f:book={k:f['rgb_'+k] for k in ('words','G','kparam')}
    with np.load(HERE/'weight_input/pca.npz') as f:mean,components=f['rgb_mean'],f['rgb_components']
    with (crops/'manifest.csv').open() as f:rows=list(csv.DictReader(f))
    features=[]
    for i,row in enumerate(rows):
        with Image.open(row['rgb_path']) as im:image=np.asarray(im.convert('L'))
        vector=encode(dense(image,basis),book,[1,2,4]);features.append((vector-mean)@components.T)
        if (i+1)%100==0:print('Extra RGB features',i+1,'/',len(rows),flush=True)
    features=np.asarray(features)
    np.savez_compressed(HERE/'action_features.npz',X=features,sample_ids=np.array([r['sample_id'] for r in rows]),
        labels=np.array([r['identity'] for r in rows]),subsets=np.array([r['subset'] for r in rows]))
    save_json(HERE/'action_features.json',dict(basis_sha256=sha(HERE/'weight_kdes_basis.npz'),codebook_sha256=sha(HERE/'weight_kdes_codebooks.npz'),
        PCA_sha256=sha(HERE/'weight_input/pca.npz'),manifest_sha256=sha(crops/'manifest.csv'),no_refitting=True,no_test_depth=True))
    while not (HERE/'weight_search/integrity.json').exists():time.sleep(10)
    inputs=HERE/'weight_input';search=HERE/'weight_search';protocol=json.loads((search/'protocol.json').read_text())
    if any(sha(inputs/n)!=h for n,h in protocol['input_sha256'].items()):raise ValueError('Main models/input changed')
    records=json.loads((search/'results.json').read_text())
    if len(records)!=35 or any(r['status']!='completed' for r in records):raise ValueError('Incomplete main selections')
    with np.load(inputs/'features_pca.npz') as f:normal=(f['X_normal_A'],f['labels_normal_A'])
    datasets={s:(features[[i for i,r in enumerate(rows) if r['subset']==s]],np.array([r['identity'] for r in rows if r['subset']==s])) for s in SETTINGS.values()}
    out=HERE/'action_final';out.mkdir(exist_ok=False);results=[];cross=[]
    for record in records:
        with np.load(search/record['round']/f"{record['method']}_model.npz") as model:
            for setting,(X,labels) in datasets.items():
                result=rank_scores(X,labels,model,20);results.append(dict(round=record['round'],method=record['method'],setting=setting,**result))
                for direction,(A,al),(B,bl) in [('normal_A_to_'+setting,normal,(X,labels)),(setting+'_to_normal_A',(X,labels),normal)]:
                    q=np.array([np.flatnonzero(al==p)[0] for p in dict.fromkeys(al.tolist())]);combined=np.vstack([A,B])
                    pairs=np.array([(i,len(A)+j,1 if al[i]==bl[j] else -1) for i in q for j in range(len(B))])
                    scores=decision_scores(combined,pairs,model).reshape(len(q),len(B));order=np.argsort(-scores,axis=1,kind='stable')
                    cmc=np.maximum.accumulate(bl[order]==al[q,None],axis=1).mean(0)
                    cross.append(dict(round=record['round'],method=record['method'],direction=direction,rank1=float(cmc[0]),CMC=cmc.tolist()))
    save_json(out/'results.json',results);save_json(out/'cross_action_results.json',cross)
    summary=[dict(method=m,setting=s,mean_rank1=float(np.mean([r['rank1'] for r in results if r['method']==m and r['setting']==s])),
        std_rank1=float(np.std([r['rank1'] for r in results if r['method']==m and r['setting']==s]))) for m in dict.fromkeys(r['method'] for r in records) for s in datasets]
    save_json(out/'summary.json',summary)
    save_json(out/'completion.json',dict(test_RGB_samples=len(rows),frozen_models=35,within_condition_evaluations=len(results),cross_action_evaluations=len(cross),
        scope=plan['scope'],new_training=False,new_parameter_selection=False,extra_test_depth_used=False))
    print('FROZEN CROSS-ACTION TESTS COMPLETED',flush=True)

if __name__=='__main__':main()
