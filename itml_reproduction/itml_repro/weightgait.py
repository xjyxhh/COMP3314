"""Frozen 20-shot alternative retrieval study using released RGB/depth videos.

Released depth videos are 8-bit colorized/compressed displays, NOT metric
sensor arrays. This limitation is intrinsic to these public archives.
RGB crop uses only row/column pose coordinates, never the third component.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .run import save_json,sha

def frame_box(joints,height=240,width=424):
    """Source Image_Processor draws (joint[1],joint[0]): stored ROW,COLUMN."""
    a=np.asarray(joints,float)[:,:2]
    valid=(a[:,0]>0)&(a[:,0]<height)&(a[:,1]>0)&(a[:,1]<width)
    if valid.sum()<12:return None
    if not valid[0] or not any(valid[[5,6]]) or not any(valid[[11,12]]) or not any(valid[[15,16]]):return None
    head=a[0,0]
    shoulder=np.median(a[[i for i in (5,6) if valid[i]],0])
    hip=np.median(a[[i for i in (11,12) if valid[i]],0])
    ankle=np.median(a[[i for i in (15,16) if valid[i]],0])
    if not (head+5<shoulder and shoulder+15<hip and hip+20<ankle):return None
    y0,x0=a[valid].min(0);y1,x1=a[valid].max(0)
    if (y1-y0<80 or x1-x0<12 or x1-x0>.85*(y1-y0) or
            min(x0,y0)<5 or x1>width-5 or y1>height-3):return None
    pad_y=.2*(y1-y0);pad_x=max(8.,.2*(x1-x0))
    return [max(0,int(x0-pad_x)),max(0,int(y0-pad_y)),min(width,int(np.ceil(x1+pad_x))),min(height,int(np.ceil(y1+pad_y)))]

def select(source,out):
    with zipfile.ZipFile(source/'All Joints.zip') as z:
        raw=z.read('All Joints/Absolute_Data.csv').decode()
    clips={}
    for r in csv.reader(io.StringIO(raw)):
        n=int(r[0]);person=int(r[5]);condition=list(map(int,r[2:5]))
        clip=clips.setdefault(n,dict(identity=person,condition=condition,frames=[],max_frame=0))
        if clip['identity']!=person or clip['condition']!=condition:raise ValueError('Conflicting clip metadata')
        clip['max_frame']=max(clip['max_frame'],int(r[1]))
        if condition!=[0,0,0]:continue
        joints=np.array([[float(x) for x in s.strip('[]').split(',')] for s in r[6:]])
        box=frame_box(joints)
        if box is not None:clip['frames'].append(dict(frame=int(r[1]),box=box))
    eligible={p:sorted(n for n,c in clips.items() if c['identity']==p and c['condition']==[0,0,0]) for p in range(32)}
    groups={};counts={}
    for person,names in eligible.items():
        if len(names)!=10:raise ValueError('Expected ten normal no-freeze/no-obstacle clips')
        groups[person]={}
        for subset,indices in [('train',names[:4]),('normal_A',names[4:7]),('normal_B',names[7:])]:
            groups[person][subset]=[(n,frame) for n in indices for frame in clips[n]['frames']]
        counts[person]={n:len(v) for n,v in groups[person].items()}
    included=sorted(p for p,c in counts.items() if min(c.values())>=20)
    if len(included)<16:raise ValueError('Too few people for PCA150/20-shot protocol after strict RGB QC')
    ntrain=len(included)//2;nval=len(included)-ntrain
    order=np.random.default_rng(2026).permutation(included).tolist();training=set(order[:ntrain])
    out.mkdir(parents=True,exist_ok=False);selected=[]
    for p in included:
        for subset,pool in groups[p].items():
            frames=[pool[i] for i in np.linspace(0,len(pool)-1,20,dtype=int)]
            for n in sorted({n for n,_ in frames}):
                if clips[n]['max_frame']>=1000:raise ValueError('Composite frame encoding needs adjustment')
                selected.append(dict(identity=p,subset=subset,clip=n,frames=[f for clip,f in frames if clip==n],expected_frames=clips[n]['max_frame']+1,
                    training_depth=(subset=='train' and p in training)))
    save_json(out/'selection.json',dict(seed=2026,identity_order=[str(p) for p in order],identity_split=[ntrain,nval],
        shots=20,rounds=5,test_identities=len(included),test_subsets=['normal_A','normal_B'],pair_counts=[190*ntrain,190*ntrain],pca_dimension=150,require_kdes=True,
        selected_clips=selected,eligible_frame_counts={str(p):c for p,c in counts.items()},excluded={str(p):counts[p] for p in range(32) if p not in included},
        exclusion_reason='Before model/test evaluation: fewer than20 anatomically plausible, visible RGB frames in any fixed group',
        QC='>=12 valid RGB joints; nose/shoulder/hip/ankle ordering; height>=80; width/height<=.85; edge checks; box padding20%; third component ignored',
        scope='WeightGait alternative retrieval using20 valid frames pooled from disjoint clip groups; display-depth privilege, not metric-depth BIWI',
        grouping='first4 normal clips train/validation, next3 normal_A, last3 normal_B; uniform sample20 across good frames in each group; not one-video20-shot BIWI',
        depth='Released MPEG4 uint8 BGR display, converted via PIL L; no claim to recover raw depth units',
        pose_reference='ChrisLochhead/GaitMonitor Code/Programs/Data_Processing/Image_Processor.py blob 67342fea5605954ea11e6121d93d66a47d1be071',
        sources_sha256={n:sha(source/n) for n in ('All Joints.zip','Readme.md')},test_evaluated=False))
    print('Selected',len(included),'people,',len(selected),'distinct clip contributions,',60*len(included),'RGB frames;',20*ntrain,'training depth frames',flush=True)

def extract(source,selection,out,write_reid_config=True):
    plan=json.loads((selection/'selection.json').read_text());out.mkdir(parents=True,exist_ok=False)
    rows=[];checks=[];temp=out/'temporary_clip.mp4'
    try:
        with zipfile.ZipFile(source/'All RGB Videos.zip') as rgb,zipfile.ZipFile(source/'All Depth Videos.zip') as depth:
            for k,clip in enumerate(plan['selected_clips']):
                selected={r['frame']:r['box'] for r in clip['frames']}
                for modality,archive,member in [('rgb',rgb,f"All RGB Videos/Instance_{clip['clip']}.0.mp4"),
                                               ('depth',depth,f"All Depth Videos/Instance_{clip['clip']}.0_depth.mp4")]:
                    if modality=='depth' and not clip['training_depth']:continue
                    raw=archive.read(member);temp.write_bytes(raw);cap=cv2.VideoCapture(str(temp))
                    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));saved=0
                    if count!=clip['expected_frames']:raise ValueError(f'Video/pose frame mismatch {member}: {count} vs {clip["expected_frames"]}')
                    for frame in range(max(selected)+1):
                        ok,a=cap.read()
                        if not ok:raise ValueError('Premature video end')
                        if frame not in selected:continue
                        if a.shape!=(240,424,3) or a.dtype!=np.uint8:raise ValueError('Unexpected released video format')
                        x0,y0,x1,y1=selected[frame];crop=cv2.cvtColor(a[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
                        image=Image.fromarray(crop)
                        if max(image.size)>300:
                            ratio=300/max(image.size);image=image.resize(tuple(max(1,int(np.floor(d*ratio))) for d in image.size),Image.Resampling.BILINEAR)
                        if min(image.size)<16:
                            ratio=16/min(image.size);image=image.resize(tuple(int(np.ceil(d*ratio)) for d in image.size),Image.Resampling.BILINEAR)
                        if max(image.size)>300:raise ValueError('Unexpected crop size')
                        path=out/modality/f"p{clip['identity']:02d}_c{clip['clip']:04d}_f{frame:03d}.png";path.parent.mkdir(exist_ok=True)
                        if modality=='depth':image=image.convert('L')
                        image.save(path);saved+=1
                        if modality=='rgb':
                            dep=out/'depth'/path.name
                            rows.append(dict(sample_id=path.stem,identity=str(clip['identity']),subset=clip['subset'],frame=clip['clip']*1000+frame,source_frame=frame,clip=clip['clip'],
                                rgb_path=str(path.resolve()),depth_path=str(dep.resolve()) if clip['training_depth'] else ''))
                    cap.release()
                    if saved!=len(selected):raise ValueError('Incomplete sampled clip contribution')
                    checks.append(dict(member=member,archive_crc_verified=True,video_frames=count,video_fps=fps,selected_frames=saved,
                                       sample_dtype='uint8',channel_count=3,member_sha256=__import__('hashlib').sha256(raw).hexdigest()))
                if (k+1)%20==0:print('Extracted',k+1,'/',len(plan['selected_clips']),'clip contributions',flush=True)
    finally:
        if temp.exists():temp.unlink()
    with (out/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'training_images.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(r for r in rows if r['subset']=='train' and r['depth_path'])
    if write_reid_config:
        config={n:plan[n] for n in ('seed','identity_order','identity_split','shots','rounds','test_identities','test_subsets','pair_counts','pca_dimension','require_kdes')}
        save_json(out/'reid_config.json',config)
    save_json(out/'video_checks.json',checks)
    save_json(out/'protocol.json',dict(selection_sha256=sha(selection/'selection.json'),source_archives_sha256={n:sha(source/n) for n in ('All RGB Videos.zip','All Depth Videos.zip')},
        scope=plan['scope'],depth=plan['depth'],pose_reference=plan['pose_reference'],samples=len(rows),training_depth_samples=sum(len(c['frames']) for c in plan['selected_clips'] if c['training_depth']),
        grouping=plan['grouping'],
        resizing='preserve aspect, cap longest side at300; increase rare minimum side to16 for KDES patch support',test_evaluated=False))
    print('WeightGait crop extraction complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['select','extract'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--selection',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='select':select(a.source,a.output)
    else:extract(a.source,a.selection,a.output)
