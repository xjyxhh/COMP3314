"""Predeclared 14-shot/40-person FaceGrabber subset; training-only geometry."""
import csv
import hashlib
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from itml_repro.external_data import HTTPFile
from itml_repro.run import save_json,sha

ROOT=Path(__file__).resolve().parents[2]/'data/external_rgbd_20261031/facegrabber'
BASE='https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=00a74ddc&path=Datenbanken%5C'
ARCHIVES=dict(rgb='KinectColorRaw',landmarks2d='Landmarks2D',landmarks3d='Landmarks3D',facecloud='FaceRaw')
PATTERN=re.compile(r'(ID_\d{3})/(\d{2}_[^/]+)/[^/]+/[^/]+_(\d{6})\.(?:jpeg|pcd)$')

def acquire():
    selection=ROOT/'selection.json'
    if selection.exists():raise FileExistsError(selection)
    indices={};urls={}
    for kind,suffix in ARCHIVES.items():
        url=BASE+'FaceGrabberDB_'+suffix+'.zip';urls[kind]=url
        with zipfile.ZipFile(HTTPFile(url)) as z:
            found={}
            for item in z.infolist():
                match=PATTERN.search(item.filename)
                if match:found[match.groups()]=item.filename
            indices[kind]=found
        print(kind,'indexed',len(found),flush=True)
    identities=sorted({key[0] for key in indices['rgb']})
    if len(identities)!=40:raise ValueError('Expected 40 official identities')
    order=np.random.default_rng(2026).permutation(identities)
    split={p:('train' if i<20 else 'val' if i<30 else 'test') for i,p in enumerate(order)}
    selected=[]
    for person in identities:
        sequences=sorted({k[1] for k in indices['rgb'] if k[0]==person and not k[1].startswith('08_')})
        if len(sequences)!=7:raise ValueError('Expected headscan + six emotions')
        for seq in sequences:
            keys=sorted(k for k in indices['rgb'] if k[:2]==(person,seq) and all(k in idx for idx in indices.values()))
            if len(keys)<3:raise ValueError('Insufficient synchronized frames')
            for index in (len(keys)//3,2*len(keys)//3):
                key=keys[index]
                selected.append(dict(sample_id='_'.join(key),identity=person,sequence=seq,frame=int(key[2]),split=split[person],
                    sources={kind:idx[key] for kind,idx in indices.items() if kind in ('rgb','landmarks2d') or split[person]=='train'}))
    save_json(selection,dict(seed=2026,identity_split=[20,10,10],sampling='two inclusive-third frames of each of seven nonrandom sequences',
        rows=selected,urls=urls,scope='FaceGrabber alternative RGB-D face verification, not original dataset reproduction',test_evaluated=False))
    def download(kind):
        receipts=[]
        with zipfile.ZipFile(HTTPFile(urls[kind])) as z:
            z.setpassword(b'FaceGrabber@MMK')
            for i,row in enumerate(selected):
                if kind not in row['sources']:continue
                source=row['sources'][kind];path=ROOT/'selected'/source
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():path.write_bytes(z.read(source))
                receipts.append(dict(path=str(path.relative_to(ROOT)),archive_member=source,sha256=sha(path),bytes=path.stat().st_size))
                if i%70==0:print(kind,i+1,'/',len(selected),flush=True)
        save_json(ROOT/(kind+'_members.json'),receipts)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download,ARCHIVES))
    save_json(ROOT/'acquisition_complete.json',dict(samples=len(selected),all_rgb_only_val_test=True,
        train_depth_samples=sum(r['split']=='train' for r in selected),selection_sha256=sha(selection)))
    print('FaceGrabber selected acquisition complete',flush=True)

if __name__=='__main__':acquire()
