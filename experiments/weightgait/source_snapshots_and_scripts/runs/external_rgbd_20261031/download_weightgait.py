"""Download the three official WeightGait archives; preserve source receipts."""
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]/'data/external_rgbd_20261031/weightgait'
ITEMS=[('All Joints.zip','35832174-ac7d-4df0-ac5e-587e0a33cd3d'),
       ('All RGB Videos.zip','2b29c4cc-04c6-4de3-8088-9cdec291cbb7'),
       ('All Depth Videos.zip','58bedc92-e7ad-4cd1-91c0-d07ffd99f469')]

def download(item):
    name,uid=item;path=ROOT/name
    if path.exists():raise FileExistsError(path)
    if shutil.disk_usage(ROOT).free<3*1024**3:raise OSError('Keep at least 3 GiB free for experiment processing')
    url=f'https://datashare.ed.ac.uk/server/api/core/bitstreams/{uid}/content'
    receipt=dict(url=url,file=name,started_unix=time.time(),source='University of Edinburgh DataShare DOI 10.7488/ds/7897')
    try:
        with requests.get(url,stream=True,timeout=(20,60)) as response:
            receipt.update(status_code=response.status_code,headers=dict(response.headers));response.raise_for_status()
            h=hashlib.sha256();count=0;last=time.monotonic()
            with path.with_suffix('.zip.part').open('xb') as file:
                for chunk in response.iter_content(1024*1024):
                    file.write(chunk);h.update(chunk);count+=len(chunk)
                    if count>1200*1024**2:raise ValueError('Unexpected archive size')
                    if time.monotonic()-last>25:
                        print(name,round(count/1024**2),'MiB',flush=True);last=time.monotonic()
            path.with_suffix('.zip.part').rename(path)
            receipt.update(status='downloaded',bytes=count,sha256=h.hexdigest())
    except Exception as error:receipt.update(status='failed',error=str(error))
    (ROOT/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(name,receipt['status'],receipt.get('bytes',receipt.get('error')),flush=True)
    return receipt['status']=='downloaded'

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        if not all(pool.map(download,ITEMS)):raise SystemExit(1)
