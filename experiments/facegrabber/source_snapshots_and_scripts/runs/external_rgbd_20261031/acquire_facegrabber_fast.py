"""Resume frozen subset with one validated HTTP request per ZIP member."""
import binascii
import argparse
import json
import re
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from itml_repro.external_data import HTTPFile
from itml_repro.run import save_json,sha

ROOT=Path(__file__).resolve().parents[2]/'data/external_rgbd_20261031/facegrabber'
LOCAL=threading.local()

def acquire(all_depth=False):
    selection=json.loads((ROOT/'selection.json').read_text());urls=selection['urls'];rows=selection['rows']
    def index(kind):
        cache=ROOT/(kind+'_zip_index.json')
        if cache.exists():return kind,json.loads(cache.read_text())
        remote=HTTPFile(urls[kind])
        with zipfile.ZipFile(remote) as archive:
            result=dict(size=remote.size,entries={i.filename:dict(offset=i.header_offset,compressed=i.compress_size,
                size=i.file_size,crc=i.CRC,method=i.compress_type) for i in archive.infolist() if not i.is_dir()})
        save_json(cache,result);print('indexed',kind,flush=True);return kind,result
    with ThreadPoolExecutor(max_workers=4) as pool:indices=dict(pool.map(index,urls))
    if all_depth:
        for kind in ('landmarks3d','facecloud'):
            lookup={}
            for name in indices[kind]['entries']:
                match=re.search(r'(ID_\d{3})/(\d{2}_[^/]+)/[^/]+/[^/]+_(\d{6})\.pcd$',name)
                if match:lookup[match.groups()]=name
            for row in rows:row['sources'][kind]=lookup[(row['identity'],row['sequence'],f"{row['frame']:06d}")]
        save_json(ROOT/'secondary_selection_all_depth.json',selection)
    jobs=[(kind,source) for row in rows for kind,source in row['sources'].items()]
    def member(job):
        kind,name=job;info=indices[kind]['entries'][name];path=ROOT/'selected'/name
        if path.exists() and path.stat().st_size==info['size'] and binascii.crc32(path.read_bytes())==info['crc']:
            return dict(kind=kind,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=info['size'],crc_verified=True)
        if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('Unsafe archive member')
        if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
        start=info['offset'];end=min(indices[kind]['size']-1,start+30+len(name.encode())+info['compressed']+4096-1)
        for attempt in range(3):
            try:
                response=LOCAL.session.get(urls[kind],headers={'Range':f'bytes={start}-{end}'},timeout=(30,60))
                response.raise_for_status()
                if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('Range not honored')
                raw=response.content
                values=struct.unpack('<IHHHHHIIIHH',raw[:30]);offset=30+values[-2]+values[-1]
                if values[0]!=0x04034b50 or raw[30:30+values[-2]]!=name.encode():raise ValueError('ZIP header mismatch')
                data=raw[offset:offset+info['compressed']]
                if values[2]&1:data=zipfile._ZipDecrypter(b'FaceGrabber@MMK')(data)[12:]
                if info['method']==8:data=zlib.decompress(data,-15)
                elif info['method']!=0:raise ValueError('Unsupported member compression')
                if len(data)!=info['size'] or binascii.crc32(data)!=info['crc']:raise ValueError('Member CRC mismatch')
                break
            except (requests.RequestException,ValueError,zlib.error):
                if attempt==2:raise
                time.sleep(1)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():path.rename(path.with_suffix(path.suffix+'.incomplete'))
        temporary=path.with_suffix(path.suffix+'.part');temporary.write_bytes(data);temporary.rename(path)
        return dict(kind=kind,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=len(data),crc_verified=True)
    receipts=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,result in enumerate(pool.map(member,jobs)):
            receipts.append(result)
            if (i+1)%50==0:print('members',i+1,'/',len(jobs),flush=True)
    prefix='secondary_' if all_depth else ''
    for kind in urls:save_json(ROOT/(prefix+kind+'_members.json'),[r for r in receipts if r['kind']==kind])
    save_json(ROOT/(prefix+'acquisition_complete.json'),dict(samples=len(rows),train_depth_samples=sum(r['split']=='train' for r in rows),
        val_test_3d_not_downloaded=not all_depth,val_test_HD2D_downloaded_but_not_used_by_RGB_preprocessing=True,
        selection_sha256=sha(ROOT/'selection.json'),members=len(receipts),all_member_CRCs_verified=True))
    print('FaceGrabber selected acquisition complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-depth',action='store_true');a=p.parse_args();acquire(a.all_depth)
