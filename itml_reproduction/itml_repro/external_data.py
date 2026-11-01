"""Read official ranged ZIPs and FaceGrabber's binary PCD without full archives."""
import io
import time
import numpy as np
import requests


class HTTPFile(io.RawIOBase):
    """Seekable public HTTP file; validate every byte-range response."""
    def __init__(self,url):
        self.url=url;self.position=0;self.cache_start=0;self.cache=b'';self.session=requests.Session()
        response=self.fetch(0,0)
        self.size=int(response.headers['Content-Range'].split('/')[-1])

    def fetch(self,start,end):
        for attempt in range(3):
            try:
                r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,60))
                r.raise_for_status()
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('Server did not honor requested range')
                if len(r.content)!=end-start+1:raise ValueError('Truncated range')
                return r
            except (requests.RequestException,ValueError):
                if attempt==2:raise
                time.sleep(1)

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative seek')
        return self.position

    def read(self,size=-1):
        size=min(self.size-self.position,size if size>=0 else self.size)
        if size<=0:return b''
        start=self.position
        if not (self.cache_start<=start and start+size<=self.cache_start+len(self.cache)):
            end=min(self.size-1,start+max(size,65536)-1)
            self.cache=self.fetch(start,end).content;self.cache_start=start
        offset=start-self.cache_start;self.position+=size
        return self.cache[offset:offset+size]


def pcd(raw):
    """FaceGrabber PCL binary x/y/z + padded float RGB records; reject guesses."""
    marker=b'DATA binary\n'
    if marker not in raw:raise ValueError('Expected binary PCD')
    header,payload=raw.split(marker,1)
    fields={line.split()[0].decode():line.split()[1:] for line in header.splitlines() if line and not line.startswith(b'#')}
    if fields['FIELDS']!=[b'x',b'y',b'z',b'_',b'rgb',b'_'] or fields['COUNT']!=[b'1',b'1',b'1',b'4',b'1',b'12']:
        raise ValueError('Unexpected PCD layout')
    n=int(fields['POINTS'][0])
    if len(payload)!=32*n:raise ValueError('PCD byte count mismatch')
    data=np.frombuffer(payload,dtype=np.dtype(dict(names=['x','y','z','rgb'],
        formats=['<f4','<f4','<f4','<u4'],offsets=[0,4,8,16],itemsize=32)))
    xyz=np.column_stack([data[k] for k in ('x','y','z')]).astype(float)
    rgb=np.column_stack([(data['rgb']>>shift)&255 for shift in (16,8,0)]).astype('uint8')
    return xyz,rgb
