"""Python port of the pinned Bo/Ren MATLAB gradient-KDES + CKSVD encoder.

Derived from BSD-licensed code copyright (c) 2012 Liefeng Bo, University
of Washington, and Xiaofeng Ren, Intel Corporation. Full notice/terms:
third_party/kdes_matlab/BSDLicense.txt. Mirror commit and source files are
retained there. Not claimed bit-identical to an unavailable MATLAB runtime.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.signal import correlate2d
from scipy.sparse.linalg import eigsh
from scipy.io import loadmat
from .run import save_json,sha


def rbf(A,B,parameter):
    A,B=np.asarray(A,float),np.asarray(B,float)
    p=np.broadcast_to(np.asarray(parameter,float),(A.shape[1],))
    return np.exp(-(np.sum(A*A*p,axis=1)[:,None]+np.sum(B*B*p,axis=1)[None,:]-2*(A*p)@B.T))


def generate_basis(dimension=200,seed=2026):
    """gen_basis.m / gradkdes_basis.m: 100 square orientation x 25 positions.

    The mirrored 2012 implementation differs from the 2010 paper's circular
    25-point illustration. This function follows the PINNED SOURCE version.
    """
    if not 1<=dimension<2500:raise ValueError('Invalid KDES dimension')
    g=np.linspace(-1,1,10);s=np.linspace(0,1,5)
    gx,gy=np.meshgrid(g,g,indexing='ij');sx,sy=np.meshgrid(s,s,indexing='ij')
    gb=np.column_stack((gx.ravel(order='F'),gy.ravel(order='F')))
    sb=np.column_stack((sx.ravel(order='F'),sy.ravel(order='F')))
    K=np.kron(rbf(sb,sb,[3,3]),rbf(gb,gb,[5,5]))
    K=np.maximum(K,K.T)
    means=K.mean(axis=0)
    K=K-means[:,None]-means[None,:]+means.mean();K=np.maximum(K,K.T)
    values,vectors=eigsh(K,k=dimension,which='LA',tol=1e-10,v0=np.random.default_rng(seed).normal(size=len(K)))
    order=np.argsort(-values)
    # MATLAB kpca.m uses L2-normalized eigenvectors, WITHOUT eigenvalue whitening.
    return dict(gpoints=gb,spoints=sb,eigvectors=vectors[:,order].T,
                eigvalues=values[order],kparam=np.array([5.,5.,3.,3.]))


def load_basis(path):
    if Path(path).suffix=='.mat':
        data=loadmat(path,simplify_cells=True)['gradkdes_params']
        return dict(gpoints=np.asarray(data['gpoints']).T,spoints=np.asarray(data['spoints']).T,
                    eigvectors=np.asarray(data['eigvectors']),kparam=np.asarray(data['kparam']).ravel())
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def source_gradients(image):
    """gen_dgauss(.8), MATLAB filter2 'same' (zero-padded correlation)."""
    x=np.arange(-2,3,dtype=float);yy,xx=np.meshgrid(x,x,indexing='ij')
    G=np.exp(-(xx*xx+yy*yy)/(2*.8**2));G/=G.sum()
    gy,gx=np.gradient(G)
    gx*=2/np.abs(gx).sum();gy*=2/np.abs(gy).sum()
    return correlate2d(image,gx,mode='same'),correlate2d(image,gy,mode='same')


def dense(image,basis):
    """Already grayscale, <=300 side. uint8/16 im2double, floats unscaled."""
    image=np.asarray(image)
    if image.ndim!=2 or min(image.shape)<16 or max(image.shape)>300 or not np.isfinite(image).all():
        raise ValueError('Supply preprocessed grayscale/depth image, minimum16 maximum300')
    if np.issubdtype(image.dtype,np.unsignedinteger):
        image=image.astype(float)/np.iinfo(image.dtype).max
    else:image=image.astype(float)
    gb,sb,projection=(np.asarray(basis[k],float) for k in ('gpoints','spoints','eigvectors'))
    p=np.asarray(basis['kparam']).ravel()
    if (gb.ndim!=2 or sb.ndim!=2 or gb.shape[1]!=2 or sb.shape[1]!=2
            or projection.ndim!=2 or projection.shape[1]!=len(gb)*len(sb) or p.shape!=(4,)
            or not all(np.isfinite(a).all() for a in (gb,sb,projection,p))):raise ValueError('Invalid KDES basis')
    dx,dy=source_gradients(image)
    magnitude=np.maximum(np.hypot(dx,dy),1e-5)
    orient=np.stack((dx/magnitude,dy/magnitude),axis=2)
    orient_kernel=rbf(orient.reshape(-1,2),gb,p[:2]).reshape(*image.shape,len(gb))
    yy,xx=np.mgrid[:16,:16]
    positions=np.column_stack((yy.ravel(order='F'),xx.ravel(order='F')))/16
    spatial_kernel=rbf(sb,positions,p[2:])
    xstarts=np.arange(((image.shape[1]-16)%8)//2,image.shape[1]-15,8)
    ystarts=np.arange(((image.shape[0]-16)%8)//2,image.shape[0]-15,8)
    descriptors,xs,ys=[],[],[]
    for x in xstarts:
        for y in ystarts:
            weight=magnitude[y:y+16,x:x+16].ravel(order='F')
            weight=weight/max(np.linalg.norm(weight),.8)
            # MATLAB reshape uses column-major pixels independently per channel.
            kernels=np.column_stack([orient_kernel[y:y+16,x:x+16,j].ravel(order='F') for j in range(len(gb))])
            joint=((spatial_kernel*weight)@kernels).T
            descriptors.append(projection@joint.ravel(order='F'))
            xs.append(float(x+8.5));ys.append(float(y+8.5))
    return dict(descriptors=np.asarray(descriptors),x=np.array(xs),y=np.array(ys),
                width=image.shape[1],height=image.shape[0])


def fit_codebook(feature_sets,words=1000,seed=2026,samples_per_image=10,iterations=50):
    """sample_kdes + kmeans_bo: random initialization, empty centers retained.

    Uses NumPy RNG (explicit difference from MATLAB randperm/randsample).
    Caller supplies TRAINING images only. Word count is never reduced.
    """
    rng=np.random.default_rng(seed)
    samples=np.concatenate([f['descriptors'][rng.choice(len(f['descriptors']),min(samples_per_image,len(f['descriptors'])),replace=False)] for f in feature_sets])
    if len(samples)<words:raise ValueError('Not enough training patches for requested visual words')
    centers=samples[rng.permutation(len(samples))[:words]].copy()
    for _ in range(iterations):
        sums=np.zeros_like(centers);counts=np.zeros(words,dtype=int)
        for offset in range(0,len(samples),2048):
            A=samples[offset:offset+2048]
            d=np.sum(A*A,axis=1)[:,None]+np.sum(centers*centers,axis=1)[None,:]-2*A@centers.T
            group=np.argmin(d,axis=1)
            np.add.at(sums,group,A);np.add.at(counts,group,1)
        valid=counts>0;centers[valid]=sums[valid]/counts[valid,None]
    K=rbf(centers,centers,.001)+1e-6*np.eye(words)
    G=np.linalg.cholesky(np.linalg.inv(K)).T
    return dict(words=centers,G=G,kparam=np.array(.001))


def encode(features,codebook,pyramid):
    """cksvd_emk.m: per-word max suppression, mean kernel, G, level weights."""
    words,G=codebook['words'],codebook['G']
    K=rbf(features['descriptors'],words,codebook['kparam'])
    pyramid=np.asarray(pyramid,dtype=int)
    if np.any(pyramid<1):raise ValueError('Invalid pyramid')
    weights=1/pyramid.astype(float)**2;weights/=weights.sum()
    output=[]
    for n,weight in zip(pyramid,weights):
        cellx=np.ceil(features['x']/(features['width']/n)).astype(int)
        celly=np.ceil(features['y']/(features['height']/n)).astype(int)
        cell=(celly-1)*n+cellx
        for index in range(1,n*n+1):
            values=K[cell==index]
            if not len(values):output.append(np.zeros(len(words)));continue
            best_word=np.argmax(values,axis=1)
            confidence=np.max(values,axis=1)
            order=np.argsort(-confidence,kind='stable')
            # MATLAB unique sorts word IDs; keep best patch for each assigned word.
            selected=[order[np.flatnonzero(best_word[order]==word)[0]] for word in np.unique(best_word)]
            output.append(weight*(G@values[selected].mean(axis=0)))
    return np.concatenate(output)


def extract_archive(manifest,training_manifest,out,depth_unit,parameters=None):
    """Generate image-level RGB21000/depth14000 without fitting on test data."""
    if depth_unit not in ('native_im2double','millimeters_to_meters') or out.suffix!='.npz':
        raise ValueError('Declare depth units and use a .npz output path')
    with manifest.open(newline='') as f:rows=list(csv.DictReader(f))
    with training_manifest.open(newline='') as f:train_rows=list(csv.DictReader(f))
    ids=[r['sample_id'] for r in rows];train_ids={r['sample_id'] for r in train_rows}
    if len(set(ids))!=len(ids) or len(train_ids)!=len(train_rows) or not train_ids or not train_ids<=set(ids):
        raise ValueError('Invalid training sample IDs')
    artifacts=[out,out.with_suffix('.json'),out.with_name(out.stem+'_basis.npz'),out.with_name(out.stem+'_codebooks.npz')]
    if any(p.exists() for p in artifacts):raise FileExistsError('Refusing to overwrite feature artifacts')
    basis=generate_basis() if parameters is None else load_basis(parameters)
    if basis['eigvectors'].shape[0]!=200:raise ValueError('Paper pipeline requires 200-dimensional local KDES')
    features={};books={}
    for modality in ('rgb','depth'):
        data=[];selected=[]
        for i,row in enumerate(rows):
            if modality=='depth' and row['sample_id'] not in train_ids:continue
            path=manifest.parent/row[modality+'_path']
            with Image.open(path) as im:
                image=np.asarray(im.convert('L')) if modality=='rgb' else np.asarray(im)
            if modality=='depth' and depth_unit=='millimeters_to_meters':image=image.astype(float)/1000
            data.append(dense(image,basis));selected.append(i)
            if len(data)%100==0:print('KDES local',modality,len(data),flush=True)
        training=[f for f,i in zip(data,selected) if ids[i] in train_ids]
        book=fit_codebook(training)
        print('KDES codebook fitted',modality,'training images',len(training),flush=True)
        books[modality]=book
        dimension=21000 if modality=='rgb' else 14000
        matrix=np.full((len(rows),dimension),np.nan)
        for n,(f,i) in enumerate(zip(data,selected)):
            matrix[i]=encode(f,book,[1,2,4] if modality=='rgb' else [1,2,3])
            if (n+1)%100==0:print('KDES encoded',modality,n+1,flush=True)
        features['X' if modality=='rgb' else 'Z']=matrix
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,sample_ids=np.array(ids),**features)
    codebook_path=out.with_name(out.stem+'_codebooks.npz')
    np.savez(codebook_path,**{mod+'_'+k:a for mod,b in books.items() for k,a in b.items()})
    np.savez(out.with_name(out.stem+'_basis.npz'),**basis)
    save_json(out.with_suffix('.json'),dict(feature_name='gradient_KDES',producer_sha256=sha(Path(__file__)),
        codebook_sha256=sha(codebook_path),codebook_source='training_only',codebook_training_sample_ids=sorted(train_ids),
        source_mirror_commit='7973cb37319449c1dc76593a2973412d72ce5bc2',depth_unit=depth_unit,
        manifest_sha256=sha(manifest),training_manifest_sha256=sha(training_manifest),
        preprocessing='pre-cropped <=300 images; RGB via PIL L, depth units explicit; no implicit resize',
        scope='Pinned-source Python port; MATLAB runtime parity and 2015 author-package version unverified'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--training-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--parameters',type=Path)
    p.add_argument('--depth-unit',choices=['native_im2double','millimeters_to_meters'],required=True)
    a=p.parse_args();extract_archive(a.manifest,a.training_manifest,a.output,a.depth_unit,a.parameters)
