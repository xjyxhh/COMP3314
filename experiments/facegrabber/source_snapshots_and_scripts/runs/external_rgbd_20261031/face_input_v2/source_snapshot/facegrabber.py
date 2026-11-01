"""Alternative RGB-D face study; ordinary features never use 3-D geometry.

Depth is rasterized from recorded FaceRaw SENSOR points (not HD model z).
Corresponding HD3D/HD2D vertices estimate the per-frame color projection.
Eye indices: Kinect/Docs commit 553f7c37, HighDetailFacePoints enumeration.
This conversion/target-eye convention is an explicit deviation from Xu.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import griddata
from .external_data import pcd
from .glbp import align_face,glbp
from .pca import fit_pca,transform
from .data import make_pairs
from .run import save_json,sha

EYE_INDICES=((210,469),(843,1117))
TARGET=np.array([[30.,42.],[75.,42.]])

def rgb_eyes(gray):
    """Pinned YuNet RGB-only eye localization; no Kinect geometry is read."""
    gray=np.asarray(gray,dtype='uint8');scale=min(1.,960/gray.shape[1])
    small=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    image=cv2.cvtColor(small,cv2.COLOR_GRAY2BGR)
    model=Path(__file__).resolve().parents[1]/'third_party/yunet/face_detection_yunet_2023mar.onnx'
    if sha(model)!='8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4':
        raise ValueError('Face detector model hash mismatch')
    detector=cv2.FaceDetectorYN.create(str(model),'',(small.shape[1],small.shape[0]),.8,.3,5000)
    _,faces=detector.detect(image)
    if faces is None or not len(faces):raise ValueError('RGB-only YuNet found no face')
    face=max(faces,key=lambda b:float(b[2])*float(b[3]))
    eye=face[4:8].reshape(2,2)/scale
    return eye[np.argsort(eye[:,0])],'RGB_only_YuNet_2023mar'


def eyes(points):
    if len(points)!=1347 or not np.isfinite(points).all():raise ValueError('Expected all 1347 HD vertices')
    result=np.array([points[list(pair),:2].mean(0) for pair in EYE_INDICES])
    return result[np.argsort(result[:,0])]

def camera(world,pixels):
    """Normalized DLT, fit 4/5 vertices; hold out every fifth for calibration QA."""
    world,pixels=np.asarray(world,float),np.asarray(pixels,float)
    if world.ndim!=2 or world.shape[1]!=3 or pixels.shape!=(len(world),2) or len(world)<20:
        raise ValueError('Invalid camera correspondences')
    train=np.arange(len(world))%5!=0
    def normalize(a,target):
        mean=a.mean(0);scale=target/np.sqrt(np.mean(np.sum((a-mean)**2,axis=1)))
        T=np.eye(a.shape[1]+1);T[:-1,:-1]*=scale;T[:-1,-1]=-scale*mean
        return T
    T3,T2=normalize(world[train],np.sqrt(3)),normalize(pixels[train],np.sqrt(2))
    a=np.column_stack([world,np.ones(len(world))])@T3.T
    b=np.column_stack([pixels,np.ones(len(world))])@T2.T
    zero=np.zeros_like(a);u,v=b[:,0,None],b[:,1,None]
    A=np.vstack([np.column_stack([a,zero,-u*a])[train],np.column_stack([zero,a,-v*a])[train]])
    _,_,V=np.linalg.svd(A,full_matrices=False)
    P=np.linalg.inv(T2)@V[-1].reshape(3,4)@T3
    prediction=project(world,P)
    error=np.linalg.norm(prediction[~train]-pixels[~train],axis=1)
    rms=float(np.sqrt(np.mean(error**2)))
    if not np.isfinite(rms) or rms>2:raise ValueError(f'Projection holdout RMS too large: {rms}')
    return P,rms

def project(points,P):
    a=np.column_stack([points,np.ones(len(points))])@P.T
    return a[:,:2]/a[:,2,None]

def depth_image(points,pixels,source_eyes):
    """Linear interpolation of true sensor z in the face convex hull, millimeters."""
    source_eyes=np.asarray(source_eyes)
    delta=source_eyes[1]-source_eyes[0];target=TARGET[1]-TARGET[0]
    ratio=complex(*target)/complex(*delta)
    A=np.array([[ratio.real,-ratio.imag],[ratio.imag,ratio.real]])
    xy=(pixels-source_eyes[0])@A.T+TARGET[0]
    valid=np.isfinite(points).all(1)&np.isfinite(xy).all(1)&(points[:,2]>0)
    xy,z=xy[valid],points[valid,2]*1000
    # Same projected location: retain the nearer observed sensor point.
    order=np.argsort(z,kind='stable');xy,z=xy[order],z[order]
    _,keep=np.unique(np.round(xy,6),axis=0,return_index=True);xy,z=xy[keep],z[keep]
    yy,xx=np.mgrid[:120,:105]
    values=griddata(xy,z,(xx,yy),method='linear',fill_value=0.)
    if np.mean(values>0)<.2:raise ValueError('Insufficient face depth coverage')
    return np.rint(values).astype('uint16')

def prepare(source,out):
    cv2.setNumThreads(1)
    acquisition=json.loads((source/'acquisition_complete.json').read_text())
    selection=json.loads((source/'selection.json').read_text());rows=selection['rows']
    if acquisition['samples']!=560:raise ValueError('Expected frozen 40 x 14 subset')
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',dict(scope='FaceGrabber real sensor-depth alternative face verification; not EURECOM/Curtin reproduction',
        seed=2026,rounds=5,identity_split=[20,10,10],shots=14,pca_dimensions=150,
        ordinary='Native JPEG grayscale, RGB-only pretrained YuNet2023mar eye points; target [30,42],[75,42]; no HD2D/3D/depth access for val/test',
        privileged='Recorded FaceRaw sensor z in meters converted to millimeters; per-frame DLT from HD3D/HD2D, linear face-hull raster at120x105; not model z',
        feature='G-LBP6888 histogram_l1, train-only PCA150 independently per modality',
        depth_conversion_deviation='point-cloud projection and interpolation rather than native depth PNG; original target eyes/conventions unavailable',
        source_selection_sha256=sha(source/'selection.json'),primary_metric='validation AP; frozen test AP/AUC',
        prior_test_exposure=False,test_evaluated=False))
    X=[];Z=np.full((len(rows),6888),np.nan);checks=[];manifest=[]
    for i,row in enumerate(rows):
        paths={k:source/'selected'/v for k,v in row['sources'].items()}
        with Image.open(paths['rgb']) as im:gray=np.asarray(im.convert('L'))
        eye,eye_method=rgb_eyes(gray);ordinary=align_face(gray,eye,TARGET)
        X.append(glbp(ordinary,'histogram_l1'))
        item=dict(sample_id=row['sample_id'],identity=row['identity'],sequence=row['sequence'],frame=row['frame'],
                  rgb_path=str(paths['rgb'].resolve()),depth_path='',eye_method=eye_method)
        if row['split']=='train':
            landmarks,_=pcd(paths['landmarks2d'].read_bytes())
            model,_=pcd(paths['landmarks3d'].read_bytes());P,rms=camera(model,landmarks[:,:2])
            points,_=pcd(paths['facecloud'].read_bytes());depth=depth_image(points,project(points,P),eye)
            Z[i]=glbp(depth,'histogram_l1');checks.append(dict(sample_id=row['sample_id'],projection_holdout_rms=rms,depth_coverage=float(np.mean(depth>0))))
            if len(checks)<=2:
                Image.fromarray(ordinary.astype('uint8')).save(out/f'qa_rgb_{len(checks)}.png')
                Image.fromarray(depth).save(out/f'qa_depth_{len(checks)}.png')
        manifest.append(item)
        if i%70==0:print('Face features',i+1,'/',len(rows),flush=True)
    X=np.asarray(X);split={n:np.array([i for i,r in enumerate(rows) if r['split']==n]) for n in ('train','val','test')}
    for name,indices in split.items():
        with (out/f'{name}_manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest[i] for i in indices)
    np.savez(out/'raw_glbp.npz',X=X,Z=Z,sample_ids=np.array([r['sample_id'] for r in rows]))
    np.savez(out/'split.npz',**split);save_json(out/'projection_checks.json',checks)
    px,pz=fit_pca(X[split['train']],150),fit_pca(Z[split['train']],150)
    np.savez(out/'pca.npz',rgb_mean=px[0],rgb_components=px[1],depth_mean=pz[0],depth_components=pz[1])
    xs={n:transform(X[idx],px) for n,idx in split.items()}
    np.savez(out/'features_pca.npz',**{'X_'+n:a for n,a in xs.items()},Z_train=transform(Z[split['train']],pz))
    a,b=np.triu_indices(len(xs['train']),1);u,l=np.quantile(np.linalg.norm(xs['train'][a]-xs['train'][b],axis=1),[.03,.97])
    save_json(out/'bounds.json',dict(u=float(u),l=float(l)))
    for n in range(5):
        folder=out/f'round_{n:02d}';folder.mkdir();pairs={}
        for k,name in enumerate(split):
            counts=(1820,5872) if name=='train' else (910,2936)
            pairs[name]=make_pairs([manifest[i]['identity'] for i in split[name]],*counts,2026+1000*n+k,positive_seed=9000+k)
        np.savez(folder/'pairs.npz',**pairs);np.save(folder/'xi0.npy',np.where(pairs['train'][:,2]==1,u,l))
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/baselines_paper.json').read_text())
    config.update(lmnn_backend='projected',lmnn_max_iterations=1000,lmnn_tolerance=1e-7)
    save_json(out/'baseline_config.json',config)
    print('FaceGrabber preparation complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.output)
