"""Gradient KDES Eq.12 patch projection with EXPLICIT supplied parameters.

This is a mathematical reference, not a replacement for the unavailable
author package/defaults. The original gradient operator, learned codebook
and image-level EMK/CKSVD parameters must be supplied/verified externally.
"""
import numpy as np


def dense_gradient_kdes(image, gradient_x, gradient_y, params):
    """16x16 patches, stride8, unit orientation Gaussian x position kernel.

    params: orientation_basis (O,2), position_basis (P,2),
    coefficients (O*P,D), gamma_orientation, gamma_position, epsilon.
    Gradients are supplied so no unreported SIFT/finite-difference choice
    is silently substituted. Input must already be grayscale in [0,1],
    maximum side <=300. Returns descriptors and patch-center positions.
    """
    image,dx,dy=(np.asarray(x,float) for x in (image,gradient_x,gradient_y))
    if (image.ndim!=2 or min(image.shape)<16 or max(image.shape)>300 or dx.shape!=image.shape
            or dy.shape!=image.shape or not all(np.isfinite(x).all() for x in (image,dx,dy))
            or image.min()<0 or image.max()>1):
        raise ValueError('Expected <=300 grayscale image in [0,1] and aligned finite gradients')
    ob,pb,alpha=(np.asarray(params[k],float) for k in ('orientation_basis','position_basis','coefficients'))
    go,gp,eps=(float(params[k]) for k in ('gamma_orientation','gamma_position','epsilon'))
    if (ob.ndim!=2 or ob.shape[1]!=2 or pb.ndim!=2 or pb.shape[1]!=2
            or alpha.ndim!=2 or alpha.shape[0]!=len(ob)*len(pb)
            or not all(np.isfinite(x).all() for x in (ob,pb,alpha))
            or not np.isfinite([go,gp,eps]).all() or min(go,gp,eps)<=0
            or not np.allclose(np.linalg.norm(ob,axis=1),1.) or np.any(pb<0) or np.any(pb>1)):
        raise ValueError('Invalid explicit kernel descriptor parameters')
    yy,xx=np.mgrid[0:16,0:16]
    positions=np.column_stack((xx.ravel(),yy.ravel()))/15
    kp=np.exp(-gp*np.sum((positions[:,None]-pb[None,:])**2,axis=2))
    descriptors,centers=[],[]
    for y in range(0,image.shape[0]-15,8):
        for x in range(0,image.shape[1]-15,8):
            gx,gy=dx[y:y+16,x:x+16].ravel(),dy[y:y+16,x:x+16].ravel()
            mag=np.hypot(gx,gy)
            direction=np.zeros((256,2));direction[:,1]=1
            valid=mag>0
            direction[valid]=np.column_stack((gy[valid],gx[valid]))/mag[valid,None]
            ko=np.exp(-go*np.sum((direction[:,None]-ob[None,:])**2,axis=2))
            weight=mag/np.sqrt(mag@mag+eps)
            feature=((ko.T*weight)@kp).ravel()@alpha
            descriptors.append(feature)
            centers.append([(x+7.5)/(image.shape[1]-1),(y+7.5)/(image.shape[0]-1)])
    return np.asarray(descriptors),np.asarray(centers)


def validate_kdes_archive(path, metadata, training_sample_ids):
    """Validate precomputed image-level KDES dimensionality/provenance contract.

    Shapes alone cannot prove that KDES/EMK was implemented correctly.
    This gate explicitly requires codebook source and fitting sample IDs.
    """
    import json
    from pathlib import Path
    meta=json.loads(Path(metadata).read_text())
    if meta.get('feature_name')!='gradient_KDES' or not meta.get('producer_sha256') or not meta.get('codebook_sha256'):
        raise ValueError('Missing KDES producer/codebook provenance')
    if meta.get('codebook_source')=='training_only':
        fitted=set(meta.get('codebook_training_sample_ids',[]))
        if not fitted or not fitted<=set(training_sample_ids):
            raise ValueError('Codebook fit includes nontraining data or has no fit IDs')
    elif meta.get('codebook_source')!='external_frozen':
        raise ValueError('Declare training_only or external_frozen codebook')
    with np.load(path,allow_pickle=False) as data:
        if data['X'].ndim!=2 or data['X'].shape[1]!=21000 or data['Z'].ndim!=2 or data['Z'].shape[1]!=14000:
            raise ValueError('Paper image-level KDES requires RGB21000/depth14000 dimensions')
    return meta
