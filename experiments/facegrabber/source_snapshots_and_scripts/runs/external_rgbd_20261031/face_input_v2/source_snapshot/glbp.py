"""Huynh et al. 2012 G-LBP, 123 bins/block; Xu 2015, 56 blocks.

Uses integer 3x3 grid neighbors (Fig.3), ascending-code U2 bins, valid
centers within each block, raw histogram counts. These boundary/order/
normalization conventions have not been checked against an author fixture.
Depth levels are preserved; no automatic conversion of depth to 8 bits.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform

OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
_codes=np.arange(256,dtype=np.uint16)
_bits=(_codes[:,None] >> np.arange(8)) & 1
_uniform=np.sum(_bits != np.roll(_bits,1,axis=1),axis=1)<=2
U2=np.full(256,58,dtype=np.uint8)
U2[_uniform]=np.arange(58)


def block_histogram(image, normalization='counts'):
    image=np.asarray(image)
    if (image.ndim!=2 or min(image.shape)<3 or not np.isfinite(image).all()
            or np.any(image != np.rint(image))):
        raise ValueError('G-LBP requires a finite integer-valued grayscale/depth block')
    a=image.astype(np.float64)
    center=a[1:-1,1:-1]
    h,w=center.shape
    differences=np.array([a[1+dy:1+dy+h,1+dx:1+dx+w]-center for dy,dx in OFFSETS])
    code=np.sum((differences>=0).astype(np.uint16)*(1 << np.arange(8))[:,None,None],axis=0)
    parts=[np.bincount(U2[code].ravel(),minlength=59)]
    # Eq.5: preserve the sign and clip to the sixteen levels -8,...,7.
    parts += [np.bincount((np.clip(d,-8,7)+8).astype(int).ravel(),minlength=16) for d in differences[:4]]
    result=np.concatenate(parts).astype(float)
    if normalization=='histogram_l1':
        result/=center.size
    elif normalization!='counts':
        raise ValueError('Unknown histogram normalization')
    return result


def glbp(image, normalization='counts'):
    """Prealigned 120x105 image -> 8x7 blocks of 15x15 -> 6888 dims."""
    image=np.asarray(image)
    if image.shape != (120,105):
        raise ValueError('Expected prealigned 120x105 input; no implicit crop/resize')
    return np.concatenate([block_histogram(image[y:y+15,x:x+15],normalization)
                           for y in range(0,120,15) for x in range(0,105,15)])


def align_face(image, source_eyes, target_eyes):
    """Two-eye similarity alignment to 120x105, explicit destination landmarks.

    Eye coordinates are (x,y). The original destination eye positions are
    not public; callers must provide and record their convention.
    Bilinear sampling, nearest border; round to native integer levels.
    """
    image=np.asarray(image,dtype=float)
    source,target=np.asarray(source_eyes,float),np.asarray(target_eyes,float)
    if (image.ndim!=2 or not np.isfinite(image).all() or source.shape!=(2,2)
            or target.shape!=(2,2) or not np.isfinite([source,target]).all()):
        raise ValueError('Invalid image or eye landmarks')
    ds,dt=source[1]-source[0],target[1]-target[0]
    if min(ds@ds,dt@dt)<=0:
        raise ValueError('Eye centers must differ')
    # Complex multiplication maps target coordinates back to source.
    ratio=complex(*ds)/complex(*dt)
    A=np.array([[ratio.real,-ratio.imag],[ratio.imag,ratio.real]])
    b=source[0]-A@target[0]
    swap=np.array([[0,1],[1,0]])
    return np.rint(affine_transform(image,swap@A@swap,offset=swap@b,
                    output_shape=(120,105),order=1,mode='nearest',prefilter=False))


def extract(path, modality):
    """Existing --extractor adapter; expects already aligned images on disk."""
    with Image.open(path) as im:
        if modality=='rgb':
            array=np.asarray(im.convert('L'))
        elif modality=='depth':
            array=np.asarray(im)
            if array.ndim!=2:
                raise ValueError('Depth must be a single native-depth channel')
        else:
            raise ValueError('Unknown modality')
    return glbp(array)
