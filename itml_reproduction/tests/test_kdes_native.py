import unittest
import math
import numpy as np
from itml_repro.kdes_native import dense,source_gradients,encode,fit_codebook


class KDESNativeTests(unittest.TestCase):
    def test_source_gradients_and_dense_against_scalar_oracle(self):
        image=np.arange(256,dtype=float).reshape(16,16)/255
        g=np.array([[math.exp(-(x*x+y*y)/(2*.8**2)) for x in range(-2,3)] for y in range(-2,3)])
        g/=g.sum()
        kx=np.empty((5,5));ky=np.empty((5,5))
        for y in range(5):
            for x in range(5):
                kx[y,x]=g[y,min(x+1,4)]-g[y,max(x-1,0)]
                ky[y,x]=g[min(y+1,4),x]-g[max(y-1,0),x]
                if 0<x<4:kx[y,x]/=2
                if 0<y<4:ky[y,x]/=2
        kx*=2/np.abs(kx).sum();ky*=2/np.abs(ky).sum()
        dx=np.zeros_like(image);dy=np.zeros_like(image)
        for y in range(16):
            for x in range(16):
                for j in range(5):
                    for i in range(5):
                        yy,xx=y+j-2,x+i-2
                        if 0<=yy<16 and 0<=xx<16:
                            dx[y,x]+=image[yy,xx]*kx[j,i];dy[y,x]+=image[yy,xx]*ky[j,i]
        actual_x,actual_y=source_gradients(image)
        np.testing.assert_allclose(actual_x,dx,atol=1e-15);np.testing.assert_allclose(actual_y,dy,atol=1e-15)
        basis=dict(gpoints=np.array([[-1.,0.],[1.,0.]]),spoints=np.array([[0.,0.],[1.,1.]]),
                   eigvectors=np.eye(4),kparam=np.array([5.,5.,3.,3.]))
        feature=dense(image,basis)
        magnitude=np.maximum(np.hypot(dx,dy),1e-5);den=max(np.linalg.norm(magnitude),.8)
        reference=[]
        for sy,sx in basis['spoints']:
            for gx,gy in basis['gpoints']:
                total=0.
                for x in range(16):
                    for y in range(16):
                        direction=[dx[y,x]/magnitude[y,x],dy[y,x]/magnitude[y,x]]
                        total+=magnitude[y,x]/den*math.exp(-5*((direction[0]-gx)**2+(direction[1]-gy)**2))*math.exp(-3*((y/16-sy)**2+(x/16-sx)**2))
                reference.append(total)
        np.testing.assert_allclose(feature['descriptors'][0],reference,rtol=1e-12,atol=1e-14)
        np.testing.assert_array_equal(feature['x'],[8.5]);np.testing.assert_array_equal(feature['y'],[8.5])
        flat=dense(np.zeros((16,16)),basis)
        self.assertTrue(np.all(flat['descriptors']>0))  # native 1e-5 floor, deliberately preserved

    def test_emk_suppression_and_paper_output_dimensions(self):
        features=dict(descriptors=np.array([[0.,0.],[1.,0.],[10.,0.]]),x=np.array([8.5]*3),y=np.array([8.5]*3),width=16,height=16)
        words=np.array([[0.,0.],[10.,0.]])
        book=dict(words=words,G=np.eye(2),kparam=np.array(.001))
        expected=np.array([.5*(1+np.exp(-.1))]*2)
        np.testing.assert_allclose(encode(features,book,[1]),expected)
        big=dict(words=np.column_stack((np.arange(1000)/100,np.zeros(1000))),G=np.eye(1000),kparam=np.array(.001))
        self.assertEqual(encode(features,big,[1,2,4]).shape,(21000,))
        self.assertEqual(encode(features,big,[1,2,3]).shape,(14000,))

    def test_codebook_determinism_and_no_silent_word_reduction(self):
        rng=np.random.default_rng(2)
        feature=dict(descriptors=rng.normal(size=(12,3)))
        first=fit_codebook([feature],words=3,seed=9,iterations=5)
        again=fit_codebook([feature],words=3,seed=9,iterations=5)
        np.testing.assert_array_equal(first['words'],again['words'])
        from itml_repro.kdes_native import rbf
        K=rbf(first['words'],first['words'],.001)+1e-6*np.eye(3)
        np.testing.assert_allclose(first['G'].T@first['G'],np.linalg.inv(K),rtol=1e-10,atol=1e-10)
        with self.assertRaises(ValueError):fit_codebook([feature],words=1000)
