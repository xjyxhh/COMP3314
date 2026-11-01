import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from itml_repro.olivetti import image_features, evaluate


class OlivettiTests(unittest.TestCase):
    def test_feature_diagnostic_ignores_test_features(self):
        from itml_repro.diagnose_olivetti import feature_diagnostic
        from itml_repro.pca import fit_pca, transform
        rng=np.random.default_rng(9)
        x,z=rng.normal(size=(40,20)),rng.normal(size=(40,20))
        train,val,test=np.arange(20),np.arange(20,30),np.arange(30,40)
        px,pz=fit_pca(x[train],16),fit_pca(z[train],16)
        xx,zz=transform(x,px),transform(z,pz)
        x[test]=np.nan;z[test]=np.nan
        pairs=np.array([[0,1,1],[2,3,1],[4,5,-1],[6,7,-1]])
        with tempfile.TemporaryDirectory() as tmp:
            source,out=Path(tmp)/'source',Path(tmp)/'out'
            source.mkdir();out.mkdir()
            np.savez(source/'image_features.npz',X=x,Z=z)
            for index in range(3):
                inputs=source/f'split_{index:02d}'/'input'
                (inputs/'round_00').mkdir(parents=True)
                np.savez(inputs/'split.npz',train=train,val=val,test=test)
                np.savez(inputs/'pca.npz',Z_mean=pz[0],Z_components=pz[1])
                np.savez(inputs/'features_pca.npz',X_train=xx[train],X_val=xx[val],Z_train=zz[train])
                np.savez(inputs/'round_00/pairs.npz',train=pairs,val=pairs)
            with patch('builtins.print'):
                results=feature_diagnostic(source,out)
            self.assertEqual(len(results),6)
            self.assertTrue(all(np.isfinite(r['euclidean']['AP']) for r in results))

    def test_checkpoints_preserve_solver_trajectory(self):
        from itml_repro.metrics import fit
        from itml_repro.corrected import fit_corrected
        rng=np.random.default_rng(17)
        X,Z=rng.normal(size=(8,2)),rng.normal(size=(8,2))
        i,j=np.triu_indices(8,1)
        r,s=np.sum((X[i]-X[j])**2,axis=1),np.sum((Z[i]-Z[j])**2,axis=1)
        pairs=np.column_stack((i,j,np.where(r<s,1,-1)))
        xi=np.where(pairs[:,2]==1,.5,4.)
        for solver in (fit,fit_corrected):
            checkpoints={}
            def capture(sweep,state):
                if sweep==5:
                    checkpoints[sweep]={k:state[k].copy() for k in ('M','P','xi')}
            final=solver(X,pairs,xi,.1,Z=Z,max_sweeps=10,tol=0.,on_sweep=capture)
            prefix=solver(X,pairs,xi,.1,Z=Z,max_sweeps=5,tol=0.)
            plain=solver(X,pairs,xi,.1,Z=Z,max_sweeps=10,tol=0.)
            for key in ('M','P','xi'):
                np.testing.assert_array_equal(checkpoints[5][key],prefix[key])
                np.testing.assert_array_equal(final[key],plain[key])

    def test_no_test_evaluation_before_all_searches_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with self.assertRaisesRegex(ValueError,'Incomplete validation search'):
                evaluate(out)
            self.assertFalse((out/'results.json').exists())

    def test_clean_privilege_and_fixed_noise(self):
        im = Image.fromarray(np.full((112,92),128,dtype=np.uint8))
        x,z = image_features(im,41)
        self.assertEqual(x.shape,(64,))
        self.assertEqual(z.shape,(1024,))
        np.testing.assert_allclose(z,128/255)
        repeated, clean = image_features(im,41)
        np.testing.assert_array_equal(x,repeated)
        np.testing.assert_array_equal(z,clean)
        other, clean = image_features(im,42)
        self.assertFalse(np.array_equal(x,other))
        np.testing.assert_array_equal(z,clean)
        self.assertGreater(np.std(x),.02)
        self.assertTrue(np.all((x>=0)&(x<=1)))


if __name__=='__main__':
    unittest.main()
