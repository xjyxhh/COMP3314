import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HAS_CVXPY = importlib.util.find_spec('cvxpy') is not None


@unittest.skipUnless(HAS_CVXPY,'Optional baseline CVXPY dependency is not installed')
class BaselineTests(unittest.TestCase):
    def test_pair_representation_and_inference_use_no_privileged_data(self):
        from itml_repro.baselines import pair_features,decision_scores
        X=np.array([[0.,2.],[2.,0.]])
        pairs=np.array([[0,1,1]])
        g=np.exp(-.5)/np.sqrt(2*np.pi)
        np.testing.assert_allclose(pair_features(X,pairs),[[2*g,2*g,0,0]])
        np.testing.assert_array_equal(pair_features(X,pairs),pair_features(X,np.array([[1,0,1]])))
        model=dict(w=np.ones(4),b=np.array(.3),w_privileged=np.array([np.nan]),b_privileged=np.array(np.nan))
        self.assertAlmostEqual(decision_scores(X,pairs,model)[0],4*g+.3)

    def test_svm_and_svm_plus_against_independent_dual(self):
        import cvxpy as cp
        from itml_repro.baselines import fit_svm,pair_features
        from itml_repro.data import make_pairs
        rng=np.random.default_rng(31)
        X,Z=rng.normal(size=(8,2)),rng.normal(size=(8,3))
        pairs=make_pairs(np.repeat(np.arange(4),2),4,8,17)
        y=pairs[:,2];F=pair_features(X,pairs)
        C,gamma=.7,1.3
        for privileged in (None,Z):
            model=fit_svm(X,pairs,C,Z=privileged,gamma=gamma)
            a=cp.Variable(len(y))
            constraints=[a>=0,y@a==0]
            obj=cp.sum(a)-.5*cp.sum_squares(F.T@cp.multiply(y,a))
            if privileged is None:
                constraints.append(a<=C)
            else:
                beta=cp.Variable(len(y))
                G=pair_features(Z,pairs)
                constraints.extend([beta>=0,cp.sum(a+beta-C)==0])
                obj-=cp.sum_squares(G.T@(a+beta-C))/(2*gamma)
            problem=cp.Problem(cp.Maximize(obj),constraints)
            problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
            self.assertEqual(problem.status,'optimal')
            self.assertAlmostEqual(model['diagnostics']['objective'],problem.value,places=6)
            np.testing.assert_allclose(model['w'],F.T@(y*a.value),atol=2e-5)
            for key,value in model['diagnostics'].items():
                if key!='objective':
                    self.assertLess(value,2e-5,key)
            if privileged is not None:
                np.testing.assert_allclose(model['xi'],G@model['w_privileged']+model['b_privileged'],atol=1e-12)

    def test_lmnn_scalar_optimum_targets_and_limit(self):
        from itml_repro.baselines import fit_lmnn,target_neighbors
        X=np.array([[0.],[.1],[1.],[1.1]])
        labels=np.array([0,0,1,1])
        np.testing.assert_array_equal(target_neighbors(X,labels,1),[[1],[0],[3],[2]])
        model=fit_lmnn(X,labels,.5,k=1)
        # All eight impostor hinge terms vanish at m=1/(.9²-.1²).
        self.assertAlmostEqual(model['M'][0,0],1.25,places=5)
        self.assertAlmostEqual(model['diagnostics']['objective'],.025,places=6)
        self.assertEqual(model['diagnostics']['triplets'],8)
        self.assertLess(model['diagnostics']['stationarity_M'],1e-6)
        self.assertLess(model['diagnostics']['complementarity'],1e-6)
        with self.assertRaises(ValueError):
            fit_lmnn(X,labels,.5,k=1,max_triplets=7)
        with self.assertRaises(ValueError):
            target_neighbors(X,labels,5)

    def test_itml_s_filter_preserves_order_and_shared_gamma(self):
        from itml_repro.baselines import fit_itml_s
        pairs=np.array([[0,1,1],[0,2,-1],[1,3,1],[1,2,-1]])
        X=np.array([[0.],[.1],[2.],[2.1]])
        xi0=np.array([.02,1.,.02,1.])
        calls=[]
        def fake_fit(A,p,x,gamma,**kwargs):
            calls.append((p.copy(),x.copy(),gamma,kwargs))
            return dict(M=np.eye(1),history=[],stop='fixed_fixture')
        with patch('itml_repro.baselines.fit',side_effect=fake_fit):
            result=fit_itml_s(X,pairs,xi0,.3,Z=X,filter_rule='shared_xi0')
        np.testing.assert_array_equal(result['keep_mask'],[True,True,False,True])
        np.testing.assert_array_equal(calls[1][0],pairs[[0,1,3]])
        np.testing.assert_array_equal(calls[1][1],xi0[[0,1,3]])
        self.assertEqual([c[2] for c in calls],[.3,.3])
        self.assertEqual(result['diagnostics']['removed'],1)
        with self.assertRaises(ValueError):
            fit_itml_s(X,pairs,xi0,.3,Z=X,filter_rule='unknown')

    def test_common_score_metrics_and_validation_runner_without_test_inputs(self):
        from itml_repro.baseline_compare import run,METHODS
        from itml_repro.data import make_pairs
        from itml_repro.evaluation import verification_scores
        self.assertEqual(verification_scores([0.,0.,0.,0.],[1,-1,1,-1]),dict(AP=.5,AUC=.5))
        rng=np.random.default_rng(31)
        labels=np.repeat(np.arange(3),6)
        X=np.array([[-1.,-1.],[0.,1.],[1.,-1.]])[labels]+.02*rng.normal(size=(18,2))
        pairs=make_pairs(labels,12,16,3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input/round_00').mkdir(parents=True)
            np.savez(root/'input/features_pca.npz',X_train=X,Z_train=X,X_val=X)
            np.savez(root/'input/round_00/pairs.npz',train=pairs,val=pairs)
            np.save(root/'input/round_00/xi0.npy',np.where(pairs[:,2]==1,.2,2.5))
            (root/'input/train_manifest.csv').write_text('identity\n'+'\n'.join(map(str,labels))+'\n')
            config=dict(methods=METHODS,gamma_grid=[.1],lambda_grid=[1.],C_grid=[1.],
                svm_plus_gamma_grid=[1.],svm_kernel='linear',lmnn_mu_grid=[.5],lmnn_k=5,
                itml_s_filter_rule='shared_xi0',max_sweeps=3,tolerance=0.,selection_metric='AP')
            (root/'config.json').write_text(json.dumps(config))
            records=run(Namespace(input=root/'input',output=root/'out',config=root/'config.json'))
            self.assertEqual([r['method'] for r in records],METHODS)
            self.assertTrue(all(r['status']=='completed' and 'test' not in r for r in records))
            self.assertEqual(json.loads((root/'out/integrity.json').read_text()),dict(source_unchanged=True,test_evaluated=False))
            with np.load(root/'out/round_00/svm_plus_model.npz') as model:
                self.assertIn('w',model.files)
                self.assertIn('w_privileged',model.files)
            with self.assertRaises(FileExistsError):
                run(Namespace(input=root/'input',output=root/'out',config=root/'config.json'))


if __name__=='__main__':
    unittest.main()
