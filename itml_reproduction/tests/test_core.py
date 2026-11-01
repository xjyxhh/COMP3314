"""Run: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from itml_repro.metrics import plus_step, fit, distances, burg, logdet
from itml_repro.evaluation import verification
from itml_repro.data import make_pairs, split_identities
from itml_repro.pca import fit_pca, transform


class CoreTests(unittest.TestCase):
    def test_feasibility_audit_known_cases(self):
        import importlib.util
        if importlib.util.find_spec('cvxpy') is None:
            self.skipTest('Optional CVXPY is not installed')
        import tempfile
        from pathlib import Path
        from itml_repro.audit_washington import feasibility
        X = np.array([[0.], [2.], [1.]])
        Z = np.array([[0.], [1.], [1.]])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            bad = feasibility(X, Z, np.array([[0, 1, 1], [0, 2, -1]]), folder)
            self.assertEqual(bad['classification'], 'solver_reported_infeasible')
            self.assertEqual(bad['status'], 'infeasible')
            good = feasibility(X, Z, np.array([[0, 1, -1], [0, 2, 1]]), folder)
            self.assertEqual(good['classification'], 'numerically_spd_feasible')

    def test_corrected_multisweep_stationarity(self):
        from itml_repro.corrected import fit_corrected
        rng = np.random.default_rng(17)
        X, Z = rng.normal(size=(8, 2)), rng.normal(size=(8, 2))
        i, j = np.triu_indices(8, 1)
        r, s = ((X[i]-X[j])**2).sum(1), ((Z[i]-Z[j])**2).sum(1)
        pairs = np.column_stack((i, j, np.where(r < s, 1, -1)))
        xi0 = np.where(pairs[:, 2] == 1, .5, 4.)
        result = fit_corrected(X, pairs, xi0, .1, Z, max_sweeps=100, tol=1e-7)
        self.assertEqual(result['stop'], 'kkt_tolerance')
        for row in result['history']:
            for key in ('stationarity_M', 'stationarity_P', 'stationarity_xi',
                        'dual_nonnegative_violation'):
                self.assertLess(row[key], 1e-10)
        self.assertLess(result['history'][-1]['relative_equality'], 1e-7)

    def test_shared_entry_matches_direct_solvers(self):
        import json
        import tempfile
        from argparse import Namespace
        from pathlib import Path
        from itml_repro.demo import generate
        from itml_repro.run import run
        from itml_repro.corrected import fit_corrected
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            generate(root/'data')
            config=json.loads((Path(__file__).resolve().parents[1]/'configs/compare.json').read_text())
            config.update(rounds=1,max_sweeps=2)
            (root/'config.json').write_text(json.dumps(config))
            run(Namespace(config=root/'config.json',manifest=root/'data/manifest.csv',
                          features=root/'data/features.npz',extractor=None,output=root/'out'))
            folder=root/'out/round_00'
            with np.load(root/'out/features_pca.npz') as f:
                X,Z=f['X_train'],f['Z_train']
            with np.load(folder/'pairs.npz') as f:
                pairs=f['train']
            xi=np.load(folder/'xi0.npy')
            for name,solver in [('itml_plus',fit),('itml_plus_corrected',fit_corrected)]:
                direct=solver(X,pairs,xi,.1,Z=Z,lam=1.,max_sweeps=2,tol=config['tolerance'])
                with np.load(folder/f'{name}_model.npz') as f:
                    np.testing.assert_allclose(f['M'],direct['M'])
                    np.testing.assert_allclose(f['xi'],direct['xi'])
                record=json.loads((folder/f'{name}.json').read_text())
                self.assertEqual(record['result']['selected']['sweeps'],2)
                self.assertIn('relative_equality_residual',record['result']['selected']['diagnostics'])
            self.assertTrue((root/'out/comparison.csv').exists())

    def test_plus_diagnostics_and_fixed_budget_prefix(self):
        rng = np.random.default_rng(21)
        X, Z = rng.normal(size=(8, 3)), rng.normal(size=(8, 2))
        pairs = make_pairs(np.repeat(np.arange(4), 2), 4, 8, 9)
        xi0 = np.where(pairs[:, 2] == 1, .5, 4.)
        short = fit(X, pairs, xi0, .1, Z=Z, max_sweeps=2, tol=0)
        long = fit(X, pairs, xi0, .1, Z=Z, max_sweeps=5, tol=0)
        self.assertEqual(short['history'], long['history'][:2])
        row = short['history'][-1]
        r, s = distances(X, pairs, short['M']), distances(Z, pairs, short['P'])
        violation = np.maximum(pairs[:, 2] * (r-s), 0)
        self.assertAlmostEqual(row['relative_inequality_violation'],
                               np.max(violation / np.maximum(r, s)))
        # A common scale contraction can hide absolute errors, not relative ones.
        self.assertAlmostEqual(row['relative_inequality_violation'],
                               np.max((violation * 1e-20) / np.maximum(r * 1e-20, s * 1e-20)))
        self.assertAlmostEqual(row['relative_equality_residual'],
                               np.max(np.abs(short['xi']-s) / np.maximum(short['xi'], s)))
        self.assertAlmostEqual(row['trace_M'], np.linalg.eigvalsh(short['M']).sum())
        self.assertAlmostEqual(row['max_eigenvalue_P'], np.linalg.eigvalsh(short['P'])[-1])

    def test_plus_projection_kkt_and_inverse_updates(self):
        rng=np.random.default_rng(1)
        for y in (-1,1):
            for xi in (.02,1.,100.):
                for _ in range(10):
                    A=rng.normal(size=(3,3));M=A@A.T+np.eye(3)
                    A=rng.normal(size=(2,2));P=A@A.T+np.eye(2)
                    v,w=rng.normal(size=3),rng.normal(size=2)
                    gamma,lam=.3,2.
                    N,Q,t,a,b=plus_step(M,P,xi,v,w,y,gamma,lam)
                    np.testing.assert_allclose(np.linalg.inv(N),np.linalg.inv(M)+y*a*np.outer(v,v),rtol=1e-9,atol=1e-9)
                    np.testing.assert_allclose(lam*np.linalg.inv(Q),lam*np.linalg.inv(P)-b*np.outer(w,w),rtol=1e-9,atol=1e-9)
                    self.assertAlmostEqual(t,w@Q@w,places=9)
                    self.assertLessEqual(y*(v@N@v-t),1e-8)
                    self.assertAlmostEqual(a*y*(v@N@v-t),0.,places=8)
                    self.assertAlmostEqual(gamma/t,gamma/xi-y*a+b,places=8)
                    np.linalg.cholesky(N);np.linalg.cholesky(Q)

    def test_itml_scalar_analytic_optimum(self):
        # One active constraint: exact minimizer m=(1+gamma)/(1+gamma*r/xi0).
        for y,xi0 in [(1,.2),(-1,8.)]:
            X=np.array([[0.],[2.]])
            pairs=np.array([[0,1,y]])
            gamma=.7;r=4.
            model=fit(X,pairs,np.array([xi0]),gamma,max_sweeps=10)
            expected=(1+gamma)/(1+gamma*r/xi0)
            self.assertAlmostEqual(model['M'][0,0],expected)
            self.assertAlmostEqual(model['xi'][0],r*expected)

    def test_itml_global_kkt(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(8,3))
        pairs=make_pairs(np.repeat(np.arange(4),2),4,10,7)
        xi0=np.where(pairs[:,2]==1,.7,5.)
        result=fit(X,pairs,xi0,.5,max_sweeps=500,tol=0)
        M,xi,dual=result['M'],result['xi'],result['dual']
        v=X[pairs[:,0]]-X[pairs[:,1]]
        rhs=np.eye(3)+np.einsum('i,ij,ik->jk',pairs[:,2]*dual,v,v)
        np.testing.assert_allclose(np.linalg.inv(M),rhs,atol=1e-8)
        np.testing.assert_allclose(.5/xi,.5/xi0-pairs[:,2]*dual,atol=1e-8)
        slack=pairs[:,2]*(distances(X,pairs,M)-xi)
        self.assertLess(slack.max(),1e-7)
        np.testing.assert_allclose(dual*slack,0,atol=1e-7)

    def test_evaluation_ties(self):
        pairs=np.array([[0,1,1],[0,2,-1],[1,2,-1],[2,3,1]])
        result=verification(np.zeros((4,1)),pairs,np.eye(1))
        self.assertEqual(result,dict(AP=.5,AUC=.5))
        X=np.array([[0.],[.1],[10.],[10.1]])
        result=verification(X,pairs,np.eye(1))
        self.assertEqual(result,dict(AP=1.,AUC=1.))

    def test_split_pairs_pca(self):
        rows=[dict(identity=str(i//3)) for i in range(18)]
        split=split_identities(rows,[2,2,2],1)
        sets=[{rows[i]['identity'] for i in idx} for idx in split.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        labels=np.repeat(np.arange(3),3)
        pairs=make_pairs(labels,9,12,3)
        self.assertEqual(len(set(map(tuple,pairs[:,:2]))),len(pairs))
        np.testing.assert_array_equal(pairs[:,2],np.where(labels[pairs[:,0]]==labels[pairs[:,1]],1,-1))
        X=np.random.default_rng(2).normal(size=(10,5))
        pca=fit_pca(X,3)
        np.testing.assert_allclose(transform(X,pca).mean(axis=0),0,atol=1e-12)
        before=pca[0].copy();transform(X+100,pca)
        np.testing.assert_array_equal(pca[0],before)
        self.assertEqual(burg(np.array([2.]),np.array([2.])),0.)
        self.assertEqual(logdet(np.eye(3)),0.)


if __name__=='__main__':
    unittest.main()
