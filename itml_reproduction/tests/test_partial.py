import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from itml_repro.metrics import fit, distances, logdet, burg
from itml_repro.partial import privileged_prefix, run
from itml_repro.data import make_pairs


class PartialTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3314)
        self.X = rng.normal(size=(8, 2))
        self.Z = rng.normal(size=(8, 3))
        self.pairs = make_pairs(np.repeat(np.arange(4), 2), 4, 8, 17)
        self.xi0 = np.where(self.pairs[:, 2] == 1, .5, 4.)

    def test_endpoints_preserve_existing_multisweep_trajectories(self):
        for fraction, depth in ((0., None), (1., self.Z)):
            expected = fit(self.X, self.pairs, self.xi0, .1, Z=depth, max_sweeps=5, tol=0.)
            actual = fit(self.X, self.pairs, self.xi0, .1, Z=self.Z, max_sweeps=5, tol=0.,
                         privileged_mask=privileged_prefix(self.pairs, fraction))
            for key in ('M', 'xi', 'dual', 'beta'):
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(actual['history'], expected['history'])
            if depth is not None:
                np.testing.assert_array_equal(actual['P'], expected['P'])

    def test_prefix_is_label_stratified_nested_and_uses_saved_order(self):
        previous = np.zeros(len(self.pairs), bool)
        for fraction in (0., .25, .5, .75, 1.):
            mask = privileged_prefix(self.pairs, fraction)
            self.assertTrue(np.all(mask[previous]))
            for label in (1, -1):
                rows = np.flatnonzero(self.pairs[:, 2] == label)
                count = int(np.floor(len(rows)*fraction))
                np.testing.assert_array_equal(np.flatnonzero(mask & (self.pairs[:, 2]==label)), rows[:count])
            previous = mask
        for fraction in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                privileged_prefix(self.pairs, fraction)

    def test_missing_depth_is_not_read_and_mixed_objective_is_eq5(self):
        pairs = np.array([[0, 1, 1], [2, 3, -1], [0, 2, -1]])
        mask = np.array([True, False, False])
        Z = self.Z.copy()
        Z[2:] = np.nan
        model = fit(self.X, pairs, np.array([.5, 4., 4.]), .3, Z=Z, lam=2.,
                    max_sweeps=3, tol=0., privileged_mask=mask)
        target = model['xi'].copy()
        target[mask] = distances(Z, pairs[mask], model['P'])
        expected = logdet(model['M'])+2*logdet(model['P'])+.3*burg(target, np.array([.5, 4., 4.]))
        self.assertAlmostEqual(model['history'][-1]['objective'], expected)
        # Only the first pair has depth; subsequent pairs must leave P untouched.
        with patch('itml_repro.metrics.plus_step', wraps=__import__('itml_repro.metrics', fromlist=['plus_step']).plus_step) as step:
            once = fit(self.X, pairs, np.array([.5, 4., 4.]), .3, Z=Z, lam=2.,
                       max_sweeps=1, tol=0., privileged_mask=mask)
            self.assertEqual(step.call_count, 1)
            first = fit(self.X, pairs[:1], np.array([.5]), .3, Z=Z, lam=2.,
                        max_sweeps=1, tol=0., privileged_mask=mask[:1])
        np.testing.assert_array_equal(once['P'], first['P'])
        with self.assertRaises(ValueError):
            fit(self.X, pairs, np.array([.5, 4., 4.]), .3, Z=Z,
                privileged_mask=np.array([True, True, False]))
        with self.assertRaises(ValueError):
            fit(self.X, pairs, np.array([.5, 4., 4.]), .3, privileged_mask=mask)
        with self.assertRaises(ValueError):
            fit(self.X, pairs, np.array([.5, 4., 4.]), .3, Z=Z, privileged_mask=np.array([1, 0, 0]))

    def test_mixed_first_sweep_against_independent_primal_projections(self):
        try:
            import cvxpy as cp
        except ImportError:
            self.skipTest('Optional CVXPY is not installed')
        mask = privileged_prefix(self.pairs, .5)
        gamma, lam = .3, 2.
        # First visit to ordinary pairs has zero accumulated dual. Compare each
        # mixed branch with its actual primal projection, not a second copy of
        # the closed-form update. This does not claim global optimality.
        M, P, xi = np.eye(2), np.eye(3), self.xi0.copy()
        for k, (i, j, y) in enumerate(self.pairs):
            v, w = self.X[i]-self.X[j], self.Z[i]-self.Z[j]
            N, Q, t = cp.Variable((2, 2), symmetric=True), cp.Variable((3, 3), symmetric=True), cp.Variable(pos=True)
            r = cp.sum(cp.multiply(np.outer(v, v), N))
            objective = (cp.trace(np.linalg.inv(M)@N)-cp.log_det(N)+np.linalg.slogdet(M)[1]-2
                +lam*(cp.trace(np.linalg.inv(P)@Q)-cp.log_det(Q)+np.linalg.slogdet(P)[1]-3)
                +gamma*(t/xi[k]-cp.log(t/xi[k])-1))
            constraints = [N >> 0, Q >> 0, y*(r-t) <= 0]
            if mask[k]:
                constraints.append(t == cp.sum(cp.multiply(np.outer(w, w), Q)))
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
            self.assertEqual(problem.status, 'optimal')
            M, P, xi[k] = N.value, Q.value, t.value
        actual = fit(self.X, self.pairs, self.xi0, gamma, Z=self.Z, lam=lam,
                     max_sweeps=1, tol=0., privileged_mask=mask)
        np.testing.assert_allclose(actual['M'], M, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(actual['P'], P, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(actual['xi'], xi, rtol=2e-4, atol=2e-5)

    def test_runner_ignores_test_arrays_and_persists_all_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root/'input'
            (inputs/'round_00').mkdir(parents=True)
            # Test arrays intentionally absent: cannot accidentally evaluate them.
            np.savez(inputs/'features_pca.npz', X_train=self.X, Z_train=self.Z, X_val=self.X)
            np.savez(inputs/'round_00/pairs.npz', train=self.pairs, val=self.pairs)
            np.save(inputs/'round_00/xi0.npy', self.xi0)
            config = root/'config.json'
            config.write_text(json.dumps(dict(gamma_grid=[.1],lambda_grid=[1.],max_sweeps=2,tolerance=0.)))
            result = run(Namespace(input=inputs, output=root/'out', config=config, fractions=[0., .25, .5, .75, 1.]))
            self.assertEqual(len(result), 5)
            self.assertTrue(all(r['status']=='completed' and 'test' not in r for r in result))
            self.assertEqual(json.loads((root/'out/integrity.json').read_text()),
                             dict(source_unchanged=True, test_evaluated=False))
            with self.assertRaises(FileExistsError):
                run(Namespace(input=inputs, output=root/'out', config=config, fractions=[0.,1.]))


if __name__ == '__main__':
    unittest.main()
