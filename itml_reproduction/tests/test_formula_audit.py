import unittest
import numpy as np
from itml_repro.metrics import burg,logdet,plus_step,fit
from itml_repro.audit_formulas import printed_davis_counterexample
from itml_repro.evaluation import verification,retrieval
from itml_repro.corrected import fit_corrected


class FormulaAuditTests(unittest.TestCase):
    def test_ap_auc_mixed_ties_and_distance_order(self):
        X=np.sqrt(np.array([[0.],[.1],[.1],[.8],[.9]]))
        pairs=np.array([[0,1,1],[0,2,-1],[0,3,1],[0,4,-1]])
        result=verification(X,pairs,np.eye(1))
        self.assertAlmostEqual(result['AP'],7/12)
        self.assertAlmostEqual(result['AUC'],.625)

    def test_rank1_and_cmc_hand_computed(self):
        result=retrieval(np.array([[0.],[10.]]),np.array([[.1],[1.],[10.1]]),
                         ['A','B'],['B','A','B'],np.eye(1))
        self.assertEqual(result['rank1'],.5)
        np.testing.assert_array_equal(result['CMC'],[.5,1.,1.])

    def test_logdet_rejects_non_spd_domain(self):
        for matrix in (np.array([[1.,100.],[0.,1.]]),np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                logdet(matrix)
        with self.assertRaises(np.linalg.LinAlgError):
            logdet(np.diag([1.,-1.]))
        with self.assertRaises(ValueError):
            plus_step(np.eye(1),np.eye(1),1.,np.ones(1),np.ones(1),0,1.,1.)

    def test_burg_positive_domain_and_nonfinite_parameters(self):
        for x,x0 in [([-1.],[-1.]),([0.],[1.]),([1.],[np.inf]),([np.nan],[1.])]:
            with self.assertRaises(ValueError):
                burg(x,x0)
        for gamma in (np.nan,np.inf,-1.):
            with self.assertRaises(ValueError):
                plus_step(np.eye(1),np.eye(1),1.,np.ones(1),np.ones(1),1,gamma,1.)
            with self.assertRaises(ValueError):
                fit(np.array([[0.],[1.]]),np.array([[0,1,1]]),np.ones(1),gamma)
        for solver in (fit,fit_corrected):
            for tol in (np.nan,np.inf):
                with self.assertRaises(ValueError):
                    solver(np.array([[0.],[1.]]),np.array([[0,1,1]]),np.ones(1),1.,
                           Z=np.array([[0.],[1.]]),tol=tol)

    def test_printed_davis_step_is_not_eq45_solution(self):
        result=printed_davis_counterexample()
        self.assertGreater(result['printed_step_constraint_violation'],.1)
        self.assertAlmostEqual(result['objective_consistent_M'],result['analytic_optimum_M'])
        self.assertAlmostEqual(4*result['objective_consistent_M'],result['objective_consistent_xi'])


if __name__=='__main__':
    unittest.main()
