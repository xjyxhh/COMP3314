import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from itml_repro.viewpoint import main
from itml_repro.evaluation import verification


class ViewpointTests(unittest.TestCase):
    def test_augmented_feasible_witness(self):
        from itml_repro.improve_depth import augmented_witness
        from itml_repro.metrics import distances
        x, z = np.zeros((4, 8)), np.zeros((4, 8))
        x[:, 0], z[:, 0] = [0, 1, 2, 3], [0, .5, 4, 4.5]
        pairs = np.array([[0, 2, 1], [0, 1, -1]])
        augmented = np.column_stack((z*2, np.random.default_rng(7).normal(size=(4, 8))))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            np.savez(folder/'original.npz', M=np.eye(8)/16, P=np.eye(8)/16)
            result = augmented_witness(x, augmented, pairs, folder/'original.npz', 2., folder)
            self.assertEqual(result['classification'], 'numerically_spd_feasible')
            with np.load(folder/'feasibility_certificate.npz') as model:
                self.assertLess(np.max(pairs[:, 2]*(distances(x, pairs, model['M'])-distances(augmented, pairs, model['P']))), 0)
                self.assertGreater(np.linalg.eigvalsh(model['P']).min(), 0)

    def test_depth_shape_validity_and_camera_offset(self):
        from itml_repro.improve_depth import depth_shape
        depth = (np.arange(64).reshape(8, 8)+500).astype(np.uint16)
        mask = np.ones((8, 8), dtype=bool)
        actual = depth_shape(depth, mask)
        self.assertEqual(actual.shape, (115,))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, depth_shape(depth+100, mask))
        depth[:2, :2] = 0
        self.assertTrue(np.isfinite(depth_shape(depth, mask)).all())
        with self.assertRaises(ValueError):
            depth_shape(np.zeros_like(depth), mask)

    def test_saved_pairs_and_validation_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/'input', Path(tmp)/'output'
            (source/'round_00').mkdir(parents=True)
            x = np.array([[0.], [.2], [2.], [2.3]])
            pairs = np.array([[0, 1, 1], [0, 2, -1], [2, 3, 1], [1, 3, -1]])
            np.savez(source/'features_pca.npz', X_train=x, X_val=x, X_test=x, Z_train=x*2)
            np.savez(source/'split.npz', train=np.arange(4), val=np.arange(4, 8), test=np.arange(8, 12))
            np.savez(source/'round_00/pairs.npz', train=pairs, val=pairs, test=pairs)
            np.save(source/'round_00/xi0.npy', np.array([.1, 4., .1, 4.]))
            with patch('sys.argv', ['viewpoint', '--input', str(source), '--output', str(out)]), patch('builtins.print'):
                main()
            records = json.loads((out/'results.json').read_text())
            self.assertEqual(len(records), 4)
            for record in records:
                self.assertEqual(record['status'], 'completed')
                search = json.loads((out/'round_00'/f"{record['method']}_search.json").read_text())
                self.assertEqual(record['selected']['validation']['AP'], max(r['validation']['AP'] for r in search if r['status']=='completed'))
                with np.load(out/'round_00'/f"{record['method']}_model.npz") as model:
                    self.assertEqual(record['test'], verification(x, pairs, model['M']))
            self.assertTrue(json.loads((out/'integrity.json').read_text())['source_unchanged'])
            deferred = Path(tmp)/'deferred'
            with patch('sys.argv', ['viewpoint', '--input', str(source), '--output', str(deferred), '--defer-test']), patch('builtins.print'):
                main()
            self.assertTrue(all('test' not in r for r in json.loads((deferred/'results.json').read_text())))


if __name__ == '__main__':
    unittest.main()
