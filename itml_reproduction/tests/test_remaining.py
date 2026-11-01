import unittest
import numpy as np
from itml_repro.glbp import U2,block_histogram,glbp,align_face


class RemainingTests(unittest.TestCase):
    def test_glbp_counts_and_clipped_signed_differences(self):
        self.assertEqual(len(np.unique(U2)),59)
        self.assertEqual(np.sum(U2==58),198)
        a=np.ones((120,105),dtype=int)*100
        v=glbp(a).reshape(56,123)
        self.assertTrue(np.all(v[:,U2[255]]==169))
        self.assertTrue(np.all(v[:,[59+8,75+8,91+8,107+8]]==169))
        self.assertTrue(np.all(v.sum(axis=1)==845))
        b=np.array([[0,1,2],[30,20,4],[100,6,7]])
        h=block_histogram(b)
        expected_code=sum((b[y,x]>=20)*(1<<k) for k,(y,x) in enumerate([(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0)]))
        self.assertEqual(h[U2[expected_code]],1)
        self.assertTrue(np.all(h[[59,75,91,107]]==1))
        np.testing.assert_array_equal(glbp(a),glbp(a+1000))
        np.testing.assert_allclose(glbp(a,'histogram_l1').reshape(56,123).sum(1),5.)
        with self.assertRaises(ValueError):glbp(a[:100])

    def test_identity_eye_alignment(self):
        image=np.arange(120*105).reshape(120,105)
        eyes=[[30,40],[75,40]]
        np.testing.assert_array_equal(align_face(image,eyes,eyes),image)

    def test_memory_bounded_lmnn_gradient_and_reference(self):
        from itml_repro.baselines import target_neighbors,lmnn_loss_gradient,fit_lmnn,fit_lmnn_projected
        rng=np.random.default_rng(8);X=rng.normal(size=(6,2));labels=np.repeat(np.arange(3),2)
        t=target_neighbors(X,labels,1);M=np.array([[1.,.2],[.2,.7]])
        _,g=lmnn_loss_gradient(X,labels,t,M,.3)
        for H in (np.eye(2),np.array([[0,1.],[1.,0]])):
            numerical=(lmnn_loss_gradient(X,labels,t,M+1e-6*H,.3)[0]-lmnn_loss_gradient(X,labels,t,M-1e-6*H,.3)[0])/2e-6
            self.assertAlmostEqual(numerical,np.sum(g*H),places=5)
        X=np.array([[0.],[.1],[1.],[1.1]]);labels=np.array([0,0,1,1])
        exact=fit_lmnn(X,labels,.5,k=1)
        scalable=fit_lmnn_projected(X,labels,.5,k=1)
        self.assertLess(abs(exact['diagnostics']['objective']-scalable['diagnostics']['objective']),1e-6)
        self.assertFalse(scalable['diagnostics']['global_optimum_certified'])

    def test_kdes_patch_against_direct_sum(self):
        from itml_repro.kdes import dense_gradient_kdes
        a=np.ones((16,16))*.5
        params=dict(orientation_basis=np.array([[0.,1.]]),position_basis=np.array([[.5,.5]]),
                    coefficients=np.ones((1,1)),gamma_orientation=3.,gamma_position=2.,epsilon=.01)
        f,p=dense_gradient_kdes(a,np.ones_like(a),np.zeros_like(a),params)
        expected=sum(np.exp(-2*((x/15-.5)**2+(y/15-.5)**2)) for y in range(16) for x in range(16))/np.sqrt(256+.01)
        self.assertAlmostEqual(f[0,0],expected)
        np.testing.assert_array_equal(p,[[.5,.5]])
        f,_=dense_gradient_kdes(a,np.zeros_like(a),np.zeros_like(a),params)
        self.assertEqual(f[0,0],0.)

    def test_retrieval_protocol_and_classifier_direction(self):
        from itml_repro.reid import probe_gallery,rank_scores
        X=np.array([[0.],[.1],[3.],[3.1]])
        labels=np.array(['a','a','b','b'])
        q,g=probe_gallery(labels,2)
        np.testing.assert_array_equal(q,[0,2]);np.testing.assert_array_equal(g,[1,3])
        self.assertEqual(rank_scores(X,labels,dict(M=np.eye(1)),2)['rank1'],1.)
        # Larger classifier confidence means more similar; do not negate it.
        model=dict(w=np.array([-1.,0.]),b=np.array(0.))
        self.assertEqual(rank_scores(np.array([[-.1],[.1],[.9],[1.1]]),labels,model,2)['rank1'],1.)

    def test_raw_reid_frame_selection_precedes_codebook_fitting(self):
        import csv,json,tempfile
        from pathlib import Path
        from itml_repro.reid import select_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for subset in ('train','walking','still'):
                for identity in ('a','b'):
                    for frame in range(4):
                        rows.append(dict(sample_id=f'{subset}_{identity}_{frame}',identity=identity,subset=subset,
                            frame=frame,rgb_path=f'{subset}_{identity}_{frame}.png',depth_path=''))
            with (root/'manifest.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            select_manifest(root/'manifest.csv',root/'selected',dict(identity_split=[1,1],test_identities=2,shots=2))
            with (root/'selected/training_images.csv').open() as f:training=list(csv.DictReader(f))
            self.assertEqual([r['sample_id'] for r in training],['train_a_0','train_a_3'])
            self.assertTrue(all(Path(r['rgb_path']).is_absolute() for r in training))
            self.assertTrue((root/'selected/all_selected.csv').exists())

    def test_reid_prepare_and_rank1_selection_without_test_access(self):
        import csv,json,tempfile
        from pathlib import Path
        from itml_repro.reid import prepare,fit_reid,evaluate_reid
        rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[];X=[]
            for subset in ('train','walking','still'):
                for identity in range(4):
                    for frame in range(6):
                        rows.append(dict(sample_id=f'{subset}_{identity}_{frame}',identity=str(identity),subset=subset,frame=frame))
                        X.append(np.array([identity*3.,identity%2])+rng.normal(0,.03,2))
            with (root/'manifest.csv').open('w',newline='') as file:
                w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            np.savez(root/'raw.npz',X=np.asarray(X),Z=np.asarray(X),sample_ids=np.array([r['sample_id'] for r in rows]))
            protocol=dict(identity_split=[2,2],test_identities=4,shots=6,pca_dimension=2,rounds=1,
                          pair_counts=[30,30],seed=2,require_kdes=False)
            prepare(root/'manifest.csv',root/'raw.npz',root/'input',protocol)
            with np.load(root/'input/features_pca.npz') as data:arrays={k:data[k] for k in data.files}
            arrays['X_walking'][:]=np.nan;arrays['X_still'][:]=np.nan
            np.savez(root/'input/features_pca.npz',**arrays)
            config=dict(gamma_grid=[.01],lambda_grid=[1.],C_grid=[1.],svm_plus_gamma_grid=[1.],
                        svm_kernel='linear',lmnn_mu_grid=[.5],lmnn_k=5,itml_s_filter_rule='shared_xi0',
                        max_sweeps=2,tolerance=0.)
            (root/'config.json').write_text(json.dumps(config))
            fit_reid(root/'input',root/'search',root/'config.json')
            records=json.loads((root/'search/results.json').read_text())
            self.assertEqual(len(records),7)
            self.assertTrue(all('rank1' in r['selected']['validation'] for r in records))
            # Test corruption is rejected only by the separate evaluation stage.
            with self.assertRaises(ValueError):evaluate_reid(root/'input',root/'search',root/'eval')

    def test_full_analysis_searches_do_not_require_test_inputs(self):
        import json,tempfile
        from pathlib import Path
        from argparse import Namespace
        from itml_repro.baseline_compare import run,METHODS
        from itml_repro.paper_analyses import train_analyses
        from itml_repro.finalize_suite import finalize
        from itml_repro.data import make_pairs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input/round_00').mkdir(parents=True)
            labels=np.repeat(np.arange(3),6)
            X=np.array([[-1.,-1.],[0.,1.],[1.,-1.]])[labels]+.02*np.random.default_rng(31).normal(size=(18,2))
            pairs=make_pairs(labels,12,16,3)
            np.savez(root/'input/features_pca.npz',X_train=X,Z_train=X,X_val=X)
            np.savez(root/'input/round_00/pairs.npz',train=pairs,val=pairs)
            np.save(root/'input/round_00/xi0.npy',np.where(pairs[:,2]==1,.2,2.5))
            (root/'input/train_manifest.csv').write_text('identity\n'+'\n'.join(map(str,labels))+'\n')
            config=dict(methods=METHODS,gamma_grid=[.01,.1],lambda_grid=[.1,1.],C_grid=[1.],
                svm_plus_gamma_grid=[1.],svm_kernel='linear',lmnn_mu_grid=[.5],lmnn_k=5,
                itml_s_filter_rule='shared_xi0',max_sweeps=2,tolerance=0.,selection_metric='AP')
            (root/'config.json').write_text(json.dumps(config))
            run(Namespace(input=root/'input',output=root/'base',config=root/'config.json'))
            train_analyses(root/'input',root/'base',root/'analyses',root/'config.json')
            records=json.loads((root/'analyses/results.json').read_text())
            self.assertEqual(len(records),15)
            base=json.loads((root/'base/results.json').read_text())
            self.assertEqual(json.loads((root/'analyses/protocol.json').read_text())['parameter_policy'],'fixed_main')
            for row in records:
                if row.get('cached_from'):
                    with np.load(root/'analyses'/row['folder']/'model.npz') as a,np.load(root/'base/round_00'/f"{row['cached_from']}_model.npz") as b:
                        np.testing.assert_array_equal(a['M'],b['M'])
                else:
                    search=json.loads((root/'analyses'/row['folder']/'search.json').read_text())
                    self.assertEqual(len(search),1)
                    original=next(r for r in base if r['method']==row['method'])['selected']['parameters']
                    self.assertEqual(search[0]['parameters']['gamma'],original['gamma'])
                    if row['method']=='itml_plus':self.assertEqual(search[0]['parameters']['lam'],original['lam'])
            train_analyses(root/'input',root/'base',root/'retuned',root/'config.json',parameter_policy='retune_each_fraction')
            for row in json.loads((root/'retuned/results.json').read_text()):
                if not row.get('cached_from'):
                    search=json.loads((root/'retuned'/row['folder']/'search.json').read_text())
                    self.assertEqual(len(search),4 if row['method']=='itml_plus' else 2)
            with self.assertRaises(KeyError):finalize(root/'input',root/'base',root/'analyses',root/'eval')
            self.assertFalse((root/'eval').exists())
