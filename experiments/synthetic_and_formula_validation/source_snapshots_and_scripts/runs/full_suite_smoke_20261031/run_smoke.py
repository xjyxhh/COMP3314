"""Complete small end-to-end acceptance incl. plots. Explicitly synthetic."""
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
from itml_repro.data import make_pairs
from itml_repro.run import save_json
from itml_repro.baseline_compare import run,METHODS
from itml_repro.paper_analyses import train_analyses
from itml_repro.finalize_suite import finalize


def main():
    root=Path(__file__).resolve().parent
    inputs=root/'input';(inputs/'round_00').mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(3314);labels=np.repeat(np.arange(3),18)
    centers=np.array([[-1.,-1.],[0.,1.],[1.,-1.]])[labels]
    X=centers+.03*rng.normal(size=(54,2))
    val=centers+.03*rng.normal(size=(54,2));test=centers+.03*rng.normal(size=(54,2))
    np.savez(inputs/'features_pca.npz',X_train=X,Z_train=X,X_val=val,X_test=test)
    np.savez(inputs/'round_00/pairs.npz',train=make_pairs(labels,200,200,3314),
             val=make_pairs(labels,200,200,3315),test=make_pairs(labels,200,200,3316))
    np.save(inputs/'round_00/xi0.npy',np.where(np.load(inputs/'round_00/pairs.npz')['train'][:,2]==1,.2,2.5))
    (inputs/'train_manifest.csv').write_text('identity\n'+'\n'.join(map(str,labels))+'\n')
    save_json(inputs/'protocol.json',dict(scope='SYNTHETIC workflow acceptance, 3 artificial classes, no real faces/depth'))
    config=dict(methods=METHODS,gamma_grid=[.1],lambda_grid=[1.],C_grid=[1.],
        svm_plus_gamma_grid=[1.],svm_kernel='linear',lmnn_mu_grid=[.5],lmnn_k=5,
        lmnn_backend='projected',lmnn_max_iterations=100,itml_s_filter_rule='shared_xi0',
        max_sweeps=2,tolerance=0.,selection_metric='AP')
    save_json(root/'config.json',config)
    run(Namespace(input=inputs,output=root/'baselines',config=root/'config.json'))
    train_analyses(inputs,root/'baselines',root/'analyses',root/'config.json')
    finalize(inputs,root/'baselines',root/'analyses',root/'final')
    completion=json.loads((root/'final/completion.json').read_text())
    assert completion['test_models']==22 and completion['all_timing_models_match']
    assert all((root/'final'/name).stat().st_size>1000 for name in
        ('partial_curve.png','sample_count_curve.png','distance_analysis.png','training_time.png'))
    print('Full synthetic acceptance passed',completion,flush=True)


if __name__=='__main__':main()
