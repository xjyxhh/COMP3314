"""Create small synthetic train/validation inputs; no test arrays exist.

Run from itml_reproduction. This is a workflow check, not a benchmark.
"""
import json
from pathlib import Path
import numpy as np
from itml_repro.data import make_pairs
from itml_repro.run import save_json

root=Path(__file__).resolve().parent
out=root/'input'
out.mkdir(exist_ok=False)
rng=np.random.default_rng(3314)
labels=np.repeat(np.arange(3),6)
centers=np.array([[-1.,-1.],[0.,1.],[1.,-1.]])[labels]
X=centers+.05*rng.normal(size=(18,2))
Z=centers+.03*rng.normal(size=(18,2))
val=centers+.05*rng.normal(size=(18,2))
np.savez(out/'features_pca.npz',X_train=X,Z_train=Z,X_val=val)
(out/'train_manifest.csv').write_text('identity\n'+'\n'.join(map(str,labels))+'\n')
for n in range(2):
    folder=out/f'round_{n:02d}'
    folder.mkdir()
    pairs=make_pairs(labels,12,16,3314+n,positive_seed=3314)
    vp=make_pairs(labels,12,16,4314+n,positive_seed=4314)
    np.savez(folder/'pairs.npz',train=pairs,val=vp)
    np.save(folder/'xi0.npy',np.where(pairs[:,2]==1,.2,2.5))
config=json.loads((root.parents[1]/'configs/baselines_paper.json').read_text())
config.update(gamma_grid=[.1,1.],lambda_grid=[.1,1.],C_grid=[.1,1.],
              svm_plus_gamma_grid=[.1,1.])
save_json(root/'config.json',config)
save_json(root/'input_scope.json',dict(seed=3314,scope='Synthetic workflow only, no test input',
    identities=3,training_samples=18,validation_samples=18,rounds=2,
    feature_dimensions=2,xi0_positive=.2,xi0_negative=2.5,
    centers_shared_across_train_validation=True,
    note='Not a face/RGB-D dataset; fixed illustrative bounds, no PCA fitting or original sample counts'))
