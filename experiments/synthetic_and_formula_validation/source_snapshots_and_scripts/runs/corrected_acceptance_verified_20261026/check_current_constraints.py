import json
from pathlib import Path
import cvxpy as cp
import numpy as np
from itml_repro.corrected import fit_corrected
root=Path('runs/shared_compare_20261026')
with np.load(root/'features_pca.npz') as f: X,Z=f['X_train'],f['Z_train']
records=[]
for rd in range(2):
    folder=root/f'round_{rd:02d}'
    with np.load(folder/'pairs.npz') as f: pairs=f['train']
    xi0=np.load(folder/'xi0.npy')
    v,w=X[pairs[:,0]]-X[pairs[:,1]],Z[pairs[:,0]]-Z[pairs[:,1]]
    M=cp.Variable((X.shape[1],X.shape[1]),symmetric=True);P=cp.Variable((Z.shape[1],Z.shape[1]),symmetric=True);t=cp.Variable()
    c=[M >> t*np.eye(X.shape[1]),P >> t*np.eye(Z.shape[1]),cp.trace(M)+cp.trace(P)==1]
    c += [y*(cp.sum(cp.multiply(np.outer(a,a),M))-cp.sum(cp.multiply(np.outer(b,b),P)))<=0 for a,b,y in zip(v,w,pairs[:,2])]
    prob=cp.Problem(cp.Maximize(t),c)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
    record=dict(round=rd,status=prob.status,normalized_spd_margin=float(t.value),solver_dual_gap=prob.solver_stats.extra_stats)
    try:
        result=fit_corrected(X,pairs,xi0,.1,Z,1.,max_sweeps=1000,tol=0)
        record['checkpoints']=[result['history'][k-1] for k in (100,500,1000)]
    except (RuntimeError,np.linalg.LinAlgError) as exc:
        record['solver_error']=str(exc)
    records.append(record)
    Path('runs/corrected_acceptance_verified_20261026/current_constraints.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(record),flush=True)
