"""Training-only solver check; no changes to the running primary studies."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import cvxpy as cp
from itml_repro.baselines import pair_features

HERE=Path(__file__).resolve().parent

def main():
    root=HERE/'face_input_v2'
    with np.load(root/'features_pca.npz') as f:X,Z=f['X_train'],f['Z_train']
    with np.load(root/'round_00/pairs.npz') as f:p=f['train']
    F,G=pair_features(X,p),pair_features(Z,p);y=p[:,2];C=.01;gamma=.01
    results=[]
    for solver in ('HIGHS','OSQP'):
        w,wp,b,bp=cp.Variable(F.shape[1]),cp.Variable(G.shape[1]),cp.Variable(),cp.Variable()
        slack=G@wp+bp;constraints=[cp.multiply(y,F@w+b)+slack>=1,slack>=0]
        problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(w)+.5*gamma*cp.sum_squares(wp)+C*cp.sum(slack)),constraints)
        start=perf_counter();row=dict(solver=solver)
        try:
            options=dict(highs_options={'time_limit':60.}) if solver=='HIGHS' else dict(time_limit=60.,max_iter=10000,eps_abs=1e-7,eps_rel=1e-7,polishing=True)
            problem.solve(solver=solver,**options)
            row.update(status=problem.status,objective=problem.value,iterations=problem.solver_stats.num_iters)
            if w.value is not None:
                a,beta=[np.asarray(c.dual_value) for c in constraints]
                margin=y*(F@w.value+b.value)+slack.value-1
                row.update(primal_violation=float(max(0.,-margin.min(),-slack.value.min())),
                    stationarity_w=float(np.linalg.norm(w.value-F.T@(y*a))),
                    stationarity_wp=float(np.linalg.norm(gamma*wp.value+G.T@(C-a-beta))),
                    complementarity=float(max(np.max(np.abs(a*margin)),np.max(np.abs(beta*slack.value)))))
        except Exception as error:row.update(error=str(error))
        row['seconds']=perf_counter()-start;results.append(row);print(row,flush=True)
        (HERE/'solver_benchmark.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
