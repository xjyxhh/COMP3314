"""Known-feasible acceptance: python -m itml_repro.verify_corrected --output runs/check"""
import argparse
import json
from pathlib import Path
import cvxpy as cp
import numpy as np
from .corrected import fit_corrected
from .metrics import fit


def stationarity(X,Z,pairs,xi0,gamma,lam,model):
    v,w=X[pairs[:,0]]-X[pairs[:,1]],Z[pairs[:,0]]-Z[pairs[:,1]]
    a,b=model['dual'],model['beta']
    rhs=[np.eye(X.shape[1])+np.einsum('i,ij,ik->jk',pairs[:,2]*a,v,v),
         lam*np.eye(Z.shape[1])-np.einsum('i,ij,ik->jk',b,w,w),
         gamma/xi0-pairs[:,2]*a+b]
    lhs=[np.linalg.inv(model['M']),lam*np.linalg.inv(model['P']),gamma/model['xi']]
    errors={k:float(np.linalg.norm(l-r)/max(1.,np.linalg.norm(l),np.linalg.norm(r)))
            for k,l,r in zip(('M','P','xi'),lhs,rhs)}
    errors['dual_nonnegative_violation']=float(max(0.,-a.min()))
    return errors


def reference(X,Z,pairs,xi0,gamma,lam):
    M=cp.Variable((X.shape[1],X.shape[1]),symmetric=True)
    P=cp.Variable((Z.shape[1],Z.shape[1]),symmetric=True)
    v,w=X[pairs[:,0]]-X[pairs[:,1]],Z[pairs[:,0]]-Z[pairs[:,1]]
    r=cp.hstack([cp.sum(cp.multiply(np.outer(t,t),M)) for t in v])
    s=cp.hstack([cp.sum(cp.multiply(np.outer(t,t),P)) for t in w])
    obj=(cp.trace(M)-cp.log_det(M)-X.shape[1]
         +lam*(cp.trace(P)-cp.log_det(P)-Z.shape[1])
         +gamma*cp.sum(s/xi0-cp.log(s/xi0)-1))
    problem=cp.Problem(cp.Minimize(obj),[M >> 0,P >> 0,cp.multiply(pairs[:,2],r-s)<=0])
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=500)
    if problem.status!='optimal':
        raise RuntimeError(f'Uncertified reference: {problem.status}')
    rv,sv=np.asarray(r.value),np.asarray(s.value)
    residual=float(np.max(np.maximum(pairs[:,2]*(rv-sv),0)/np.maximum(rv,sv)))
    if residual>1e-6 or min(np.linalg.eigvalsh(M.value).min(),np.linalg.eigvalsh(P.value).min())<=0:
        raise RuntimeError('Reference feasibility check failed')
    return dict(objective=float(problem.value),relative_violation=residual,M=M.value,P=P.value,status=problem.status)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for seed,gamma,lam in [(17,.1,1.),(23,.7,2.),(42,1.,.2)]:
        rng=np.random.default_rng(seed)
        X,Z=rng.normal(size=(8,2)),rng.normal(size=(8,2))
        i,j=np.triu_indices(8,1)
        r,s=((X[i]-X[j])**2).sum(1),((Z[i]-Z[j])**2).sum(1)
        # I,I explicitly satisfies all inequalities strictly.
        pairs=np.column_stack((i,j,np.where(r<s,1,-1)))
        xi0=np.where(pairs[:,2]==1,.5,4.)
        np.savez(args.output/f'input_{seed}.npz',X=X,Z=Z,pairs=pairs,xi0=xi0)
        ref=reference(X,Z,pairs,xi0,gamma,lam)
        model=fit_corrected(X,pairs,xi0,gamma,Z,lam,max_sweeps=1000,tol=1e-7)
        checks=stationarity(X,Z,pairs,xi0,gamma,lam,model)
        row=model['history'][-1]
        gap=abs(row['objective']-ref['objective'])/max(1.,abs(ref['objective']))
        original=fit(X,pairs,xi0,gamma,Z,lam,max_sweeps=1000,tol=0)
        passed=(max(checks.values())<1e-7 and gap<1e-5
                and max(row['relative_violation'],row['relative_equality'],row['complementarity'])<1e-6
                and min(row['min_eigenvalue_M'],row['min_eigenvalue_P'])>0)
        record=dict(seed=seed,gamma=gamma,lam=lam,passed=bool(passed),corrected=row,
                    stationarity=checks,relative_objective_gap=gap,
                    reference={k:v for k,v in ref.items() if k not in ('M','P')},
                    relative_matrix_difference={k:float(np.linalg.norm(model[k]-ref[k])/np.linalg.norm(ref[k])) for k in ('M','P')},
                    original=original['history'][-1])
        records.append(record)
        np.savez(args.output/f'model_{seed}.npz',M=model['M'],P=model['P'],xi=model['xi'],alpha=model['dual'],beta=model['beta'],reference_M=ref['M'],reference_P=ref['P'])
        (args.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    if not all(r['passed'] for r in records):
        raise SystemExit('Acceptance failed')


if __name__=='__main__':
    main()
