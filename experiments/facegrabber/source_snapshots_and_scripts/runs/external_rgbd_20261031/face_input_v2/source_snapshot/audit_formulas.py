"""Independent primal checks of Xu (2015) and Davis (2007) formulas."""
import argparse
from pathlib import Path
import numpy as np
import cvxpy as cp
from .metrics import plus_step, fit, distances
from .run import save_json, sha


def matrix_burg(A, B):
    return float(np.trace(np.linalg.solve(B,A))-np.linalg.slogdet(A)[1]+np.linalg.slogdet(B)[1]-len(A))


def solve_projection(M0,P0,xi0,v,w,y,gamma,lam):
    M=cp.Variable(M0.shape,symmetric=True);P=cp.Variable(P0.shape,symmetric=True);xi=cp.Variable(pos=True)
    r=cp.sum(cp.multiply(np.outer(v,v),M));s=cp.sum(cp.multiply(np.outer(w,w),P))
    objective=(cp.trace(np.linalg.inv(M0)@M)-cp.log_det(M)+np.linalg.slogdet(M0)[1]-len(M0)
        +lam*(cp.trace(np.linalg.inv(P0)@P)-cp.log_det(P)+np.linalg.slogdet(P0)[1]-len(P0))
        +gamma*(xi/xi0-cp.log(xi/xi0)-1))
    problem=cp.Problem(cp.Minimize(objective),[M>>0,P>>0,y*(r-xi)<=0,xi==s])
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=500)
    if problem.status!='optimal':
        raise RuntimeError('Projection reference status: '+problem.status)
    return dict(objective=float(problem.value),M=M.value,P=P.value,xi=float(xi.value),status=problem.status)


def projection_checks(out):
    rng=np.random.default_rng(3314);records=[]
    for gamma,lam in ((.1,.2),(.7,2.),(1.8,.4)):
        for y in (-1,1):
            for xi0 in (.1,10.):
                B=rng.normal(size=(2,2));M=B@B.T+np.eye(2)
                B=rng.normal(size=(3,3));P=B@B.T+np.eye(3)
                v,w=rng.normal(size=2),rng.normal(size=3)
                v/=np.sqrt(v@M@v);w/=np.sqrt(w@P@w)
                N,Q,xi,alpha,beta=plus_step(M,P,xi0,v,w,y,gamma,lam)
                ref=solve_projection(M,P,xi0,v,w,y,gamma,lam)
                objective=matrix_burg(N,M)+lam*matrix_burg(Q,P)+gamma*(xi/xi0-np.log(xi/xi0)-1)
                gap=abs(objective-ref['objective'])/max(1.,abs(ref['objective']))
                residuals=dict(eq17=float(np.linalg.norm(np.linalg.inv(N)-np.linalg.inv(M)-y*alpha*np.outer(v,v))),
                    eq18=float(np.linalg.norm(lam*np.linalg.inv(Q)-lam*np.linalg.inv(P)+beta*np.outer(w,w))),
                    eq21_22=float(abs(xi-w@Q@w)),
                    eq25_26=float(abs(gamma/xi-gamma/xi0+y*alpha-beta)),
                    inequality=float(max(0.,y*(v@N@v-xi))),
                    complementarity=float(abs(alpha*y*(v@N@v-xi))))
                eigenvalue=float(min(np.linalg.eigvalsh(N).min(),np.linalg.eigvalsh(Q).min()))
                passed=gap<1e-6 and max(residuals.values())<1e-8 and eigenvalue>0 and alpha>=0
                index=len(records)
                np.savez(out/f'projection_{index:02d}.npz',M0=M,P0=P,xi0=xi0,v=v,w=w,y=y,gamma=gamma,lam=lam,
                    M=N,P=Q,xi=xi,alpha=alpha,beta=beta,reference_M=ref['M'],reference_P=ref['P'],reference_xi=ref['xi'])
                records.append(dict(case=index,gamma=gamma,lam=lam,y=y,xi0=xi0,active=bool(alpha>1e-12),
                    objective=objective,reference_objective=ref['objective'],relative_objective_gap=gap,
                    residuals=residuals,min_eigenvalue=eigenvalue,passed=bool(passed)))
    save_json(out/'projections.json',records)
    if not all(r['passed'] for r in records):
        raise AssertionError('Single-pair projection formula check failed')
    return records


def itml_checks(out):
    records=[]
    for seed,gamma in ((17,.01),(23,.7),(42,2.)):
        rng=np.random.default_rng(seed);X=rng.normal(size=(8,2))
        i,j=np.triu_indices(8,1)
        pairs=np.column_stack((i,j,np.where(i//2==j//2,1,-1)))
        xi0=np.where(pairs[:,2]==1,.7,5.)
        M=cp.Variable((2,2),symmetric=True);xi=cp.Variable(len(pairs),pos=True)
        v=X[i]-X[j];r=cp.hstack([cp.sum(cp.multiply(np.outer(t,t),M)) for t in v])
        objective=cp.trace(M)-cp.log_det(M)-2+gamma*cp.sum(xi/xi0-cp.log(xi/xi0)-1)
        problem=cp.Problem(cp.Minimize(objective),[M>>0,cp.multiply(pairs[:,2],r-xi)<=0])
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=500)
        if problem.status!='optimal':
            raise RuntimeError('ITML reference status: '+problem.status)
        model=fit(X,pairs,xi0,gamma,max_sweeps=1000,tol=0.)
        q=model['dual'];slack=pairs[:,2]*(distances(X,pairs,model['M'])-model['xi'])
        stationarity_M=float(np.linalg.norm(np.linalg.inv(model['M'])-np.eye(2)-np.einsum('i,ij,ik->jk',pairs[:,2]*q,v,v)))
        stationarity_xi=float(np.max(np.abs(gamma/model['xi']-gamma/xi0+pairs[:,2]*q)))
        gap=float(abs(model['history'][-1]['objective']-problem.value)/max(1.,abs(problem.value)))
        residuals=dict(stationarity_M=stationarity_M,stationarity_xi=stationarity_xi,
            primal_violation=float(max(0.,slack.max())),complementarity=float(np.max(np.abs(q*slack))),dual_violation=float(max(0.,-q.min())))
        passed=gap<1e-6 and max(residuals.values())<1e-7
        records.append(dict(seed=seed,gamma=gamma,objective=model['history'][-1]['objective'],
            reference_objective=float(problem.value),relative_objective_gap=gap,residuals=residuals,passed=bool(passed)))
        np.savez(out/f'itml_{seed}.npz',X=X,pairs=pairs,xi0=xi0,M=model['M'],xi=model['xi'],dual=q,reference_M=M.value,reference_xi=xi.value)
    save_json(out/'itml.json',records)
    if not all(r['passed'] for r in records):
        raise AssertionError('ITML global objective check failed')
    return records


def printed_davis_counterexample():
    gamma=.7;r=4.;xi0=.2
    a=min(0.,.5*(1/r-gamma/xi0))
    printed_m=1/(1-a*r);printed_xi=gamma*xi0/(gamma+a*xi0)
    actual=fit(np.array([[0.],[2.]]),np.array([[0,1,1]]),np.array([xi0]),gamma,max_sweeps=1,tol=0.)
    return dict(gamma=gamma,r=r,xi0=xi0,printed_step_alpha=a,
        printed_step_M=printed_m,printed_step_xi=printed_xi,printed_step_constraint_violation=r*printed_m-printed_xi,
        objective_consistent_M=float(actual['M'][0,0]),objective_consistent_xi=float(actual['xi'][0]),
        analytic_optimum_M=(1+gamma)/(1+gamma*r/xi0),
        interpretation='Printed Davis Algorithm 1 line 3.4 is incompatible with its Eq.4.5 for this gamma; no claim of an author-issued erratum.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    projections=projection_checks(args.output);itml=itml_checks(args.output)
    save_json(args.output/'davis_printed_counterexample.json',printed_davis_counterexample())
    save_json(args.output/'summary.json',dict(projection_cases=len(projections),projection_passed=sum(r['passed'] for r in projections),
        active_cases=sum(r['active'] for r in projections),inactive_cases=sum(not r['active'] for r in projections),
        itml_global_cases=len(itml),itml_global_passed=sum(r['passed'] for r in itml),
        code_sha256={n:sha(Path(__file__).parent/n) for n in ('metrics.py','corrected.py','audit_formulas.py')}))
    print('Independent projection and ITML objective checks passed',flush=True)


if __name__=='__main__':
    main()
