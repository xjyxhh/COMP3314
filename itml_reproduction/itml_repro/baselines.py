"""Explicit linear SVM/SVM+, LMNN SDP, and ITML-S two-stage baselines.

Source equations and unresolved paper choices: BASELINES_PROGRESS.md.
CVXPY/CLARABEL solve the convex reference formulations, not the authors'
original software. No PSD clipping, constraint subsampling or test depth.
"""
import numpy as np
import cvxpy as cp
from .metrics import fit, distances


def _features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError('Expected a nonempty finite feature matrix')
    return X


def pair_features(X, pairs):
    """Xu V-A4: [|xi-xj|*g, xi*xj*g], g=standard-normal PDF((xi+xj)/2)."""
    X, pairs = _features(X), np.asarray(pairs)
    if (pairs.ndim != 2 or pairs.shape[1] != 3 or not len(pairs)
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs[:, :2] < 0) or np.any(pairs[:, :2] >= len(X))
            or not np.isin(pairs[:, 2], [-1, 1]).all()):
        raise ValueError('Invalid pair indices/labels')
    left, right = X[pairs[:, 0]], X[pairs[:, 1]]
    gate = np.exp(-.5*((left+right)/2)**2)/np.sqrt(2*np.pi)
    result = np.column_stack((np.abs(left-right)*gate, left*right*gate))
    if not np.isfinite(result).all():
        raise ValueError('Pair representation overflowed')
    return result


def _solve(problem):
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-8, tol_gap_rel=1e-8,
                  tol_feas=1e-8, max_iter=500)
    if problem.status != 'optimal' or not np.isfinite(problem.value):
        raise RuntimeError('Uncertified convex baseline: '+str(problem.status))


def fit_svm(X, pairs, C, Z=None, gamma=1.):
    """Linear hinge SVM; with Z, original linear correcting-function SVM+.

    SVM+: min .5||w||² + .5*gamma||w*||² + C*sum(Gw*+b*)
    subject to y(Fw+b) >= 1-(Gw*+b*), Gw*+b* >= 0.
    Biases are unregularized. C multiplies the SUM, not mean loss.
    """
    if not np.isfinite([C, gamma]).all() or min(C, gamma) <= 0:
        raise ValueError('C and gamma must be finite and positive')
    F = pair_features(X, pairs)
    y = np.asarray(pairs)[:, 2]
    if len(np.unique(y)) != 2:
        raise ValueError('SVM training requires both pair labels')
    w, b = cp.Variable(F.shape[1]), cp.Variable()
    objective = .5*cp.sum_squares(w)
    if Z is None:
        slack = cp.Variable(len(pairs))
    else:
        Z = _features(Z)
        if len(Z) != len(X):
            raise ValueError('Privileged features must align with ordinary samples')
        G = pair_features(Z, pairs)
        wp, bp = cp.Variable(G.shape[1]), cp.Variable()
        slack = G@wp+bp
        objective += .5*gamma*cp.sum_squares(wp)
    margin = cp.multiply(y, F@w+b)+slack-1
    constraints = [margin >= 0, slack >= 0]
    problem = cp.Problem(cp.Minimize(objective+C*cp.sum(slack)), constraints)
    _solve(problem)
    wv, bv, sv = np.asarray(w.value), float(b.value), np.asarray(slack.value)
    a, beta = (np.asarray(c.dual_value) for c in constraints)
    mv = y*(F@wv+bv)+sv-1
    diagnostics = dict(objective=float(.5*wv@wv+C*sv.sum()),
        primal_violation=float(max(0., -mv.min(), -sv.min())),
        stationarity_w=float(np.linalg.norm(wv-F.T@(y*a))/max(1.,np.linalg.norm(wv))),
        stationarity_b=float(abs(y@a)),
        complementarity=float(max(np.max(np.abs(a*mv)),np.max(np.abs(beta*sv)))),
        dual_violation=float(max(0.,-a.min(),-beta.min())))
    model = dict(w=wv,b=np.array(bv),xi=sv,dual=a,beta=beta,diagnostics=diagnostics,stop='convex_optimal')
    if Z is not None:
        wpv = np.asarray(wp.value)
        diagnostics['objective'] += float(.5*gamma*wpv@wpv)
        diagnostics['stationarity_privileged'] = float(np.linalg.norm(gamma*wpv+G.T@(C-a-beta)) /
                                                       max(1.,gamma*np.linalg.norm(wpv)))
        diagnostics['stationarity_privileged_bias'] = float(abs(np.sum(C-a-beta)))
        model.update(w_privileged=wpv,b_privileged=np.array(float(bp.value)))
    else:
        diagnostics['stationarity_slack'] = float(np.max(np.abs(C-a-beta)))
    diagnostics['relative_objective_recheck'] = abs(diagnostics['objective']-problem.value)/max(1.,abs(problem.value))
    if diagnostics['primal_violation'] > 1e-6 or diagnostics['relative_objective_recheck'] > 1e-6:
        raise RuntimeError('Convex SVM solution failed direct primal check')
    return model


def target_neighbors(X, labels, k=5):
    X, labels = _features(X), np.asarray(labels)
    if labels.shape != (len(X),) or not isinstance(k, (int,np.integer)) or k < 1:
        raise ValueError('Invalid LMNN identity labels or k')
    if len(np.unique(labels)) < 2:
        raise ValueError('LMNN needs at least two identities')
    targets = []
    for i in range(len(X)):
        same = np.flatnonzero((labels==labels[i]) & (np.arange(len(X)) != i))
        if len(same) < k:
            raise ValueError('LMNN needs k other samples per identity; k is not silently reduced')
        order = np.argsort(np.sum((X[same]-X[i])**2,axis=1), kind='stable')
        targets.append(same[order[:k]])
    return np.asarray(targets)


def fit_lmnn(X, labels, mu, k=5, max_triplets=1000000):
    """Weinberger/Saul Eq.14: (1-mu)*pull + mu*SUM of all impostor hinges.

    Fixed same-label Euclidean target neighbors; all differently labeled
    samples used, unit margin and M>=0. Numerical backend is a direct SDP.
    """
    if not np.isfinite(mu) or not 0 < mu <= 1 or max_triplets < 1:
        raise ValueError('Invalid LMNN mu or triplet limit')
    X, labels = _features(X), np.asarray(labels)
    targets = target_neighbors(X, labels, k)
    count = sum(k*int(np.sum(labels != label)) for label in labels)
    # ponytail: explicit SDP ceiling; use an active-set LMNN solver for larger runs.
    if count > max_triplets:
        raise ValueError(f'Exact LMNN needs {count} triplets, exceeds explicit limit {max_triplets}; none sampled')
    M = cp.Variable((X.shape[1], X.shape[1]), symmetric=True)
    constraints = [M >> 0]
    pull, push, blocks = 0, 0, []
    for i in range(len(X)):
        vneg = X[i]-X[labels != labels[i]]
        dneg = cp.sum(cp.multiply(vneg@M,vneg),axis=1)
        for j in targets[i]:
            v = X[i]-X[j]
            dpos = cp.sum(cp.multiply(np.outer(v,v),M))
            slack = cp.Variable(len(vneg))
            margin = dneg-dpos+slack-1
            cm, cs = margin >= 0, slack >= 0
            constraints.extend([cm,cs])
            pull += dpos
            push += cp.sum(slack)
            blocks.append((v,vneg,slack,cm,cs))
    problem = cp.Problem(cp.Minimize((1-mu)*pull+mu*push), constraints)
    _solve(problem)
    matrix = np.asarray(M.value)
    pull_value, push_value, violation, complementarity = 0., 0., 0., 0.
    grad = np.zeros_like(matrix)
    slack_stationarity = 0.
    for v,vneg,slack,cm,cs in blocks:
        dp, dn = float(v@matrix@v), np.sum((vneg@matrix)*vneg,axis=1)
        sv, a, beta = np.asarray(slack.value), np.asarray(cm.dual_value), np.asarray(cs.dual_value)
        pull_value += dp
        push_value += float(np.maximum(1+dp-dn,0).sum())
        residual = dn-dp+sv-1
        violation = max(violation,float(max(0.,-residual.min(),-sv.min())))
        complementarity = max(complementarity,float(np.max(np.abs(a*residual))),float(np.max(np.abs(beta*sv))))
        grad += (1-mu+a.sum())*np.outer(v,v)-(vneg.T*a)@vneg
        slack_stationarity = max(slack_stationarity,float(np.max(np.abs(mu-a-beta))))
    psd_dual = constraints[0].dual_value
    objective = (1-mu)*pull_value+mu*push_value
    diagnostics = dict(objective=objective,pull=pull_value,push=push_value,triplets=count,
        min_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),primal_violation=violation,
        stationarity_M=float(np.linalg.norm(grad-psd_dual)/max(1.,np.linalg.norm(grad),np.linalg.norm(psd_dual))),
        stationarity_slack=slack_stationarity,
        complementarity=max(complementarity,float(abs(np.sum(matrix*psd_dual)))),
        relative_objective_recheck=float(abs(objective-problem.value)/max(1.,abs(problem.value))))
    if violation>1e-6 or diagnostics['min_eigenvalue'] < -1e-6 or diagnostics['relative_objective_recheck'] > 1e-6:
        raise RuntimeError('LMNN solution failed direct primal/objective check')
    return dict(M=matrix,target_neighbors=targets,diagnostics=diagnostics,stop='convex_optimal')


def fit_itml_s(X, pairs, xi0, gamma, Z, *, filter_rule, max_sweeps=10, tol=1e-3):
    """Xu V-A6 two-stage removal, with an explicitly chosen threshold rule.

    shared_xi0: both stages use supplied xi0; remove depth distances violating
    each original hard boundary. This threshold choice is not specified by Xu.
    """
    if filter_rule != 'shared_xi0':
        raise ValueError('ITML-S requires explicit filter_rule=shared_xi0')
    pairs, xi0 = np.asarray(pairs), np.asarray(xi0, dtype=float)
    Z = _features(Z)
    if len(Z) != len(X):
        raise ValueError('Privileged features must align with ordinary samples')
    depth = fit(Z,pairs,xi0,gamma,max_sweeps=max_sweeps,tol=tol)
    keep = pairs[:,2]*(distances(Z,pairs,depth['M'])-xi0) <= 0
    if len(np.unique(pairs[keep,2])) != 2:
        raise ValueError('ITML-S filtering removed every pair of at least one label; no fallback')
    model = fit(X,pairs[keep],xi0[keep],gamma,max_sweeps=max_sweeps,tol=tol)
    model.update(depth_M=depth['M'],keep_mask=keep,depth_history=depth['history'],
        diagnostics=dict(filter_rule=filter_rule,retained_positive=int(np.sum(keep & (pairs[:,2]==1))),
                         retained_negative=int(np.sum(keep & (pairs[:,2]==-1))),
                         removed=int(np.sum(~keep)),depth_stop=depth['stop']))
    return model


def decision_scores(X, pairs, model):
    """Inference always consumes ordinary input only, including SVM+."""
    if 'w' in model:
        return pair_features(X,pairs)@model['w']+float(model['b'])
    return -distances(X,pairs,model['M'])


def lmnn_loss_gradient(X, labels, targets, M, mu):
    """Exact all-impostor Eq.14 loss/gradient, O(n²) memory, no triplet tensor."""
    gram=X@M@X.T
    distance=np.diag(gram)[:,None]+np.diag(gram)[None,:]-2*gram
    weights=np.zeros((len(X),len(X)))
    loss=0.
    for i,neighbors in enumerate(targets):
        negative=np.flatnonzero(labels!=labels[i])
        dp=distance[i,neighbors]
        hinge=1+dp[:,None]-distance[i,negative][None,:]
        active=hinge>0
        loss+=(1-mu)*dp.sum()+mu*np.maximum(hinge,0).sum()
        weights[i,neighbors]+=(1-mu)+mu*active.sum(axis=1)
        weights[i,negative]-=mu*active.sum(axis=0)
    laplacian=np.diag(weights.sum(axis=0)+weights.sum(axis=1))-weights-weights.T
    return float(loss),X.T@laplacian@X


def fit_lmnn_projected(X, labels, mu, k=5, max_iterations=1000, tol=1e-7):
    """Memory-bounded projected subgradient for the SAME convex LMNN objective.

    This numerical backend is not the author's active-set software. Stop
    labels describe numerical progress, not a global optimality certificate.
    PSD projection is part of LMNN optimization, not a repair to ITML.
    """
    if not np.isfinite([mu,tol]).all() or not 0<mu<=1 or tol<0 or max_iterations<1:
        raise ValueError('Invalid projected LMNN parameters')
    X,labels=_features(X),np.asarray(labels)
    targets=target_neighbors(X,labels,k)
    M=np.eye(X.shape[1]);history=[];stop='max_iterations'
    value,gradient=lmnn_loss_gradient(X,labels,targets,M,mu)
    step=1/max(1.,np.linalg.norm(gradient))
    for iteration in range(max_iterations):
        for backtrack in range(45):
            values,vectors=np.linalg.eigh(M-step*gradient)
            candidate=(vectors*np.maximum(values,0))@vectors.T
            next_value,next_gradient=lmnn_loss_gradient(X,labels,targets,candidate,mu)
            motion=np.linalg.norm(candidate-M)
            if next_value<=value-1e-4*motion**2/step+1e-12*max(1.,abs(value)):
                break
            step*=.5
        else:
            stop='line_search_stalled'
            break
        change=abs(next_value-value)/max(1.,abs(value))
        M,value,gradient=candidate,next_value,next_gradient
        history.append(dict(iteration=iteration+1,objective=value,relative_objective_change=change,
                            step=step,projected_step_norm=float(motion)))
        if change<tol:
            stop='objective_tolerance'
            break
        step*=1.2
    return dict(M=M,target_neighbors=targets,history=history,stop=stop,
        diagnostics=dict(objective=value,iterations=len(history),backend='exact_loss_projected_subgradient',
            min_eigenvalue=float(np.linalg.eigvalsh(M).min()),global_optimum_certified=False,
            triplets=sum(k*int(np.sum(labels!=label)) for label in labels)))
