"""Experimental dual block ascent for Eq. (7), NOT literal Algorithm 1.

Each block optimizes the TOTAL alpha>=0 and unrestricted beta for one pair.
Inverse matrices and inverse xi retain contributions from all other blocks.
SciPy is an optional dependency for this solver only.
"""
import numpy as np
from scipy.optimize import minimize

from .metrics import fit, distances, logdet, burg


def fit_corrected(X, pairs, xi0, gamma, Z, lam=1., max_sweeps=1000, tol=1e-7, on_sweep=None):
    # Reuse the original public input validation (one discarded sweep).
    fit(X, pairs, xi0, gamma, Z=Z, lam=lam, max_sweeps=1, tol=0.)
    X, Z, xi0 = np.asarray(X, float), np.asarray(Z, float), np.asarray(xi0, float)
    pairs = np.asarray(pairs)
    if max_sweeps < 1 or not np.isfinite(tol) or tol < 0:
        raise ValueError('Invalid iteration budget or tolerance')
    M, P, xi = np.eye(X.shape[1]), np.eye(Z.shape[1]), xi0.copy()
    alpha, beta = np.zeros(len(pairs)), np.zeros(len(pairs))
    v_all = X[pairs[:, 0]] - X[pairs[:, 1]]
    w_all = Z[pairs[:, 0]] - Z[pairs[:, 1]]
    history = []
    stop = 'max_sweeps'
    for epoch in range(max_sweeps):
        for k, (i, j, y) in enumerate(pairs):
            v, w = X[i]-X[j], Z[i]-Z[j]
            mv, pw = M@v, P@w
            r, s = float(v@mv), float(w@pw)
            # u=y*delta_alpha*r, v=delta_beta*s/lambda.
            q = np.array([-xi[k]/(gamma*r), xi[k]*lam/(gamma*s)])

            def objective(z):
                den = np.array([1+z[0], 1-z[1], 1+q@z])
                if np.any(den <= 0):
                    return 1e100, np.zeros(2)
                value = -np.log(den[0])-lam*np.log(den[1])-gamma*np.log(den[2])
                grad = np.array([-1/den[0], lam/den[1]])-gamma*q/den[2]
                return value, grad

            lower = -alpha[k]*r if y == 1 else None
            upper = alpha[k]*r if y == -1 else None
            result = minimize(objective, np.zeros(2), jac=True, method='SLSQP',
                bounds=[(lower, upper), (None, None)],
                constraints=[dict(type='ineq',
                    fun=lambda z: np.array([1+z[0], 1-z[1], 1+q@z])-1e-12,
                    jac=lambda z: np.array([[1., 0.], [0., -1.], q]))],
                options=dict(ftol=1e-16, maxiter=100))
            if not result.success:
                raise RuntimeError(f'Block {epoch}/{k}: {result.message}')
            u, b = result.x
            if objective(result.x)[0] > 1e-11:
                raise RuntimeError('Dual block objective decreased')
            M -= (u/r)/(1+u)*np.outer(mv, mv)
            P += (b/s)/(1-b)*np.outer(pw, pw)
            xi[k] /= 1+q@result.x
            alpha[k] += y*u/r
            beta[k] += lam*b/s
        r, s = distances(X, pairs, M), distances(Z, pairs, P)
        c = pairs[:, 2]*(r-s)
        row = dict(sweep=epoch+1,
            objective=logdet(M)+lam*logdet(P)+gamma*burg(s, xi0),
            relative_violation=float(np.max(np.maximum(c, 0)/np.maximum(r,s))),
            relative_equality=float(np.max(np.abs(xi-s)/np.maximum(xi,s))),
            complementarity=float(np.max(np.abs(alpha*c))),
            min_eigenvalue_M=float(np.linalg.eigvalsh(M).min()),
            min_eigenvalue_P=float(np.linalg.eigvalsh(P).min()))
        # Global stationarity must survive every sweep, not only one pair update.
        lhs = [np.linalg.inv(M), lam*np.linalg.inv(P), gamma/xi]
        rhs = [np.eye(X.shape[1])+np.einsum('i,ij,ik->jk', pairs[:, 2]*alpha, v_all, v_all),
               lam*np.eye(Z.shape[1])-np.einsum('i,ij,ik->jk', beta, w_all, w_all),
               gamma/xi0-pairs[:, 2]*alpha+beta]
        for name, left, right in zip(('M', 'P', 'xi'), lhs, rhs):
            scale = max(1., np.max(np.abs(left)), np.max(np.abs(right)))
            row['stationarity_'+name] = float(np.linalg.norm(left/scale-right/scale) /
                max(1/scale, np.linalg.norm(left/scale), np.linalg.norm(right/scale)))
        row['dual_nonnegative_violation'] = float(max(0., -alpha.min()))
        history.append(row)
        if on_sweep is not None:
            on_sweep(epoch+1, dict(M=M, P=P, xi=xi, dual=alpha, beta=beta, history=history))
        if max(row['relative_violation'], row['relative_equality'], row['complementarity'], row['stationarity_M'], row['stationarity_P'],
               row['stationarity_xi'], row['dual_nonnegative_violation']) < tol:
            stop = 'kkt_tolerance'
            break
    return dict(M=M, P=P, xi=xi, dual=alpha, beta=beta, history=history, stop=stop)
