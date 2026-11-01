"""Equations refer to Xu et al. (2015), unless Davis et al. is named."""
import numpy as np


def distances(X, pairs, M):
    """Eq. (1): rows are x_i; output is squared Mahalanobis distance."""
    v = X[pairs[:, 0]] - X[pairs[:, 1]]
    return np.einsum('ij,ij->i', v @ M, v)


def logdet(A):
    """D_ld(A, I) from Eq. (3), rather than log(det(A))."""
    A = np.asarray(A, dtype=float)
    if (A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(A).all()
            or not np.allclose(A, A.T, rtol=1e-12, atol=1e-12)):
        raise ValueError('LogDet divergence requires a finite symmetric square matrix')
    # Eq. (3), reference I. Cholesky enforces the finite-LogDet SPD domain.
    C = np.linalg.cholesky(A)
    return float(np.trace(A) - 2 * np.log(np.diag(C)).sum() - len(A))


def burg(x, x0):
    x, x0 = np.asarray(x, dtype=float), np.asarray(x0, dtype=float)
    if (not np.isfinite(x).all() or not np.isfinite(x0).all()
            or np.any(x <= 0) or np.any(x0 <= 0)):
        raise ValueError('Burg loss requires finite strictly positive arguments')
    q = x / x0
    if np.any(q <= 0):
        raise ValueError('Burg loss requires strictly positive arguments')
    return float(np.sum(q - np.log(q) - 1))


def plus_step(M, P, xi, v, w, y, gamma, lam):
    """Algorithm 1, steps 4–9. All RHS quantities use the OLD state.

    v=x_i-x_j, w=z_i-z_j; y=+1 (S), -1 (D).
    No clipping, PSD projection, surrogate loss, or learned depth at inference.
    """
    mv, pw = M @ v, P @ w
    r, s = float(v @ mv), float(w @ pw)
    if y not in (-1, 1):
        raise ValueError('Pair label must be +1 or -1')
    if not np.isfinite([r, s, xi, gamma, lam]).all() or min(r, s, xi, gamma, lam) <= 0:
        raise ValueError('Eqs. (23)/(24) require finite positive r,s,xi,gamma,lambda')
    alpha = max(0., (gamma / xi + lam / s - (lam + gamma) / r)
                / (y * (lam + gamma + 1)))  # Eq. (23)
    beta = lam / (lam + gamma) * (gamma / s - gamma / xi + y * alpha)  # (24)
    dm, dp = 1 + y * alpha * r, lam - beta * s
    if min(dm, dp) <= 0:
        raise FloatingPointError('Nonpositive rank-one denominator')
    return (M - y * alpha / dm * np.outer(mv, mv),  # (11)
            P + beta / dp * np.outer(pw, pw),      # (12)
            lam * s / dp, alpha, beta)             # (13)


def _relative_norm(new, old):
    # IV-D leaves comparison timing/zero denominator unspecified: compare sweeps.
    a, b = np.linalg.norm(new), np.linalg.norm(old)
    return abs(a - b) / max(b, np.finfo(float).eps)


def fit(X, pairs, xi0, gamma, Z=None, lam=1., max_sweeps=10, tol=1e-3, on_sweep=None,
        privileged_mask=None):
    """ITML: Davis et al. (2007), corrected soft-constraint dual coordinate update.

    ITML+: literal Xu Algorithm 1 / (11)–(13),(23)–(24), with persistent xi.
    Deliberately no unreported Dykstra correction added to the ITML+ algorithm.
    A finite update budget or stable dual norms is NOT a feasibility certificate.
    on_sweep receives live state after each sweep; copy arrays if retaining them.
    Partial ITML+ (Eqs. 5/28): a Boolean mask selects pairs with depth.
    Other pairs use the ITML update and leave P unchanged (Section IV-E).
    """
    X = np.asarray(X, dtype=float)
    pairs = np.asarray(pairs)
    xi0 = np.asarray(xi0, dtype=float)
    if (X.ndim != 2 or not np.isfinite(X).all() or not np.isfinite([gamma, lam, tol]).all() or gamma <= 0 or lam <= 0
            or max_sweeps < 1 or tol < 0):
        raise ValueError('Invalid features or solver parameters')
    if (pairs.ndim != 2 or pairs.shape[1] != 3 or len(pairs) == 0
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs[:, :2] < 0) or np.any(pairs[:, :2] >= len(X))
            or not np.isin(pairs[:, 2], [-1, 1]).all()
            or xi0.shape != (len(pairs),) or not np.isfinite(xi0).all()
            or np.any(xi0 <= 0)):
        raise ValueError('Invalid pairs or xi0')
    if privileged_mask is None:
        privileged_mask = np.full(len(pairs), Z is not None, dtype=bool)
    else:
        privileged_mask = np.asarray(privileged_mask)
        if privileged_mask.dtype != np.bool_ or privileged_mask.shape != (len(pairs),):
            raise ValueError('privileged_mask must be a Boolean vector aligned with pairs')
    if not privileged_mask.any():
        Z = None  # Eq. (5) at 0%: P is unused, its minimizing value is I.
    elif Z is None:
        raise ValueError('Privileged pairs require depth features')
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        used = np.unique(pairs[privileged_mask, :2])
        if Z.ndim != 2 or len(Z) != len(X) or not np.isfinite(Z[used]).all():
            raise ValueError('Depth must be aligned and finite for every privileged pair')
    M = np.eye(X.shape[1])
    P = None if Z is None else np.eye(Z.shape[1])
    xi = xi0.copy()
    dual = np.zeros(len(pairs))
    beta_vec = np.zeros(len(pairs))
    history = []
    stop = 'max_sweeps'
    for epoch in range(max_sweeps):
        old_dual, old_beta = dual.copy(), beta_vec.copy()
        for k, (i, j, y) in enumerate(pairs):
            v = X[i] - X[j]
            if privileged_mask[k]:
                M, P, xi[k], dual[k], beta_vec[k] = plus_step(
                    M, P, xi[k], v, Z[i] - Z[j], y, gamma, lam)
            else:
                mv = M @ v
                r = float(v @ mv)
                if r <= 0:
                    raise ValueError('ITML requires positive pair distances')
                # Davis soft ITML: delta=y, dual=lambda_ij (NOT Xu's lambda).
                # Step a has opposite sign to the nonnegative constraint multiplier.
                # gamma/(gamma+1) follows by equating new distance and new xi.
                a = min(dual[k], y * gamma / (gamma + 1) * (1/r - 1/xi[k]))
                denom = 1 - y * a * r
                xi_denom = gamma + y * a * xi[k]
                if min(denom, xi_denom) <= 0:
                    raise FloatingPointError('ITML nonpositive denominator')
                M += y * a / denom * np.outer(mv, mv)
                xi[k] = gamma * xi[k] / xi_denom
                dual[k] -= a  # essential dual correction, Davis Algorithm 1
        rgb = distances(X, pairs, M)
        target = xi.copy()
        if P is not None:
            target[privileged_mask] = distances(Z, pairs[privileged_mask], P)
        obj = logdet(M) + gamma * burg(target, xi0)
        if P is not None:
            obj += lam * logdet(P)
        row = dict(sweep=epoch+1, objective=obj,
                   max_inequality_violation=float(np.maximum(pairs[:, 2]*(rgb-target), 0).max()),
                   max_equality_residual=0. if P is None else float(np.max(
                       np.abs(xi[privileged_mask]-target[privileged_mask]))),
                   min_eigenvalue_M=float(np.linalg.eigvalsh(M).min()),
                   dual_norm_change=_relative_norm(dual, old_dual))
        if P is not None:
            row['min_eigenvalue_P'] = float(np.linalg.eigvalsh(P).min())
            row['beta_norm_change'] = _relative_norm(beta_vec, old_beta)
            for name, matrix in [('M', M), ('P', P)]:
                eigenvalues = np.linalg.eigvalsh(matrix)
                row['max_eigenvalue_' + name] = float(eigenvalues[-1])
                row['trace_' + name] = float(np.trace(matrix))
                row['condition_number_' + name] = float(eigenvalues[-1] / eigenvalues[0])
            for name, values in [('rgb', rgb), ('depth', target[privileged_mask])]:
                row[name + '_distance_min'] = float(values.min())
                row[name + '_distance_median'] = float(np.median(values))
                row[name + '_distance_max'] = float(values.max())
            # Symmetric, pairwise scale normalization; no epsilon changes distances.
            row['relative_inequality_violation'] = float(np.max(
                np.maximum(pairs[:, 2] * (rgb - target), 0) / np.maximum(rgb, target)))
            row['relative_equality_residual'] = float(np.max(
                np.abs(xi[privileged_mask] - target[privileged_mask]) /
                np.maximum(xi[privileged_mask], target[privileged_mask])))
        history.append(row)
        if on_sweep is not None:
            on_sweep(epoch+1, dict(M=M, P=P, xi=xi, dual=dual, beta=beta_vec, history=history))
        if epoch > 0 and row['dual_norm_change'] < tol and (P is None or row['beta_norm_change'] < tol):
            stop = 'dual_norm_tolerance'
            break
    return dict(M=M, P=P, xi=xi, dual=dual, beta=beta_vec, history=history, stop=stop)
