import numpy as np


def fit_pca(X, dimension):
    """Centered, unwhitened PCA; fit ONLY training rows, independently per modality."""
    if not np.isfinite(X).all() or not 1 <= dimension <= min(len(X)-1, X.shape[1]):
        raise ValueError('PCA dimension exceeds training-sample/feature limit')
    mean = X.mean(axis=0)
    _, singular, vt = np.linalg.svd(X-mean, full_matrices=False)
    if singular[dimension-1] <= np.finfo(float).eps * max(X.shape) * singular[0]:
        raise ValueError('PCA dimension exceeds numerical training rank')
    return mean, vt[:dimension]


def transform(X, model):
    mean, components = model
    return (X-mean) @ components.T
