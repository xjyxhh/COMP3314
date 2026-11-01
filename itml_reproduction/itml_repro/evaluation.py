import numpy as np
from .metrics import distances


def verification(X, pairs, M):
    """AP (step integral) and ROC AUC; equal scores grouped, same identity positive."""
    score = -distances(X, pairs, M)
    return verification_scores(score, pairs[:, 2])


def verification_scores(score, labels):
    """Common AP/AUC for metric scores or classifier decision values; +1 positive."""
    score, labels = np.asarray(score), np.asarray(labels)
    if (score.ndim != 1 or labels.shape != score.shape or not np.isfinite(score).all()
            or not np.isin(labels, [-1, 1]).all()):
        raise ValueError('Expected finite scores and aligned +/-1 labels')
    truth = labels == 1
    if not truth.any() or truth.all():
        raise ValueError('AP/AUC evaluation needs both labels')
    order = np.argsort(-score, kind='stable')
    score, truth = score[order], truth[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(score)-1]
    tp = np.cumsum(truth)[ends]
    fp = ends+1-tp
    recall = tp/truth.sum()
    precision = tp/(tp+fp)
    ap = np.sum(np.diff(np.r_[0.,recall])*precision)
    tpr, fpr = np.r_[0.,recall], np.r_[0.,fp/(~truth).sum()]
    auc = np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)
    return dict(AP=float(ap), AUC=float(auc))


def retrieval(X_query, X_gallery, query_ids, gallery_ids, M):
    """BIWI-compatible Rank-1/CMC; caller supplies explicit query/gallery protocol.

    Stable gallery-order tie break. Query and gallery must exclude the same sample.
    """
    d = X_query[:,None,:]-X_gallery[None,:,:]
    distance = np.einsum('qgd,df,qgf->qg',d,M,d)
    matches = np.asarray(query_ids)[:,None] == np.asarray(gallery_ids)[None,:]
    if not matches.any(axis=1).all():
        raise ValueError('Every query must have a gallery match')
    order = np.argsort(distance, axis=1, kind='stable')
    ranked = np.take_along_axis(matches,order,axis=1)
    cmc = np.maximum.accumulate(ranked,axis=1).mean(axis=0)
    return dict(rank1=float(cmc[0]), CMC=cmc.tolist())
