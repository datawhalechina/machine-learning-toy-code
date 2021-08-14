import numpy as np


def _estimate_py(y, n_classes, w):
    prob = np.zeros(n_classes)
    for i in range(y.shape[0]):
        prob[y[i]] += w[i]
    return prob / prob.sum()


def gini(y, n_classes, w):
    prob = _estimate_py(y, n_classes, w)
    return 1 - (prob**2).sum()


def entropy(y, n_classes, w):
    prob = _estimate_py(y, n_classes, w)
    log_prob = np.log2(prob, out=np.zeros_like(prob), where=(prob != 0))
    return - (prob * log_prob).sum()


def mse(y, w=None):
    return ((w * y**2).sum()) / w.sum() - ((w * y).sum() / w.sum()) ** 2


def mae(y, w=None):
    return np.abs(w*y - np.median(w*y)).sum() / w.sum()
