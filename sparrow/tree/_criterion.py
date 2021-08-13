import numpy as np


def _estimate_py(y, n_classes):
    res = np.r_[y, [n_classes-1]]
    res = np.bincount(res)
    res[-1] -= 1
    res = res / res.sum()
    return res


def gini(y, n_classes):
    prob = _estimate_py(y, n_classes)
    return 1 - (prob**2).sum()


def entropy(y, n_classes):
    prob = _estimate_py(y, n_classes)
    log_prob = np.log2(prob, out=np.zeros_like(prob), where=(prob != 0))
    return - (prob * log_prob).sum()


def mse(y, n_classes=None):
    return ((y - y.mean())**2).mean()


def mae(y, n_classes=None):
    return np.abs(y - np.median(y))
