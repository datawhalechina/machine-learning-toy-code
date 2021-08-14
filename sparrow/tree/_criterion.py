import numpy as np
import pandas as pd


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


def mse(y, n_classes=None, w=None):
    return ((y - y.mean())**2).mean()


def mae(y, n_classes=None, w=None):
    return np.abs(y - np.median(y))
