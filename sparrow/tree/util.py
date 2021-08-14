from sparrow.tree._criterion import gini, entropy, mse, mae
from sparrow.tree._splitter import random_split, best_split

import numpy as np


def get_criterion(func_type):
    if func_type == "gini":
        return gini
    elif func_type == "entropy":
        return entropy
    elif func_type == "mse":
        return mse
    elif func_type == "mae":
        return mae


def get_feature_id(fea_num, random_state, max_feature):
    np.random.seed(random_state)
    if max_feature == "log":
        select_num = int(np.log(fea_num))
    else:
        select_num = fea_num
    return np.random.choice(
        np.arange(fea_num),
        select_num,
        replace=False,
    )


def get_score(y, w, idx, n_classes, criterion, tree_type):
    if tree_type == "cls":
        return criterion(y[idx == 1], n_classes, w[idx == 1])
    elif tree_type == "reg":
        return criterion(y[idx == 1], w[idx == 1])


def get_conditional_score(
    X, y, w, idx, splitter,
    n_classes, criterion, feature_id, random_state, tree_type
):
    Hyx = np.infty
    best_pivot = None
    for i in feature_id:
        data = X[:, i]
        if splitter == "random":
            inner_Hyx, inner_idx_left, inner_idx_right, pivot = random_split(
                data, y, w, idx, n_classes, criterion, random_state, tree_type
            )
        elif splitter == "best":
            inner_Hyx, inner_idx_left, inner_idx_right, pivot = best_split(
                data, y, w, idx, n_classes, criterion, tree_type
            )
        if Hyx > inner_Hyx:
            Hyx, idx_left, idx_right = (
                inner_Hyx,
                inner_idx_left,
                inner_idx_right,
            )
            Hyx = inner_Hyx
            l_num = idx_left.sum()
            r_num = idx_right.sum()
            feature_id = i
            best_pivot = pivot
    return Hyx, idx_left, idx_right, l_num, r_num, feature_id, best_pivot
