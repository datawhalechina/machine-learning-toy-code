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


def get_class_weight(class_weight, target, n_classes):
    if class_weight == "balanced":
        class_weight = (
            target.shape[0] / (n_classes * np.bincount(target))
        )
        class_weight = class_weight[target]
    elif class_weight == "equal":
        class_weight = np.ones(target.shape[0])
    else:
        class_weight = class_weight.values()[target]
    return class_weight


def get_score(y, idx, n_classes, criterion):
    return criterion(y[idx == 1], n_classes)


def get_conditional_score(
    X, y, w, idx, splitter,
    n_classes, criterion, feature_id, random_state
):
    Hyx = np.infty
    for i in feature_id:
        data = X[:, i]
        if splitter == "random":
            inner_Hyx, inner_idx_left, inner_idx_right = random_split(
                data, y, w, idx, n_classes, criterion, random_state
            )
        elif splitter == "best":
            inner_Hyx, inner_idx_left, inner_idx_right = best_split(
                data, y, w, idx, n_classes, criterion
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
    return Hyx, idx_left, idx_right, l_num, r_num, feature_id
