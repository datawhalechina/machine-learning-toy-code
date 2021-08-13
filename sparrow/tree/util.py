import numpy as np
import pandas as pd


def get_class_weight(class_weight, target, n_classes):
    if class_weight == "balanced":
        class_weight = (
            target.shape[0] / (n_classes * np.bincount(target))
        )
    else:
        class_weight = class_weight.values()[target]
    return class_weight


def _estimate_py(y, n_classes):
    res = np.r_[y, [n_classes-1]]
    res = np.bincount(res)
    res[-1] -= 1
    res = res / res.sum()
    return res


def gini(prob):
    return 1 - (prob**2).sum()


def get_info_gain(y, idx, n_classes):
    prob = _estimate_py(y[idx == 1], n_classes)
    return gini(prob)


def _make_discrete(X):
    pass


def get_conditional_info_gain(X, y, w, idx, n_classes):
    Hyx = -np.infty
    for i in range(X.shape[1]):
        data = X[:, i]
        cat_X = data <= np.median(data)
        df = pd.DataFrame({"X": cat_X, "y": y, "w": w}).loc[idx == 1]
        res = df.groupby("X")[["y", "w"]].apply(
            lambda _df: (
                _df.w.sum() / w.sum() * (
                    gini(_estimate_py(y, n_classes))
                )
            )
        ).sum()
        if Hyx < res:
            Hyx = res
            idx_left = (idx == 1) & cat_X
            idx_right = (idx == 1) & ~cat_X
            l_num = idx_left.sum()
            r_num = idx_right.sum()
            feature_id = i
    return Hyx, idx_left, idx_right, l_num, r_num, feature_id
