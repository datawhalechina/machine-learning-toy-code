import numpy as np
import pandas as pd


def _compute_score(_df, n_classes, criterion, tree_type, df):
    if tree_type == "cls":
        res = _df.w.sum() / df.w.sum() * (
            criterion(_df.y.values, n_classes, _df.w.values)
        )
    elif tree_type == "reg":
        res = _df.w.sum() / df.w.sum() * (
            criterion(_df.y.values, _df.w.values)
        )
    return res


def _get_split_score(X, y, w, idx, n_classes, criterion, tree_type):
    df = pd.DataFrame({"X": X[idx == 1], "y": y[idx == 1], "w": w[idx == 1]})
    res = df.groupby("X")[["y", "w"]].apply(
        lambda _df: _compute_score(_df, n_classes, criterion, tree_type, df)
    )
    return res.sum()


def random_split(
    data, y, w, idx, n_classes, criterion, random_state, tree_type
):
    cat_X = data <= np.random.uniform(data.min(), data.max())
    Hyx = _get_split_score(cat_X, y, w, idx, n_classes, criterion, tree_type)
    return Hyx, (idx == 1) & cat_X, (idx == 1) & ~cat_X


def best_split(data, y, w, idx, n_classes, criterion, tree_type):
    best_Hyx = np.infty
    for item in data[:-1]:
        cat_X_iter = data <= item
        Hyx = _get_split_score(
            cat_X_iter, y, w, idx, n_classes, criterion, tree_type
        )
        if best_Hyx > Hyx:
            best_Hyx = Hyx
            cat_X = cat_X_iter
    return best_Hyx, (idx == 1) & cat_X, (idx == 1) & ~cat_X
