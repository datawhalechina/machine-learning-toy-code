from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

import numpy as np
import pandas as pd

m1 = KNeighborsRegressor()
m2 = DecisionTreeRegressor()
m3 = LinearRegression()

models = [m1, m2, m3]

from sklearn.svm import LinearSVR

final_model = LinearSVR()

k, m = 4, len(models)

if __name__ == "__main__":

    # 模拟回归数据集
    X, y = make_regression(
        n_samples=1000, n_features=8, n_informative=4, random_state=0
    )

    final_X, _ = make_regression(
        n_samples=500, n_features=8, n_informative=4, random_state=0
    )

    final_train = pd.DataFrame(np.zeros((X.shape[0], m)))
    final_test = pd.DataFrame(np.zeros((final_X.shape[0], m)))

    kf = KFold(n_splits=k)
    for model_id in range(m):
        model = models[model_id]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            final_train.iloc[test_index, model_id] = model.predict(X_test)
            final_test.iloc[:, model_id] += model.predict(final_X)
        final_test.iloc[:, model_id] /= m

    final_model.fit(final_train, y)
    res = final_model.predict(final_test)
