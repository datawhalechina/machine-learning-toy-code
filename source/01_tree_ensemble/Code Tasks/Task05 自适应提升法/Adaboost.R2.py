import numpy as np

class AdaboostRegressor:

    def __init__(self, base_estimator, n_estimator):
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

        self.booster = []
        self.weight = []

    def fit(self, X, y, **kwargs):
        w = np.ones(X.shape[0]) / X.shape[0]
        for n in range(self.n_estimator):
            cur_reg = self.base_estimator(**kwargs)
            cur_reg.fit(X, y)
            y_pred = cur_reg.predict(X)
            e = np.abs(y - y_pred)
            e /= e.max()
            err = (w*e).sum()
            beta = (1-err)/err
            alpha = np.log( + 1e-6)
            w *= np.power(beta, 1-e)
            w /= w.sum()
            self.booster.append(cur_reg)
            self.weight.append(alpha)
    
    def predict(self, X):
        pass