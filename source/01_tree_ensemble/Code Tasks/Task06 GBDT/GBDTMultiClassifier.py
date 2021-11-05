from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

def one_hot(y):
    res = np.zeros((y.size, y.max()+1))
    res[np.arange(y.size), y] = 1
    return res

class GBDTMultiClassifier:

    def __init__(self, max_depth=4, n_estimator=1000, lr=0.2):
        self.max_depth = max_depth
        self.n_estimator = n_estimator
        self.lr = lr
        self.booster = []

        self.n_classes = None
        self.best_round = None

    def get_init_val(self, y):
        init_val = []
        y = np.argmax(y, axis=1)
        for c in range(self.n_classes):
            init_val.append(np.log((y==c).mean()))
        return np.full((y.shape[0], self.n_classes), init_val)

    def record_score(self, y_train, y_val, train_predict, val_predict, i):
        train_predict = np.exp(train_predict) / np.exp(
            train_predict).sum(1).reshape(-1, 1)
        val_predict = np.exp(val_predict) / np.exp(
            val_predict).sum(1).reshape(-1, 1)
        auc_val = roc_auc_score(y_val, val_predict)
        if (i+1)%10==0:
            auc_train = roc_auc_score(y_train, train_predict)
            print("第%d轮\t训练集： %.4f\t"
                "验证集： %.4f"%(i+1, auc_train, auc_val))
        return auc_val

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=0)
        self.n_classes = y.shape[1]
        train_predict = np.zeros((X_train.shape[0], self.n_classes))
        val_predict = np.zeros((X_val.shape[0], self.n_classes))
        next_fit_val = self.get_init_val(y_train)
        last_val_score = - np.infty
        for i in range(self.n_estimator):
            last_train = train_predict.copy()
            self.booster.append([])
            for m in range(self.n_classes):
                cur_booster = DT(max_depth=self.max_depth)
                cur_booster.fit(X_train, next_fit_val[:, m])
                train_predict[:, m] += cur_booster.predict(X_train) * self.lr
                val_predict[:, m] += cur_booster.predict(X_val) * self.lr
                next_fit_val[:, m] = y_train[:, m] - np.exp(
                    last_train[:, m]) / np.exp(last_train).sum(1)
                self.booster[-1].append(cur_booster)
            cur_val_score = self.record_score(
                y_train, y_val, train_predict, val_predict, i)
            if cur_val_score < last_val_score:
                self.best_round = i
                print("\n训练结束！最佳轮数为%d"%(i+1))
                break
            last_val_score = cur_val_score

    def predict(self, X):
        cur_predict = np.zeros((X.shape[0], self.n_classes))
        for i in range(self.best_round):
            for m in range(self.n_classes):
                cur_predict[:, m] += self.lr * self.booster[i][m].predict(X)
        return np.exp(cur_predict) / np.exp(cur_predict).sum(1).reshape(-1, 1)


if __name__ == "__main__":

    X, y = make_classification(
        n_samples=10000, n_features=50, n_informative=20,
        n_classes=3, random_state=1)
    y = one_hot(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    model = GBDTMultiClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    auc = roc_auc_score(y_test, prediction)
    print("\n测试集的AUC为 %.4f"%(auc))