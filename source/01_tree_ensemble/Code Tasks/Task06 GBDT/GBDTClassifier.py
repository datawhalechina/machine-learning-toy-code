from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

class GBDTClassifier:

    def __init__(self, max_depth=4, n_estimator=1000, lr=0.2):
        self.max_depth = max_depth
        self.n_estimator = n_estimator
        self.lr = lr
        self.booster = []

        self.best_round = None

    def record_score(self, y_train, y_val, train_predict, val_predict, i):
        train_predict = np.exp(train_predict) / (1 + np.exp(train_predict))
        val_predict = np.exp(val_predict) / (1 + np.exp(val_predict))
        auc_val = roc_auc_score(y_val, val_predict)
        if (i+1)%10==0:
            auc_train = roc_auc_score(y_train, train_predict)
            print("第%d轮\t训练集： %.4f\t"
                "验证集： %.4f"%(i+1, auc_train, auc_val))
        return auc_val

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=0)
        train_predict, val_predict = 0, 0
        # 按照二分类比例的初始化公式计算
        fit_val = np.log(y_train.mean() / (1 - y_train.mean()))
        next_fit_val = np.full(X_train.shape[0], fit_val)
        last_val_score = - np.infty
        for i in range(self.n_estimator):
            cur_booster = DT(max_depth=self.max_depth)
            cur_booster.fit(X_train, next_fit_val)
            train_predict += cur_booster.predict(X_train) * self.lr
            val_predict += cur_booster.predict(X_val) * self.lr
            next_fit_val = y_train - np.exp(
                train_predict) / (1 + np.exp(train_predict))
            self.booster.append(cur_booster)
            cur_val_score = self.record_score(
                y_train, y_val, train_predict, val_predict, i)
            if cur_val_score < last_val_score:
                self.best_round = i
                print("\n训练结束！最佳轮数为%d"%(i+1))
                break
            last_val_score = cur_val_score

    def predict(self, X):
        cur_predict = 0
        for i in range(self.best_round):
            cur_predict += self.lr * self.booster[i].predict(X)
        return np.exp(cur_predict) / (1 + np.exp(cur_predict))

if __name__ == "__main__":

    X, y = make_classification(
        n_samples=10000, n_features=50, n_informative=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    model = GBDTClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    auc = roc_auc_score(y_test, prediction)
    print("\n测试集的AUC为 %.4f"%(auc))