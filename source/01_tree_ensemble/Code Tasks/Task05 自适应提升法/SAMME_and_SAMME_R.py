import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class AdaboostClassifier:

    def __init__(self, base_estimator, n_estimators, algorithm):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.algorithm = algorithm

        self.boostors = []
        if self.algorithm == "SAMME":
            self.boostor_weights = []

        self.classes = None

    def fit(self, X, y, **kwargs):
        w = np.repeat(1/X.shape[0], X.shape[0])
        self.classes = np.unique(y.reshape(-1)).shape[0]
        output = 0
        for n in range(self.n_estimators):
            cur_boostor = self.base_estimator(**kwargs)
            cur_boostor.fit(X, y, w)
            if self.algorithm == "SAMME":
                y_pred = cur_boostor.predict(X)
                err = (w*(y != y_pred)).sum()
                alpha = np.log((1-err)/err) + np.log(self.classes-1)
                temp_output = np.full(
                    (X.shape[0], self.classes), -1/(self.classes-1))
                temp_output[np.arange(X.shape[0]), y_pred] = 1
                self.boostors.append(cur_boostor)
                self.boostor_weights.append(alpha)
                w *= np.exp(alpha * (y != y_pred))
                w /= w.sum()
                output += temp_output * alpha
            elif self.algorithm == "SAMME.R":
                y_pred = cur_boostor.predict_proba(X)
                log_proba = np.log(y_pred + 1e-6)
                temp_output = (
                    self.classes-1)*(log_proba-log_proba.mean(1).reshape(-1,1))
                temp_y = np.full(
                    (X.shape[0], self.classes), -1/(self.classes-1))
                temp_y[np.arange(X.shape[0]), y] = 1
                self.boostors.append(cur_boostor)
                w *= np.exp(
                    (1-self.classes)/self.classes * (temp_y*log_proba).sum(1))
                w /= w.sum()
                output += temp_output
            #acc = accuracy_score(y, np.argmax(output, axis=1))
            #print(acc)

    def predict(self, X):
        result = 0
        if self.algorithm == "SAMME":
            for n in range(self.n_estimators):
                cur_pred = self.boostors[n].predict(X)
                temp_output = np.full(
                    (X.shape[0], self.classes), -1/(self.classes-1))
                temp_output[np.arange(X.shape[0]), cur_pred] = 1
                result += self.boostor_weights[n] * temp_output
        elif self.algorithm == "SAMME.R":
            for n in range(self.n_estimators):
                y_pred = self.boostors[n].predict_proba(X)
                log_proba = np.log(y_pred + 1e-6)
                temp_output = (
                    self.classes-1)*(log_proba-log_proba.mean(1).reshape(-1,1))
                result += temp_output
        return np.argmax(result, axis=1)


if __name__ == "__main__":

    X, y = make_classification(
        n_samples=10000, n_features=10,
        n_informative=5, random_state=0, n_classes=2
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    from sklearn.ensemble import AdaBoostClassifier as ABC
    clf = ABC(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=20, algorithm="SAMME"
    )
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    print("sklearn中SAMME的验证集得分为: ", accuracy_score(y_test, result))

    clf = AdaboostClassifier(
        DecisionTreeClassifier,
        20, "SAMME"
    )
    clf.fit(X_train, y_train, max_depth=1)
    result = clf.predict(X_test)
    print("使用SAMME.R集成的验证集得分为: ", accuracy_score(y_test, result))

    clf = ABC(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=20, algorithm="SAMME.R"
    )
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    print("sklearn中SAMME.R的验证集得分为: ", accuracy_score(y_test, result))

    clf = AdaboostClassifier(
        DecisionTreeClassifier,
        20, "SAMME.R"
    )
    clf.fit(X_train, y_train, max_depth=1)
    result = clf.predict(X_test)
    print("使用SAMME.R集成的验证集得分为: ", accuracy_score(y_test, result))

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    print("使用决策树桩的验证集得分为: ", accuracy_score(y_test, result))