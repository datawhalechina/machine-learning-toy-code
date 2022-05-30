import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression



# 数据
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Test score with L1 penalty: %.4f" % score)