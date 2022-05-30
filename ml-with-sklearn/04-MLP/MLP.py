from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)


clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(15,15), random_state=1)

clf.fit(X_train, y_train)


score = clf.score(X_test, y_test)


