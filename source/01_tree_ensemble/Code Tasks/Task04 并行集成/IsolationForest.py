from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
import numpy as np


class Node:

    def __init__(self, depth):
        self.depth = depth
        self.left = None
        self.right = None
        self.feature = None
        self.pivot = None


class Tree:

    def __init__(self, max_height):
        self.root = Node(0)
        self.max_height = max_height
        self.c = None

    def _build(self, node, X,):
        if X.shape[0] == 1:
            return
        if node.depth+1 > self.max_height:
            node.depth += self._c(X.shape[0])
            return
        node.feature = np.random.randint(X.shape[1])
        pivot_min = X[:, node.feature].min()
        pivot_max = X[:, node.feature].max()
        node.pivot = np.random.uniform(pivot_min, pivot_max)
        node.left, node.right = Node(node.depth+1), Node(node.depth+1)
        self._build(node.left, X[X[:, node.feature]<node.pivot])
        self._build(node.right, X[X[:, node.feature]>=node.pivot])

    def build(self, X):
        self.c = self._c(X.shape[0])
        self._build(self.root, X)

    def _c(self, n):
        if n == 1:
            return 0
        else:
            return 2 * ((np.log(n-1) + 0.5772) - (n-1)/n)

    def _get_h_score(self, node, x):
        if node.left is None and node.right is None:
            return node.depth
        if x[node.feature] < node.pivot:
            return self._get_h_score(node.left, x)
        else:
            return self._get_h_score(node.right, x)

    def get_h_score(self, x):
        return self._get_h_score(self.root, x)


class IsolationForest:

    def __init__(self, n_estimators=100, max_samples=256):
        self.n_estimator = n_estimators
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X):
        for tree_id in range(self.n_estimator):
            random_X = X[np.random.randint(0, X.shape[0], self.max_samples)]
            tree = Tree(np.log(random_X.shape[0]))
            tree.build(X)
            self.trees.append(tree)

    def predict(self, X):
        result = []
        for x in X:
            h = 0
            for tree in self.trees:
                h += tree.get_h_score(x) / tree.c
            score = np.power(2, - h/len(self.trees))
            result.append(score)
        return np.array(result)


if __name__ == "__main__":

    np.random.seed(0)

    # 5%异常点
    X_train, X_test, y_train, y_test = generate_data(
        n_train=1000, n_test=500, 
        contamination=0.05, behaviour="new", random_state=0
    )

    IF = IsolationForest()
    IF.fit(X_train)
    res = IF.predict(X_test)

    abnormal_X = X_test[res > np.quantile(res, 0.95)]

    plt.scatter(X_test[:, 0], X_test[:, 1], s=5)
    plt.scatter(
        abnormal_X[:, 0], abnormal_X[:, 1],
        s=30, edgecolors="Red", facecolor="none"
    )
    plt.show()