from sparrow.tree._classes import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn import tree
import matplotlib.pyplot as plt

import numpy as np


def main():
    X, y = make_classification(
        100, 6, 3, 0, 0, 3,
        random_state=0
    )
    #X, y = make_regression(
    #    100, 6, 3, 1,
    #    random_state=0, weights=[0.65, 0.35]
    #)
    #y *= 1000
    #X, y = np.array([[0,2,3,5,6,2,5,6,2,2],[2,3,6,2,2,1,4,4,6,3],[5,8,1,4,2,6,3,5,6,2]]).T, np.array([0.3,0.5,0.1,0.2,0.1,0.3,0.4,0.1,0.6,0.1])*1
    np.random.seed(1)

    #X, y = np.random.randint(0,10,(100, 4)), np.random.rand(100) * 100
    clf1 = DecisionTreeClassifier(criterion="gini", max_depth=3)
    #w = np.random.randint(5,10,100)
    w = np.ones(y.shape[0])
    clf1.fit(X, y, sample_weight=w)
    clf2 = dt(criterion="gini", max_depth=3)
    clf2.fit(X, y, sample_weight=w)
    print(clf1.tree.feature_importances_)
    print(clf2.feature_importances_)
    print((clf1.predict(X[:, :])==clf2.predict(X[:, :])).mean())
    print(clf1.tree.root.right.right.right.leaf_idx.mean())
    tree.plot_tree(clf2)
    plt.show()


if __name__ == "__main__":
    main()
