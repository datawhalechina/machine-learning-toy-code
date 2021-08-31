from sparrow.tree._classes import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn import tree
import matplotlib.pyplot as plt

import numpy as np

def main():
    pass

"""
def main():
    X, y = make_regression(
        100, 6, 3, 1,
        random_state=0
    )
    #X, y = make_regression(
    #    100, 6, 3, 1,
    #    random_state=0, weights=[0.65, 0.35]
    #)
    y *= 1

    np.random.seed(2)
    #X, y = np.random.rand(100, 4), np.random.rand(100)*10

    #X, y = np.random.randint(0,10,(100, 4)), np.random.rand(100) * 100
    clf1 = DecisionTreeRegressor(criterion="mse", max_depth=3, ccp_alpha=100000)
    w = np.random.randint(5,10,100)
    #w = np.ones(y.shape[0])
    clf1.fit(X, y, sample_weight=w)
    clf2 = dt(criterion="mse", max_depth=3, ccp_alpha=100000)
    clf2.fit(X, y, sample_weight=w)
    #print(clf1.tree.feature_importances_)
    #print(clf2.feature_importances_)
    print(clf1.predict(X[-5:, :]),clf2.predict(X[-5:, :]))
    def p(node):
        if node is None:
            return
        print(node.depth, node.child_mccp_value)
        p(node.left)
        p(node.right)
    p(clf1.tree.root)
    tree.plot_tree(clf2)
    plt.show()
"""

if __name__ == "__main__":
    main()
