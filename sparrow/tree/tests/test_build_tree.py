from sparrow.tree._classes import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn import tree
import matplotlib.pyplot as plt

import numpy as np


def main():
    #X, y = make_classification(
    #    500, 4, 2, 0, 0, 2,
    #    random_state=10, weights=[0.75, 0.25]
    #)
    X, y = np.array([[0,2,3,5,6,2,5,6,2,2],[2,3,6,2,3,5,2,4,6,3],[5,8,1,4,2,6,3,5,6,2]]).T, np.array([0,1,1,0,0,1,1,0,1,1])

    clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=2, class_weight={0:150, 1:100})
    np.random.seed(0)
    w = np.random.randint(5,10,10)
    clf1.fit(X, y, sample_weight=w)
    clf2 = dt(criterion="entropy", max_depth=2,class_weight={0:150, 1:100})
    clf2.fit(X, y, sample_weight=w)
    print(clf1.tree.feature_importances_)
    print(clf1.tree.depth)
    print(clf1.tree.left_nodes_num)
    print(clf2.feature_importances_)
    tree.plot_tree(clf2)
    plt.show()


if __name__ == "__main__":
    main()
