from sparrow.tree._classes import DecisionTreeClassifier
from sklearn.datasets import make_classification

import numpy as np


def main():
    X, y = make_classification(
        500, 4, 2, 0, 0, 2,
        random_state=10, weights=[0.75, 0.25]
    )
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf.fit(X, y)
    print(X.shape)
    print(y.mean())
    print(clf.tree.depth)


if __name__ == "__main__":
    main()
