from sparrow.tree._classes import DecisionTreeClassifier

import numpy as np


def main():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)


if __name__ == "__main__":
    main()
