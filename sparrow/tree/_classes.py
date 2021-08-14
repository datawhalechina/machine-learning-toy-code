from sparrow.tree.util import get_class_weight
from sparrow.tree._tree import Tree

import numpy as np


class DecisionTreeClassifier:

    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="all",
        max_leaf_nodes=2**(6+1)-1,
        random_state=0,
        min_impurity_decrease=0.0,
        class_weight="balanced",
        ccp_alpha=0.0,
    ):
        self.tree = Tree(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        self.tree.X = X
        self.tree.y = y
        self.tree.n_classes = y.max() + 1
        if sample_weight is None:
            sample_weight = np.full(X.shape[0], 1 / X.shape[0])
        final_weight = sample_weight * get_class_weight(
            self.class_weight, y, self.tree.n_classes
        )
        self.tree.weight = final_weight
        self.tree.init_node()
        self.tree.build(self.tree.root, np.ones(X.shape[0]))
        self.tree.pruning()
        self.tree.feature_importances_ = (
            self.tree.feature_importances_ /
            self.tree.feature_importances_.sum()
        )
        return self

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
