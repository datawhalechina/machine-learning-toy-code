from sparrow.tree._tree import Tree

import numpy as np


class BaseDecisionTree:

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
        class_weight="equal",
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

    def _adjust_sample_weight(self, sample_weight):
        if sample_weight is None:
            sample_weight = np.full(
                self.tree.X.shape[0], 1 / self.tree.X.shape[0]
            )
        if self.tree.tree_type == "cls":
            sample_weight = sample_weight * self._get_class_weight(
                self.class_weight, self.tree.y, self.tree.n_classes
            )
        return sample_weight

    def _get_class_weight(self, class_weight, target, n_classes):
        if class_weight == "balanced":
            class_weight = (
                target.shape[0] / (n_classes * np.bincount(target))
            )
            class_weight = class_weight[target]
        elif class_weight == "equal":
            class_weight = np.ones(target.shape[0])
        elif isinstance(class_weight, dict):
            class_weight = np.array(list(class_weight.values()))[target]
        return class_weight

    def fit(self, X, y, sample_weight=None):
        self.tree.X = X
        self.tree.y = y
        self.tree.n_classes = y.max() + 1
        self.tree.weight = self._adjust_sample_weight(sample_weight)
        self.tree.init_node()
        self.tree.build(self.tree.root, np.ones(X.shape[0]))
        self.tree.pruning()
        self.tree.feature_importances_ = (
            self.tree.feature_importances_ /
            self.tree.feature_importances_.sum()
        )
        return self


class DecisionTreeClassifier(BaseDecisionTree):

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
        class_weight="equal",
        ccp_alpha=0.0,
    ):
        super().__init__(
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
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )
        self.tree.tree_type = "cls"
        self.class_weight = class_weight

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


class DecisionTreeRegressor(BaseDecisionTree):

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
        class_weight="equal",
        ccp_alpha=0.0,
    ):
        super().__init__(
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
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )
        self.tree.tree_type = "reg"

    def predict(self, X):
        pass
