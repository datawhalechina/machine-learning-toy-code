from sparrow.tree.util import (
    get_score,
    get_conditional_score,
    get_criterion,
    get_feature_id,
)

import numpy as np


class Tree:

    def __init__(
        self,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        ccp_alpha,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha,

        self.X = None
        self.y = None
        self.weight = None
        self.n_classes = None

        self.left_nodes_num = 1
        self.depth = 0

    def _pruning(self):
        pass

    def _able_to_split(self, node):
        return (
            (node.depth < self.max_depth) &
            (node.data_size >= self.min_samples_split) &
            (node.weight_frac >= self.min_weight_fraction_leaf) &
            (self.left_nodes_num < self.max_leaf_nodes)
        )

    def _split(self, node, idx):
        criterion = get_criterion(self.criterion)
        Hy = get_score(self.y, idx, self.n_classes, criterion)
        node.mccp_value = Hy + self.ccp_alpha
        if not self._able_to_split(node):
            return None, None
        feature_ids = get_feature_id(
            self.X.shape[1],
            self.random_state,
            self.max_features
        )
        (
            Hyx, idx_left, idx_right, l_num, r_num, feature_id
        ) = get_conditional_score(
            self.X, self.y, self.weight, idx, self.splitter,
            self.n_classes, criterion, feature_ids, self.random_state
        )
        info_gain = Hy - Hyx
        relative_gain = (
            self.weight[idx == 1].sum() / self.weight.sum() * info_gain
        )
        if (l_num < self.min_samples_leaf) or (r_num < self.min_samples_leaf):
            return None, None
        if relative_gain < self.min_impurity_decrease:
            return None, None
        self.feature_importances_[feature_id] += relative_gain
        left_weight_frac, right_weight_frac = (
            self.weight[idx_left == 1].sum() / self.weight.sum(),
            self.weight[idx_right == 1].sum() / self.weight.sum()
        )
        node.left = Node(node.depth+1, idx_left, left_weight_frac, self)
        node.right = Node(node.depth+1, idx_right, right_weight_frac, self)
        self.left_nodes_num += 1
        self.depth = max(node.depth+1, self.depth)
        return idx_left, idx_right

    def _init_node(self):
        self.feature_importances_ = np.zeros(self.X.shape[1])
        self.root = Node(
            depth=0,
            idx=np.ones(self.X.shape[0]) == 1.,
            weight_frac=1,
            tree=self,
        )

    def _build(self, mid, idx):
        if mid is None:
            return
        idx_left, idx_right = self._split(mid, idx)
        self._build(mid.left, idx_left)
        self._build(mid.right, idx_right)


class Node:

    def __init__(
        self,
        depth,
        idx,
        weight_frac,
        tree,
        mccp_value=None,
        left=None,
        right=None,
    ):
        self.depth = depth
        self.idx = idx
        self.data_size = self.idx.sum()
        self.weight_frac = weight_frac
        self.tree = tree
        self.mccp_value = mccp_value
        self.left = left
        self.right = right
