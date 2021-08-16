from sparrow.tree.util import (
    get_score,
    get_conditional_score,
    get_criterion,
    get_feature_id,
)

import numpy as np
from weightedstats import numpy_weighted_median as wmd


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
        self.ccp_alpha = ccp_alpha

        self.X = None
        self.y = None
        self.weight = None
        self.n_classes = None

    def _get_child_mccp_value(self, node):
        if node is None:
            return 0, 0
        elif node.left is None and node.right is None:
            return node.Hy * self.weight[node.idx].sum(), 1
        left_w, left_n = self._get_child_mccp_value(node.left)
        right_w, right_n = self._get_child_mccp_value(node.right)
        leaf_w = left_w + right_w
        leaf_node_num = left_n + right_n
        node.child_mccp_value = (
            self.ccp_alpha * leaf_node_num + (
                leaf_w / self.weight[node.idx].sum()))
        return leaf_w, leaf_node_num

    def _pruning_subtree(self, node):
        if node is None or node.left is None or node.right is None:
            return False
        elif node.Hy < node.child_mccp_value:
            node.left, node.right = None, None
            return True
        else:
            bool_res = (
                self._pruning_subtree(node.left) &
                self._pruning_subtree(node.right))
            return bool_res

    def pruning(self):
        self._get_child_mccp_value(self.root)
        need_prune = True
        while need_prune:
            need_prune = self._pruning_subtree(self.root)

    def _able_to_split(self, node):
        return (
            (node.depth < self.max_depth) &
            (node.data_size >= self.min_samples_split) &
            (node.weight_frac >= self.min_weight_fraction_leaf) &
            (self.left_nodes_num < self.max_leaf_nodes)
        )

    def _split(self, node, idx):
        criterion = get_criterion(self.criterion)
        Hy = get_score(
            self.y, self.weight, idx, self.n_classes,
            criterion, self.tree_type)
        node.Hy = Hy  # 不纯度
        if not self._able_to_split(node):
            return None, None, None, None
        feature_ids = get_feature_id(
            self.X.shape[1],
            self.random_state,
            self.max_features)
        (
            Hyx, idx_left, idx_right, l_num, r_num, feature_id, pivot
        ) = get_conditional_score(
            self.X, self.y, self.weight, idx, self.splitter,
            self.n_classes, criterion, feature_ids,
            self.random_state, self.tree_type)
        info_gain = Hy - Hyx
        relative_gain = (
            self.weight[idx == 1].sum() / self.weight.sum() * info_gain)
        if (l_num < self.min_samples_leaf) or (r_num < self.min_samples_leaf):
            return None, None, None, None
        if relative_gain < self.min_impurity_decrease:
            return None, None, None, None
        self.feature_importances_[feature_id] += relative_gain
        left_weight_frac, right_weight_frac = (
            self.weight[idx_left == 1].sum() / self.weight.sum(),
            self.weight[idx_right == 1].sum() / self.weight.sum())
        node.left = Node(node.depth+1, idx_left, left_weight_frac, self)
        node.right = Node(node.depth+1, idx_right, right_weight_frac, self)
        self.left_nodes_num += 1
        self.depth = max(node.depth+1, self.depth)
        return idx_left, idx_right, pivot, feature_id

    def init_node(self):
        self.depth = 0
        self.left_nodes_num = 1
        self.feature_importances_ = np.zeros(self.X.shape[1])
        self.root = Node(
            depth=0,
            idx=np.ones(self.X.shape[0]) == 1.,
            weight_frac=1,
            tree=self)

    def build(self, mid, idx):
        if mid is None:
            return
        idx_left, idx_right, split_pivot, split_feature = self._split(mid, idx)
        self.build(mid.left, idx_left)
        self.build(mid.right, idx_right)
        mid.split_pivot = split_pivot
        mid.split_feature = split_feature
        if mid.left is None and mid.right is None:
            mid.leaf_idx = idx

    def _search_prediction(self, mid, x):
        if mid.left is None and mid.right is None:
            return self._get_predict(mid)
        if x[mid.split_feature] <= mid.split_pivot:
            node = mid.left
        else:
            node = mid.right
        return self._search_prediction(node, x)

    def _get_predict(self, node):
        if self.tree_type == "cls":
            return np.argmax(np.bincount(self.y[node.leaf_idx]))
        elif self.tree_type == "reg":
            if self.criterion == "mse":
                res = (self.weight[node.leaf_idx] *
                       self.y[node.leaf_idx]).sum()
                res /= self.weight[node.leaf_idx].sum()
            elif self.criterion == "mae":
                res = wmd(self.y[node.leaf_idx], self.weight[node.leaf_idx])
            return res

    def predict(self, x):
        return self._search_prediction(self.root, x)


class Node:

    def __init__(
        self,
        depth,
        idx,
        weight_frac,
        tree,
        left=None,
        right=None,
    ):
        self.depth = depth
        self.idx = idx
        self.data_size = self.idx.sum()
        self.weight_frac = weight_frac
        self.tree = tree
        self.left = left
        self.right = right

        self.split_pivot = None
        self.split_feature = None
        self.leaf_idx = None
        self.Hy = None
        self.child_mccp_value = None
        self.leaf_node_num = None
