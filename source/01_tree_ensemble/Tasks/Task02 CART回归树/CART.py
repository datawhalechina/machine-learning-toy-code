import numpy as np


def MSE(y):
    return ((y - y.mean())**2).sum() / y.shape[0]

class Node:

    def __init__(self, depth, idx):
        self.depth = depth
        self.idx = idx

        self.left = None
        self.right = None
        self.feature = None
        self.pivot = None


class Tree:

    def __init__(self, max_depth):
        self.max_depth = max_depth

        self.X = None
        self.y = None
        self.feature_importances_ = None

    def _able_to_split(self, node):
        return (node.depth < self.max_depth) & (node.idx.sum() >= 2)

    def _get_inner_split_score(self, to_left, to_right):
        total_num = to_left.sum() + to_right.sum()
        left_val = to_left.sum() / total_num * MSE(self.y[to_left])
        right_val = to_right.sum() / total_num * MSE(self.y[to_right])
        return left_val + right_val

    def _inner_split(self, col, idx):
        data = self.X[:, col]
        best_val = np.infty
        for pivot in data[:-1]:
            to_left = (idx==1) & (data<=pivot)
            to_right = (idx==1) & (~to_left)
            if to_left.sum() == 0 or to_left.sum() == idx.sum():
                continue
            Hyx = self._get_inner_split_score(to_left, to_right)
            if best_val > Hyx:
                best_val, best_pivot = Hyx, pivot
                best_to_left, best_to_right = to_left, to_right
        return best_val, best_to_left, best_to_right, best_pivot

    def _get_conditional_entropy(self, idx):
        best_val = np.infty
        for col in range(self.X.shape[1]):
            Hyx, _idx_left, _idx_right, pivot = self._inner_split(col, idx)
            if best_val > Hyx:
                best_val, idx_left, idx_right = Hyx, _idx_left, _idx_right
                best_feature, best_pivot = col, pivot
        return best_val, idx_left, idx_right, best_feature, best_pivot

    def split(self, node):
        # 首先判断本节点是不是符合分裂的条件
        if not self._able_to_split(node):
            return None, None, None, None
        # 计算H(Y)
        entropy = MSE(self.y[node.idx==1])
        # 计算最小的H(Y|X)
        (
            conditional_entropy,
            idx_left,
            idx_right,
            feature,
            pivot
        ) = self._get_conditional_entropy(node.idx)
        # 计算信息增益G(Y, X)
        info_gain = entropy - conditional_entropy
        # 计算相对信息增益
        relative_gain = node.idx.sum() / self.X.shape[0] * info_gain
        # 更新特征重要性
        self.feature_importances_[feature] += relative_gain
        # 新建左右节点并更新深度
        node.left = Node(node.depth+1, idx_left)
        node.right = Node(node.depth+1, idx_right)
        self.depth = max(node.depth+1, self.depth)
        return idx_left, idx_right, feature, pivot

    def build_prepare(self):
        self.depth = 0
        self.feature_importances_ = np.zeros(self.X.shape[1])
        self.root = Node(depth=0, idx=np.ones(self.X.shape[0]) == 1)

    def build_node(self, cur_node):
        if cur_node is None:
            return
        idx_left, idx_right, feature, pivot = self.split(cur_node)
        cur_node.feature, cur_node.pivot = feature, pivot
        self.build_node(cur_node.left)
        self.build_node(cur_node.right)

    def build(self):
        self.build_prepare()
        self.build_node(self.root)

    def _search_prediction(self, node, x):
        if node.left is None and node.right is None:
            return self.y[node.idx].mean()
        if x[node.feature] <= node.pivot:
            node = node.left
        else:
            node = node.right
        return self._search_prediction(node, x)

    def predict(self, x):
        return self._search_prediction(self.root, x)


class DecisionTreeRegressor:
    """
    max_depth控制最大深度，类功能与sklearn默认参数下的功能实现一致
    """

    def __init__(self, max_depth):
        self.tree = Tree(max_depth=max_depth)

    def fit(self, X, y):
        self.tree.X = X
        self.tree.y = y
        self.tree.build()
        self.feature_importances_ = (
            self.tree.feature_importances_ 
            / self.tree.feature_importances_.sum()
        )
        return self

    def predict(self, X):
        return np.array([self.tree.predict(x) for x in X])
