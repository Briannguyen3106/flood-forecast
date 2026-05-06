# src/model/decision_tree_classifier.py

import numpy as np
from src.model.base_model import BaseModel


# ─────────────────────────────────────────
#  Node — một nút trong cây
# ─────────────────────────────────────────
class Node:
    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature_index = feature_index  # Split theo cột nào
        self.threshold     = threshold      # Ngưỡng split
        self.left          = left           # Node con trái  (≤ threshold)
        self.right         = right          # Node con phải  (> threshold)
        self.value         = value          # Nếu là leaf: class dự đoán


# ─────────────────────────────────────────
#  DecisionTreeClassifierModel
# ─────────────────────────────────────────
class DecisionTreeClassifierModel(BaseModel):
    """
    Decision Tree tự code từ scratch bằng numpy.
    Kế thừa BaseModel để tương thích với Trainer của project.
    """

    def __init__(self, random_state: int = 42,
                 max_depth=None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = "gini"):
        super().__init__()
        self.random_state      = random_state
        self.pipeline_type     = "tree"   # dùng data/processed/tree/

        # Hyperparameters — nhận từ __init__ để sklearn clone() hoạt động
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.criterion         = criterion

        self.root = None  # Gốc cây sau khi fit

    # ── 1. Hyperparameter search space ──────────────────────────────
    def get_param_distributions(self) -> dict:
        return {
            "max_depth"         : [3, 5, 10, 15, 20, None],
            "min_samples_split" : [2, 5, 10, 20, 50],
            "min_samples_leaf"  : [1, 2, 5, 10, 20],
            "criterion"         : ["gini", "entropy"],
        }

    # ── 2. Khởi tạo model với params ────────────────────────────────
    def build(self, **params):
        self.max_depth         = params.get("max_depth",         None)
        self.min_samples_split = params.get("min_samples_split", 2)
        self.min_samples_leaf  = params.get("min_samples_leaf",  1)
        self.criterion         = params.get("criterion",         "gini")
        self.root              = None
        self.model             = self   # Trainer cần self.model tồn tại
        return self

    # ── 3. Fit ───────────────────────────────────────────────────────
    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    # ── 4. Predict ──────────────────────────────────────────────────
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # ════════════════════════════════════════════════════════════════
    #  Các hàm nội bộ
    # ════════════════════════════════════════════════════════════════

    def _impurity(self, y) -> float:
        """Tính Gini hoặc Entropy tuỳ self.criterion."""
        if self.criterion == "gini":
            return self._gini(y)
        return self._entropy(y)

    def _gini(self, y) -> float:
        """Gini impurity = 1 - Σ pᵢ²"""
        classes = np.unique(y)
        gini = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            gini -= p ** 2
        return gini

    def _entropy(self, y) -> float:
        """Entropy = - Σ pᵢ log₂(pᵢ)"""
        classes = np.unique(y)
        ent = 0.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            if p > 0:
                ent -= p * np.log2(p)
        return ent

    def _best_split(self, X, y):
        """
        Duyệt tất cả feature và threshold,
        trả về cặp (feature_index, threshold) cho impurity thấp nhất.
        """
        best_impurity  = float("inf")
        best_feature   = None
        best_threshold = None

        n = len(y)

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_mask  = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                # Bỏ qua nếu một bên quá nhỏ
                if left_mask.sum() < self.min_samples_leaf or \
                   right_mask.sum() < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # Weighted impurity
                impurity = (len(y_left)  / n) * self._impurity(y_left) + \
                           (len(y_right) / n) * self._impurity(y_right)

                if impurity < best_impurity:
                    best_impurity  = impurity
                    best_feature   = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth: int) -> Node:
        """Đệ quy xây cây."""

        # ── Điều kiện dừng ──
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(np.unique(y)) == 1 or \
           len(y) < self.min_samples_split:
            return Node(value=self._most_common(y))

        # ── Tìm split tốt nhất ──
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=self._most_common(y))

        # ── Chia data và đệ quy ──
        left_mask = X[:, feature] <= threshold
        left  = self._grow_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature_index=feature, threshold=threshold,
                    left=left, right=right)

    def _traverse_tree(self, x, node: Node):
        """Đệ quy dự đoán một mẫu x."""
        if node.value is not None:   # Leaf node
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common(self, y):
        """Trả về class xuất hiện nhiều nhất trong y."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    # ── Cho ImbPipeline ─────────────────────────────────────────────
    def get_params(self, deep: bool = True):
        return {
            "random_state"      : self.random_state,
            "max_depth"         : self.max_depth,
            "min_samples_split" : self.min_samples_split,
            "min_samples_leaf"  : self.min_samples_leaf,
            "criterion"         : self.criterion,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = self
        return self
