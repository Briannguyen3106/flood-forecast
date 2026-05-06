# src/model/lgbm_scratch.py
#
# LightGBM từ scratch bằng numpy thuần túy.
#
# Các thành phần chính được implement:
#   1. Histogram-based splitting  — thay vì duyệt mọi threshold như DT thông thường,
#      ta bin features thành max_bin bucket → tìm split nhanh hơn O(n) → O(max_bin)
#   2. Leaf-wise tree growth      — mỗi bước chỉ split leaf có gain lớn nhất
#                                   (khác level-wise của XGBoost)
#   3. Gradient Boosting          — mỗi tree mới fit trên negative gradient của loss
#   4. Softmax loss (multiclass)  — phù hợp với bài toán 3 class của project
#   5. L2 regularization (lambda) — penalize leaf weights để tránh overfit

import numpy as np
from src.model.base_model import BaseModel


# ══════════════════════════════════════════════════════════════════════
#  1. HISTOGRAM NODE
# ══════════════════════════════════════════════════════════════════════
class HistNode:
    """Một node trong Histogram Tree."""

    def __init__(self):
        self.feature_index = None   # Feature dùng để split
        self.bin_threshold = None   # Split tại bin nào
        self.real_threshold = None  # Giá trị thực tương ứng
        self.left  = None
        self.right = None
        self.weight = None          # Leaf weight (chỉ có ở leaf)
        self.is_leaf = False


# ══════════════════════════════════════════════════════════════════════
#  2. HISTOGRAM TREE (một weak learner)
# ══════════════════════════════════════════════════════════════════════
class HistogramTree:
    """
    Decision Tree dùng histogram-based split.

    Thay vì duyệt mọi unique value của feature (như DT thông thường),
    ta chia feature thành max_bin bucket → tìm best split trong max_bin threshold.

    Leaf-wise growth:
    - Bắt đầu từ root (toàn bộ data)
    - Mỗi bước: tìm leaf có gain lớn nhất → split leaf đó
    - Lặp cho đến khi đủ num_leaves hoặc không còn gain dương
    """

    def __init__(self, max_depth: int = 6, num_leaves: int = 31,
                 min_child_samples: int = 20, reg_lambda: float = 1.0,
                 max_bin: int = 255):
        self.max_depth        = max_depth
        self.num_leaves       = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda       = reg_lambda
        self.max_bin          = max_bin

        self.root = None
        self.bin_edges = None   # Lưu bin edges để transform X lúc predict

    # ── Histogram construction ───────────────────────────────────────
    def _build_bin_edges(self, X: np.ndarray):
        """Tạo bin edges cho từng feature dựa trên training data."""
        n_features = X.shape[1]
        bin_edges = []
        for f in range(n_features):
            # Lấy quantile để chia đều dữ liệu vào bins
            percentiles = np.linspace(0, 100, self.max_bin + 1)
            edges = np.unique(np.percentile(X[:, f], percentiles))
            bin_edges.append(edges)
        return bin_edges

    def _bin_data(self, X: np.ndarray) -> np.ndarray:
        """Chuyển X về dạng bin indices."""
        n_samples, n_features = X.shape
        X_binned = np.zeros((n_samples, n_features), dtype=np.int32)
        for f in range(n_features):
            X_binned[:, f] = np.searchsorted(
                self.bin_edges[f][1:-1], X[:, f], side='right'
            )
        return X_binned

    # ── Gain computation ─────────────────────────────────────────────
    def _leaf_weight(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        """
        Optimal leaf weight theo Newton step:
        w* = -ΣGᵢ / (ΣHᵢ + λ)
        """
        return -gradients.sum() / (hessians.sum() + self.reg_lambda)

    def _split_gain(self, G_L, H_L, G_R, H_R) -> float:
        """
        Gain khi split một node thành left/right:
        Gain = 0.5 * [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)]
        """
        gain  =  G_L**2 / (H_L + self.reg_lambda)
        gain +=  G_R**2 / (H_R + self.reg_lambda)
        gain -= (G_L + G_R)**2 / (H_L + H_R + self.reg_lambda)
        return 0.5 * gain

    # ── Best split for a node ────────────────────────────────────────
    def _best_split(self, X_bin, gradients, hessians, indices):
        """
        Tìm best (feature, bin_threshold) cho tập mẫu `indices`
        bằng cách dùng histogram: tổng G, H theo từng bin.
        """
        best_gain      = -np.inf
        best_feature   = None
        best_bin       = None
        best_left_idx  = None
        best_right_idx = None

        G_total = gradients[indices].sum()
        H_total = hessians[indices].sum()

        for f in range(X_bin.shape[1]):
            bins = X_bin[indices, f]
            n_bins = self.bin_edges[f].shape[0]

            # Tích lũy G, H theo bin
            G_hist = np.zeros(n_bins)
            H_hist = np.zeros(n_bins)
            np.add.at(G_hist, bins, gradients[indices])
            np.add.at(H_hist, bins, hessians[indices])

            # Quét từ trái → phải, tính cumulative sum
            G_left, H_left = 0.0, 0.0
            for b in range(n_bins - 1):
                G_left += G_hist[b]
                H_left += H_hist[b]
                G_right = G_total - G_left
                H_right = H_total - H_left

                # Bỏ qua nếu một bên quá ít mẫu
                n_left  = (bins <= b).sum()
                n_right = len(bins) - n_left
                if n_left < self.min_child_samples or \
                   n_right < self.min_child_samples:
                    continue

                gain = self._split_gain(G_left, H_left, G_right, H_right)
                if gain > best_gain:
                    best_gain    = gain
                    best_feature = f
                    best_bin     = b
                    mask_left    = X_bin[indices, f] <= b
                    best_left_idx  = indices[mask_left]
                    best_right_idx = indices[~mask_left]

        return best_gain, best_feature, best_bin, best_left_idx, best_right_idx

    # ── Leaf-wise tree building ──────────────────────────────────────
    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        """
        Build một histogram tree theo leaf-wise strategy:
        1. Tạo root node với toàn bộ data
        2. Mỗi bước: tìm leaf có split gain lớn nhất → split
        3. Dừng khi đủ num_leaves hoặc max_depth hoặc không còn gain dương
        """
        # Xây bin edges từ X (chỉ lần đầu)
        self.bin_edges = self._build_bin_edges(X)
        X_bin = self._bin_data(X)

        n_samples = X.shape[0]
        all_indices = np.arange(n_samples)

        # Root node
        root = HistNode()
        # leaves: list of (node, indices, depth)
        leaves = [(root, all_indices, 0)]
        n_leaves = 1

        while n_leaves < self.num_leaves and leaves:
            # Tìm leaf có gain lớn nhất
            best_overall_gain = -np.inf
            best_leaf_idx     = None
            best_split_info   = None

            for i, (node, indices, depth) in enumerate(leaves):
                if depth >= self.max_depth:
                    continue
                if len(indices) < 2 * self.min_child_samples:
                    continue

                gain, feat, bin_t, left_idx, right_idx = \
                    self._best_split(X_bin, gradients, hessians, indices)

                if gain > best_overall_gain and gain > 0:
                    best_overall_gain = gain
                    best_leaf_idx     = i
                    best_split_info   = (feat, bin_t, left_idx, right_idx, depth)

            # Không còn leaf nào có gain dương → dừng
            if best_leaf_idx is None:
                break

            # Thực hiện split
            node, indices, depth = leaves.pop(best_leaf_idx)
            feat, bin_t, left_idx, right_idx, depth = best_split_info

            node.feature_index  = feat
            node.bin_threshold  = bin_t
            node.real_threshold = self.bin_edges[feat][bin_t]
            node.is_leaf        = False

            node.left  = HistNode()
            node.right = HistNode()

            leaves.append((node.left,  left_idx,  depth + 1))
            leaves.append((node.right, right_idx, depth + 1))
            n_leaves += 1  # split 1 leaf → thêm 1 leaf mới

        # Gán leaf weights cho tất cả leaf còn lại
        for node, indices, _ in leaves:
            node.is_leaf = True
            node.weight  = self._leaf_weight(
                gradients[indices], hessians[indices]
            )

        self.root = root
        return self

    # ── Predict ─────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bin = self._bin_data(X)
        return np.array([self._traverse(X_bin[i], self.root)
                         for i in range(X_bin.shape[0])])

    def _traverse(self, x_bin, node: HistNode):
        if node.is_leaf:
            return node.weight
        if x_bin[node.feature_index] <= node.bin_threshold:
            return self._traverse(x_bin, node.left)
        return self._traverse(x_bin, node.right)


# ══════════════════════════════════════════════════════════════════════
#  3. GBDT MULTICLASS (Gradient Boosting)
# ══════════════════════════════════════════════════════════════════════
class GBDTMulticlass:
    """
    Gradient Boosting cho bài toán multiclass dùng Softmax loss.

    Với K class, mỗi iteration ta train K cây riêng biệt —
    mỗi cây fit trên gradient của class tương ứng.

    Softmax:
        pₖ = exp(Fₖ) / Σ exp(Fⱼ)

    Gradient (negative gradient = pseudo-residual):
        gᵢₖ = pᵢₖ - 1(yᵢ = k)

    Hessian:
        hᵢₖ = pᵢₖ(1 - pᵢₖ)
    """

    def __init__(self, n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 num_leaves: int = 31,
                 min_child_samples: int = 20,
                 reg_lambda: float = 1.0,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 max_bin: int = 255,
                 random_state: int = 42):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.num_leaves        = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda        = reg_lambda
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.max_bin           = max_bin
        self.random_state      = random_state

        self.trees_per_class = []   # List of list: [iter][class] = HistogramTree
        self.n_classes       = None
        self.init_score      = None  # Log-odds ban đầu
        self.loss_history    = []   # Cross-entropy mỗi iter
        self.rng             = np.random.default_rng(random_state)

    # ── Softmax ─────────────────────────────────────────────────────
    def _softmax(self, F: np.ndarray) -> np.ndarray:
        """F shape: (n_samples, n_classes)"""
        F_shifted = F - F.max(axis=1, keepdims=True)  # numerical stability
        exp_F = np.exp(F_shifted)
        return exp_F / exp_F.sum(axis=1, keepdims=True)

    # ── Fit ─────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        K = self.n_classes

        # One-hot encode y
        Y = np.zeros((n_samples, K))
        Y[np.arange(n_samples), y.astype(int)] = 1.0

        # Init score: log(class_freq) → softmax → khởi điểm hợp lý
        class_counts  = Y.sum(axis=0) + 1e-8
        self.init_score = np.log(class_counts / class_counts.sum())

        # F: raw score matrix, shape (n_samples, K)
        F = np.tile(self.init_score, (n_samples, 1))

        self.trees_per_class = []

        print(f"  Training GBDT: {self.n_estimators} iterations, "
              f"{K} classes, leaf-wise (num_leaves={self.num_leaves})")

        for t in range(self.n_estimators):
            P = self._softmax(F)  # (n_samples, K)

            # Gradient & Hessian cho từng class
            G = P - Y              # (n_samples, K)
            H = P * (1.0 - P)      # (n_samples, K)

            # Subsample rows
            if self.subsample < 1.0:
                n_sub = max(1, int(n_samples * self.subsample))
                row_idx = self.rng.choice(n_samples, size=n_sub, replace=False)
            else:
                row_idx = np.arange(n_samples)

            # Subsample cols (features)
            if self.colsample_bytree < 1.0:
                n_col = max(1, int(n_features * self.colsample_bytree))
                col_idx = self.rng.choice(n_features, size=n_col, replace=False)
                col_idx = np.sort(col_idx)
            else:
                col_idx = np.arange(n_features)

            X_sub = X[np.ix_(row_idx, col_idx)]

            iter_trees = []
            for k in range(K):
                tree = HistogramTree(
                    max_depth         = self.max_depth,
                    num_leaves        = self.num_leaves,
                    min_child_samples = self.min_child_samples,
                    reg_lambda        = self.reg_lambda,
                    max_bin           = self.max_bin,
                )
                tree.fit(X_sub, G[row_idx, k], H[row_idx, k])

                # Update F cho toàn bộ samples (dùng toàn bộ features)
                # Cần build bin_edges từ full X với col_idx
                tree_full = HistogramTree(
                    max_depth         = self.max_depth,
                    num_leaves        = self.num_leaves,
                    min_child_samples = self.min_child_samples,
                    reg_lambda        = self.reg_lambda,
                    max_bin           = self.max_bin,
                )
                tree_full.bin_edges = tree.bin_edges
                tree_full.root      = tree.root
                tree_full._build_bin_edges = tree._build_bin_edges

                # Predict trên full X với đúng cols
                pred = tree._bin_data(X[:, col_idx])
                preds = np.array([tree._traverse(pred[i], tree.root)
                                  for i in range(n_samples)])
                F[:, k] += self.learning_rate * preds

                iter_trees.append((tree, col_idx))

            self.trees_per_class.append(iter_trees)

            # Tính train loss (cross-entropy) mỗi iter
            P_log = np.clip(self._softmax(F), 1e-8, 1.0)
            loss  = -np.sum(Y * np.log(P_log)) / n_samples
            self.loss_history.append(loss)
            print(f"    Iter {t+1:>4}/{self.n_estimators} | "
                  f"Train loss (cross-entropy): {loss:.6f}")

        return self

    # ── Predict proba ───────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        F = np.tile(self.init_score, (n_samples, 1))

        for iter_trees in self.trees_per_class:
            for k, (tree, col_idx) in enumerate(iter_trees):
                X_sub_bin = tree._bin_data(X[:, col_idx])
                preds = np.array([tree._traverse(X_sub_bin[i], tree.root)
                                  for i in range(n_samples)])
                F[:, k] += self.learning_rate * preds

        return self._softmax(F)

    # ── Predict ─────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ── sklearn interface ────────────────────────────────────────────
    def get_params(self, deep=True):
        return {
            'n_estimators'     : self.n_estimators,
            'learning_rate'    : self.learning_rate,
            'max_depth'        : self.max_depth,
            'num_leaves'       : self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'reg_lambda'       : self.reg_lambda,
            'subsample'        : self.subsample,
            'colsample_bytree' : self.colsample_bytree,
            'max_bin'          : self.max_bin,
            'random_state'     : self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ══════════════════════════════════════════════════════════════════════
#  4. WRAPPER — tương thích BaseModel + Trainer
# ══════════════════════════════════════════════════════════════════════
class LGBMScratchModel(BaseModel):
    """
    LightGBM tự implement từ scratch.
    Kế thừa BaseModel để tương thích với Trainer của project.

    Các kỹ thuật được implement:
    - Histogram-based splitting (O(max_bin) thay vì O(n))
    - Leaf-wise tree growth (chọn leaf gain lớn nhất để split)
    - Gradient Boosting với Softmax loss (multiclass)
    - Newton step để tính leaf weight (dùng cả G và H)
    - Subsampling rows & cols (giống bagging + feature sampling)
    - L2 regularization (reg_lambda)
    """

    def __init__(self,
                 n_estimators     : int   = 100,
                 learning_rate    : float = 0.1,
                 max_depth        : int   = 6,
                 num_leaves       : int   = 31,
                 min_child_samples: int   = 20,
                 reg_lambda       : float = 1.0,
                 subsample        : float = 1.0,
                 colsample_bytree : float = 1.0,
                 max_bin          : int   = 255,
                 random_state     : int   = 42):
        super().__init__()
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.num_leaves        = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda        = reg_lambda
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.max_bin           = max_bin
        self.random_state      = random_state
        self.pipeline_type     = 'tree'

    # ── BaseModel interface ─────────────────────────────────────────
    def get_param_distributions(self) -> dict:
        return {
            'n_estimators'     : [50, 100, 200, 300],
            'learning_rate'    : [0.01, 0.05, 0.1, 0.2],
            'max_depth'        : [3, 4, 5, 6, 8],
            'num_leaves'       : [15, 31, 63],
            'min_child_samples': [10, 20, 30, 50],
            'reg_lambda'       : [0.1, 1.0, 5.0, 10.0],
            'subsample'        : [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree' : [0.7, 0.8, 0.9, 1.0],
        }

    def build(self, **params) -> 'LGBMScratchModel':
        cfg = {
            'n_estimators'     : self.n_estimators,
            'learning_rate'    : self.learning_rate,
            'max_depth'        : self.max_depth,
            'num_leaves'       : self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'reg_lambda'       : self.reg_lambda,
            'subsample'        : self.subsample,
            'colsample_bytree' : self.colsample_bytree,
            'max_bin'          : self.max_bin,
            'random_state'     : self.random_state,
        }
        cfg.update(params)
        self.model = GBDTMulticlass(**cfg)
        return self

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    # ── sklearn clone() interface ───────────────────────────────────
    def get_params(self, deep=True):
        return {
            'n_estimators'     : self.n_estimators,
            'learning_rate'    : self.learning_rate,
            'max_depth'        : self.max_depth,
            'num_leaves'       : self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'reg_lambda'       : self.reg_lambda,
            'subsample'        : self.subsample,
            'colsample_bytree' : self.colsample_bytree,
            'max_bin'          : self.max_bin,
            'random_state'     : self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        if self.model is not None:
            self.model.set_params(**params)
        return self
