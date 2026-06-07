# src/model/lgbm_scratch_v2.py
#
# LightGBM từ scratch — phiên bản đầy đủ, sát với thư viện LightGBM gốc.
#
# Các kỹ thuật được implement (so với v1):
#   GĐ1 ─ L1 regularization (reg_alpha)
#          min_split_gain
#          feature_importance (split count + gain)
#          predict(num_iteration=N)
#
#   GĐ2 ─ Early stopping (validation set + patience)
#          Multiple loss functions: softmax, binary, mse
#
#   GĐ3 ─ GOSS (Gradient-based One-Side Sampling)
#          Giữ top-a% mẫu gradient lớn + random b% mẫu nhỏ có trọng số bù
#
#   GĐ4 ─ EFB (Exclusive Feature Bundling)
#          Conflict graph → greedy coloring → rebinning histogram
#
# Tất cả các class đều tương thích ngược với v1.

import numpy as np
from src.model.base_model import BaseModel


# ══════════════════════════════════════════════════════════════════════
#  PHẦN 1 — HISTOGRAM NODE  (không đổi so với v1)
# ══════════════════════════════════════════════════════════════════════
class HistNode:
    """Một node trong Histogram Tree."""

    def __init__(self):
        self.feature_index  = None   # Chỉ số feature gốc (sau khi unbundle)
        self.bin_threshold  = None   # Ngưỡng split theo bin index
        self.real_threshold = None   # Giá trị thực tương ứng
        self.left           = None
        self.right          = None
        self.weight         = None   # Leaf weight — chỉ có ở leaf node
        self.is_leaf        = False


# ══════════════════════════════════════════════════════════════════════
#  PHẦN 2 — EFB: Exclusive Feature Bundling  (GĐ4)
# ══════════════════════════════════════════════════════════════════════
class EFBBundler:
    """
    Exclusive Feature Bundling — giảm số feature bằng cách gộp các feature
    không bao giờ cùng khác 0 vào một bundle duy nhất.

    Thuật toán:
      1. Xây conflict graph: feature i và j conflict nếu cùng ≠ 0 trên
         nhiều hơn `max_conflict_rate` tỉ lệ mẫu.
      2. Greedy graph coloring: sắp xếp feature theo số conflict giảm dần,
         gán vào bundle đầu tiên còn đủ chỗ.
      3. Rebinning: mỗi feature trong bundle được offset vào vùng bin riêng
         → bundle có tổng số bin = sum(max_bin_per_feature).

    Tham chiếu: LightGBM paper (Ke et al., 2017), Section 3.2.
    """

    def __init__(self, max_bin: int = 255, max_conflict_rate: float = 0.0):
        self.max_bin           = max_bin
        self.max_conflict_rate = max_conflict_rate

        # Kết quả sau fit
        self.bundles_          = []   # List[List[int]] — index feature gốc trong mỗi bundle
        self.bundle_offsets_   = []   # List[List[int]] — offset bin của từng feature trong bundle
        self.bundle_max_bins_  = []   # List[int]       — số bin thực của mỗi bundle
        self.n_original_       = 0

    def fit(self, X: np.ndarray) -> 'EFBBundler':
        """
        Học cấu trúc bundle từ data X (chỉ gọi một lần trên training data).
        """
        n_samples, n_features = X.shape
        self.n_original_ = n_features

        # ── Bước 1: Xây conflict graph ──────────────────────────────
        # conflict[i, j] = tỉ lệ mẫu mà cả feature i và j đều ≠ 0
        nonzero = (X != 0).astype(np.float32)   # (n, d)
        # conflict_count[i,j] = số mẫu cả 2 cùng ≠ 0
        # Dùng matmul: (d, n) @ (n, d) → (d, d)
        conflict_count = nonzero.T @ nonzero     # (d, d)
        conflict_rate  = conflict_count / n_samples

        # Loại bỏ đường chéo (mỗi feature conflict với chính nó = 1.0)
        np.fill_diagonal(conflict_rate, 0.0)

        # ── Bước 2: Greedy graph coloring ───────────────────────────
        # Sắp xếp feature theo số neighbor conflict giảm dần
        n_conflicts   = (conflict_rate > self.max_conflict_rate).sum(axis=1)
        feature_order = np.argsort(-n_conflicts)   # giảm dần

        bundles = []   # List[List[int]] — feature indices trong mỗi bundle

        for f in feature_order:
            placed = False
            for bundle in bundles:
                # Kiểm tra f có conflict với bất kỳ feature nào trong bundle không
                conflicts_with_bundle = any(
                    conflict_rate[f, g] > self.max_conflict_rate
                    for g in bundle
                )
                if not conflicts_with_bundle:
                    bundle.append(f)
                    placed = True
                    break
            if not placed:
                bundles.append([f])

        self.bundles_ = bundles

        # ── Bước 3: Tính offset bin cho từng feature trong mỗi bundle ─
        self.bundle_offsets_  = []
        self.bundle_max_bins_ = []

        for bundle in bundles:
            offsets  = []
            cur_offset = 0
            for f in bundle:
                offsets.append(cur_offset)
                # Số bin của feature f: đếm unique values (tối đa max_bin)
                n_unique = min(len(np.unique(X[:, f])), self.max_bin)
                cur_offset += n_unique
            self.bundle_offsets_.append(offsets)
            # Tổng bin của bundle, capped tại max_bin
            self.bundle_max_bins_.append(min(cur_offset, self.max_bin))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Chuyển X (n_samples, n_features_gốc) → X_bundled (n_samples, n_bundles).

        Mỗi bundle feature = giá trị thực của feature trong bundle + offset tương ứng.
        Nếu nhiều feature trong bundle cùng ≠ 0 (rất hiếm nếu conflict_rate thấp),
        ưu tiên feature nào ≠ 0 đầu tiên.
        """
        n_samples = X.shape[0]
        n_bundles = len(self.bundles_)
        X_bundled = np.zeros((n_samples, n_bundles), dtype=X.dtype)

        for b_idx, (bundle, offsets) in enumerate(
            zip(self.bundles_, self.bundle_offsets_)
        ):
            for feat_pos, (f, offset) in enumerate(zip(bundle, offsets)):
                col = X[:, f]
                # Chỉ ghi vào bundle ở những mẫu chưa được ghi (= 0)
                mask = (X_bundled[:, b_idx] == 0) & (col != 0)
                X_bundled[mask, b_idx] = col[mask] + offset

        return X_bundled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_bundles(self) -> int:
        return len(self.bundles_)

    def compression_ratio(self) -> float:
        """Tỉ lệ nén feature: n_bundles / n_features_gốc."""
        return self.n_bundles / max(self.n_original_, 1)


# ══════════════════════════════════════════════════════════════════════
#  PHẦN 3 — LOSS FUNCTIONS  (GĐ2: multiple loss)
# ══════════════════════════════════════════════════════════════════════
class SoftmaxLoss:
    """Cross-entropy loss cho multiclass classification."""

    name = "softmax"

    @staticmethod
    def _softmax(F: np.ndarray) -> np.ndarray:
        F_s = F - F.max(axis=1, keepdims=True)
        e   = np.exp(F_s)
        return e / e.sum(axis=1, keepdims=True)

    def init_score(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y.astype(int)] = 1.0
        counts = Y.sum(axis=0) + 1e-8
        return np.log(counts / counts.sum())

    def gradients_hessians(self, F: np.ndarray, Y: np.ndarray):
        P = self._softmax(F)
        G = P - Y             # (n, K)
        H = P * (1.0 - P)     # (n, K)
        return G, H

    def loss(self, F: np.ndarray, Y: np.ndarray) -> float:
        P = np.clip(self._softmax(F), 1e-8, 1.0)
        return -np.sum(Y * np.log(P)) / len(F)

    def predict(self, F: np.ndarray) -> np.ndarray:
        return self._softmax(F).argmax(axis=1)

    def n_outputs(self, n_classes: int) -> int:
        return n_classes


class BinaryLoss:
    """Binary cross-entropy loss (sigmoid)."""

    name = "binary"

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))

    def init_score(self, y: np.ndarray, n_classes: int = 2) -> np.ndarray:
        pos_rate = np.mean(y)
        pos_rate = np.clip(pos_rate, 1e-6, 1 - 1e-6)
        return np.array([np.log(pos_rate / (1 - pos_rate))])

    def gradients_hessians(self, F: np.ndarray, Y: np.ndarray):
        P = self._sigmoid(F[:, 0])[:, None]
        G = P - Y
        H = P * (1.0 - P)
        return G, H

    def loss(self, F: np.ndarray, Y: np.ndarray) -> float:
        P = np.clip(self._sigmoid(F[:, 0]), 1e-8, 1 - 1e-8)
        return -np.mean(Y[:, 0] * np.log(P) + (1 - Y[:, 0]) * np.log(1 - P))

    def predict(self, F: np.ndarray) -> np.ndarray:
        return (self._sigmoid(F[:, 0]) >= 0.5).astype(int)

    def n_outputs(self, n_classes: int) -> int:
        return 1


class MSELoss:
    """Mean squared error loss cho regression."""

    name = "mse"

    def init_score(self, y: np.ndarray, n_classes: int = 1) -> np.ndarray:
        return np.array([np.mean(y)])

    def gradients_hessians(self, F: np.ndarray, Y: np.ndarray):
        G = F - Y               # residual
        H = np.ones_like(G)    # hessian = 1 với MSE
        return G, H

    def loss(self, F: np.ndarray, Y: np.ndarray) -> float:
        return np.mean((F - Y) ** 2)

    def predict(self, F: np.ndarray) -> np.ndarray:
        return F[:, 0]

    def n_outputs(self, n_classes: int) -> int:
        return 1


LOSS_REGISTRY = {
    "softmax": SoftmaxLoss,
    "binary":  BinaryLoss,
    "mse":     MSELoss,
    "multiclass": SoftmaxLoss,
    "regression": MSELoss,
}


# ══════════════════════════════════════════════════════════════════════
#  PHẦN 4 — HISTOGRAM TREE  (GĐ1: L1, min_split_gain; GĐ4: EFB bins)
# ══════════════════════════════════════════════════════════════════════
class HistogramTree:
    """
    Decision Tree với histogram-based split + leaf-wise growth.

    Thêm so với v1:
      - reg_alpha (L1): ảnh hưởng leaf weight và split gain
      - min_split_gain: ngưỡng gain tối thiểu để split
      - sample_weights: hỗ trợ trọng số mẫu (dùng cho GOSS)
      - feature_importance_split / feature_importance_gain: đếm tích lũy
    """

    def __init__(self, max_depth: int = 6, num_leaves: int = 31,
                 min_child_samples: int = 20,
                 reg_lambda: float = 1.0, reg_alpha: float = 0.0,
                 min_split_gain: float = 0.0,
                 max_bin: int = 255):
        self.max_depth         = max_depth
        self.num_leaves        = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda        = reg_lambda
        self.reg_alpha         = reg_alpha   # L1
        self.min_split_gain    = min_split_gain
        self.max_bin           = max_bin

        self.root      = None
        self.bin_edges = None

        # Feature importance tích lũy từ bên ngoài
        self._fi_split = None   # np.ndarray — số lần feature được chọn split
        self._fi_gain  = None   # np.ndarray — tổng gain

    # ── Histogram construction ───────────────────────────────────────
    def _build_bin_edges(self, X: np.ndarray):
        n_features = X.shape[1]
        bin_edges  = []
        for f in range(n_features):
            percentiles = np.linspace(0, 100, self.max_bin + 1)
            edges = np.unique(np.percentile(X[:, f], percentiles))
            bin_edges.append(edges)
        return bin_edges

    def _bin_data(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        X_binned = np.zeros((n_samples, n_features), dtype=np.int32)
        for f in range(n_features):
            X_binned[:, f] = np.searchsorted(
                self.bin_edges[f][1:-1], X[:, f], side='right'
            )
        return X_binned

    # ── Leaf weight với L1 (GĐ1) ────────────────────────────────────
    def _leaf_weight(self, gradients: np.ndarray,
                     hessians: np.ndarray,
                     weights: np.ndarray | None = None) -> float:
        """
        Newton step với L1 + L2:
          w* = -clip(ΣG, -reg_alpha, reg_alpha→ soft-threshold) / (ΣH + λ)

        Soft-thresholding cho L1:
          nếu |ΣG| ≤ reg_alpha → w* = 0
          nếu ΣG > reg_alpha   → w* = -(ΣG - reg_alpha) / (ΣH + λ)
          nếu ΣG < -reg_alpha  → w* = -(ΣG + reg_alpha) / (ΣH + λ)
        """
        if weights is not None:
            G_sum = (gradients * weights).sum()
            H_sum = (hessians  * weights).sum()
        else:
            G_sum = gradients.sum()
            H_sum = hessians.sum()

        # Soft-threshold L1
        if self.reg_alpha > 0:
            if G_sum > self.reg_alpha:
                G_eff = G_sum - self.reg_alpha
            elif G_sum < -self.reg_alpha:
                G_eff = G_sum + self.reg_alpha
            else:
                return 0.0
        else:
            G_eff = G_sum

        return -G_eff / (H_sum + self.reg_lambda)

    # ── Split gain với L1 (GĐ1) ─────────────────────────────────────
    def _score(self, G, H) -> float:
        """
        Score của một node: G² / (H + λ) sau khi soft-threshold L1.
        """
        if self.reg_alpha > 0:
            G_eff = max(0.0, abs(G) - self.reg_alpha) * np.sign(G) if G != 0 else 0.0
        else:
            G_eff = G
        return G_eff ** 2 / (H + self.reg_lambda)

    def _split_gain(self, G_L, H_L, G_R, H_R) -> float:
        return 0.5 * (self._score(G_L, H_L) + self._score(G_R, H_R)
                      - self._score(G_L + G_R, H_L + H_R))

    # ── Best split ──────────────────────────────────────────────────
    def _best_split(self, X_bin, gradients, hessians, indices,
                    weights=None):
        best_gain      = -np.inf
        best_feature   = None
        best_bin       = None
        best_left_idx  = None
        best_right_idx = None

        if weights is not None:
            w = weights[indices]
            G_total = (gradients[indices] * w).sum()
            H_total = (hessians[indices]  * w).sum()
        else:
            w = None
            G_total = gradients[indices].sum()
            H_total = hessians[indices].sum()

        for f in range(X_bin.shape[1]):
            bins   = X_bin[indices, f]
            n_bins = self.bin_edges[f].shape[0]

            if w is not None:
                G_hist = np.bincount(
                    bins, weights=gradients[indices] * w, minlength=n_bins
                )
                H_hist = np.bincount(
                    bins, weights=hessians[indices] * w, minlength=n_bins
                )
            else:
                G_hist = np.bincount(
                    bins, weights=gradients[indices], minlength=n_bins
                )
                H_hist = np.bincount(
                    bins, weights=hessians[indices], minlength=n_bins
                )

            count_hist = np.bincount(bins, minlength=n_bins)
            G_cum = np.cumsum(G_hist)
            H_cum = np.cumsum(H_hist)
            count_cum = np.cumsum(count_hist)

            for b in range(n_bins - 1):
                G_left = G_cum[b]
                H_left = H_cum[b]
                G_right = G_total - G_left
                H_right = H_total - H_left

                n_left  = count_cum[b]
                n_right = len(bins) - n_left
                if (n_left  < self.min_child_samples or
                        n_right < self.min_child_samples):
                    continue

                gain = self._split_gain(G_left, H_left, G_right, H_right)

                # GĐ1: kiểm tra min_split_gain
                if gain > best_gain and gain > self.min_split_gain:
                    best_gain    = gain
                    best_feature = f
                    best_bin     = b

        if best_feature is not None:
            mask_left = X_bin[indices, best_feature] <= best_bin
            best_left_idx = indices[mask_left]
            best_right_idx = indices[~mask_left]

        return best_gain, best_feature, best_bin, best_left_idx, best_right_idx

    # ── Leaf-wise tree building ──────────────────────────────────────
    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray,
            weights: np.ndarray | None = None,
            fi_split: np.ndarray | None = None,
            fi_gain:  np.ndarray | None = None,
            bin_edges: list[np.ndarray] | None = None,
            X_bin: np.ndarray | None = None):
        """
        fit() mở rộng: nhận thêm sample weights (GOSS) và mảng feature importance.

        Parameters
        ----------
        X          : (n_samples, n_features) — đã bin hoá hoặc raw
        gradients  : (n_samples,) — gradient của loss
        hessians   : (n_samples,) — hessian của loss
        weights    : (n_samples,) hoặc None — sample weight (GOSS)
        fi_split   : (n_features,) — tích lũy split count (in-place update)
        fi_gain    : (n_features,) — tích lũy gain (in-place update)
        """
        self.bin_edges = bin_edges if bin_edges is not None else self._build_bin_edges(X)
        X_bin = X_bin if X_bin is not None else self._bin_data(X)

        n_samples   = X.shape[0]
        all_indices = np.arange(n_samples)

        root   = HistNode()
        leaves = [(root, all_indices, 0)]
        n_leaves = 1

        while n_leaves < self.num_leaves and leaves:
            best_overall_gain = -np.inf
            best_leaf_idx     = None
            best_split_info   = None

            for i, (node, indices, depth) in enumerate(leaves):
                if depth >= self.max_depth:
                    continue
                if len(indices) < 2 * self.min_child_samples:
                    continue

                gain, feat, bin_t, left_idx, right_idx = self._best_split(
                    X_bin, gradients, hessians, indices, weights
                )

                if gain > best_overall_gain and gain > self.min_split_gain:
                    best_overall_gain = gain
                    best_leaf_idx     = i
                    best_split_info   = (feat, bin_t, left_idx, right_idx, depth, gain)

            if best_leaf_idx is None:
                break

            node, indices, depth = leaves.pop(best_leaf_idx)
            feat, bin_t, left_idx, right_idx, depth, gain = best_split_info

            node.feature_index  = feat
            node.bin_threshold  = bin_t
            node.real_threshold = self.bin_edges[feat][bin_t]
            node.is_leaf        = False
            node.left           = HistNode()
            node.right          = HistNode()

            leaves.append((node.left,  left_idx,  depth + 1))
            leaves.append((node.right, right_idx, depth + 1))
            n_leaves += 1

            # GĐ1: cập nhật feature importance
            if fi_split is not None:
                fi_split[feat] += 1
            if fi_gain is not None:
                fi_gain[feat] += gain

        # Gán leaf weights
        for node, indices, _ in leaves:
            node.is_leaf = True
            node.weight  = self._leaf_weight(
                gradients[indices], hessians[indices],
                weights[indices] if weights is not None else None
            )

        self.root = root
        return self

    # ── Predict / traverse ──────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bin = self._bin_data(X)
        return self.predict_binned(X_bin)

    def predict_binned(self, X_bin: np.ndarray) -> np.ndarray:
        """Predict from pre-binned rows using vectorized node partitioning."""
        predictions = np.empty(X_bin.shape[0], dtype=float)
        stack = [(self.root, np.arange(X_bin.shape[0]))]

        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue
            if node.is_leaf:
                predictions[indices] = node.weight
                continue

            left_mask = (
                X_bin[indices, node.feature_index] <= node.bin_threshold
            )
            stack.append((node.left, indices[left_mask]))
            stack.append((node.right, indices[~left_mask]))

        return predictions

    def _traverse(self, x_bin, node: HistNode):
        if node.is_leaf:
            return node.weight
        if x_bin[node.feature_index] <= node.bin_threshold:
            return self._traverse(x_bin, node.left)
        return self._traverse(x_bin, node.right)


# ══════════════════════════════════════════════════════════════════════
#  PHẦN 5 — GBDT MULTICLASS  (tích hợp toàn bộ GĐ1–4)
# ══════════════════════════════════════════════════════════════════════
class GBDTMulticlass:
    """
    Gradient Boosting với đầy đủ kỹ thuật LightGBM:

    GĐ1 — L1 (reg_alpha), min_split_gain, feature importance, predict(num_iteration)
    GĐ2 — Early stopping, multiple loss functions
    GĐ3 — GOSS sampling
    GĐ4 — EFB bundling
    """

    def __init__(self,
                 n_estimators      : int   = 100,
                 learning_rate     : float = 0.1,
                 max_depth         : int   = 6,
                 num_leaves        : int   = 31,
                 min_child_samples : int   = 20,
                 reg_lambda        : float = 1.0,
                 reg_alpha         : float = 0.0,      # GĐ1: L1
                 min_split_gain    : float = 0.0,      # GĐ1
                 subsample         : float = 1.0,
                 colsample_bytree  : float = 1.0,
                 max_bin           : int   = 255,
                 # GĐ3 — GOSS
                 use_goss          : bool  = False,
                 top_rate          : float = 0.2,      # a: tỉ lệ giữ gradient lớn
                 other_rate        : float = 0.1,      # b: tỉ lệ sample gradient nhỏ
                 # GĐ4 — EFB
                 use_efb           : bool  = False,
                 max_conflict_rate : float = 0.0,
                 # GĐ2 — Early stopping
                 early_stopping_rounds : int | None = None,
                 # GĐ2 — Loss function
                 objective         : str   = "softmax",
                 random_state      : int   = 42):

        self.n_estimators         = n_estimators
        self.learning_rate        = learning_rate
        self.max_depth            = max_depth
        self.num_leaves           = num_leaves
        self.min_child_samples    = min_child_samples
        self.reg_lambda           = reg_lambda
        self.reg_alpha            = reg_alpha
        self.min_split_gain       = min_split_gain
        self.subsample            = subsample
        self.colsample_bytree     = colsample_bytree
        self.max_bin              = max_bin
        self.use_goss             = use_goss
        self.top_rate             = top_rate
        self.other_rate           = other_rate
        self.use_efb              = use_efb
        self.max_conflict_rate    = max_conflict_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.objective            = objective
        self.random_state         = random_state

        # Trạng thái sau fit
        self.trees_per_class   = []
        self.n_classes_        = None
        self.init_score_       = None
        self.loss_history      = []
        self.val_loss_history  = []
        self.best_iteration_   = None

        # GĐ1: feature importance
        self.feature_importances_split_ = None
        self.feature_importances_gain_  = None
        self.n_features_in_            = None

        # GĐ4: EFB
        self.efb_bundler_ = None

        # Loss function object
        self._loss_fn = None

        self.rng = np.random.default_rng(random_state)

    # ── GOSS sampling  (GĐ3) ────────────────────────────────────────
    def _goss_sample(self, G: np.ndarray, n_samples: int):
        """
        Gradient-based One-Side Sampling.

        Returns
        -------
        row_idx    : np.ndarray — chỉ số hàng được chọn
        weights    : np.ndarray — trọng số bù cho mỗi hàng
                     (top-a% có weight=1, b% còn lại có weight=(1-a)/b)
        """
        # Dùng |G| trung bình trên tất cả class nếu G là 2D
        if G.ndim == 2:
            abs_g = np.abs(G).mean(axis=1)
        else:
            abs_g = np.abs(G)

        sorted_idx = np.argsort(-abs_g)   # giảm dần

        n_top   = max(1, int(n_samples * self.top_rate))
        n_other = max(1, int(n_samples * self.other_rate))

        top_idx   = sorted_idx[:n_top]
        other_pool = sorted_idx[n_top:]

        if len(other_pool) > n_other:
            other_idx = self.rng.choice(other_pool, size=n_other, replace=False)
        else:
            other_idx = other_pool

        row_idx = np.concatenate([top_idx, other_idx])

        # Trọng số: top → 1.0; other → amplify để bù cho việc bỏ bớt
        amp = (1.0 - self.top_rate) / (self.other_rate + 1e-10)
        weights = np.ones(len(row_idx))
        weights[n_top:] = amp   # phần other được khuếch đại

        return row_idx, weights

    # ── Softmax (dùng nội bộ khi predict proba) ─────────────────────
    @staticmethod
    def _softmax(F: np.ndarray) -> np.ndarray:
        F_s = F - F.max(axis=1, keepdims=True)
        e   = np.exp(F_s)
        return e / e.sum(axis=1, keepdims=True)

    # ── Fit ─────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: tuple | None = None):
        """
        Train GBDT.

        Parameters
        ----------
        X        : (n_samples, n_features)
        y        : (n_samples,) — nhãn class (int) hoặc giá trị (float cho mse)
        eval_set : (X_val, y_val) hoặc None — dùng cho early stopping (GĐ2)
        """
        n_samples, n_features = X.shape
        self.n_features_in_   = n_features

        # ── Khởi tạo loss function (GĐ2) ────────────────────────────
        loss_cls      = LOSS_REGISTRY.get(self.objective, SoftmaxLoss)
        self._loss_fn = loss_cls()

        self.n_classes_ = int(np.max(y) + 1) if self.objective == "softmax" else 1
        K = self._loss_fn.n_outputs(self.n_classes_)

        # One-hot encode y cho softmax / binary
        if self.objective in ("softmax", "multiclass"):
            Y = np.zeros((n_samples, K))
            Y[np.arange(n_samples), y.astype(int)] = 1.0
        elif self.objective == "binary":
            Y = y.reshape(-1, 1).astype(float)
        else:  # mse / regression
            Y = y.reshape(-1, 1).astype(float)

        # Init score
        self.init_score_ = self._loss_fn.init_score(y, self.n_classes_)
        F = np.tile(self.init_score_, (n_samples, 1))

        # Validation set (GĐ2 — early stopping)
        X_val = Y_val = F_val = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if self.objective in ("softmax", "multiclass"):
                n_val = len(y_val)
                Y_val = np.zeros((n_val, K))
                Y_val[np.arange(n_val), y_val.astype(int)] = 1.0
            else:
                Y_val = y_val.reshape(-1, 1).astype(float)
            F_val = np.tile(self.init_score_, (len(y_val), 1))

        # GĐ4 — EFB: bundling features
        if self.use_efb:
            self.efb_bundler_ = EFBBundler(
                max_bin=self.max_bin,
                max_conflict_rate=self.max_conflict_rate
            )
            X_work = self.efb_bundler_.fit_transform(X)
            if X_val is not None:
                X_val_work = self.efb_bundler_.transform(X_val)
            n_features_work = X_work.shape[1]
            
        else:
            X_work         = X
            X_val_work     = X_val
            n_features_work = n_features

        # GĐ1 — feature importance arrays
        self.feature_importances_split_ = np.zeros(n_features_work)
        self.feature_importances_gain_  = np.zeros(n_features_work)

        self.trees_per_class  = []
        self.loss_history     = []
        self.val_loss_history = []

        # GĐ2 — early stopping state
        best_val_loss   = np.inf
        no_improve_cnt  = 0
        self.best_iteration_ = self.n_estimators

        print(f"  Training GBDT [{self._loss_fn.name}]: "
              f"{self.n_estimators} iter, K={K}, "
              f"GOSS={'on' if self.use_goss else 'off'}, "
              f"EFB={'on' if self.use_efb else 'off'}")

        for t in range(self.n_estimators):

            # ── Tính gradient & hessian ──────────────────────────────
            G, H = self._loss_fn.gradients_hessians(F, Y)

            # ── GĐ3: GOSS hoặc random sampling ──────────────────────
            if self.use_goss:
                row_idx, sample_weights = self._goss_sample(G, n_samples)
            else:
                if self.subsample < 1.0:
                    n_sub   = max(1, int(n_samples * self.subsample))
                    row_idx = self.rng.choice(n_samples, size=n_sub, replace=False)
                else:
                    row_idx = np.arange(n_samples)
                sample_weights = None

            # ── Column sampling ──────────────────────────────────────
            if self.colsample_bytree < 1.0:
                n_col   = max(1, int(n_features_work * self.colsample_bytree))
                col_idx = np.sort(self.rng.choice(n_features_work, size=n_col, replace=False))
            else:
                col_idx = np.arange(n_features_work)

            X_sub = X_work[np.ix_(row_idx, col_idx)]

            # All K class trees use the same rows and columns in this
            # iteration, so their percentile edges and binned matrices are
            # identical. Build them once instead of once per class.
            shared_binner = HistogramTree(max_bin=self.max_bin)
            shared_bin_edges = shared_binner._build_bin_edges(X_sub)
            shared_binner.bin_edges = shared_bin_edges
            X_sub_bin = shared_binner._bin_data(X_sub)
            X_work_bin = shared_binner._bin_data(X_work[:, col_idx])
            X_val_bin = (
                shared_binner._bin_data(X_val_work[:, col_idx])
                if X_val_work is not None else None
            )

            # ── Train K cây ─────────────────────────────────────────
            iter_trees = []
            for k in range(K):
                g_k = G[row_idx, k] if G.ndim == 2 else G[row_idx, 0]
                h_k = H[row_idx, k] if H.ndim == 2 else H[row_idx, 0]

                # Feature importance arrays cho cột được chọn
                fi_split_sub = self.feature_importances_split_[col_idx]
                fi_gain_sub  = self.feature_importances_gain_[col_idx]

                tree = HistogramTree(
                    max_depth         = self.max_depth,
                    num_leaves        = self.num_leaves,
                    min_child_samples = self.min_child_samples,
                    reg_lambda        = self.reg_lambda,
                    reg_alpha         = self.reg_alpha,
                    min_split_gain    = self.min_split_gain,
                    max_bin           = self.max_bin,
                )
                tree.fit(
                    X_sub, g_k, h_k,
                    weights  = sample_weights,
                    fi_split = fi_split_sub,
                    fi_gain  = fi_gain_sub,
                    bin_edges=shared_bin_edges,
                    X_bin=X_sub_bin,
                )

                # Ghi importance back
                self.feature_importances_split_[col_idx] = fi_split_sub
                self.feature_importances_gain_[col_idx]  = fi_gain_sub

                # Update F trên toàn bộ mẫu
                preds = tree.predict_binned(X_work_bin)
                if G.ndim == 2:
                    F[:, k] += self.learning_rate * preds
                else:
                    F[:, 0] += self.learning_rate * preds

                iter_trees.append((tree, col_idx))

            self.trees_per_class.append(iter_trees)

            # ── Train loss ───────────────────────────────────────────
            train_loss = self._loss_fn.loss(F, Y)
            self.loss_history.append(train_loss)

            # ── Val loss + early stopping (GĐ2) ─────────────────────
            val_msg = ""
            if F_val is not None:
                for iter_trees_t in [iter_trees]:
                    for k, (tree, col_idx) in enumerate(iter_trees_t):
                        vpreds = tree.predict_binned(X_val_bin)
                        if G.ndim == 2:
                            F_val[:, k] += self.learning_rate * vpreds
                        else:
                            F_val[:, 0] += self.learning_rate * vpreds

                val_loss = self._loss_fn.loss(F_val, Y_val)
                self.val_loss_history.append(val_loss)
                val_msg = f" | Val loss: {val_loss:.6f}"

                if val_loss < best_val_loss:
                    best_val_loss        = val_loss
                    no_improve_cnt       = 0
                    self.best_iteration_ = t + 1
                else:
                    no_improve_cnt += 1

                if (self.early_stopping_rounds is not None and
                        no_improve_cnt >= self.early_stopping_rounds):
            
                    break

            

        return self

    # ── Predict theo số iteration (GĐ1) ────────────────────────────
    def _predict_raw(self, X: np.ndarray,
                     num_iteration: int | None = None) -> np.ndarray:
        """Trả về raw score F (trước softmax)."""
        n_samples = X.shape[0]

        if self.use_efb and self.efb_bundler_ is not None:
            X_work = self.efb_bundler_.transform(X)
        else:
            X_work = X

        F = np.tile(self.init_score_, (n_samples, 1))

        iters = self.trees_per_class
        if num_iteration is not None:
            iters = iters[:num_iteration]

        for iter_trees in iters:
            for k, (tree, col_idx) in enumerate(iter_trees):
                xb = tree._bin_data(X_work[:, col_idx])
                preds = tree.predict_binned(xb)
                F[:, k] += self.learning_rate * preds

        return F

    def predict_proba(self, X: np.ndarray,
                      num_iteration: int | None = None) -> np.ndarray:
        F = self._predict_raw(X, num_iteration)
        return self._softmax(F)

    def predict(self, X: np.ndarray,
                num_iteration: int | None = None) -> np.ndarray:
        F = self._predict_raw(X, num_iteration)
        return self._loss_fn.predict(F)

    # ── GĐ1: Feature importance ─────────────────────────────────────
    def feature_importance(self, importance_type: str = "split") -> np.ndarray:
        """
        Trả về mảng feature importance.

        Parameters
        ----------
        importance_type : "split" (số lần feature được split)
                          hoặc "gain" (tổng gain khi split feature đó)
        """
        if importance_type == "split":
            return self.feature_importances_split_.copy()
        elif importance_type == "gain":
            return self.feature_importances_gain_.copy()
        else:
            raise ValueError(f"importance_type phải là 'split' hoặc 'gain', "
                             f"không phải '{importance_type}'")

    # ── sklearn interface ────────────────────────────────────────────
    def get_params(self, deep=True):
        return {
            'n_estimators'         : self.n_estimators,
            'learning_rate'        : self.learning_rate,
            'max_depth'            : self.max_depth,
            'num_leaves'           : self.num_leaves,
            'min_child_samples'    : self.min_child_samples,
            'reg_lambda'           : self.reg_lambda,
            'reg_alpha'            : self.reg_alpha,
            'min_split_gain'       : self.min_split_gain,
            'subsample'            : self.subsample,
            'colsample_bytree'     : self.colsample_bytree,
            'max_bin'              : self.max_bin,
            'use_goss'             : self.use_goss,
            'top_rate'             : self.top_rate,
            'other_rate'           : self.other_rate,
            'use_efb'              : self.use_efb,
            'max_conflict_rate'    : self.max_conflict_rate,
            'early_stopping_rounds': self.early_stopping_rounds,
            'objective'            : self.objective,
            'random_state'         : self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ══════════════════════════════════════════════════════════════════════
#  PHẦN 6 — WRAPPER  (tương thích BaseModel + Trainer)
# ══════════════════════════════════════════════════════════════════════
class LGBMScratchModel(BaseModel):
    """
    LightGBM từ scratch — đầy đủ 4 giai đoạn.
    Kế thừa BaseModel để tương thích với Trainer của project.

    Kỹ thuật implement:
      GĐ1 — L1 regularization, min_split_gain, feature importance,
              predict(num_iteration=N)
      GĐ2 — Early stopping, multiple objectives (softmax/binary/mse)
      GĐ3 — GOSS: Gradient-based One-Side Sampling
      GĐ4 — EFB: Exclusive Feature Bundling
    """

    def __init__(self,
                 n_estimators         : int   = 100,
                 learning_rate        : float = 0.1,
                 max_depth            : int   = 6,
                 num_leaves           : int   = 31,
                 min_child_samples    : int   = 20,
                 reg_lambda           : float = 1.0,
                 reg_alpha            : float = 0.0,
                 min_split_gain       : float = 0.0,
                 subsample            : float = 1.0,
                 colsample_bytree     : float = 1.0,
                 max_bin              : int   = 255,
                 use_goss             : bool  = False,
                 top_rate             : float = 0.2,
                 other_rate           : float = 0.1,
                 use_efb              : bool  = False,
                 max_conflict_rate    : float = 0.0,
                 early_stopping_rounds: int | None = None,
                 objective            : str   = "softmax",
                 random_state         : int   = 42):
        super().__init__()
        self.n_estimators          = n_estimators
        self.learning_rate         = learning_rate
        self.max_depth             = max_depth
        self.num_leaves            = num_leaves
        self.min_child_samples     = min_child_samples
        self.reg_lambda            = reg_lambda
        self.reg_alpha             = reg_alpha
        self.min_split_gain        = min_split_gain
        self.subsample             = subsample
        self.colsample_bytree      = colsample_bytree
        self.max_bin               = max_bin
        self.use_goss              = use_goss
        self.top_rate              = top_rate
        self.other_rate            = other_rate
        self.use_efb               = use_efb
        self.max_conflict_rate     = max_conflict_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.objective             = objective
        self.random_state          = random_state
        self.pipeline_type         = 'tree'

    # ── BaseModel interface ─────────────────────────────────────────
    def get_param_distributions(self) -> dict:
        return {
            'n_estimators'     : [50, 100, 150],
            'learning_rate'    : [0.05, 0.1, 0.2],
            'max_depth'        : [3, 4, 5],
            'num_leaves'       : [15, 31],
            'min_child_samples': [20, 30, 50],
            'reg_lambda'       : [0.1, 1.0, 5.0, 10.0],
            'reg_alpha'        : [0.0, 0.1, 0.5, 1.0],
            'min_split_gain'   : [0.0, 0.01, 0.1],
            'subsample'        : [0.8, 1.0],
            'colsample_bytree' : [0.8, 1.0],
            'use_goss'         : [True, False],
            #'use_efb'          : [True, False],
        }

    def build(self, **params) -> 'LGBMScratchModel':
        cfg = self.get_params()
        cfg.pop('random_state', None)
        cfg['random_state'] = self.random_state
        cfg.update(params)
        self.model = GBDTMulticlass(**cfg)
        return self

    def fit(self, X, y, eval_set=None):
        self.model.fit(X, y, eval_set=eval_set)
        return self

    def predict(self, X, num_iteration=None):
        return self.model.predict(X, num_iteration=num_iteration)

    def predict_proba(self, X, num_iteration=None):
        return self.model.predict_proba(X, num_iteration=num_iteration)

    def feature_importance(self, importance_type="split"):
        return self.model.feature_importance(importance_type)

    # ── sklearn clone() interface ───────────────────────────────────
    def get_params(self, deep=True):
        return {
            'n_estimators'         : self.n_estimators,
            'learning_rate'        : self.learning_rate,
            'max_depth'            : self.max_depth,
            'num_leaves'           : self.num_leaves,
            'min_child_samples'    : self.min_child_samples,
            'reg_lambda'           : self.reg_lambda,
            'reg_alpha'            : self.reg_alpha,
            'min_split_gain'       : self.min_split_gain,
            'subsample'            : self.subsample,
            'colsample_bytree'     : self.colsample_bytree,
            'max_bin'              : self.max_bin,
            'use_goss'             : self.use_goss,
            'top_rate'             : self.top_rate,
            'other_rate'           : self.other_rate,
            'use_efb'              : self.use_efb,
            'max_conflict_rate'    : self.max_conflict_rate,
            'early_stopping_rounds': self.early_stopping_rounds,
            'objective'            : self.objective,
            'random_state'         : self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        if self.model is not None:
            self.model.set_params(**params)
        return self
