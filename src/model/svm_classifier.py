"""src/model/svm_classifier.py

A Support Vector Machine (SVM) classifier implemented **from scratch** using the
**dual formulation** and a custom **SMO (Sequential Minimal Optimization)** solver.

Why the dual?
- The primal soft-margin SVM is:

    minimize_{w, b, xi}  1/2 ||w||^2 + C * sum_i xi_i
    subject to          y_i (w^T phi(x_i) + b) >= 1 - xi_i,   xi_i >= 0

- The dual replaces the explicit feature map phi(.) with a kernel K(x, z) = phi(x)^T phi(z):

    maximize_{alpha}    sum_i alpha_i - 1/2 * sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
    subject to          0 <= alpha_i <= C
                        sum_i alpha_i y_i = 0

Once we have the optimal Lagrange multipliers alpha*, the decision function is:

    f(x) = sum_i alpha_i y_i K(x_i, x) + b

The "support vectors" are the training points with alpha_i > 0.

Optimization method (SMO)
-------------------------
The dual is a convex Quadratic Program (QP). A generic QP solver (e.g.
`scipy.optimize`) could be used, but SMO is a classic specialized method for SVMs:
- It updates two multipliers (alpha_i, alpha_j) at a time while respecting the
  equality constraint sum alpha_i y_i = 0.
- Each 2-variable sub-problem has a closed-form update + clipping to box constraints.

This implementation is designed to be compatible with this codebase’s training stack:
- It is a scikit-learn style estimator (`fit`, `predict`, `get_params`, `set_params`)
  so it can be used inside `imblearn.Pipeline` and `RandomizedSearchCV`.
- For custom models in this repo, the convention is `self.model = self` in `build()`.

Limitations / scope
-------------------
- Implements C-SVM classification (soft margin) for:
  - binary classification
  - multi-class via One-vs-Rest (OVR)
- Does NOT implement probabilities (Platt scaling), nor One-vs-One multi-class.

No usage of sklearn’s SVM implementation is made in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from src.model.base_model import BaseModel


KernelCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]
KernelName = Literal["linear", "poly", "rbf", "sigmoid"]
KernelSpec = KernelName | KernelCallable


@dataclass
class _BinarySVMState:
    """Holds the fitted parameters for one binary SVM classifier."""

    # Training metadata
    classes: tuple[object, object]  # (negative_class, positive_class)

    # Support vectors + dual coefficients
    support_indices: np.ndarray  # (n_support,)
    support_vectors: np.ndarray  # (n_support, n_features)
    support_y_pm1: np.ndarray  # (n_support,) in {-1, +1}
    dual_coef: np.ndarray  # (n_support,) = alpha_i * y_i

    # Intercept term
    intercept: float

    # Full alphas (optional, mainly for debugging / validation)
    alphas: np.ndarray  # (n_samples,)

    # Training diagnostics
    n_updates: int


class SVMClassifierModel(BaseModel):
    """Kernel SVM classifier (dual form) with a custom SMO solver.

    Parameters mirror (roughly) `sklearn.svm.SVC` for familiarity.

    Notes on labels
    ---------------
    - For each binary problem, we map labels to y in {-1, +1}.
    - In multi-class mode, we train One-vs-Rest classifiers and pick the class
      with the largest decision function value.
    """

    def __init__(
        self,
        kernel: KernelSpec = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        eps: float = 1e-5,
        max_passes: int = 10,
        max_iter: int = 100_000,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__()

        # Hyperparameters (kept as attributes for sklearn compatibility)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.eps = eps
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        # This repo uses `pipeline_type` to choose the preprocessing pipeline.
        # SVM belongs to PipelineB (linear / distance-based), which includes scaling.
        self.pipeline_type = "linear"

        # Fitted attributes (set during fit)
        self.classes_: np.ndarray | None = None
        self._gamma_value_: float | None = None
        self._binary_state_: _BinarySVMState | None = None
        self._ovr_states_: list[_BinarySVMState] | None = None

    # ====================================================================== #
    # Hyperparameter tuning integration
    # ====================================================================== #
    def get_param_distributions(self) -> list[dict]:
        """Hyperparameter search space for `RandomizedSearchCV`.

        Returns a list of dictionaries to avoid redundant parameter combinations
        (e.g., `degree` is only used by the 'poly' kernel).
        """
        # Stronger regularization (smaller C) to combat overfitting.
        common = {
            "C": [0.01, 0.1, 1.0],
            "tol": [1e-3, 1e-4],
            "max_passes": [5, 10],
        }

        return [
            {"kernel": ["linear"], **common},
            {
                "kernel": ["rbf"],
                "gamma": ["scale", "auto", 0.01, 0.1],
                **common,
            },
            {
                "kernel": ["poly"],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "degree": [2, 3],
                "coef0": [0.0, 1.0],
                **common,
            },
            {
                "kernel": ["sigmoid"],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "coef0": [0.0, 1.0],
                **common,
            },
        ]

    def build(self, **params):
        """Prepare the estimator for training.

        In this repo, custom-from-scratch models conventionally set `self.model = self`
        so the imblearn Pipeline can call `fit/predict/get_params/set_params` directly.

        `params` overrides any attributes, enabling the trainer to pass tuned values.
        """

        for key, value in params.items():
            setattr(self, key, value)

        # IMPORTANT: This makes the pipeline step refer to this estimator itself.
        self.model = self

        # Reset fitted state so re-fitting starts cleanly.
        self.classes_ = None
        self._gamma_value_ = None
        self._binary_state_ = None
        self._ovr_states_ = None

        return self

    def get_params(self, deep: bool = True):
        # Must include all __init__ args for sklearn clone() compatibility.
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "tol": self.tol,
            "eps": self.eps,
            "max_passes": self.max_passes,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        # Keep the pipeline convention consistent.
        self.model = self
        return self

    # ====================================================================== #
    # Kernels
    # ====================================================================== #
    @staticmethod
    def _kernel_linear(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Linear kernel: K(A, B) = A B^T"""

        return A @ B.T

    @staticmethod
    def _kernel_poly(
        A: np.ndarray,
        B: np.ndarray,
        gamma: float,
        degree: int,
        coef0: float,
    ) -> np.ndarray:
        """Polynomial kernel: K(x, z) = (gamma * x^T z + coef0) ^ degree"""

        return (gamma * (A @ B.T) + coef0) ** degree

    @staticmethod
    def _kernel_sigmoid(A: np.ndarray, B: np.ndarray, gamma: float, coef0: float) -> np.ndarray:
        """Sigmoid kernel (a.k.a. tanh kernel): K(x, z) = tanh(gamma * x^T z + coef0).

        This matches the common formulation used by scikit-learn (`sklearn.svm.SVC`).
        """

        return np.tanh(gamma * (A @ B.T) + coef0)

    @staticmethod
    def _kernel_rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        """RBF (Gaussian) kernel: K(x, z) = exp(-gamma * ||x - z||^2)

        Efficient vectorized computation:
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a^T b
        """

        A_sq = np.sum(A * A, axis=1)[:, None]  # (nA, 1)
        B_sq = np.sum(B * B, axis=1)[None, :]  # (1, nB)
        sq_dists = A_sq + B_sq - 2.0 * (A @ B.T)
        # Numerical safety: small negative values can appear due to rounding.
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-gamma * sq_dists)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """Resolve `gamma` when it’s 'scale' or 'auto'.

        Match scikit-learn semantics:
        - 'scale': 1 / (n_features * X.var())
        - 'auto' : 1 / n_features
        """

        n_features = X.shape[1]

        if isinstance(self.gamma, str):
            g = self.gamma.lower().strip()
            if g == "scale":
                x_var = float(X.var())
                if x_var == 0.0:
                    return 1.0
                return 1.0 / (n_features * x_var)
            if g == "auto":
                return 1.0 / n_features
            raise ValueError(f"Unknown gamma string: {self.gamma}")

        gamma_val = float(self.gamma)
        if gamma_val <= 0:
            raise ValueError("gamma must be > 0")
        return gamma_val

    def _kernel_fn(self, gamma_value: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a function K(A, B) based on current kernel settings.

        Supports:
        - built-in kernels via strings: 'linear', 'poly', 'rbf', 'sigmoid'
        - user-provided callable: kernel(A, B) -> Gram matrix
        """

        # User-provided callable kernel
        if callable(self.kernel):
            user_kernel = self.kernel

            def _kuser(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                K = user_kernel(A, B)
                K = np.asarray(K, dtype=np.float64)
                expected = (A.shape[0], B.shape[0])
                if K.shape != expected:
                    raise ValueError(
                        "Callable kernel must return a matrix with shape "
                        f"{expected}, got {K.shape}"
                    )
                return K

            return _kuser

        k = str(self.kernel).lower().strip()
        if k == "linear":
            return self._kernel_linear
        if k == "poly":
            degree = int(self.degree)
            coef0 = float(self.coef0)

            def _kpoly(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                return self._kernel_poly(A, B, gamma=gamma_value, degree=degree, coef0=coef0)

            return _kpoly
        if k == "rbf":

            def _krbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                return self._kernel_rbf(A, B, gamma=gamma_value)

            return _krbf

        if k == "sigmoid":
            coef0 = float(self.coef0)

            def _ksig(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                return self._kernel_sigmoid(A, B, gamma=gamma_value, coef0=coef0)

            return _ksig

        raise ValueError(
            f"Unknown kernel: {self.kernel}. Use 'linear', 'poly', 'rbf', 'sigmoid', or a callable."
        )

    # ====================================================================== #
    # SMO solver (binary)
    # ====================================================================== #
    def _fit_binary_smo(
        self,
        X: np.ndarray,
        y_pm1: np.ndarray,
        classes: tuple[object, object],
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        K: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _BinarySVMState:
        """Fit a *binary* SVM using the simplified SMO algorithm.

        Inputs
        ------
        X      : (n_samples, n_features)
        y_pm1  : (n_samples,) labels in {-1, +1}

        Output
        ------
        _BinarySVMState with support vectors, dual coefficients, intercept.

        KKT conditions (intuition)
        --------------------------
        At the solution, each training point satisfies:
        - If alpha_i == 0      → point is outside margin / correctly classified
        - If 0 < alpha_i < C   → point is ON the margin (support vector)
        - If alpha_i == C      → point is inside margin or misclassified

        SMO tries to enforce these conditions by iteratively fixing violating points.
        """

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y_pm1.ndim != 1:
            raise ValueError("y must be 1D")

        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to train SVM")

        C = float(self.C)
        if C <= 0:
            raise ValueError("C must be > 0")
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != (n_samples,) or np.any(sample_weight <= 0):
            raise ValueError("sample_weight must contain one positive value per row")
        C_bounds = C * sample_weight

        tol = float(self.tol)
        eps = float(self.eps)
        max_passes = int(self.max_passes)
        max_iter = int(self.max_iter)

        rng = np.random.default_rng(self.random_state)

        # Precompute the full kernel matrix K_ij = K(x_i, x_j).
        # This costs O(n^2) memory, but here n~2k so it’s acceptable and speeds up SMO.
        #
        # For multi-class OVR training, we can reuse the same Gram matrix across the
        # per-class binary problems. If a precomputed K is provided, validate and use it.
        if K is None:
            K = kernel(X, X).astype(np.float64, copy=False)
        else:
            K = np.asarray(K, dtype=np.float64)
            expected = (n_samples, n_samples)
            if K.shape != expected:
                raise ValueError(f"Precomputed K must have shape {expected}, got {K.shape}")

        # Lagrange multipliers
        alphas = np.zeros(n_samples, dtype=np.float64)

        # Intercept (bias)
        b = 0.0

        # Decision values on the training set: f_i = sum_j alpha_j y_j K_ji + b
        # Start at 0 because alphas=0 and b=0.
        f = np.zeros(n_samples, dtype=np.float64)

        # Error cache: E_i = f_i - y_i
        E = f - y_pm1

        passes = 0
        n_updates = 0

        # ------------------------------------------------------------------ #
        # Helper: pick second index j for the pair update.
        # A common heuristic: choose the point with maximum |E_i - E_j|.
        # ------------------------------------------------------------------ #
        def select_j(i: int, E_i: float) -> int:
            nonlocal rng

            # Prefer non-bound examples (0 < alpha < C) because they are more informative.
            non_bound = np.where(
                (alphas > eps) & (alphas < C_bounds - eps)
            )[0]
            if non_bound.size > 1:
                # Choose j that maximizes the step size |E_i - E_j|.
                j = int(non_bound[np.argmax(np.abs(E_i - E[non_bound]))])
                if j != i:
                    return j

            # Fallback: random j != i
            j = int(rng.integers(0, n_samples - 1))
            if j >= i:
                j += 1
            return j

        # ------------------------------------------------------------------ #
        # Main SMO loop
        # ------------------------------------------------------------------ #
        iter_count = 0
        while passes < max_passes and iter_count < max_iter:
            num_changed = 0

            for i in range(n_samples):
                iter_count += 1
                if iter_count >= max_iter:
                    break

                E_i = float(E[i])
                y_i = float(y_pm1[i])
                alpha_i_old = float(alphas[i])
                C_i = float(C_bounds[i])

                # KKT violation check (simplified)
                # If y_i * E_i < -tol and alpha_i < C  → point violates margin (needs larger alpha)
                # If y_i * E_i >  tol and alpha_i > 0  → point violates (needs smaller alpha)
                r_i = y_i * E_i
                if not ((r_i < -tol and alpha_i_old < C_i - eps) or (r_i > tol and alpha_i_old > eps)):
                    continue

                j = select_j(i, E_i)
                E_j = float(E[j])
                y_j = float(y_pm1[j])

                alpha_j_old = float(alphas[j])
                C_j = float(C_bounds[j])

                # Compute L and H to satisfy 0 <= alpha <= C and sum alpha_i y_i = 0.
                if y_i != y_j:
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(C_j, C_i + alpha_j_old - alpha_i_old)
                else:
                    L = max(0.0, alpha_i_old + alpha_j_old - C_i)
                    H = min(C_j, alpha_i_old + alpha_j_old)

                if L == H:
                    continue

                # eta is the second derivative of the objective along the update direction.
                # Using the standard SMO derivation:
                #   eta = K_ii + K_jj - 2 K_ij
                # Must be > 0 for the update to make progress.
                K_ii = float(K[i, i])
                K_jj = float(K[j, j])
                K_ij = float(K[i, j])
                eta = K_ii + K_jj - 2.0 * K_ij
                if eta <= 0.0:
                    continue

                # Unconstrained update for alpha_j, then clip to [L, H]
                alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / eta
                alpha_j_new = float(np.clip(alpha_j_new, L, H))

                if abs(alpha_j_new - alpha_j_old) < eps:
                    continue

                # Update alpha_i to maintain equality constraint
                alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

                # Compute updated bias term candidates (Platt’s SMO)
                b_old = b
                b1 = (
                    b_old
                    - E_i
                    - y_i * (alpha_i_new - alpha_i_old) * K_ii
                    - y_j * (alpha_j_new - alpha_j_old) * K_ij
                )
                b2 = (
                    b_old
                    - E_j
                    - y_i * (alpha_i_new - alpha_i_old) * K_ij
                    - y_j * (alpha_j_new - alpha_j_old) * K_jj
                )

                # Choose b based on whether alphas are non-bound
                if 0.0 + eps < alpha_i_new < C_i - eps:
                    b = float(b1)
                elif 0.0 + eps < alpha_j_new < C_j - eps:
                    b = float(b2)
                else:
                    b = float(0.5 * (b1 + b2))

                # Commit alpha updates
                alphas[i] = alpha_i_new
                alphas[j] = alpha_j_new

                # Efficiently update f and E for ALL points:
                #   f_k = sum_m alpha_m y_m K_mk + b
                # Only alpha_i, alpha_j, and b changed, so:
                #   f += (delta_ai * y_i) * K_i + (delta_aj * y_j) * K_j + delta_b
                delta_ai = alpha_i_new - alpha_i_old
                delta_aj = alpha_j_new - alpha_j_old
                delta_b = b - b_old

                f += (delta_ai * y_i) * K[i, :] + (delta_aj * y_j) * K[j, :] + delta_b
                E = f - y_pm1

                num_changed += 1
                n_updates += 1

            

            # SMO stopping criterion: if we loop over all i and change nothing,
            # increment `passes`. Otherwise reset `passes`.
            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # ------------------------------------------------------------------ #
        # Post-processing: compute a more stable intercept b from KKT conditions.
        # For any i with 0 < alpha_i < C, we should have:
        #   y_i (sum_j alpha_j y_j K_ji + b) = 1
        # Rearranged:
        #   b = y_i - sum_j alpha_j y_j K_ji
        # We average b across such margin support vectors.
        # ------------------------------------------------------------------ #
        ay = alphas * y_pm1  # (n_samples,)

        margin_sv = np.where(
            (alphas > eps) & (alphas < C_bounds - eps)
        )[0]
        if margin_sv.size > 0:
            # Compute decision value without b for each margin SV index i:
            #   sum_j alpha_j y_j K_ji = (ay @ K[:, i])
            decision_wo_b = ay @ K[:, margin_sv]
            b = float(np.mean(y_pm1[margin_sv] - decision_wo_b))
        else:
            # Fallback: use any support vector(s).
            sv = np.where(alphas > eps)[0]
            if sv.size > 0:
                decision_wo_b = ay @ K[:, sv]
                b = float(np.mean(y_pm1[sv] - decision_wo_b))
            else:
                b = 0.0

        # Identify support vectors (alpha > 0)
        support_indices = np.where(alphas > eps)[0]
        support_vectors = X[support_indices]
        support_y_pm1 = y_pm1[support_indices]
        dual_coef = (alphas[support_indices] * support_y_pm1).astype(np.float64, copy=False)

        return _BinarySVMState(
            classes=classes,
            support_indices=support_indices,
            support_vectors=support_vectors,
            support_y_pm1=support_y_pm1,
            dual_coef=dual_coef,
            intercept=b,
            alphas=alphas,
            n_updates=n_updates,
        )

    # ====================================================================== #
    # Public API: fit / predict
    # ====================================================================== #
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """Fit the SVM.

        Multi-class handling:
        - If y has exactly 2 unique classes: train one binary SVM.
        - Else: train OVR SVMs, one per class.

        Important: `SMOTE` in the trainer can change class counts in folds.
        This implementation supports that as long as y contains >=2 classes.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != (X.shape[0],) or np.any(sample_weight <= 0):
            raise ValueError("sample_weight must contain one positive value per row")

        unique_classes = np.unique(y)
        if unique_classes.size < 2:
            raise ValueError("SVM requires at least 2 classes")

        self.classes_ = unique_classes

        # Resolve gamma once per fit (depends on the data if gamma='scale').
        gamma_value = self._resolve_gamma(X)
        self._gamma_value_ = gamma_value
        kernel = self._kernel_fn(gamma_value)

        # Binary classification
        if unique_classes.size == 2:
            neg_class = unique_classes[0]
            pos_class = unique_classes[1]
            y_pm1 = np.where(y == pos_class, 1.0, -1.0).astype(np.float64)

            self._binary_state_ = self._fit_binary_smo(
                X=X,
                y_pm1=y_pm1,
                classes=(neg_class, pos_class),
                kernel=kernel,
                sample_weight=sample_weight,
            )
            self._ovr_states_ = None
            return self

        # Multi-class (OVR)
        self._binary_state_ = None
        self._ovr_states_ = []

        # Compute the Gram matrix once and reuse across OVR classifiers.
        K_train = kernel(X, X).astype(np.float64, copy=False)
        for cls in unique_classes:
            # Positive class = cls, negative = rest.
            y_pm1 = np.where(y == cls, 1.0, -1.0).astype(np.float64)
            state = self._fit_binary_smo(
                X=X,
                y_pm1=y_pm1,
                classes=(f"not_{cls}", cls),
                kernel=kernel,
                K=K_train,
                sample_weight=sample_weight,
            )
            self._ovr_states_.append(state)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute raw decision scores f(x).

        Returns
        -------
        - Binary: shape (n_samples,)
        - Multi-class OVR: shape (n_samples, n_classes)

        For binary classification, predicted label is sign(f(x)).
        """

        if self.classes_ is None:
            raise ValueError("Model is not fitted yet")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        gamma_value = float(self._gamma_value_ if self._gamma_value_ is not None else 1.0)
        kernel = self._kernel_fn(gamma_value)

        # Binary
        if self._binary_state_ is not None:
            st = self._binary_state_
            K_sv = kernel(st.support_vectors, X)  # (n_support, n_samples)
            # f(x) = sum_i dual_coef_i K(x_i, x) + b
            scores = st.dual_coef @ K_sv + st.intercept
            return scores

        # Multi-class OVR
        assert self._ovr_states_ is not None
        scores_all = []
        for st in self._ovr_states_:
            K_sv = kernel(st.support_vectors, X)
            scores_all.append(st.dual_coef @ K_sv + st.intercept)

        # Stack to shape (n_classes, n_samples) then transpose
        return np.vstack(scores_all).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""

        if self.classes_ is None:
            raise ValueError("Model is not fitted yet")

        scores = self.decision_function(X)

        # Binary
        if scores.ndim == 1:
            assert self._binary_state_ is not None
            neg_class, pos_class = self._binary_state_.classes
            # NOTE (typing): `_BinarySVMState.classes` stores labels as `object` to
            # support arbitrary label types. NumPy’s type stubs for `np.where` are
            # strict about x/y being ArrayLike, so we wrap labels as 0-d arrays.
            pos = np.asarray(pos_class)
            neg = np.asarray(neg_class)
            return np.where(scores >= 0.0, pos, neg)

        # Multi-class OVR: pick class with max score
        # Order of columns corresponds to self.classes_ order.
        best = np.argmax(scores, axis=1)
        return self.classes_[best]

    # ====================================================================== #
    # Debug / inspection helpers (useful for tests)
    # ====================================================================== #
    @property
    def support_indices_(self) -> np.ndarray:
        """Support vector indices (binary only).

        For multiclass, each OVR classifier has its own support vectors; inspect
        `ovr_support_indices_` instead.
        """

        if self._binary_state_ is None:
            raise AttributeError("support_indices_ is only available for binary classification")
        return self._binary_state_.support_indices

    @property
    def support_vectors_(self) -> np.ndarray:
        if self._binary_state_ is None:
            raise AttributeError("support_vectors_ is only available for binary classification")
        return self._binary_state_.support_vectors

    @property
    def intercept_(self) -> float:
        if self._binary_state_ is None:
            raise AttributeError("intercept_ is only available for binary classification")
        return float(self._binary_state_.intercept)

    @property
    def ovr_support_indices_(self) -> list[np.ndarray]:
        if self._ovr_states_ is None:
            raise AttributeError("ovr_support_indices_ is only available for multiclass")
        return [st.support_indices for st in self._ovr_states_]
