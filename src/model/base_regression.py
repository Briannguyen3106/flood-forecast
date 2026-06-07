import numpy as np
from itertools import product
from sklearn.metrics import fbeta_score
from src.model.base_model import BaseModel

class BaseOrdinalRegression(BaseModel):
    def __init__(self, learning_rate = 0.01,
                 max_iter: int=1000,
                 tol: float = 1e-6,
                 random_state: int=42):
        super().__init__()
        self.learning_rate  = learning_rate
        self.max_iter       = max_iter
        self.tol            = tol
        self.random_state   = random_state
        self.pipeline_type  = 'linear'

        self.weights        = None
        self.bias           = None
        self.threshold1     = None
        self.threshold2     = None
        self.loss_history   = []

    def _predict_continuous(self, X):
        return np.dot(X, self.weights) + self.bias
        
    def _compute_loss(self, y, y_pred, weights):
        raise NotImplementedError
    def _compute_gradient(self, X, y, y_pred, weights):
        raise NotImplementedError

    def _regularization_curvature(self):
        return 0.0

    def _stable_learning_rate(self, X, sample_weight):
        row_squared_norms = np.einsum('ij,ij->i', X, X)
        data_curvature = 2.0 * np.average(
            row_squared_norms, weights=sample_weight
        )
        curvature = data_curvature + self._regularization_curvature()
        if not np.isfinite(curvature):
            raise ValueError("Training features produce non-finite curvature")
        if curvature <= np.finfo(float).eps:
            return self.learning_rate
        return min(self.learning_rate, 0.5 / curvature)
        
    def _gradient_descent(self, X, y, sample_weight=None):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values")

        self.weights = np.random.randn(n_features)*0.01
        self.bias = 0.0
        self.loss_history = []

        batch_size = min(32, n_samples)
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != (n_samples,) or np.any(sample_weight <= 0):
            raise ValueError("sample_weight must contain one positive value per row")
        prev_loss = np.inf

        for iteration in range(self.max_iter):

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            weight_shuffled = sample_weight[indices]

            for start  in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start: end]
                y_batch = y_shuffled[start: end]
                weight_batch = weight_shuffled[start:end]

                y_pred_batch = self._predict_continuous(X_batch)
                grad_w, grad_b = self._compute_gradient(
                    X_batch, y_batch, y_pred_batch, self.weights,
                    sample_weight=weight_batch
                )

                step_size = self._stable_learning_rate(X_batch, weight_batch)
                self.weights -= step_size * grad_w
                self.bias   -= step_size * grad_b

                if (not np.all(np.isfinite(self.weights))
                        or not np.isfinite(self.bias)):
                    raise RuntimeError(
                        "Gradient descent diverged; reduce learning_rate or scale features"
                    )

            y_pred_full = self._predict_continuous(X)
            loss = self._compute_loss(
                y, y_pred_full, self.weights, sample_weight=sample_weight
            )
            if not np.isfinite(loss):
                raise RuntimeError(
                    "Gradient descent produced a non-finite loss; "
                    "reduce learning_rate or scale features"
                )
            self.loss_history.append(loss)

            if abs(prev_loss - loss)< self.tol:
                break
            prev_loss = loss
            
    def _optimize_thresholds(self, X, y, sample_weight=None):
        y_pred_continuous  = self._predict_continuous(X)

        if not np.all(np.isfinite(y_pred_continuous)):
            raise RuntimeError("Cannot optimize thresholds from non-finite predictions")

        y_min, y_max = y_pred_continuous.min(), y_pred_continuous.max()
        candidates = np.linspace(y_min, y_max, 50)

        best_score = -np.inf
        best_1, best_2 = 0.5, 1.5
        for t1, t2 in product(candidates, candidates):
            if t1>=t2:
                continue
            y_class = self._apply_threshold(y_pred_continuous, t1, t2)
            score = fbeta_score(
                y, y_class, beta=2, average='macro', zero_division=0,
                sample_weight=sample_weight
            )

            if score>best_score:
                best_score = score
                best_1, best_2 = t1, t2
        self.threshold1 = best_1
        self.threshold2 = best_2

    def _apply_threshold(self, y_pred_continuous, t1, t2):
        y_class = np.zeros(len(y_pred_continuous), dtype=int)
        y_class[y_pred_continuous >= t1] = 1
        y_class[y_pred_continuous >= t2] = 2
        return y_class


    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        self._gradient_descent(X, y, sample_weight=sample_weight)

        # Sau khi đã có weights và bias
        self._optimize_thresholds(X, y, sample_weight=sample_weight)
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred_continuous = self._predict_continuous(X)
        return self._apply_threshold(
            y_pred_continuous, self.threshold1, self.threshold2
        )
 
    def predict_continuous(self, X: np.ndarray) -> np.ndarray:
        """Dùng để visualize regression output"""
        return self._predict_continuous(X)  
    
    #==================================================
    #Cho imbPipeline
    #==================================================
    
    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'max_iter'     : self.max_iter,
            'tol'          : self.tol,
            'random_state' : self.random_state
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = self
        return self
    
    def get_param_distributions(self):
        raise NotImplementedError
    
    
    def build(self, **params):
        # Reset weights để fit lại từ đầu
        self.weights    = None
        self.bias       = None
        self.threshold1 = None
        self.threshold2 = None
        # Set params mới
        for key, value in params.items():
            setattr(self, key, value)
        self.model = self  # Chính nó là model → ImbPipeline dùng được
        return self
