from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csc_array, csr_matrix
from sklearn.linear_model import SGDRegressor


@dataclass(frozen=True)
class ItemLinearModel:
    """Zero-diagonal item-to-item predictor; rows predict movies from likes."""

    weights: np.ndarray
    regularization: float
    method: str

    def predict(self, liked: np.ndarray) -> np.ndarray:
        """Score movies for one viewer's binary liked-movie vector."""
        liked = np.asarray(liked)
        if liked.ndim != 1 or liked.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"liked must be a vector of length {self.weights.shape[1]}."
            )
        scores = self.weights @ liked
        scores[liked.astype(bool)] = -np.inf
        return scores


def _validate_training_data(interactions: csc_array, regularization: float) -> None:
    if interactions.shape[0] < 1 or interactions.shape[1] < 1:
        raise ValueError("interactions must have at least one movie and one viewer.")
    if not np.all(np.isfinite(interactions.data)):
        raise ValueError("interactions must contain only finite values.")
    if type(regularization) not in (int, float):
        raise TypeError("regularization must be a number.")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive.")


def fit_analytical(
    interactions: csc_array, regularization: float = 100.0
) -> ItemLinearModel:
    """Fit the exact zero-diagonal ridge solution using a movie Gram matrix."""
    _validate_training_data(interactions, regularization)
    floating_interactions = interactions.astype(np.float64)
    gram = (floating_interactions @ floating_interactions.T).toarray()
    system = gram.copy()
    system.flat[:: system.shape[0] + 1] += regularization
    precision = solve(
        system,
        np.eye(system.shape[0]),
        assume_a="pos",
        check_finite=False,
    )
    diagonal = np.diag(precision)
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        raise ArithmeticError("The regularized Gram matrix is not positive definite.")
    weights = -precision / diagonal[:, np.newaxis]
    np.fill_diagonal(weights, 0.0)
    return ItemLinearModel(weights, float(regularization), "analytical")


def fit_sgd(
    interactions: csc_array,
    regularization: float = 100.0,
    *,
    max_iter: int = 1_000,
    tolerance: float = 1e-3,
    seed: int = 42,
) -> ItemLinearModel:
    """Fit one sklearn SGDRegressor per target movie, excluding self-input."""
    _validate_training_data(interactions, regularization)
    if type(max_iter) is not int:
        raise TypeError("max_iter must be an integer.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if type(tolerance) not in (int, float):
        raise TypeError("tolerance must be a number.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    if type(seed) is not int:
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    num_movies, num_viewers = interactions.shape
    viewer_features = csr_matrix(interactions.T, dtype=np.float64)
    targets = interactions.astype(np.float64)
    weights = np.zeros((num_movies, num_movies), dtype=np.float64)
    alpha = regularization / num_viewers
    for movie_id in range(num_movies):
        feature_ids = np.concatenate(
            (np.arange(movie_id), np.arange(movie_id + 1, num_movies))
        )
        regressor = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=alpha,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tolerance,
            random_state=seed + movie_id,
        )
        _ = regressor.fit(
            viewer_features[:, feature_ids],
            np.asarray(targets[movie_id, :].toarray()).ravel(),
        )
        weights[movie_id, feature_ids] = regressor.coef_

    return ItemLinearModel(weights, float(regularization), "sgd")


def reconstruction_loss(
    interactions: csc_array,
    model: ItemLinearModel,
    *,
    batch_size: int = 256,
) -> float:
    """Return squared reconstruction error plus ridge penalty in bounded memory."""
    if interactions.shape[0] != model.weights.shape[1]:
        raise ValueError("Model and interaction movie dimensions must match.")
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    squared_error = 0.0
    for start in range(0, interactions.shape[1], batch_size):
        end = min(start + batch_size, interactions.shape[1])
        target: np.ndarray = np.asarray(
            interactions[:, start:end].toarray(), dtype=np.float64
        )
        residual = target - model.weights @ target
        residual_norm = float(np.linalg.norm(residual))
        squared_error += residual_norm**2
    weight_norm = float(np.linalg.norm(model.weights))
    return squared_error + model.regularization * weight_norm**2
