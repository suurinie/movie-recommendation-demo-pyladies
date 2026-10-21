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
        # A liked movie is already known to the viewer, so it is not a
        # recommendation candidate. -inf guarantees it sorts below every unseen
        # movie, regardless of its predictor score.
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


# ---------------------------------------------------------------------------
# ANALYTICAL APPROACH
# Closed-form solution used by model selection and the Streamlit demo. It
# computes the constrained ridge model directly, with one regularized solve.
# ---------------------------------------------------------------------------
def fit_analytical(
    interactions: csc_array, regularization: float = 100.0
) -> ItemLinearModel:
    """Fit the exact zero-diagonal ridge solution using a movie Gram matrix.

    With H shaped (movies, viewers), the objective is
    ||H - XH||_F^2 + lambda * ||X||_F^2, subject to diag(X) = 0.
    Each row of X is therefore a ridge regression that predicts one movie's
    likes across viewers from the likes of all other movies.
    """
    _validate_training_data(interactions, regularization)
    floating_interactions = interactions.astype(np.float64)
    # G[i, j] is the dot product of the two movies' viewer-interaction rows.
    # The product is movie-by-movie, so the solve scales with catalog size,
    # not with the number of viewers.
    gram = (floating_interactions @ floating_interactions.T).toarray()
    # Ridge regularization adds lambda to each diagonal entry of G, making the
    # system positive definite even when movie rows are correlated or sparse.
    system = gram
    diagonal_indices = np.diag_indices_from(system)
    system[diagonal_indices] += regularization
    # Let P = (G + lambda I)^-1. For a fixed target movie i, partition
    # G + lambda I by putting i first:
    #     [ a    b.T ]
    #     [ b     C  ]
    # Here C = G[-i, -i] + lambda I and b = G[-i, i]. The constrained ridge
    # normal equations for predicting movie i from all other movies are
    # C @ beta = b, so beta = C^-1 @ b.
    #
    # The block-inverse identity gives P[-i, i] = -C^-1 @ b * P[i, i].
    # Therefore beta[j] = -P[j, i] / P[i, i] = -P[i, j] / P[i, i] for j != i,
    # where the last equality uses symmetry. These coefficients solve the
    # problem with x[i, i] constrained to zero; they are not an unconstrained
    # fit followed by truncation. Solve against the identity to obtain P
    # without explicitly inverting the regularized Gram matrix.
    regularized_gram_inverse = solve(
        system,
        np.eye(system.shape[0]),
        assume_a="pos",
        check_finite=False,
    )
    diagonal = np.diag(regularized_gram_inverse)
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        raise ArithmeticError("The regularized Gram matrix is not positive definite.")
    weights = -regularized_gram_inverse / diagonal[:, np.newaxis]
    # The ratio above gives -1 on the diagonal, but those entries are only
    # algebraic intermediates, not fitted self-prediction coefficients.
    np.fill_diagonal(weights, 0.0)
    return ItemLinearModel(weights, float(regularization), "analytical")


# ---------------------------------------------------------------------------
# SGD APPROACH (separate, iterative alternative)
# This fits one approximate regression per movie using sklearn's optimizer.
# It is useful for comparing approaches, but is not used by model selection.
# ---------------------------------------------------------------------------
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
    # sklearn expects samples in rows: each row here is one viewer and each
    # column is that viewer's binary like for a movie.
    viewer_features = csr_matrix(interactions.T, dtype=np.float64)
    targets = interactions.astype(np.float64)
    weights = np.zeros((num_movies, num_movies), dtype=np.float64)
    # SGDRegressor averages loss over samples. Scale alpha by viewer count so
    # regularization is on the scale of the summed-squares objective above.
    alpha = regularization / num_viewers
    for movie_id in range(num_movies):
        # Fit this target movie using every other movie as a feature. Omitting
        # movie_id structurally enforces a zero on the predictor's diagonal.
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
    """Return ||H-XH||_F^2 + lambda*||X||_F^2 in bounded memory."""
    if interactions.shape[0] != model.weights.shape[1]:
        raise ValueError("Model and interaction movie dimensions must match.")
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    squared_error = 0.0
    for start in range(0, interactions.shape[1], batch_size):
        end = min(start + batch_size, interactions.shape[1])
        # Reconstruct only this viewer batch; summing squared residual norms
        # across batches is the full Frobenius reconstruction error.
        target: np.ndarray = np.asarray(
            interactions[:, start:end].toarray(), dtype=np.float64
        )
        residual = target - model.weights @ target
        residual_norm = float(np.linalg.norm(residual))
        squared_error += residual_norm**2
    # The model penalty is independent of viewer batching, so add it once.
    weight_norm = float(np.linalg.norm(model.weights))
    return squared_error + model.regularization * weight_norm**2
