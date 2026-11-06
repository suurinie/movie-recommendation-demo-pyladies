"""Item-to-item recommender: "viewers who liked A also tend to like B".

The model is one square table of numbers, ``weights``, with a row and a column
per movie. ``weights[i, j]`` says how much liking movie ``j`` should raise the
score of movie ``i``. A viewer's score for every movie is that table multiplied
by their likes (see ``ItemLinearModel.predict``).

Training finds the table that best re-creates the known likes of the training
viewers from each other's likes, with two rules:

* A movie may not predict itself (the diagonal is zero). Otherwise the best
  "prediction" for any liked movie is trivially itself, and nothing is learned.
* Large weights are penalised by a strength ``regularization`` (written λ,
  "lambda"). Without it the table memorises noise in the training viewers.
  Higher λ gives smaller, more cautious weights; lower λ fits the training data
  more tightly. The right value is chosen by trying several on held-out viewers
  (see ``model_selection.py``).

Three fitting methods share this shape and differ in the penalty and solver:

* ``fit_analytical``: L2 (ridge) penalty, solved exactly in one step. This is
  the default and the one the demo trains live.
* ``fit_lasso``: L1 penalty, which pushes many weights to exactly zero.
* ``fit_sgd``: L2 penalty like ``fit_analytical`` but solved approximately by
  repeated small updates. Kept for comparison.

Matrix convention used throughout: ``interactions`` has shape
``(movies, viewers)`` and entry ``[i, v]`` is 1 if viewer ``v`` liked movie
``i``, otherwise 0.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csc_array, csr_matrix
from sklearn.linear_model import Lasso, SGDRegressor


@dataclass(frozen=True)
class ItemLinearModel:
    """A trained movie-to-movie weight table plus how it was trained.

    ``weights[i, j]`` is how much liking movie ``j`` raises the score of movie
    ``i``. The diagonal is always zero. ``regularization`` (λ) and ``method``
    only record how the weights were produced; ``predict`` does not use them.
    """

    weights: np.ndarray
    regularization: float
    method: str

    def predict(self, liked: np.ndarray) -> np.ndarray:
        """Score every movie for one viewer.

        ``liked`` is a vector with one entry per movie: 1 for movies the viewer
        likes, 0 otherwise (this is not checked, only the length is). Returns
        one score per movie. Scores are only meaningful for ranking: they are
        not probabilities and can be negative.
        """
        liked = np.asarray(liked)
        if liked.ndim != 1 or liked.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"liked must be a vector of length {self.weights.shape[1]}."
            )
        # Row i of the table dotted with the likes = the summed influence of
        # everything the viewer likes on movie i.
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


def fit_analytical(
    interactions: csc_array, regularization: float = 100.0
) -> ItemLinearModel:
    """Fit the L2 (ridge) model exactly, in one matrix solve.

    ``interactions`` is the (movies, viewers) 0/1 matrix. ``regularization`` is
    λ, the penalty strength; it must be finite and positive. Memory use is
    O(movies²) because a dense movie-by-movie matrix is built, so this suits
    catalogs of thousands of movies, not millions. The number of viewers only
    affects the cost of building that matrix, not the solve.

    Formally, with H the interactions matrix and X the weight table, this
    minimises ``||H - X·H||² + λ·||X||²`` (sums of squares over all entries)
    subject to ``diag(X) = 0``. In words: each row of X is an independent ridge
    regression that predicts one movie's likes across all viewers from the
    likes of every *other* movie.

    Instead of running one regression per movie, this uses a closed-form
    shortcut (derived in the comments below) that gets every row from a single
    matrix inverse.
    """
    _validate_training_data(interactions, regularization)
    floating_interactions = interactions.astype(np.float64)
    # Gram matrix G: G[i, j] counts the viewers who liked both movie i and
    # movie j (the diagonal is each movie's own like count). Its size is
    # movies x movies regardless of how many viewers there are.
    gram = (floating_interactions @ floating_interactions.T).toarray()
    # Ridge: add λ to the diagonal of G (modifying `gram` in place). This is
    # what makes the matrix safely invertible even when movies are strongly
    # correlated or have few likes, and it is the "penalty on large weights".
    gram[np.diag_indices_from(gram)] += regularization

    # Why a single inverse is enough
    # ------------------------------
    # Let P be the inverse of (G + λI). Fix one target movie i. Predicting it
    # from all the other movies is an ordinary ridge regression whose solution
    # is beta = C^-1 @ b, where C is (G + λI) with row/column i removed and b is
    # column i of G with entry i removed. That is the "no self-prediction"
    # constraint: movie i is simply left out of its own inputs.
    #
    # A standard identity for inverting a matrix in blocks shows that this same
    # beta can be read straight off P:  beta[j] = -P[i, j] / P[i, i]  (j != i).
    # So dividing each row of P by minus its diagonal entry gives every
    # per-movie regression at once. This is exact; it is NOT an unconstrained
    # fit with the diagonal zeroed afterwards.
    #
    # We obtain P by solving (G + λI) @ P = I rather than calling an explicit
    # inverse routine, which is more numerically stable. ``assume_a="pos"``
    # tells SciPy the matrix is symmetric positive definite (true for G + λI
    # with λ > 0), allowing a faster Cholesky-based solver.
    regularized_gram_inverse = solve(
        gram,
        np.eye(gram.shape[0]),
        assume_a="pos",
        check_finite=False,
    )
    diagonal = np.diag(regularized_gram_inverse)
    # Safety net: for a positive-definite matrix the diagonal of the inverse is
    # positive. Anything else means the solve broke down numerically.
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        raise ArithmeticError("The regularized Gram matrix is not positive definite.")
    weights = -regularized_gram_inverse / diagonal[:, np.newaxis]
    # The division above puts -1 on the diagonal. Those values are a
    # by-product of the formula, not real weights, so replace them with the
    # required zero ("a movie does not predict itself").
    np.fill_diagonal(weights, 0.0)
    return ItemLinearModel(weights, float(regularization), "analytical")


def fit_lasso(
    interactions: csc_array,
    regularization: float = 100.0,
    *,
    max_iter: int = 1_000,
    tolerance: float = 1e-4,
) -> ItemLinearModel:
    """Fit the L1 (lasso) model: one sparse regression per movie.

    Same inputs and meaning of ``regularization`` (λ) as ``fit_analytical``,
    but the penalty is the sum of absolute weights instead of squared weights.
    That makes many weights exactly zero, so each movie ends up depending on
    only a few other movies, and the resulting table is easier to inspect.
    There is no closed form, so each movie is fitted iteratively by
    scikit-learn. This is much slower than ``fit_analytical``: one regression
    per movie, each over all other movies.

    ``max_iter`` caps the solver's passes per movie and ``tolerance`` is how
    small the improvement must get before it stops. If it hits ``max_iter``
    first, scikit-learn emits a ConvergenceWarning and the weights are only
    approximate.
    """
    _validate_training_data(interactions, regularization)
    if type(max_iter) is not int:
        raise TypeError("max_iter must be an integer.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if type(tolerance) not in (int, float):
        raise TypeError("tolerance must be a number.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    num_movies, num_viewers = interactions.shape
    # scikit-learn wants one row per sample. Here a sample is a viewer and the
    # features are movies, so transpose to (viewers, movies).
    viewer_features = csr_matrix(interactions.T, dtype=np.float64)
    targets = interactions.astype(np.float64)
    weights = np.zeros((num_movies, num_movies), dtype=np.float64)
    # scikit-learn's Lasso minimises
    #     (1 / (2n)) * sum_of_squared_errors + alpha * sum(|w|),  n = viewers.
    # We want sum_of_squared_errors + λ * sum(|w|), the form that
    # reconstruction_loss reports. Multiplying scikit-learn's objective by 2n
    # gives sum_of_squared_errors + 2n * alpha * sum(|w|), so alpha = λ / (2n).
    alpha = regularization / (2 * num_viewers)
    for movie_id in range(num_movies):
        # Use every movie except this one as input. Leaving it out is what
        # keeps the diagonal of the weight table at zero.
        feature_ids = np.concatenate(
            (np.arange(movie_id), np.arange(movie_id + 1, num_movies))
        )
        regressor = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tolerance,
            selection="cyclic",
        )
        _ = regressor.fit(
            viewer_features[:, feature_ids],
            np.asarray(targets[movie_id, :].toarray()).ravel(),
        )
        weights[movie_id, feature_ids] = regressor.coef_

    return ItemLinearModel(weights, float(regularization), "lasso")


def fit_sgd(
    interactions: csc_array,
    regularization: float = 100.0,
    *,
    max_iter: int = 1_000,
    tolerance: float = 1e-3,
    seed: int = 42,
) -> ItemLinearModel:
    """Fit the L2 model approximately with stochastic gradient descent (SGD).

    Targets the same objective as ``fit_analytical`` (L2 penalty, zero
    diagonal), but finds it by repeated small updates instead of an exact
    solve, one regression per movie. The result is close to, not identical
    to, the exact answer, and it is slower. It exists to show how an iterative
    solver compares with the closed form. ``seed`` makes runs reproducible;
    each movie uses ``seed + movie_id`` so movies get different shuffles.
    ``tolerance`` may be 0 here (never stop early before ``max_iter``).
    """
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
    # scikit-learn's SGDRegressor (squared_error, l2) minimises
    #     (1 / (2n)) * sum_of_squared_errors + alpha * (1/2) * sum(w^2),
    # with n = viewers. Multiplying by 2n gives
    #     sum_of_squared_errors + n * alpha * sum(w^2),
    # which equals the ridge objective in fit_analytical when alpha = λ / n.
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
    """Return the training objective: reconstruction error plus weight penalty.

    Reconstruction error is how far ``weights @ likes`` is from the real
    likes, summed squared over all movies and viewers. The penalty is
    ``λ * sum(|w|)`` for lasso models and ``λ * sum(w^2)`` for the others
    (analytical and SGD are both L2). Lower is better, and the value is
    comparable across models only if they use the same λ and penalty type.
    Viewers are processed in batches of ``batch_size`` so the dense
    (movies, batch) block stays small.
    """
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
    # The penalty depends only on the weights, not on the viewers, so add it
    # once rather than per batch. The penalty type follows the method.
    if model.method == "lasso":
        penalty = model.regularization * float(np.abs(model.weights).sum())
    else:
        weight_norm = float(np.linalg.norm(model.weights))
        penalty = model.regularization * weight_norm**2
    return squared_error + penalty
