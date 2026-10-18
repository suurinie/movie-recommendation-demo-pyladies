from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.sparse import csc_array

from movie_demo.model import ItemLinearModel, fit_analytical

DEFAULT_LAMBDAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
DEFAULT_INPUT_LIKES = 5
DEFAULT_CUTOFF = 10
DEFAULT_EVALUATION_SEED = 42
DEFAULT_EVALUATION_BATCH_SIZE = 256


@dataclass(frozen=True)
class ValidationHoldout:
    """Known likes and held-out liked targets for the same viewers."""

    inputs: csc_array
    targets: csc_array


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_k: float
    ndcg_at_k: float
    viewers: int
    target_likes: int


@dataclass(frozen=True)
class LambdaScore:
    regularization: float
    metrics: RankingMetrics


@dataclass(frozen=True)
class ModelSelectionResult:
    best_model: ItemLinearModel
    best_regularization: float
    metric: Literal["ndcg_at_k", "recall_at_k"]
    scores: tuple[LambdaScore, ...]
    popularity_metrics: RankingMetrics


def make_validation_holdout(
    interactions: csc_array,
    *,
    input_likes: int = DEFAULT_INPUT_LIKES,
    seed: int = DEFAULT_EVALUATION_SEED,
) -> ValidationHoldout:
    """Split each validation viewer's positives into fixed input and target likes."""
    if type(input_likes) is not int:
        raise TypeError("input_likes must be an integer.")
    if input_likes < 1:
        raise ValueError("input_likes must be at least 1.")
    if type(seed) is not int:
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if interactions.shape[0] < 2:
        raise ValueError("At least two movies are required for a validation holdout.")
    if interactions.shape[1] < 1:
        raise ValueError("Validation interactions must contain at least one viewer.")

    input_rows: list[np.ndarray] = []
    input_columns: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_columns: list[np.ndarray] = []
    for viewer_id in range(interactions.shape[1]):
        start = cast(int, interactions.indptr[viewer_id])
        end = cast(int, interactions.indptr[viewer_id + 1])
        movie_ids = interactions.indices[start:end]
        if len(movie_ids) <= input_likes:
            details = f"{len(movie_ids)} likes found; {input_likes + 1} required."
            raise ValueError(f"Viewer {viewer_id}: {details}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, viewer_id]))
        chosen = rng.choice(movie_ids, size=input_likes, replace=False)
        is_input = np.isin(movie_ids, chosen, assume_unique=True)
        target_ids = movie_ids[~is_input]
        input_rows.append(np.asarray(chosen, dtype=np.int32))
        input_columns.append(np.full(input_likes, viewer_id, dtype=np.int32))
        target_rows.append(np.asarray(target_ids, dtype=np.int32))
        target_columns.append(np.full(len(target_ids), viewer_id, dtype=np.int32))

    input_row_array = np.concatenate(input_rows)
    input_column_array = np.concatenate(input_columns)
    target_row_array = np.concatenate(target_rows)
    target_column_array = np.concatenate(target_columns)
    shape = interactions.shape
    inputs = csc_array(
        (
            np.ones(len(input_row_array), dtype=np.uint8),
            (input_row_array, input_column_array),
        ),
        shape=shape,
    )
    targets = csc_array(
        (
            np.ones(len(target_row_array), dtype=np.uint8),
            (target_row_array, target_column_array),
        ),
        shape=shape,
    )
    inputs.sort_indices()
    targets.sort_indices()
    return ValidationHoldout(inputs, targets)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    top = np.argpartition(-scores, kth=k - 1, axis=0)[:k]
    top_scores = np.take_along_axis(scores, top, axis=0)
    order = np.argsort(-top_scores, axis=0)
    return np.take_along_axis(top, order, axis=0)


def _measure_batch(
    scores: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> tuple[float, float]:
    scores = scores.copy()
    scores[inputs.astype(bool)] = -np.inf
    top = _top_indices(scores, k)
    columns = np.arange(cast(int, scores.shape[1]))
    hits = targets[top, columns].astype(np.float64)
    target_counts = np.asarray(targets.sum(axis=0), dtype=np.int64).ravel()
    recalls = hits.sum(axis=0) / target_counts
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    dcg = (hits * discounts[:, np.newaxis]).sum(axis=0)
    ideal_count = np.minimum(k, target_counts)
    idcg = np.cumsum(discounts)[ideal_count - 1]
    return float(recalls.sum()), float((dcg / idcg).sum())


def evaluate_ranking(
    model: ItemLinearModel,
    holdout: ValidationHoldout,
    *,
    k: int = DEFAULT_CUTOFF,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
) -> RankingMetrics:
    """Evaluate ranked held-out likes, averaging Recall@k and NDCG@k by viewer."""
    if type(k) is not int:
        raise TypeError("k must be an integer.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if holdout.inputs.shape != holdout.targets.shape:
        raise ValueError("Input and target matrices must have the same shape.")
    if model.weights.shape != (holdout.inputs.shape[0], holdout.inputs.shape[0]):
        raise ValueError("Model movie dimensions must match holdout rows.")
    if holdout.inputs.shape[1] == 0:
        raise ValueError("Cannot evaluate an empty viewer set.")
    target_counts: np.ndarray = np.asarray(
        np.diff(np.asarray(holdout.targets.indptr, dtype=np.int64)), dtype=np.int64
    )
    if 0 in target_counts:
        raise ValueError("Every evaluated viewer must have at least one target like.")

    effective_k = min(k, holdout.inputs.shape[0] - 1)
    if effective_k < 1:
        raise ValueError("At least two movies are required for ranking.")
    recall_sum = 0.0
    ndcg_sum = 0.0
    for start in range(0, holdout.inputs.shape[1], batch_size):
        end = min(start + batch_size, holdout.inputs.shape[1])
        inputs = holdout.inputs[:, start:end].toarray()
        targets = holdout.targets[:, start:end].toarray()
        scores = model.weights @ inputs
        recall_batch, ndcg_batch = _measure_batch(scores, inputs, targets, effective_k)
        recall_sum += recall_batch
        ndcg_sum += ndcg_batch
    viewers = holdout.inputs.shape[1]
    return RankingMetrics(
        recall_at_k=recall_sum / viewers,
        ndcg_at_k=ndcg_sum / viewers,
        viewers=viewers,
        target_likes=holdout.targets.nnz,
    )


def _popularity_model(training: csc_array) -> ItemLinearModel:
    popularity = np.asarray(training.sum(axis=1)).ravel().astype(np.float64)
    weights = np.repeat(popularity[:, np.newaxis], len(popularity), axis=1)
    np.fill_diagonal(weights, 0.0)
    return ItemLinearModel(weights, regularization=0.0, method="popularity")


def select_regularization(
    training: csc_array,
    validation: csc_array,
    *,
    candidates: tuple[float, ...] = DEFAULT_LAMBDAS,
    input_likes: int = DEFAULT_INPUT_LIKES,
    k: int = DEFAULT_CUTOFF,
    seed: int = DEFAULT_EVALUATION_SEED,
    metric: str = "ndcg_at_k",
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
) -> ModelSelectionResult:
    """Select ridge regularization by ranking validation viewers' held-out likes."""
    if training.shape[0] != validation.shape[0]:
        raise ValueError("Training and validation movie dimensions must match.")
    if not candidates:
        raise ValueError("At least one regularization candidate is required.")
    if any(type(value) not in (int, float) for value in candidates):
        raise TypeError("Regularization candidates must be numbers.")
    if any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("Regularization candidates must be finite and positive.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Regularization candidates must be unique.")
    if metric not in ("ndcg_at_k", "recall_at_k"):
        raise ValueError("metric must be 'ndcg_at_k' or 'recall_at_k'.")
    if validation.shape[1] < 1:
        raise ValueError("Validation interactions must contain at least one viewer.")

    holdout = make_validation_holdout(validation, input_likes=input_likes, seed=seed)
    scores: list[LambdaScore] = []
    models: list[ItemLinearModel] = []
    for regularization in candidates:
        model = fit_analytical(training, regularization=float(regularization))
        models.append(model)
        scores.append(
            LambdaScore(
                regularization=float(regularization),
                metrics=evaluate_ranking(model, holdout, k=k, batch_size=batch_size),
            )
        )
    best_index = max(
        range(len(scores)),
        key=lambda index: (
            getattr(scores[index].metrics, metric),
            scores[index].metrics.recall_at_k,
            scores[index].regularization,
        ),
    )
    popularity_metrics = evaluate_ranking(
        _popularity_model(training), holdout, k=k, batch_size=batch_size
    )
    return ModelSelectionResult(
        best_model=models[best_index],
        best_regularization=scores[best_index].regularization,
        metric=metric,
        scores=tuple(scores),
        popularity_metrics=popularity_metrics,
    )
