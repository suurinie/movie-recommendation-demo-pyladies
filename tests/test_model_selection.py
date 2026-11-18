import numpy as np
import pytest
from scipy.sparse import csc_array
from sparse_helpers import differing_entries

from movie_demo.model import ItemLinearModel
from movie_demo.model_selection import (
    DEFAULT_CUTOFF,
    DEFAULT_INPUT_LIKES,
    DEFAULT_LAMBDAS,
    LambdaScore,
    ModelSelectionResult,
    RankingMetrics,
    ValidationHoldout,
    evaluate_on_test,
    evaluate_ranking,
    make_validation_holdout,
    random_recall_at_k,
    select_regularization,
)
from movie_demo.splitting import split_viewers


def _structured_matrix(num_viewers: int = 300) -> csc_array:
    rng = np.random.default_rng(15)
    dense = np.zeros((8, num_viewers), dtype=np.uint8)
    dense[0] = 1
    dense[1] = 1
    dense[2] = rng.binomial(1, 0.6, num_viewers)
    dense[3] = dense[2] | rng.binomial(1, 0.1, num_viewers)
    dense[4] = rng.binomial(1, 0.25, num_viewers)
    dense[5] = dense[4] | rng.binomial(1, 0.1, num_viewers)
    dense[6] = 1
    dense[7] = rng.binomial(1, 0.2, num_viewers)
    return csc_array(dense)


def test_validation_holdout_is_reproducible_and_complementary() -> None:
    validation = _structured_matrix(30)

    first = make_validation_holdout(validation, input_likes=2, seed=23)
    repeated = make_validation_holdout(validation, input_likes=2, seed=23)

    assert first.inputs.shape == validation.shape
    assert first.targets.shape == validation.shape
    assert first.inputs.nnz == 60
    assert first.inputs.nnz + first.targets.nnz == validation.nnz
    assert (first.inputs.multiply(first.targets)).nnz == 0
    assert differing_entries(first.inputs, repeated.inputs) == 0
    assert differing_entries(first.targets, repeated.targets) == 0
    assert differing_entries(first.inputs + first.targets, validation) == 0


def test_validation_holdout_requires_more_likes_than_inputs() -> None:
    validation = csc_array(np.array([[1, 1], [0, 1], [0, 0]], dtype=np.uint8))

    with pytest.raises(ValueError, match="2 required"):
        make_validation_holdout(validation, input_likes=1)


def test_known_ranking_has_perfect_recall_and_ndcg() -> None:
    inputs = csc_array(np.array([[1], [0], [0], [0]], dtype=np.uint8))
    targets = csc_array(np.array([[0], [1], [1], [0]], dtype=np.uint8))
    model = ItemLinearModel(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
        regularization=1.0,
        method="fixture",
    )

    metrics = evaluate_ranking(model, ValidationHoldout(inputs, targets), k=2)

    assert metrics == RankingMetrics(1.0, 1.0, viewers=1, target_likes=2)


def test_selects_candidate_by_validation_recall_and_reports_baseline() -> None:
    matrix = _structured_matrix()
    split = split_viewers(matrix, seed=5)
    candidates = (0.1, 1.0, 10.0, 100.0)

    result = select_regularization(
        split.train,
        split.validation,
        candidates=candidates,
        input_likes=2,
        k=3,
        seed=6,
    )

    assert isinstance(result, ModelSelectionResult)
    assert result.metric == "recall_at_k"
    assert (
        result.best_regularization
        == max(
            result.scores, key=lambda score: score.metrics.recall_at_k
        ).regularization
    )
    assert result.best_regularization in candidates
    assert result.best_model.regularization == result.best_regularization
    assert len(result.scores) == len(candidates)
    assert all(isinstance(score, LambdaScore) for score in result.scores)
    assert all(
        score.metrics.viewers == split.validation.shape[1] for score in result.scores
    )
    assert result.popularity_metrics.viewers == split.validation.shape[1]
    assert all(0 <= score.metrics.recall_at_k <= 1 for score in result.scores)
    assert all(0 <= score.metrics.ndcg_at_k <= 1 for score in result.scores)


def test_select_regularization_supports_lasso_solver() -> None:
    matrix = _structured_matrix()
    split = split_viewers(matrix, seed=5)

    result = select_regularization(
        split.train,
        split.validation,
        candidates=(1.0, 10.0),
        input_likes=2,
        k=3,
        seed=6,
        solver="l1",
    )

    assert result.best_model.method == "lasso"
    assert (
        result.best_regularization
        == max(
            result.scores, key=lambda score: score.metrics.recall_at_k
        ).regularization
    )


def test_selection_is_reproducible() -> None:
    matrix = _structured_matrix()
    split = split_viewers(matrix, seed=5)

    first = select_regularization(
        split.train,
        split.validation,
        candidates=(1.0, 10.0),
        input_likes=2,
        k=3,
        seed=6,
    )
    repeated = select_regularization(
        split.train,
        split.validation,
        candidates=(1.0, 10.0),
        input_likes=2,
        k=3,
        seed=6,
    )

    assert first.best_regularization == repeated.best_regularization
    assert first.scores == repeated.scores
    assert first.popularity_metrics == repeated.popularity_metrics


def test_defaults_are_small_but_log_spaced() -> None:
    assert DEFAULT_LAMBDAS == (
        1.0,
        10.0,
        100.0,
        1_000.0,
        10_000.0,
        100_000.0,
    )
    assert DEFAULT_INPUT_LIKES == 5
    assert DEFAULT_CUTOFF == 10


@pytest.mark.parametrize(
    "candidates",
    [(), (0.0,), (-1.0,), (float("nan"),), (1.0, 1.0)],
)
def test_invalid_regularization_candidates(candidates: tuple[float, ...]) -> None:
    matrix = _structured_matrix()
    split = split_viewers(matrix, seed=5)
    with pytest.raises(ValueError):
        select_regularization(split.train, split.validation, candidates=candidates)


def test_no_validation_viewers_rejected() -> None:
    matrix = _structured_matrix()
    with pytest.raises(ValueError, match="at least one viewer"):
        make_validation_holdout(matrix[:, :0])


def test_evaluate_on_test_scores_model_and_popularity_on_same_holdout() -> None:
    split = split_viewers(_structured_matrix(), seed=3)
    result = select_regularization(
        split.train, split.validation, candidates=(1.0, 10.0), input_likes=1, k=2
    )

    scores = evaluate_on_test(
        result.best_model, split.train, split.test, input_likes=1, k=2
    )
    holdout = make_validation_holdout(split.test, input_likes=1)

    assert scores.model_metrics == evaluate_ranking(result.best_model, holdout, k=2)
    assert scores.model_metrics.viewers == split.test.shape[1]
    assert scores.popularity_metrics.viewers == split.test.shape[1]
    assert scores.popularity_metrics.target_likes == scores.model_metrics.target_likes


def test_evaluate_on_test_rejects_mismatched_movies() -> None:
    split = split_viewers(_structured_matrix(), seed=3)
    model = ItemLinearModel(np.zeros((8, 8)), 1.0, "analytical")

    with pytest.raises(ValueError, match="movie dimensions"):
        _ = evaluate_on_test(model, split.train, split.test[:7, :])


def test_random_recall_matches_simulated_random_recommendations() -> None:
    validation = _structured_matrix(200)
    holdout = make_validation_holdout(validation, input_likes=2, seed=5)
    rng = np.random.default_rng(9)
    recalls = []
    for viewer in range(validation.shape[1]):
        shown = holdout.inputs[:, [viewer]].indices
        hidden = holdout.targets[:, [viewer]].indices
        unseen = np.setdiff1d(np.arange(validation.shape[0]), shown)
        for _ in range(50):
            picked = rng.choice(unseen, size=3, replace=False)
            recalls.append(np.isin(hidden, picked).mean())

    expected = random_recall_at_k(validation.shape[0], input_likes=2, k=3)

    assert expected == pytest.approx(3 / 6)
    assert float(np.mean(recalls)) == pytest.approx(expected, abs=0.01)


def test_random_recall_rejects_no_unseen_movies() -> None:
    with pytest.raises(ValueError, match="unseen"):
        _ = random_recall_at_k(5, input_likes=5, k=10)
