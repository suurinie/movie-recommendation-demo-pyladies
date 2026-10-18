from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_array

from movie_demo.catalog import load_catalog
from movie_demo.interactions import generate_interactions
from movie_demo.model import (
    ItemLinearModel,
    fit_analytical,
    fit_sgd,
    reconstruction_loss,
)
from movie_demo.splitting import split_viewers

CatalogWriter = Callable[[list[dict[str, str]]], Path]


def _matrix(seed: int = 5) -> csc_array:
    rng = np.random.default_rng(seed)
    dense = (rng.random((6, 300)) < 0.2).astype(np.uint8)
    dense[1] |= dense[0] & (rng.random(300) < 0.5)
    dense[2] |= dense[0] & (rng.random(300) < 0.4)
    return csc_array(dense)


def test_analytical_solution_matches_independent_row_solves() -> None:
    interactions = _matrix()
    regularization = 3.0

    model = fit_analytical(interactions, regularization)

    floating_interactions = interactions.astype(np.float64)
    gram = (floating_interactions @ floating_interactions.T).toarray()
    for movie_id in range(interactions.shape[0]):
        other_ids = np.delete(np.arange(interactions.shape[0]), movie_id)
        solution = np.linalg.solve(
            gram[np.ix_(other_ids, other_ids)]
            + regularization * np.eye(len(other_ids)),
            gram[other_ids, movie_id],
        )
        expected = np.zeros(interactions.shape[0])
        expected[other_ids] = solution
        np.testing.assert_allclose(model.weights[movie_id], expected, atol=1e-10)


def test_analytical_solution_does_not_overflow_integer_interactions() -> None:
    interactions = csc_array(np.ones((3, 400), dtype=np.uint8))

    model = fit_analytical(interactions, regularization=10.0)

    expected = np.full((3, 3), 400.0 / 810.0)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(model.weights, expected)
    assert np.all(np.isfinite(model.weights))


def test_analytical_model_has_zero_diagonal_and_improves_objective() -> None:
    interactions = _matrix()
    model = fit_analytical(interactions, regularization=3.0)
    zero_model = ItemLinearModel(np.zeros((6, 6)), regularization=3.0, method="zero")

    assert model.method == "analytical"
    np.testing.assert_array_equal(np.diag(model.weights), np.zeros(6))
    assert reconstruction_loss(interactions, model, batch_size=37) < (
        reconstruction_loss(interactions, zero_model)
    )


def test_prediction_masks_input_likes() -> None:
    interactions = _matrix()
    model = fit_analytical(interactions, regularization=3.0)
    liked = interactions[:, 0].toarray().ravel()

    scores = model.predict(liked)

    assert np.all(np.isneginf(scores[liked.astype(bool)]))
    assert np.all(np.isfinite(scores[~liked.astype(bool)]))


def test_prediction_rejects_wrong_shape() -> None:
    model = ItemLinearModel(np.zeros((3, 3)), regularization=1.0, method="test")

    with pytest.raises(ValueError, match="vector of length 3"):
        model.predict(np.zeros((3, 1)))


def test_sgd_is_reproducible_and_zero_diagonal() -> None:
    interactions = _matrix()

    first = fit_sgd(interactions, regularization=3.0, max_iter=300, seed=11)
    repeated = fit_sgd(interactions, regularization=3.0, max_iter=300, seed=11)

    assert first.method == "sgd"
    np.testing.assert_array_equal(first.weights, repeated.weights)
    np.testing.assert_array_equal(np.diag(first.weights), np.zeros(6))


def test_sgd_approaches_analytical_objective() -> None:
    interactions = _matrix()
    regularization = 3.0
    analytical = fit_analytical(interactions, regularization)
    stochastic = fit_sgd(
        interactions,
        regularization,
        max_iter=1_000,
        tolerance=1e-5,
        seed=12,
    )

    assert (
        reconstruction_loss(interactions, stochastic)
        <= reconstruction_loss(interactions, analytical) * 1.2
    )


def test_model_fits_only_training_viewers(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    rows = [
        dict(
            movie_row,
            Series_Title=f"Movie {index}",
            Genre=f"Genre {index % 4}",
            Director=f"Director {index % 4}",
            Star1=f"Actor {index % 4}",
            Star2="",
            Star3="",
            Star4="",
        )
        for index in range(40)
    ]
    catalog = load_catalog(write_catalog(rows))
    interactions = generate_interactions(catalog, num_viewers=100, seed=8).matrix
    split = split_viewers(interactions, seed=9)

    model = fit_analytical(split.train, regularization=2.0)

    assert model.weights.shape == (40, 40)
    assert split.validation.shape[1] == split.test.shape[1] == 10


@pytest.mark.parametrize("regularization", [0, -1, float("nan"), float("inf")])
def test_analytical_rejects_invalid_regularization(
    regularization: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        fit_analytical(_matrix(), regularization)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"max_iter": 0}, ValueError, "max_iter"),
        ({"tolerance": -1}, ValueError, "tolerance"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_sgd_rejects_invalid_configuration(
    kwargs: dict[str, int],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        fit_sgd(_matrix(), **kwargs)
