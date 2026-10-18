from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_array

from movie_demo.catalog import Catalog, load_catalog
from movie_demo.interactions import (
    ACTOR_WEIGHT,
    DEFAULT_BATCH_SIZE,
    DIRECTOR_WEIGHT,
    GENRE_WEIGHT,
    POPULARITY_EXPLORATION_RATE,
    InteractionDataset,
    generate_interactions,
)
from movie_demo.profiles import DEFAULT_SEED, DEFAULT_VIEWERS

CatalogWriter = Callable[[list[dict[str, str]]], Path]


def _catalog_rows(movie_row: dict[str, str], count: int) -> list[dict[str, str]]:
    return [
        dict(
            movie_row,
            Series_Title=f"Movie {index}",
            Genre=f"Genre {index % 2}",
            Director=f"Director {index % 2}",
            Star1=f"Actor {index % 2} A",
            Star2=f"Actor {index % 2} B",
            Star3=f"Actor {index % 2} C",
            Star4=f"Actor {index % 2} D",
            No_of_Votes=str(100 + index),
        )
        for index in range(count)
    ]


def _interaction_catalog(
    write_catalog: CatalogWriter, movie_row: dict[str, str], count: int = 100
) -> Catalog:
    return load_catalog(write_catalog(_catalog_rows(movie_row, count)))


def test_working_defaults_and_sampling_weights() -> None:
    assert DEFAULT_VIEWERS == 500
    assert DEFAULT_SEED == 42
    assert DEFAULT_BATCH_SIZE == 128
    assert (GENRE_WEIGHT, DIRECTOR_WEIGHT, ACTOR_WEIGHT) == (3.0, 2.0, 1.5)
    assert POPULARITY_EXPLORATION_RATE == 0.1


def test_matrix_shape_binary_counts_and_csc(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _interaction_catalog(write_catalog, movie_row)
    dataset = generate_interactions(catalog, num_viewers=50, seed=10, batch_size=7)

    assert isinstance(dataset, InteractionDataset)
    assert isinstance(dataset.matrix, csc_array)
    assert dataset.matrix.shape == (100, 50)
    assert dataset.matrix.nnz == sum(
        profile.target_interactions for profile in dataset.profiles
    )
    assert set(dataset.matrix.data) == {1}
    np.testing.assert_array_equal(
        np.diff(dataset.matrix.indptr),
        [profile.target_interactions for profile in dataset.profiles],
    )
    assert [profile.viewer_id for profile in dataset.profiles] == list(range(50))


def test_seed_reproducibility_and_batch_independence(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _interaction_catalog(write_catalog, movie_row)
    first = generate_interactions(catalog, num_viewers=40, seed=81, batch_size=1)
    repeated = generate_interactions(catalog, num_viewers=40, seed=81, batch_size=13)
    other_seed = generate_interactions(catalog, num_viewers=40, seed=82, batch_size=8)

    assert first.profiles == repeated.profiles
    assert (first.matrix != repeated.matrix).nnz == 0
    assert first.profiles != other_seed.profiles
    assert (first.matrix != other_seed.matrix).nnz > 0


def test_sampling_prefers_movies_matching_user_tastes(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _interaction_catalog(write_catalog, movie_row, count=100)
    dataset = generate_interactions(catalog, num_viewers=500, seed=13)

    matching = 0
    selected = 0
    for profile in dataset.profiles:
        preferred_genres = {item.name for item in profile.preferred_genres}
        start, end = (
            dataset.matrix.indptr[profile.viewer_id],
            dataset.matrix.indptr[profile.viewer_id + 1],
        )
        movie_ids = dataset.matrix.indices[start:end]
        selected += len(movie_ids)
        matching += sum(
            bool(preferred_genres.intersection(catalog.movies[movie_id].genres))
            for movie_id in movie_ids
        )

    assert selected > 0
    assert matching / selected > 0.75


def test_missing_votes_are_supported_with_uniform_exploration(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    rows = [dict(row, No_of_Votes="") for row in _catalog_rows(movie_row, 100)]
    catalog = load_catalog(write_catalog(rows))

    dataset = generate_interactions(catalog, num_viewers=10, seed=5)

    assert dataset.matrix.shape == (100, 10)
    assert dataset.matrix.nnz == sum(
        profile.target_interactions for profile in dataset.profiles
    )


def test_activity_larger_than_catalog_is_rejected(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _interaction_catalog(write_catalog, movie_row, count=1)

    with pytest.raises(ValueError, match="requests .* interactions"):
        generate_interactions(catalog, num_viewers=1, seed=2)


def test_empty_catalog_is_rejected(write_catalog: CatalogWriter) -> None:
    with pytest.raises(ValueError, match="no movies"):
        generate_interactions(load_catalog(write_catalog([])))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "at least 1"),
        (-1, "at least 1"),
        (1.5, "integer"),
        (True, "integer"),
        ("2", "integer"),
    ],
)
def test_num_viewers_must_be_positive_integer(
    write_catalog: CatalogWriter,
    movie_row: dict[str, str],
    value: int,
    message: str,
) -> None:
    catalog = _interaction_catalog(write_catalog, movie_row)
    with pytest.raises((TypeError, ValueError), match=message):
        generate_interactions(catalog, num_viewers=value)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(
    write_catalog: CatalogWriter,
    movie_row: dict[str, str],
    batch_size: int,
) -> None:
    catalog = _interaction_catalog(write_catalog, movie_row)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        generate_interactions(catalog, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1.5, True, "2"])
def test_batch_size_must_be_integer(
    write_catalog: CatalogWriter,
    movie_row: dict[str, str],
    batch_size: int,
) -> None:
    catalog = _interaction_catalog(write_catalog, movie_row)
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        generate_interactions(catalog, batch_size=batch_size)
