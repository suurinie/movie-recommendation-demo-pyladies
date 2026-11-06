import numpy as np
import pytest
from scipy.sparse import csc_array
from sparse_helpers import differing_entries

from movie_demo.splitting import ViewerSplit, split_viewers


def _matrix(num_movies: int = 20, num_viewers: int = 100) -> csc_array:
    rows = np.arange(num_viewers, dtype=np.int32) % num_movies
    columns = np.arange(num_viewers, dtype=np.int32)
    return csc_array(
        (np.ones(num_viewers, dtype=np.uint8), (rows, columns)),
        shape=(num_movies, num_viewers),
    )


def test_splits_whole_viewers_into_disjoint_groups() -> None:
    interactions = _matrix()

    split = split_viewers(interactions, seed=13)

    assert isinstance(split, ViewerSplit)
    assert split.train.shape == (20, 80)
    assert split.validation.shape == (20, 10)
    assert split.test.shape == (20, 10)
    assert set(split.train_viewer_ids).isdisjoint(split.validation_viewer_ids)
    assert set(split.train_viewer_ids).isdisjoint(split.test_viewer_ids)
    assert set(split.validation_viewer_ids).isdisjoint(split.test_viewer_ids)
    np.testing.assert_array_equal(
        np.sort(
            np.concatenate(
                (
                    split.train_viewer_ids,
                    split.validation_viewer_ids,
                    split.test_viewer_ids,
                )
            )
        ),
        np.arange(100),
    )
    assert differing_entries(split.train, interactions[:, split.train_viewer_ids]) == 0
    assert (
        differing_entries(
            split.validation, interactions[:, split.validation_viewer_ids]
        )
        == 0
    )
    assert differing_entries(split.test, interactions[:, split.test_viewer_ids]) == 0


def test_split_is_reproducible_for_same_seed() -> None:
    interactions = _matrix()

    first = split_viewers(interactions, seed=55)
    repeated = split_viewers(interactions, seed=55)

    np.testing.assert_array_equal(first.train_viewer_ids, repeated.train_viewer_ids)
    np.testing.assert_array_equal(
        first.validation_viewer_ids, repeated.validation_viewer_ids
    )
    np.testing.assert_array_equal(first.test_viewer_ids, repeated.test_viewer_ids)


def test_different_seed_changes_assignment() -> None:
    interactions = _matrix()

    first = split_viewers(interactions, seed=55)
    other = split_viewers(interactions, seed=56)

    assert not np.array_equal(first.train_viewer_ids, other.train_viewer_ids)


def test_zero_validation_or_test_fraction_is_supported() -> None:
    split = split_viewers(
        _matrix(num_viewers=10),
        validation_fraction=0,
        test_fraction=0.2,
    )

    assert split.train.shape == (20, 8)
    assert split.validation.shape == (20, 0)
    assert split.test.shape == (20, 2)


def test_fraction_rounding_keeps_positive_groups_nonempty() -> None:
    split = split_viewers(_matrix(num_viewers=3))

    assert split.train.shape[1] == 1
    assert split.validation.shape[1] == 1
    assert split.test.shape[1] == 1


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction"),
    [
        (-0.1, 0.1),
        (1.0, 0.0),
        (float("nan"), 0.1),
        (0.1, float("inf")),
        (0.7, 0.3),
    ],
)
def test_invalid_fractions_rejected(
    validation_fraction: float, test_fraction: float
) -> None:
    with pytest.raises(ValueError):
        split_viewers(
            _matrix(),
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )


@pytest.mark.parametrize("value", [True, "0.1"])
def test_fraction_type_must_be_numeric(value: object) -> None:
    with pytest.raises(TypeError, match="validation_fraction must be a number"):
        split_viewers(_matrix(), validation_fraction=value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_seed_must_be_a_nonnegative_integer(seed: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        split_viewers(_matrix(), seed=seed)


def test_impossible_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="no viewers for training"):
        split_viewers(_matrix(num_viewers=2))
