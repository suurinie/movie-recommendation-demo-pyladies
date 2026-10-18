from collections.abc import Callable
from pathlib import Path

import pytest

from movie_demo.catalog import Catalog, load_catalog
from movie_demo.profiles import (
    DEFAULT_SEED,
    DEFAULT_VIEWERS,
    MAX_ACTIVITY,
    MIN_ACTIVITY,
    TastePreference,
    ViewerProfile,
    iter_viewer_profiles,
)

CatalogWriter = Callable[[list[dict[str, str]]], Path]


def _profile_catalog(
    write_catalog: CatalogWriter, movie_row: dict[str, str], movie_count: int = 8
) -> Catalog:
    rows = [
        dict(
            movie_row,
            Series_Title=f"Movie {index}",
            Genre=f"Genre {index}",
            Director=f"Director {index}",
            Star1=f"Actor {index} A",
            Star2=f"Actor {index} B",
            Star3=f"Actor {index} C",
            Star4=f"Actor {index} D",
        )
        for index in range(movie_count)
    ]
    return load_catalog(write_catalog(rows))


def test_default_configuration() -> None:
    assert DEFAULT_VIEWERS == 500
    assert DEFAULT_SEED == 42
    assert MIN_ACTIVITY == 12
    assert MAX_ACTIVITY == 60


def test_profiles_are_reproducible_and_seed_sensitive(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row)
    first = tuple(iter_viewer_profiles(catalog, num_viewers=30, seed=57))
    repeated = tuple(iter_viewer_profiles(catalog, num_viewers=30, seed=57))
    different = tuple(iter_viewer_profiles(catalog, num_viewers=30, seed=58))

    assert first == repeated
    assert first != different
    assert [profile.viewer_id for profile in first] == list(range(30))
    batches = tuple(iter_viewer_profiles(catalog, 10, 57)) + tuple(
        iter_viewer_profiles(catalog, 20, 57, start_viewer_id=10)
    )
    assert batches == first


def test_preferences_are_valid_unique_and_have_strengths(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row)
    genres = set(catalog.genre_features.vocabulary)
    directors = set(catalog.director_features.vocabulary)
    actors = set(catalog.actor_features.vocabulary)

    for profile in iter_viewer_profiles(catalog, num_viewers=500, seed=2):
        assert 1 <= len(profile.preferred_genres) <= min(3, len(genres))
        assert len(profile.preferred_directors) <= 2
        assert len(profile.preferred_actors) <= 4
        assert MIN_ACTIVITY <= profile.target_interactions <= MAX_ACTIVITY
        for preferences, vocabulary in (
            (profile.preferred_genres, genres),
            (profile.preferred_directors, directors),
            (profile.preferred_actors, actors),
        ):
            assert len({preference.name for preference in preferences}) == len(
                preferences
            )
            assert {preference.name for preference in preferences} <= vocabulary
            assert all(0.5 <= preference.strength <= 1.5 for preference in preferences)
            assert all(
                isinstance(preference, TastePreference) for preference in preferences
            )
        assert isinstance(profile, ViewerProfile)


def test_directors_and_actors_are_partly_genre_aligned(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row, movie_count=12)
    movie_by_genre = {movie.genres[0]: movie for movie in catalog.movies}
    matching = 0
    total = 0
    for profile in iter_viewer_profiles(catalog, num_viewers=2_000, seed=8):
        preferred_genres = {item.name for item in profile.preferred_genres}
        for preference in (*profile.preferred_directors, *profile.preferred_actors):
            total += 1
            matching += any(
                preference.name in (movie.director, *movie.actors)
                for genre, movie in movie_by_genre.items()
                if genre in preferred_genres
            )

    assert total > 0
    assert matching / total > 0.6


def test_activity_distribution_is_centered_near_twenty(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row)
    activities = [
        profile.target_interactions
        for profile in iter_viewer_profiles(catalog, num_viewers=5_000, seed=12)
    ]

    assert MIN_ACTIVITY <= min(activities)
    assert max(activities) <= MAX_ACTIVITY
    assert 19 <= sum(activities) / len(activities) <= 21


def test_profiles_support_catalog_without_features(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    row = dict.fromkeys(movie_row, "")
    row["Series_Title"] = "No metadata"
    catalog = load_catalog(write_catalog([row]))

    profiles = tuple(iter_viewer_profiles(catalog, num_viewers=3))

    assert len(profiles) == 3
    assert all(not profile.preferred_genres for profile in profiles)
    assert all(not profile.preferred_directors for profile in profiles)
    assert all(not profile.preferred_actors for profile in profiles)


@pytest.mark.parametrize("num_viewers", [0, -1])
def test_viewer_count_must_be_positive(
    write_catalog: CatalogWriter, movie_row: dict[str, str], num_viewers: int
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row)
    with pytest.raises(ValueError, match="at least 1"):
        _ = tuple(iter_viewer_profiles(catalog, num_viewers=num_viewers))


@pytest.mark.parametrize("num_viewers", [1.5, True, "2"])
def test_viewer_count_must_be_integer(
    write_catalog: CatalogWriter, movie_row: dict[str, str], num_viewers: int
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row)
    with pytest.raises(TypeError, match="num_viewers must be an integer"):
        _ = tuple(iter_viewer_profiles(catalog, num_viewers=num_viewers))


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_seed_must_be_nonnegative_integer(
    write_catalog: CatalogWriter, movie_row: dict[str, str], seed: int
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row)
    with pytest.raises((TypeError, ValueError)):
        _ = tuple(iter_viewer_profiles(catalog, seed=seed))


def test_start_viewer_id_must_be_nonnegative(
    write_catalog: CatalogWriter, movie_row: dict[str, str]
) -> None:
    catalog = _profile_catalog(write_catalog, movie_row)
    with pytest.raises(ValueError, match="start_viewer_id must be non-negative"):
        _ = tuple(iter_viewer_profiles(catalog, start_viewer_id=-1))
