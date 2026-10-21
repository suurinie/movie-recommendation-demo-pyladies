from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import csc_array

from movie_demo.catalog import Catalog
from movie_demo.profiles import (
    DEFAULT_SEED,
    DEFAULT_VIEWERS,
    TastePreference,
    ViewerProfile,
    iter_viewer_profiles,
)

GENRE_WEIGHT = 3.0
DIRECTOR_WEIGHT = 2.0
ACTOR_WEIGHT = 1.5
POPULARITY_EXPLORATION_RATE = 0.1
DEFAULT_BATCH_SIZE = 128
SamplingVersion = Literal["v1", "v2"]
DEFAULT_SAMPLING_VERSION: SamplingVersion = "v1"


@dataclass(frozen=True)
class InteractionDataset:
    """Binary movie-by-viewer interactions and the matching viewer profiles."""

    matrix: csc_array
    profiles: tuple[ViewerProfile, ...]


def _profile_vector(
    preferences: tuple[TastePreference, ...], indices: Mapping[str, int]
) -> np.ndarray:
    vector = np.zeros(len(indices), dtype=np.float64)
    for preference in preferences:
        vector[indices[preference.name]] = preference.strength
    return vector


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _popularity_distribution(
    catalog: Catalog, sampling_version: SamplingVersion
) -> np.ndarray:
    votes = np.array([movie.votes or 0 for movie in catalog.movies], dtype=np.float64)
    if sampling_version == "v1":
        popularity = np.log1p(votes)
    elif sampling_version == "v2":
        popularity = votes
    else:
        raise ValueError(f"Unknown sampling version: {sampling_version!r}.")
    total = popularity.sum()
    if total == 0:
        return np.full(len(votes), 1.0 / len(votes))
    return popularity / total


def _affinity(
    catalog: Catalog,
    profile: ViewerProfile,
    genre_indices: Mapping[str, int],
    director_indices: Mapping[str, int],
    actor_indices: Mapping[str, int],
) -> np.ndarray:
    scores = GENRE_WEIGHT * np.asarray(
        catalog.genre_features.weights
        @ _profile_vector(profile.preferred_genres, genre_indices)
    ).reshape(-1)
    scores += DIRECTOR_WEIGHT * np.asarray(
        catalog.director_features.weights
        @ _profile_vector(profile.preferred_directors, director_indices)
    ).reshape(-1)
    scores += ACTOR_WEIGHT * np.asarray(
        catalog.actor_features.weights
        @ _profile_vector(profile.preferred_actors, actor_indices)
    ).reshape(-1)
    return scores


def generate_interactions(
    catalog: Catalog,
    num_viewers: int = DEFAULT_VIEWERS,
    seed: int = DEFAULT_SEED,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sampling_version: SamplingVersion = DEFAULT_SAMPLING_VERSION,
) -> InteractionDataset:
    """Sample binary likes into a sparse movies-by-viewers matrix."""
    if type(num_viewers) is not int:
        raise TypeError("num_viewers must be an integer.")
    if num_viewers < 1:
        raise ValueError("num_viewers must be at least 1.")
    if type(seed) is not int:
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if sampling_version not in ("v1", "v2"):
        raise ValueError(f"Unknown sampling version: {sampling_version!r}.")
    num_movies = len(catalog.movies)
    if num_movies == 0:
        raise ValueError("Cannot generate interactions for an empty catalog.")

    genre_indices = {
        name: index for index, name in enumerate(catalog.genre_features.vocabulary)
    }
    director_indices = {
        name: index for index, name in enumerate(catalog.director_features.vocabulary)
    }
    actor_indices = {
        name: index for index, name in enumerate(catalog.actor_features.vocabulary)
    }
    popularity = _popularity_distribution(catalog, sampling_version)
    mixture = 1.0 - POPULARITY_EXPLORATION_RATE
    row_chunks: list[np.ndarray] = []
    column_chunks: list[np.ndarray] = []
    profiles: list[ViewerProfile] = []
    for batch_start in range(0, num_viewers, batch_size):
        batch_count = min(batch_size, num_viewers - batch_start)
        batch_profiles = tuple(
            iter_viewer_profiles(
                catalog,
                batch_count,
                seed,
                start_viewer_id=batch_start,
            )
        )
        profiles.extend(batch_profiles)
        batch_rows: list[np.ndarray] = []
        batch_columns: list[np.ndarray] = []
        for profile in batch_profiles:
            if profile.target_interactions > num_movies:
                raise ValueError(
                    f"Viewer {profile.viewer_id} requests "
                    f"{profile.target_interactions} interactions, but the catalog "
                    f"contains only {num_movies} movies."
                )
            probabilities = (
                mixture
                * _softmax(
                    _affinity(
                        catalog,
                        profile,
                        genre_indices,
                        director_indices,
                        actor_indices,
                    )
                )
                + POPULARITY_EXPLORATION_RATE * popularity
            )
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, profile.viewer_id])
            )
            selected = rng.choice(
                num_movies,
                size=profile.target_interactions,
                replace=False,
                p=probabilities,
            )
            batch_rows.append(np.asarray(selected, dtype=np.int32))
            batch_columns.append(
                np.full(profile.target_interactions, profile.viewer_id, dtype=np.int32)
            )
        row_chunks.append(np.concatenate(batch_rows))
        column_chunks.append(np.concatenate(batch_columns))

    rows = np.concatenate(row_chunks)
    columns = np.concatenate(column_chunks)
    matrix = csc_array(
        (
            np.ones(len(rows), dtype=np.uint8),
            (rows, columns),
        ),
        shape=(num_movies, num_viewers),
    )
    matrix.sort_indices()
    return InteractionDataset(matrix=matrix, profiles=tuple(profiles))
