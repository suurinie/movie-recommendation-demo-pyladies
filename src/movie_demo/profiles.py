import random
from collections.abc import Iterator
from dataclasses import dataclass

from movie_demo.catalog import Catalog

DEFAULT_VIEWERS = 50_000
DEFAULT_SEED = 42
MIN_ACTIVITY = 12
MAX_ACTIVITY = 60
MEAN_ACTIVITY = 20.0
ACTIVITY_STDDEV = 6.0
GENRE_ALIGNMENT_RATE = 0.7


@dataclass(frozen=True)
class TastePreference:
    name: str
    strength: float


@dataclass(frozen=True)
class ViewerProfile:
    viewer_id: int
    preferred_genres: tuple[TastePreference, ...]
    preferred_directors: tuple[TastePreference, ...]
    preferred_actors: tuple[TastePreference, ...]
    target_interactions: int


def _sample_preferences(
    rng: random.Random,
    candidates: tuple[str, ...],
    aligned_candidates: tuple[str, ...],
    maximum: int,
    *,
    minimum: int = 0,
) -> tuple[TastePreference, ...]:
    if not candidates:
        return ()
    count = rng.randint(minimum, min(maximum, len(candidates)))
    if count == 0:
        return ()

    aligned_count = min(
        len(aligned_candidates),
        sum(rng.random() < GENRE_ALIGNMENT_RATE for _ in range(count)),
    )
    selected = rng.sample(aligned_candidates, k=aligned_count)
    remaining = tuple(name for name in candidates if name not in selected)
    selected.extend(rng.sample(remaining, k=count - len(selected)))
    return tuple(
        TastePreference(name=name, strength=rng.uniform(0.5, 1.5)) for name in selected
    )


def _activity_target(rng: random.Random) -> int:
    target = round(rng.gauss(MEAN_ACTIVITY, ACTIVITY_STDDEV))
    return min(MAX_ACTIVITY, max(MIN_ACTIVITY, target))


def iter_viewer_profiles(
    catalog: Catalog,
    num_viewers: int = DEFAULT_VIEWERS,
    seed: int = DEFAULT_SEED,
    *,
    start_viewer_id: int = 0,
) -> Iterator[ViewerProfile]:
    """Yield stable synthetic taste profiles, independent of iteration batching."""
    if type(num_viewers) is not int:
        raise TypeError("num_viewers must be an integer.")
    if num_viewers < 1:
        raise ValueError("num_viewers must be at least 1.")
    if type(seed) is not int:
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if type(start_viewer_id) is not int:
        raise TypeError("start_viewer_id must be an integer.")
    if start_viewer_id < 0:
        raise ValueError("start_viewer_id must be non-negative.")

    all_genres = catalog.genre_features.vocabulary
    all_directors = catalog.director_features.vocabulary
    all_actors = catalog.actor_features.vocabulary
    genre_directors: dict[str, set[str]] = {genre: set() for genre in all_genres}
    genre_actors: dict[str, set[str]] = {genre: set() for genre in all_genres}
    for movie in catalog.movies:
        for genre in movie.genres:
            if movie.director is not None:
                genre_directors[genre].add(movie.director)
            genre_actors[genre].update(movie.actors)

    for viewer_id in range(start_viewer_id, start_viewer_id + num_viewers):
        rng = random.Random(f"{seed}:{viewer_id}")
        genre_count = rng.randint(1, min(3, len(all_genres))) if all_genres else 0
        genre_names = rng.sample(all_genres, k=genre_count)
        genres = tuple(
            TastePreference(name, rng.uniform(0.5, 1.5)) for name in genre_names
        )
        preferred_genres = set(genre_names)
        aligned_directors = {
            director for name in preferred_genres for director in genre_directors[name]
        }
        aligned_actors = {
            actor for name in preferred_genres for actor in genre_actors[name]
        }
        directors = _sample_preferences(
            rng,
            all_directors,
            tuple(sorted(aligned_directors)),
            maximum=2,
        )
        actors = _sample_preferences(
            rng,
            all_actors,
            tuple(sorted(aligned_actors)),
            maximum=4,
        )
        yield ViewerProfile(
            viewer_id=viewer_id,
            preferred_genres=genres,
            preferred_directors=directors,
            preferred_actors=actors,
            target_interactions=_activity_target(rng),
        )
