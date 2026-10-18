import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from movie_demo.model import ItemLinearModel


@dataclass(frozen=True)
class MovieChoice:
    movie_id: int
    title: str
    director: str
    year: int | None

    @property
    def label(self) -> str:
        details = [str(self.year) if self.year is not None else "", self.director]
        suffix = ", ".join(detail for detail in details if detail)
        return f"{self.title} ({suffix})" if suffix else self.title


def load_movie_index(path: str | Path) -> tuple[MovieChoice, ...]:
    """Load and verify the compact row-aligned movie index."""
    movies: list[MovieChoice] = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ("movie_id", "title", "director", "year")
        if reader.fieldnames != list(expected):
            raise ValueError(f"Movie index columns must be {', '.join(expected)}.")
        for row_number, row in enumerate(reader):
            raw_id = row["movie_id"]
            raw_year = row["year"]
            title = row["title"].strip()
            if raw_id is None or not raw_id.isascii() or not raw_id.isdigit():
                raise ValueError(f"Movie index row {row_number + 2} has an invalid ID.")
            if int(raw_id) != row_number:
                raise ValueError("Movie IDs must be contiguous and match matrix rows.")
            if not title:
                raise ValueError(f"Movie index row {row_number + 2} has no title.")
            if raw_year and (not raw_year.isascii() or not raw_year.isdigit()):
                raise ValueError(
                    f"Movie index row {row_number + 2} has an invalid year."
                )
            movies.append(
                MovieChoice(
                    movie_id=row_number,
                    title=title,
                    director=(row["director"] or "").strip(),
                    year=int(raw_year) if raw_year else None,
                )
            )
    if not movies:
        raise ValueError("Movie index contains no movies.")
    return tuple(movies)


def find_movies(
    movies: tuple[MovieChoice, ...],
    query: str,
    *,
    limit: int = 25,
) -> tuple[tuple[MovieChoice, ...], int]:
    """Find title substrings and cap rendered results for a responsive picker."""
    if type(limit) is not int:
        raise TypeError("limit must be an integer.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return (), 0
    matches = tuple(
        movie for movie in movies if normalized_query in movie.title.casefold()
    )
    return matches[:limit], len(matches)


def recommend_movies(
    model: ItemLinearModel,
    movies: tuple[MovieChoice, ...],
    favorite_ids: Iterable[int],
    *,
    limit: int = 10,
) -> tuple[tuple[MovieChoice, float], ...]:
    """Return highest-scoring unseen movies for a viewer's selected favorites."""
    if type(limit) is not int:
        raise TypeError("limit must be an integer.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    liked_ids = tuple(favorite_ids)
    if not liked_ids:
        raise ValueError("Select at least one favorite movie.")
    if any(type(movie_id) is not int for movie_id in liked_ids):
        raise TypeError("Favorite movie IDs must be integers.")
    if len(set(liked_ids)) != len(liked_ids):
        raise ValueError("Favorite movie IDs must be unique.")
    if any(movie_id < 0 or movie_id >= len(movies) for movie_id in liked_ids):
        raise ValueError("Favorite movie ID is outside the movie index.")
    if model.weights.shape != (len(movies), len(movies)):
        raise ValueError("Model dimensions must match the movie index.")

    liked = np.zeros(len(movies), dtype=np.uint8)
    liked[list(liked_ids)] = 1
    scores = model.predict(liked)
    score_values = cast(list[float], scores.tolist())
    order = sorted(
        range(len(movies)), key=lambda movie_id: (-score_values[movie_id], movie_id)
    )
    recommendations: list[tuple[MovieChoice, float]] = []
    for movie_id in order:
        score = score_values[movie_id]
        if not np.isfinite(score):
            continue
        recommendations.append((movies[movie_id], score))
        if len(recommendations) == limit:
            break
    return tuple(recommendations)
