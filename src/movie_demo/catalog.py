import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from scipy.sparse import csr_array

REQUIRED_COLUMNS = (
    "Series_Title",
    "Released_Year",
    "Genre",
    "Director",
    "Star1",
    "Star2",
    "Star3",
    "Star4",
    "No_of_Votes",
)


class CatalogError(ValueError):
    """Invalid movie catalog structure or required values."""


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    director: str | None
    actors: tuple[str, ...]
    votes: int | None


@dataclass(frozen=True)
class FeatureMatrix:
    vocabulary: tuple[str, ...]
    weights: csr_array


@dataclass(frozen=True)
class Catalog:
    movies: tuple[Movie, ...]
    source_sha256: str
    genre_features: FeatureMatrix
    director_features: FeatureMatrix
    actor_features: FeatureMatrix
    issues: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "movies": len(self.movies),
            "source_sha256": self.source_sha256,
            "features": {
                "genres": len(self.genre_features.vocabulary),
                "directors": len(self.director_features.vocabulary),
                "actors": len(self.actor_features.vocabulary),
            },
            "missing_metadata": {
                "year": sum(movie.year is None for movie in self.movies),
                "genres": sum(not movie.genres for movie in self.movies),
                "director": sum(movie.director is None for movie in self.movies),
                "cast": sum(not movie.actors for movie in self.movies),
                "votes": sum(movie.votes is None for movie in self.movies),
            },
            "issues": list(self.issues),
        }


def _features(labels: tuple[tuple[str, ...], ...]) -> FeatureMatrix:
    vocabulary = tuple(sorted({label for group in labels for label in group}))
    indices = {label: index for index, label in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, group in enumerate(labels):
        for label in group:
            rows.append(row)
            columns.append(indices[label])
            values.append(1.0 / len(group))
    weights = csr_array(
        (values, (rows, columns)), shape=(len(labels), len(vocabulary)), dtype=float
    )
    weights.sort_indices()
    return FeatureMatrix(vocabulary, weights)


def _labels(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def load_catalog(path: str | Path) -> Catalog:
    source = Path(path).read_bytes()
    try:
        text = source.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CatalogError("Catalog must be UTF-8 encoded.") from exc

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    movies: list[Movie] = []
    issues: list[str] = []
    try:
        header: list[str] = next(reader, list[str]())
        if len(header) != len(set(header)):
            raise CatalogError("Catalog contains duplicate column names.")
        missing = sorted(set(REQUIRED_COLUMNS) - set(header))
        if missing:
            raise CatalogError(f"Missing required columns: {', '.join(missing)}")
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise CatalogError(
                    f"CSV line {reader.line_num}: expected {len(header)} fields, "
                    + f"got {len(values)}."
                )
            row = dict(zip(header, (value.strip() for value in values), strict=True))
            title = row["Series_Title"]
            if not title:
                raise CatalogError(f"CSV line {reader.line_num}: title is required.")

            year = None
            raw_year = row["Released_Year"]
            if raw_year:
                if len(raw_year) == 4 and raw_year.isascii() and raw_year.isdigit():
                    year = int(raw_year)
                    if year == 0:
                        year = None
                if year is None:
                    issues.append(
                        f"CSV line {reader.line_num}: invalid year {raw_year!r} "
                        + f"for {title!r}; year is unavailable."
                    )

            votes = None
            raw_votes = row["No_of_Votes"]
            if raw_votes:
                if not (raw_votes.isascii() and raw_votes.isdigit()):
                    raise CatalogError(
                        f"CSV line {reader.line_num}: vote count must be "
                        + "a non-negative integer."
                    )
                votes = int(raw_votes)
            movies.append(
                Movie(
                    movie_id=len(movies),
                    title=title,
                    year=year,
                    genres=_labels(row["Genre"].split(",")),
                    director=row["Director"] or None,
                    actors=_labels([row[f"Star{i}"] for i in range(1, 5)]),
                    votes=votes,
                )
            )
    except csv.Error as exc:
        raise CatalogError(f"Malformed CSV at line {reader.line_num}: {exc}") from exc
    if not movies:
        raise CatalogError("Catalog contains no movies.")

    return Catalog(
        movies=tuple(movies),
        source_sha256=hashlib.sha256(source).hexdigest(),
        genre_features=_features(tuple(movie.genres for movie in movies)),
        director_features=_features(
            tuple((movie.director,) if movie.director else () for movie in movies)
        ),
        actor_features=_features(tuple(movie.actors for movie in movies)),
        issues=tuple(issues),
    )
