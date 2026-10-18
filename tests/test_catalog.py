import hashlib

import numpy as np
import pytest

from movie_demo.catalog import CatalogError, load_catalog


def test_load_and_normalize(write_catalog, movie_row):
    second = dict(movie_row, Series_Title="Second", Genre="Action")
    path = write_catalog([movie_row, second])
    catalog = load_catalog(path)

    assert [movie.movie_id for movie in catalog.movies] == [0, 1]
    assert [movie.title for movie in catalog.movies] == ["Example", "Second"]
    assert catalog.movies[0].year == 2001
    assert catalog.movies[0].votes == 100
    assert catalog.movies[0].genres == ("Drama", "Comedy")
    assert catalog.movies[0].actors == ("Actor A", "Actor B")
    assert catalog.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert catalog.genre_features.vocabulary == ("Action", "Comedy", "Drama")
    np.testing.assert_allclose(
        catalog.genre_features.weights.toarray(), [[0, 0.5, 0.5], [1, 0, 0]]
    )
    np.testing.assert_allclose(
        catalog.actor_features.weights.toarray(), [[0.5, 0.5], [0.5, 0.5]]
    )
    np.testing.assert_allclose(catalog.director_features.weights.toarray(), [[1], [1]])
    assert catalog.issues == ()
    assert catalog.summary()["missing_metadata"] == dict.fromkeys(
        ["year", "genres", "director", "cast", "votes"], 0
    )


def test_deterministic_features_and_hash(write_catalog, movie_row):
    path = write_catalog([movie_row])
    first, second = load_catalog(path), load_catalog(path)
    assert first.movies == second.movies
    assert first.source_sha256 == second.source_sha256
    for name in ("genre_features", "director_features", "actor_features"):
        left, right = getattr(first, name), getattr(second, name)
        assert left.vocabulary == right.vocabulary
        assert (left.weights != right.weights).nnz == 0
    path.write_bytes(path.read_bytes() + b"\n")
    assert load_catalog(path).source_sha256 != first.source_sha256


def test_missing_optional_cells(write_catalog, movie_row):
    row = dict.fromkeys(movie_row, "")
    row["Series_Title"] = "Unknown metadata"
    catalog = load_catalog(write_catalog([row]))
    movie = catalog.movies[0]
    assert movie.year is None and movie.votes is None and movie.director is None
    assert movie.genres == movie.actors == ()
    assert catalog.summary()["missing_metadata"] == dict.fromkeys(
        ["year", "genres", "director", "cast", "votes"], 1
    )
    for features in (
        catalog.genre_features,
        catalog.actor_features,
        catalog.director_features,
    ):
        assert features.weights.shape == (1, 0)
        assert features.weights.nnz == 0


def test_malformed_year_is_reported(write_catalog, movie_row):
    catalog = load_catalog(write_catalog([dict(movie_row, Released_Year="PG")]))
    assert catalog.movies[0].year is None
    assert len(catalog.issues) == 1
    assert "invalid year 'PG'" in catalog.issues[0]
    assert catalog.summary()["issues"] == list(catalog.issues)


def test_whitespace_and_duplicate_labels(write_catalog, movie_row):
    row = dict(
        movie_row,
        Series_Title="  Example  ",
        Genre=" Drama , Drama, Comedy, ",
        Star1=" Actor A ",
        Star2="Actor A",
    )
    movie = load_catalog(write_catalog([row])).movies[0]
    assert movie.title == "Example"
    assert movie.genres == ("Drama", "Comedy")
    assert movie.actors == ("Actor A",)


@pytest.mark.parametrize("votes", ["-1", "1.5", "many"])
def test_invalid_votes_rejected(write_catalog, movie_row, votes):
    with pytest.raises(CatalogError, match="vote count"):
        load_catalog(write_catalog([dict(movie_row, No_of_Votes=votes)]))


def test_blank_title_rejected(write_catalog, movie_row):
    with pytest.raises(CatalogError, match="title is required"):
        load_catalog(write_catalog([dict(movie_row, Series_Title=" ")]))


def test_empty_catalog_rejected(write_catalog):
    with pytest.raises(CatalogError, match="no movies"):
        load_catalog(write_catalog([]))


def test_missing_column_rejected(write_catalog, movie_row):
    row = dict(movie_row)
    del row["Genre"]
    with pytest.raises(CatalogError, match="Missing required columns: Genre"):
        load_catalog(write_catalog([row], columns=tuple(row)))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "Missing required columns"),
        ("Series_Title,Series_Title\n", "duplicate column"),
        ("a,b\n", "Missing required columns"),
    ],
)
def test_invalid_headers(tmp_path, text, message):
    path = tmp_path / "invalid.csv"
    path.write_text(text)
    with pytest.raises(CatalogError, match=message):
        load_catalog(path)


@pytest.mark.parametrize("suffix", ["extra,fields\n", '"unterminated\n'])
def test_malformed_csv(write_catalog, suffix):
    path = write_catalog([])
    path.write_text(path.read_text() + suffix)
    with pytest.raises(CatalogError, match="expected .* fields|Malformed CSV"):
        load_catalog(path)


def test_utf8_bom_supported(write_catalog, movie_row):
    path = write_catalog([movie_row])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert len(load_catalog(path).movies) == 1


def test_non_utf8_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"\xff")
    with pytest.raises(CatalogError, match="UTF-8"):
        load_catalog(path)
