import csv
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from scipy.sparse import csc_array, load_npz
from sparse_helpers import differing_entries

from movie_demo.catalog import Catalog, load_catalog
from movie_demo.dataset_io import save_generated_dataset
from movie_demo.interactions import generate_interactions

CatalogWriter = Callable[[list[dict[str, str]]], Path]


def _catalog(
    write_catalog: CatalogWriter, movie_row: dict[str, str], count: int
) -> Catalog:
    rows = [
        dict(
            movie_row,
            Series_Title=f"Movie, {index}",
            Director="" if index == 1 else f"Director {index}",
            Released_Year="" if index == 2 else str(1990 + index),
        )
        for index in range(count)
    ]
    return load_catalog(write_catalog(rows))


def test_saves_matrix_and_minimal_index_with_matching_ids(
    write_catalog: CatalogWriter, movie_row: dict[str, str], tmp_path: Path
) -> None:
    catalog = _catalog(write_catalog, movie_row, 100)
    dataset = generate_interactions(catalog, num_viewers=8, seed=42)
    output = tmp_path / "generated"

    matrix_path, movies_path = save_generated_dataset(catalog, dataset, output)

    assert matrix_path == output / "interactions.npz"
    assert movies_path == output / "movies.csv"
    restored = cast(csc_array, load_npz(matrix_path))
    assert differing_entries(restored, dataset.matrix) == 0
    with movies_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["movie_id", "title", "director", "year"]
    assert len(rows) == len(catalog.movies)
    assert [int(row["movie_id"]) for row in rows] == list(range(len(rows)))
    assert rows[0] == {
        "movie_id": "0",
        "title": "Movie, 0",
        "director": "Director 0",
        "year": "1990",
    }
    assert rows[1]["director"] == ""
    assert rows[2]["year"] == ""


def test_refuses_existing_output_directory(
    write_catalog: CatalogWriter, movie_row: dict[str, str], tmp_path: Path
) -> None:
    catalog = _catalog(write_catalog, movie_row, 100)
    dataset = generate_interactions(catalog, num_viewers=2, seed=1)
    output = tmp_path / "generated"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve")

    with pytest.raises(FileExistsError, match="already exists"):
        save_generated_dataset(catalog, dataset, output)

    assert sentinel.read_text() == "preserve"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


def test_rejects_matrix_with_mismatched_catalog(
    write_catalog: CatalogWriter, movie_row: dict[str, str], tmp_path: Path
) -> None:
    catalog = _catalog(write_catalog, movie_row, 100)
    other_catalog = _catalog(write_catalog, movie_row, 99)
    dataset = generate_interactions(catalog, num_viewers=2, seed=1)

    with pytest.raises(ValueError, match="rows must match"):
        save_generated_dataset(other_catalog, dataset, tmp_path / "invalid")

    assert not (tmp_path / "invalid").exists()
