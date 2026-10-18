import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from scipy.sparse import load_npz

from movie_demo import main

CatalogWriter = Callable[[list[dict[str, str]]], Path]


def _write_large_catalog(
    write_catalog: CatalogWriter, movie_row: dict[str, str], count: int = 40
) -> Path:
    return write_catalog(
        [
            dict(
                movie_row,
                Series_Title=f"Movie {index}",
                Director=f"Director {index}",
                Star1=f"Actor {index}",
            )
            for index in range(count)
        ]
    )


def test_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["movie-demo", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert "collaborative filtering" in capsys.readouterr().out


def test_no_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["movie-demo"])

    main()

    assert "usage: movie-demo" in capsys.readouterr().out


def test_unknown_argument(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["movie-demo", "--unknown"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_catalog_summary(monkeypatch, capsys, write_catalog, movie_row):
    path = write_catalog([dict(movie_row, Released_Year="PG")])
    monkeypatch.setattr(sys, "argv", ["movie-demo", "catalog", "--catalog", str(path)])

    main()

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["movies"] == 1
    assert summary["missing_metadata"]["year"] == 1
    assert "Warning:" in captured.err
    assert "invalid year 'PG'" in captured.err


@pytest.mark.parametrize("exists", [False, True])
def test_catalog_error(monkeypatch, capsys, tmp_path, exists):
    path = tmp_path / "bad.csv"
    if exists:
        path.write_text("wrong,columns\n")
    monkeypatch.setattr(sys, "argv", ["movie-demo", "catalog", "--catalog", str(path)])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


def test_generate_command_saves_matrix_and_compact_index(
    monkeypatch,
    capsys,
    tmp_path,
    write_catalog,
    movie_row,
):
    catalog_path = _write_large_catalog(write_catalog, movie_row)
    output = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "movie-demo",
            "generate",
            "--catalog",
            str(catalog_path),
            "--viewers",
            "5",
            "--output",
            str(output),
        ],
    )

    main()

    summary = json.loads(capsys.readouterr().out)
    matrix = load_npz(output / "interactions.npz")
    assert summary["shape"] == [40, 5]
    assert summary["likes"] == matrix.nnz
    assert matrix.shape == (40, 5)
    with (output / "movies.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    assert list(rows[0]) == ["movie_id", "title", "director", "year"]


def test_generate_command_preserves_existing_output(
    monkeypatch,
    capsys,
    tmp_path,
    write_catalog,
    movie_row,
):
    catalog_path = _write_large_catalog(write_catalog, movie_row)
    output = tmp_path / "generated"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "movie-demo",
            "generate",
            "--catalog",
            str(catalog_path),
            "--viewers",
            "2",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert marker.read_text() == "keep"
    assert "already exists" in capsys.readouterr().err
