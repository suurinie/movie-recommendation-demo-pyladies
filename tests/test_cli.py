import json
import sys

import pytest

from movie_demo import main


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
