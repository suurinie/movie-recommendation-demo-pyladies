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
