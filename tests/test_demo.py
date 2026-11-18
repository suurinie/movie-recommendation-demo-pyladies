from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from movie_demo.app import (
    _alphabetical_movie_ids,
    _comparison_rows,
)
from movie_demo.demo import (
    MovieChoice,
    find_movies,
    load_movie_index,
    recommend_movies,
)
from movie_demo.model import ItemLinearModel


def test_load_movie_index_checks_row_alignment(tmp_path) -> None:
    path = tmp_path / "movies.csv"
    path.write_text(
        "movie_id,title,director,year\n0,My Neighbor Totoro,Hayao Miyazaki,1988\n"
        "1,Spirited Away,Hayao Miyazaki,2001\n",
        encoding="utf-8",
    )

    movies = load_movie_index(path)

    assert movies == (
        MovieChoice(0, "My Neighbor Totoro", "Hayao Miyazaki", 1988),
        MovieChoice(1, "Spirited Away", "Hayao Miyazaki", 2001),
    )
    assert movies[0].label == "My Neighbor Totoro (1988, Hayao Miyazaki)"


def test_load_movie_index_rejects_noncontiguous_ids(tmp_path) -> None:
    path = tmp_path / "movies.csv"
    path.write_text(
        "movie_id,title,director,year\n1,My Neighbor Totoro,,1988\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        load_movie_index(path)


def test_find_movies_is_case_insensitive_and_caps_results() -> None:
    movies = tuple(
        MovieChoice(movie_id, title, "", None)
        for movie_id, title in enumerate(
            ("My Neighbor Totoro", "Totoro", "Other", "Totoro Returns")
        )
    )

    results, match_count = find_movies(movies, "  TOTORO ", limit=2)

    assert [movie.title for movie in results] == ["My Neighbor Totoro", "Totoro"]
    assert match_count == 3


def test_find_movies_does_not_match_empty_search() -> None:
    movies = (MovieChoice(0, "Movie", "", None),)

    assert find_movies(movies, "   ") == ((), 0)


def test_alphabetical_movie_ids_sort_by_title_without_changing_ids() -> None:
    movies = (
        MovieChoice(0, "zebra", "", None),
        MovieChoice(1, "Apple", "", None),
        MovieChoice(2, "banana", "", None),
    )

    assert _alphabetical_movie_ids(movies) == (1, 2, 0)


def test_app_trains_before_favorite_recommendations() -> None:
    app_path = Path(__file__).parents[1] / "src" / "movie_demo" / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    assert not app.exception
    assert not app.selectbox
    assert all(
        button.label != "Add selected movies to favorites" for button in app.button
    )
    assert not app.subheader

    app.multiselect[0].set_value([10_000.0]).run()
    next(
        button for button in app.button if button.label == "Train and compare"
    ).click().run()
    assert not app.exception
    assert app.session_state["selection_result"].best_regularization == 10_000.0
    assert app.session_state["selection_viewer_counts"] == (40_000, 5_000, 5_000)
    assert app.dataframe
    assert "Recall@10" in [item.value for item in app.subheader]
    rows = _comparison_rows(app.session_state["selection_result"], 0.01)
    assert rows[-1] == {"Model": "Random baseline", "Recall@10": 0.01}
    assert rows[-2]["Model"] == "Popularity baseline"
    assert sum(row["Model"] == "Popularity baseline" for row in rows) == 1
    assert rows[-3]["Model"] == "λ = 10000 (selected)"
    assert set(rows[-2]) == {"Model", "Recall@10"}
    assert "Random baseline" in [metric.label for metric in app.metric]
    assert app.dataframe

    favorites = next(box for box in app.multiselect if box.label == "Your favorites")
    favorites.set_value([0]).run()
    assert not app.exception
    assert len(app.session_state["favorite_movie_ids"]) == 1
    assert any(item.value == "Recommended for you" for item in app.subheader)


def test_app_uses_precomputed_l1_without_training() -> None:
    app_path = Path(__file__).parents[1] / "src" / "movie_demo" / "app.py"
    app = AppTest.from_file(str(app_path)).run()

    model_choice = next(
        radio for radio in app.radio if radio.label == "Recommendation model"
    )
    model_choice.set_value("L1 — precomputed").run()

    assert not app.exception
    assert all(button.label != "Train and compare" for button in app.button)
    assert "Validation Recall@10" in [metric.label for metric in app.metric]
    app.multiselect[0].set_value([0]).run()
    assert not app.exception
    assert any(item.value == "Recommended for you" for item in app.subheader)


def test_recommendations_rank_unseen_movies_and_exclude_favorites() -> None:
    movies = tuple(
        MovieChoice(movie_id, f"Movie {movie_id}", "", None) for movie_id in range(4)
    )
    model = ItemLinearModel(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        ),
        regularization=1.0,
        method="fixture",
    )

    results = recommend_movies(model, movies, (0,), limit=2)

    assert [movie.movie_id for movie, _ in results] == [1, 2]
    assert [score for _, score in results] == [9.0, 6.0]


@pytest.mark.parametrize(
    ("favorite_ids", "limit", "error"),
    [
        ((), 10, "at least one favorite"),
        ((4,), 10, "outside the movie index"),
        ((0, 0), 10, "unique"),
        ((0,), 0, "at least 1"),
    ],
)
def test_recommendations_reject_invalid_inputs(
    favorite_ids: tuple[int, ...],
    limit: int,
    error: str,
) -> None:
    movies = (MovieChoice(0, "Movie", "", None),)
    model = ItemLinearModel(np.zeros((1, 1)), regularization=1.0, method="fixture")

    with pytest.raises(ValueError, match=error):
        recommend_movies(model, movies, favorite_ids, limit=limit)
