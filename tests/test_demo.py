from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from movie_demo.app import _comparison_rows, _merge_favorite_ids
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


def test_merge_favorite_ids_adds_unique_movies_in_selection_order() -> None:
    assert _merge_favorite_ids([5, 2], [2, 4, 5, 3, 4]) == [5, 2, 4, 3]


def test_app_trains_before_search_recommendations() -> None:
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
    assert [item.value for item in app.subheader] == ["NDCG@10", "Recall@10"]
    rows = _comparison_rows(app.session_state["selection_result"])
    assert rows[-1]["Model"] == "Popularity baseline"
    assert sum(row["Model"] == "Popularity baseline" for row in rows) == 1
    assert rows[-2]["Model"] == "λ = 10000 (selected)"
    assert "Recall@10" in rows[-1]
    assert app.dataframe

    app.text_input[0].set_value("Totoro")
    next(button for button in app.button if button.label == "Search").click().run()

    assert not app.exception
    search_selector = next(
        box for box in app.selectbox if box.label == "Matching movies"
    )
    assert search_selector.options == ["Tonari no Totoro (1988, Hayao Miyazaki)"]
    next(
        button for button in app.button if button.label == "Add movie to favorites"
    ).click().run()
    assert len(app.session_state["favorite_movie_ids"]) == 1
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
