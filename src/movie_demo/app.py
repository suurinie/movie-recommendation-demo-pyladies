from collections.abc import Callable
from pathlib import Path
from typing import cast

import streamlit as st
from scipy.sparse import csc_array, load_npz

from movie_demo.demo import (
    MovieChoice,
    find_movies,
    load_movie_index,
    recommend_movies,
)
from movie_demo.model import ItemLinearModel, fit_analytical

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "seed-42-50000"
MATRIX_PATH = DATA_DIR / "interactions.npz"
MOVIES_PATH = DATA_DIR / "movies.csv"
DEMO_REGULARIZATION = 10_000.0


@st.cache_data
def _load_movies(path: str) -> tuple[MovieChoice, ...]:
    return load_movie_index(path)


@st.cache_resource
def _load_demo_model(path: str, regularization: float) -> ItemLinearModel:
    interactions = cast(csc_array, load_npz(path))
    return fit_analytical(interactions, regularization=regularization)


def _movie_label(movie: MovieChoice) -> str:
    return movie.label


def _movie_id_labeler(movies: tuple[MovieChoice, ...]) -> Callable[[int], str]:
    def label(movie_id: int) -> str:
        return movies[movie_id].label

    return label


def main() -> None:
    st.set_page_config(page_title="Movie favorites", page_icon="🎬")
    _ = st.title("Movie recommendations")
    _ = st.write("Search for favorites and browse the catalog for recommendations.")
    if not MATRIX_PATH.is_file() or not MOVIES_PATH.is_file():
        _ = st.error("Demo data missing. Run with data/synthetic/seed-42-50000 files.")
        st.stop()

    try:
        movies = _load_movies(str(MOVIES_PATH))
        interactions = cast(csc_array, load_npz(MATRIX_PATH))
        if interactions.shape[0] != len(movies):
            raise ValueError("Movie index rows do not match the interaction matrix.")
        model = _load_demo_model(str(MATRIX_PATH), DEMO_REGULARIZATION)
    except (OSError, ValueError) as exc:
        _ = st.error(f"Could not load the movie demo data: {exc}")
        st.stop()

    if "favorite_movie_ids" not in st.session_state:
        st.session_state.favorite_movie_ids = []

    with st.form("movie_title_search"):
        search_text = st.text_input(
            "Find a movie by title",
            placeholder="Try Totoro",
            help="Type any part of a title; matching is case-insensitive.",
        )
        submitted = st.form_submit_button("Search")
    if submitted:
        st.session_state["movie_search_query"] = search_text.strip()

    query = cast(str, st.session_state.get("movie_search_query", ""))
    if query:
        matches, match_count = find_movies(movies, query)
        if matches:
            selected_match = st.selectbox(
                "Matching movies",
                options=matches,
                format_func=_movie_label,
                key="search_movie_choice",
            )
            if match_count > len(matches):
                _ = st.caption(
                    f"Showing {len(matches)} of {match_count} matches. "
                    + "Refine your search to narrow the list."
                )
            if st.button("Add movie to favorites"):
                favorite_ids = list(
                    cast(list[int], st.session_state.favorite_movie_ids)
                )
                if selected_match.movie_id not in favorite_ids:
                    favorite_ids.append(selected_match.movie_id)
                    st.session_state.favorite_movie_ids = favorite_ids
                    st.rerun()
                else:
                    _ = st.info("That movie is already in your favorites.")
        else:
            _ = st.info(f"No movie titles contain {query!r}.")
    else:
        _ = st.caption("Enter part of a title and select Search to see matches.")

    with st.expander("Browse the movie list"):
        _ = st.caption(f"{len(movies):,} titles, sorted alphabetically.")
        _ = st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            [
                {
                    "Title": movie.title,
                    "Year": movie.year,
                    "Director": movie.director,
                }
                for movie in sorted(movies, key=lambda item: item.title.casefold())
            ],
            hide_index=True,
            width="stretch",
        )

    favorite_ids = st.multiselect(
        "Your favorites",
        options=range(len(movies)),
        format_func=_movie_id_labeler(movies),
        key="favorite_movie_ids",
        help="You can remove a selection with the x on its chip.",
    )

    if favorite_ids:
        recommendations = recommend_movies(model, movies, favorite_ids, limit=10)
        _ = st.subheader("Recommended for you")
        if recommendations:
            _ = st.dataframe(  # pyright: ignore[reportUnknownMemberType]
                [
                    {
                        "Title": movie.title,
                        "Year": movie.year,
                        "Director": movie.director,
                        "Score": round(score, 4),
                    }
                    for movie, score in recommendations
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            _ = st.info("There are no unseen movies left to recommend.")
        _ = st.caption(
            " ".join(
                (
                    "This demo model uses all synthetic "
                    + "interactions and validation-selected λ=10,000.",
                    "Scores are rankings, not probabilities.",
                )
            )
        )
    else:
        _ = st.info("Add at least one favorite to get recommendations.")


if __name__ == "__main__":
    main()
