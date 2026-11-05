import inspect
from collections.abc import Callable, Iterable
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
from movie_demo.model import fit_analytical
from movie_demo.model_selection import (
    DEFAULT_CUTOFF,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_INPUT_LIKES,
    DEFAULT_LAMBDAS,
    ModelSelectionResult,
    evaluate_ranking,
    select_regularization,
)
from movie_demo.splitting import DEFAULT_SPLIT_SEED, split_viewers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "seed-42-50000-v2"
MATRIX_PATH = DATA_DIR / "interactions.npz"
MOVIES_PATH = DATA_DIR / "movies.csv"


@st.cache_data
def _load_movies(path: str) -> tuple[MovieChoice, ...]:
    return load_movie_index(path)


@st.cache_resource
def _load_interactions(path: str) -> csc_array:
    interactions = cast(csc_array, load_npz(path))
    return interactions


@st.cache_resource
def _train_and_select(
    path: str,
    candidates: tuple[float, ...],
    split_seed: int,
    evaluation_seed: int,
    input_likes: int,
    cutoff: int,
) -> tuple[ModelSelectionResult, tuple[int, int, int]]:
    interactions = _load_interactions(path)
    split = split_viewers(interactions, seed=split_seed)
    result = select_regularization(
        split.train,
        split.validation,
        candidates=candidates,
        input_likes=input_likes,
        k=cutoff,
        seed=evaluation_seed,
    )
    counts = (
        split.train.shape[1],
        split.validation.shape[1],
        split.test.shape[1],
    )
    return result, counts


def _movie_label(movie: MovieChoice) -> str:
    return movie.label


def _movie_id_labeler(movies: tuple[MovieChoice, ...]) -> Callable[[int], str]:
    def label(movie_id: int) -> str:
        return movies[movie_id].label

    return label


def _alphabetical_movie_ids(movies: tuple[MovieChoice, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(movies)),
            key=lambda movie_id: movies[movie_id].title.casefold(),
        )
    )


def _merge_favorite_ids(
    favorite_ids: Iterable[int], movie_ids: Iterable[int]
) -> list[int]:
    merged_ids = list(favorite_ids)
    known_ids = set(merged_ids)
    for movie_id in movie_ids:
        if movie_id not in known_ids:
            merged_ids.append(movie_id)
            known_ids.add(movie_id)
    return merged_ids


def _add_favorites(movie_ids: Iterable[int]) -> bool:
    current_ids = cast(list[int], st.session_state["favorite_movie_ids"])
    updated_ids = _merge_favorite_ids(current_ids, movie_ids)
    if updated_ids == current_ids:
        return False
    st.session_state["favorite_movie_ids"] = updated_ids
    return True


def _lambda_label(value: float) -> str:
    return f"{value:g}"


def _comparison_rows(result: ModelSelectionResult) -> list[dict[str, str | float]]:
    rows = [
        {
            "Model": (
                f"λ = {score.regularization:g}"
                + (
                    " (selected)"
                    if score.regularization == result.best_regularization
                    else ""
                )
            ),
            "Recall@10": score.metrics.recall_at_k,
        }
        for score in result.scores
    ]
    rows.append(
        {
            "Model": "Popularity baseline",
            "Recall@10": result.popularity_metrics.recall_at_k,
        }
    )
    return rows


def _show_model_selection(
    movie_count: int,
    total_viewers: int,
) -> ModelSelectionResult | None:
    _ = st.header("1. Train and compare models")
    _ = st.write(
        " ".join(
            (
                "We are solving a *regularized* recommendation problem.\n\n",
                "    min_X||H - XH||_F² + λ||X||_F², subject to diag(X) = 0\n\n ",
                "`H` is movies × viewers; `X` predicts movie scores from other movies.",
                "The regularization parameter `λ` is a scalar that constitutes a ",
                "*family of models*.",
                "We select regularization values and fit with them.",
                "The best model is selected based on validation performance.",
                "We have split the viewers into training, validation, and test sets.\n",
                " 1. Fit `X` for each `λ` on training viewers\n",
                " 2. Choose best `λ` on validation viewers\n",
                " 3. Keep test viewers untouched until the model is trained.\n\n",
                "This is the way to measure real-world recommendation performance.",
            )
        )
    )
    _ = st.caption(
        " ".join(
            (
                f"Dataset: {movie_count:,} movies × {total_viewers:,} viewers.",
                "Retrain after changing λ values.",
            )
        )
    )
    candidates = st.multiselect(
        "Candidate λ values",
        options=DEFAULT_LAMBDAS,
        default=DEFAULT_LAMBDAS,
        format_func=_lambda_label,
        key="lambda_candidates",
    )
    selected_candidates = tuple(float(value) for value in candidates)
    train_clicked = st.button(
        "Train and compare",
        type="primary",
        disabled=not selected_candidates,
        key="train_models",
    )
    if not selected_candidates:
        _ = st.info("Select at least one λ value to run model selection.")
    if train_clicked:
        try:
            with st.spinner(
                f"Fitting {len(selected_candidates)} models and evaluating "
                + "validation recommendations..."
            ):
                result, viewer_counts = _train_and_select(
                    str(MATRIX_PATH),
                    selected_candidates,
                    DEFAULT_SPLIT_SEED,
                    DEFAULT_EVALUATION_SEED,
                    DEFAULT_INPUT_LIKES,
                    DEFAULT_CUTOFF,
                )
        except (OSError, ValueError, ArithmeticError) as exc:
            _ = st.error(f"Model selection failed: {exc}")
            return None
        st.session_state["selection_result"] = result
        st.session_state["selection_viewer_counts"] = viewer_counts

    result = cast(ModelSelectionResult | None, st.session_state.get("selection_result"))
    if result is None:
        _ = st.info("Run the comparison to train models and enable recommendations.")
        return None

    viewer_counts = cast(
        tuple[int, int, int],
        st.session_state.get("selection_viewer_counts", (0, 0, 0)),
    )
    train_count, validation_count, test_count = viewer_counts
    best_score = next(
        score
        for score in result.scores
        if score.regularization == result.best_regularization
    )
    baseline_metrics = result.popularity_metrics
    metric_col, recall_col, splits_col = st.columns(3)
    _ = metric_col.metric(
        "Selected λ",
        f"{result.best_regularization:g}",
        help="Chosen by validation Recall@10; not measured on test viewers.",
    )
    _ = recall_col.metric(
        "Validation Recall@10",
        f"{best_score.metrics.recall_at_k:.4f}",
        delta=(
            f"{best_score.metrics.recall_at_k - baseline_metrics.recall_at_k:+.4f} "
            "vs popularity"
        ),
    )
    _ = splits_col.metric(
        "Viewers (train / val / test)",
        f"{train_count:,} / {validation_count:,} / {test_count:,}",
    )

    score_rows = _comparison_rows(result)
    _ = st.dataframe(  # pyright: ignore[reportUnknownMemberType]
        score_rows, hide_index=True, width="stretch"
    )

    _ = st.subheader("Recall@10")
    _ = st.bar_chart(  # pyright: ignore[reportUnknownMemberType]
        score_rows, x="Model", y="Recall@10"
    )
    _ = st.caption(
        " ".join(
            (
                "Recall@10 is the fraction of held-out likes found in the top ten.",
                "The popularity baseline is included for comparison.",
            )
        )
    )
    return result


def _show_implementation_details() -> None:
    with st.expander("Under the hood: Python and the chosen tools"):
        _ = st.markdown(
            """
This demo keeps the data and model pipeline in ordinary Python modules. The
web interface is only the front end:

- **NumPy** makes the seeded viewer split and handles dense linear algebra.
- **SciPy sparse** stores the movie-by-viewer likes matrix efficiently;
  **SciPy linalg** solves the regularized linear system.
- **Streamlit** provides the interactive workflow without a separate
  front-end stack.
- **uv** manages the Python project and lockfile; **pytest** checks behavior;
  **Ruff** checks Python style.

The fit is analytical, not an epoch-based optimizer: each λ produces one
regularized solve.
"""
        )
        with st.expander("How do we split viewers?"):
            _ = st.markdown("Whole-viewer splits keep held-out likes out of training.")
            _ = st.code(inspect.getsource(split_viewers), language="python")
        with st.expander("What does the model fit?"):
            _ = st.markdown(
                " ".join(
                    (
                        "Objective: `||H - XH||_F² + λ||X||_F²`, with `diag(X) = 0`.",
                        "`H` is movies × viewers; `X` predicts movie scores.",
                    )
                )
            )
            _ = st.code(inspect.getsource(fit_analytical), language="python")
        with st.expander("How is λ selected?"):
            _ = st.markdown(
                " ".join(
                    (
                        "Use 5 validation likes as input; hold the rest out.",
                        "Compare every λ on the same split; choose highest mean Recall@10.",
                    )
                )
            )
            _ = st.code(inspect.getsource(select_regularization), language="python")
        with st.expander("How are recommendations evaluated?"):
            _ = st.markdown(
                " ".join(
                    (
                        "For each viewer, Recall@10 is the fraction of held-out likes in the top ten.",
                        "Average that fraction across validation viewers.",
                    )
                )
            )
            _ = st.code(inspect.getsource(evaluate_ranking), language="python")


def main() -> None:
    st.set_page_config(page_title="Movie favorites", page_icon="🎬")
    _ = st.title("Movie recommendations")
    _ = st.write("Search for favorites or select them from the checkbox menu.")
    _ = st.caption(
        "Experiment dataset: sampler v2 — raw vote counts drive 10% exploration."
    )
    if not MATRIX_PATH.is_file() or not MOVIES_PATH.is_file():
        _ = st.error(f"Demo data missing from {DATA_DIR}.")
        st.stop()

    try:
        movies = _load_movies(str(MOVIES_PATH))
        interactions = _load_interactions(str(MATRIX_PATH))
        if interactions.shape[0] != len(movies):
            raise ValueError("Movie index rows do not match the interaction matrix.")
    except (OSError, ValueError) as exc:
        _ = st.error(f"Could not load the movie demo data: {exc}")
        st.stop()

    selection_result = _show_model_selection(
        interactions.shape[0], interactions.shape[1]
    )
    _show_implementation_details()
    _ = st.header("2. Try your favorites")
    _ = st.write(
        "Search for movies or select favorites from the checkbox menu below. "
        + "Recommendations become available "
        + "after training and comparing models above."
    )

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
                if _add_favorites((selected_match.movie_id,)):
                    st.rerun()
                else:
                    _ = st.info("That movie is already in your favorites.")
        else:
            _ = st.info(f"No movie titles contain {query!r}.")
    else:
        _ = st.caption("Enter part of a title and select Search to see matches.")

    favorite_ids = st.multiselect(
        "Your favorites",
        options=_alphabetical_movie_ids(movies),
        format_func=_movie_id_labeler(movies),
        key="favorite_movie_ids",
        help="You can remove a selection with the x on its chip.",
    )

    if favorite_ids and selection_result is not None:
        recommendations = recommend_movies(
            selection_result.best_model, movies, favorite_ids, limit=10
        )
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
                    f"Using λ={selection_result.best_regularization:g} model.",
                    "Scores are rankings, not probabilities.",
                )
            )
        )
    elif favorite_ids:
        _ = st.info("Train and compare models above to see recommendations.")
    else:
        _ = st.info("Add at least one favorite to get recommendations.")


if __name__ == "__main__":
    main()
