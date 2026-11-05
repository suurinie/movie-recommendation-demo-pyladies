from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import streamlit as st
from scipy.sparse import csc_array, load_npz

from movie_demo.demo import (
    MovieChoice,
    load_movie_index,
    recommend_movies,
)
from movie_demo.model import ItemLinearModel
from movie_demo.model_selection import (
    DEFAULT_CUTOFF,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_INPUT_LIKES,
    DEFAULT_LAMBDAS,
    ModelSelectionResult,
    select_regularization,
)
from movie_demo.splitting import DEFAULT_SPLIT_SEED, split_viewers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "seed-42-50000-v2"
MATRIX_PATH = DATA_DIR / "interactions.npz"
MOVIES_PATH = DATA_DIR / "movies.csv"
L1_WEIGHTS_PATH = PROJECT_ROOT / "slides" / "assets" / "l1-l2-weights-full.npz"


@st.cache_data
def _load_movies(path: str) -> tuple[MovieChoice, ...]:
    return load_movie_index(path)


@st.cache_resource
def _load_interactions(path: str) -> csc_array:
    interactions = cast(csc_array, load_npz(path))
    return interactions


@st.cache_resource
def _load_precomputed_l1(
    path: str,
) -> tuple[ItemLinearModel, float, float, int, int]:
    with np.load(path, allow_pickle=False) as saved_model:
        model = ItemLinearModel(
            weights=np.asarray(saved_model["l1_weights"], dtype=np.float64),
            regularization=float(saved_model["l1_lambda"]),
            method="lasso",
        )
        recall_at_10 = float(saved_model["l1_validation_recall_at_10"])
        training_viewers = int(saved_model["training_viewers"])
        validation_viewers = int(saved_model["validation_viewers"])
    return (
        model,
        model.regularization,
        recall_at_10,
        training_viewers,
        validation_viewers,
    )


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
        "Train candidate L2 models and choose the one that finds the most "
        "held-out likes in the validation viewers' top ten."
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


def _show_precomputed_l1(movie_count: int) -> tuple[ItemLinearModel, float] | None:
    _ = st.header("1. Use the precomputed L1 model")
    try:
        model, regularization, recall_at_10, training_viewers, validation_viewers = (
            _load_precomputed_l1(str(L1_WEIGHTS_PATH))
        )
    except (OSError, KeyError, ValueError) as exc:
        _ = st.error(f"Could not load the precomputed L1 model: {exc}")
        return None
    if model.weights.shape != (movie_count, movie_count):
        _ = st.error("The precomputed L1 model does not match the movie catalog.")
        return None

    model_col, lambda_col, recall_col = st.columns(3)
    _ = model_col.metric("Model", "L1 / Lasso")
    _ = lambda_col.metric("Selected λ", f"{regularization:g}")
    _ = recall_col.metric("Validation Recall@10", f"{recall_at_10:.4f}")
    _ = st.caption(
        f"Loaded precomputed weights fitted on {training_viewers:,} viewers; "
        f"λ was selected using {validation_viewers:,} validation viewers."
    )
    return model, regularization


def _show_implementation_details() -> None:
    with st.expander("About the models"):
        _ = st.markdown(
            """
The L2 option is fitted here and selected using validation Recall@10. The L1
option loads a precomputed model trained on the same synthetic training data.
The separate test viewers are not used to select either model.
"""
        )


def main() -> None:
    st.set_page_config(page_title="Movie favorites", page_icon="🎬")
    _ = st.title("Movie recommendations")
    _ = st.write("Select favorite movies in the picker below.")
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

    model_choice = st.radio(
        "Recommendation model",
        options=("L2 — train and compare", "L1 — precomputed"),
        horizontal=True,
        key="recommendation_model",
    )
    selected_model: ItemLinearModel | None = None
    selected_regularization: float | None = None
    selected_model_name = "L2"
    if model_choice == "L2 — train and compare":
        selection_result = _show_model_selection(
            interactions.shape[0], interactions.shape[1]
        )
        if selection_result is not None:
            selected_model = selection_result.best_model
            selected_regularization = selection_result.best_regularization
    else:
        precomputed_l1 = _show_precomputed_l1(interactions.shape[0])
        if precomputed_l1 is not None:
            selected_model, selected_regularization = precomputed_l1
            selected_model_name = "L1"

    _show_implementation_details()
    _ = st.header("2. Try your favorites")
    _ = st.write(
        "Choose movies in the alphabetical favorites selector. "
        + "Recommendations use the selected model above."
    )

    if "favorite_movie_ids" not in st.session_state:
        st.session_state.favorite_movie_ids = []

    favorite_ids = st.multiselect(
        "Your favorites",
        options=_alphabetical_movie_ids(movies),
        format_func=_movie_id_labeler(movies),
        key="favorite_movie_ids",
        help="You can remove a selection with the x on its chip.",
    )

    if favorite_ids and selected_model is not None:
        recommendations = recommend_movies(
            selected_model, movies, favorite_ids, limit=10
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
                    f"Using {selected_model_name}, λ={selected_regularization:g}.",
                    "Scores are rankings, not probabilities.",
                )
            )
        )
    elif favorite_ids:
        _ = st.info("Select an available model above to see recommendations.")
    else:
        _ = st.info("Add at least one favorite to get recommendations.")


if __name__ == "__main__":
    main()
