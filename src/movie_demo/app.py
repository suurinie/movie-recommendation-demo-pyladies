"""Streamlit demo: train a movie recommender, then try it on your favorites.

Run with ``streamlit run src/movie_demo/app.py``. Streamlit re-executes
``main()`` from top to bottom on every click or widget change, so anything
that must survive a rerun lives in ``st.session_state`` or a cache.

Page flow:
1. Pick a model: L2 (trained live here) or L1 (loaded from a precomputed file).
2. Pick favorite movies and see the model's top ten unseen recommendations.

The data is synthetic (generated, not real viewers). The modelling itself is in
``model.py``; how a λ is chosen is in ``model_selection.py``.
"""

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
    FinalTestScores,
    ModelSelectionResult,
    evaluate_on_test,
    random_recall_at_k,
    select_regularization,
)
from movie_demo.splitting import DEFAULT_SPLIT_SEED, split_viewers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "seed-42-50000-v2"
MATRIX_PATH = DATA_DIR / "interactions.npz"
MOVIES_PATH = DATA_DIR / "movies.csv"
# Weights of the L1 (lasso) model, saved ahead of time by
# scripts/compare_regularization.py because fitting lasso is too slow to do
# interactively.
L1_WEIGHTS_PATH = PROJECT_ROOT / "slides" / "assets" / "l1-l2-weights-full.npz"

# Caching: Streamlit reruns this script on every interaction, so loading and
# training are wrapped in caches. ``cache_data`` returns a copy per call (safe
# for small, plain data). ``cache_resource`` shares one object across all
# reruns and sessions (right for large matrices and models; it must not be
# modified by callers). Arguments are plain strings/numbers so they hash into
# a cache key; the same arguments return the cached result without recomputing.


@st.cache_data
def _load_movies(path: str) -> tuple[MovieChoice, ...]:
    return load_movie_index(path)


@st.cache_resource
def _load_interactions(path: str) -> csc_array:
    """Load the (movies, viewers) 0/1 like matrix saved as a SciPy .npz file."""
    interactions = cast(csc_array, load_npz(path))
    return interactions


@st.cache_resource
def _load_precomputed_l1(
    path: str,
    matrix_path: str,
    split_seed: int,
) -> tuple[ItemLinearModel, float, float, int, int, FinalTestScores]:
    """Load the saved L1 model and the facts recorded when it was trained.

    Returns (model, λ, validation Recall@10, training viewer count,
    validation viewer count, test scores). The test scores are computed here,
    on the same split the model was trained with, so they always match the
    saved weights. ``allow_pickle=False`` stops the file from running
    arbitrary code on load.
    """
    with np.load(path, allow_pickle=False) as saved_model:
        model = ItemLinearModel(
            weights=np.asarray(saved_model["l1_weights"], dtype=np.float64),
            regularization=float(saved_model["l1_lambda"]),
            method="lasso",
        )
        recall_at_10 = float(saved_model["l1_validation_recall_at_10"])
        training_viewers = int(saved_model["training_viewers"])
        validation_viewers = int(saved_model["validation_viewers"])
    split = split_viewers(_load_interactions(matrix_path), seed=split_seed)
    test_scores = evaluate_on_test(model, split.train, split.test)
    return (
        model,
        model.regularization,
        recall_at_10,
        training_viewers,
        validation_viewers,
        test_scores,
    )


@st.cache_resource
def _train_and_select(
    path: str,
    candidates: tuple[float, ...],
    split_seed: int,
    evaluation_seed: int,
    input_likes: int,
    cutoff: int,
) -> tuple[ModelSelectionResult, tuple[int, int, int], FinalTestScores]:
    """Fit one L2 model per candidate λ and pick the best on validation viewers.

    Viewers are split into train / validation / test groups. Models are fitted
    on train viewers only, and compared on validation viewers (each shows a few
    likes, and the model must find the rest). Only after λ is chosen is the
    winning model scored once on the test viewers, as a final check. Returns
    the selection result, the (train, validation, test) viewer counts and the
    test scores. The seeds make the split and evaluation reproducible, and
    are part of the cache key.
    """
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
    test_scores = evaluate_on_test(
        result.best_model,
        split.train,
        split.test,
        input_likes=input_likes,
        k=cutoff,
        seed=evaluation_seed,
    )
    return result, counts, test_scores


def _movie_id_labeler(movies: tuple[MovieChoice, ...]) -> Callable[[int], str]:
    """Return a function that turns a movie ID into its display label.

    The multiselect stores integer IDs (stable, cheap to compare) and uses
    this to show readable text, e.g. "Title (1999, Director)".
    """

    def label(movie_id: int) -> str:
        return movies[movie_id].label

    return label


def _alphabetical_movie_ids(movies: tuple[MovieChoice, ...]) -> tuple[int, ...]:
    """Return all movie IDs ordered by title, ignoring letter case."""
    return tuple(
        sorted(
            range(len(movies)),
            key=lambda movie_id: movies[movie_id].title.casefold(),
        )
    )


def _lambda_label(value: float) -> str:
    return f"{value:g}"


def _comparison_rows(
    result: ModelSelectionResult, random_recall: float
) -> list[dict[str, str | float]]:
    """Build table/chart rows: one per candidate λ, plus two baselines.

    Popularity recommends the most-liked movies to everyone, ignoring the
    viewer; a model that does not beat it has not learned to personalise.
    Random is the floor: what ten random unseen movies would score.
    """
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
    rows.append({"Model": "Random baseline", "Recall@10": random_recall})
    return rows


def _show_test_scores(test_scores: FinalTestScores, random_recall: float) -> None:
    """Render the final check on test viewers, next to both baselines."""
    model_recall = test_scores.model_metrics.recall_at_k
    popularity_recall = test_scores.popularity_metrics.recall_at_k
    test_col, popularity_col, random_col = st.columns(3)
    _ = test_col.metric(
        "Test Recall@10 (final check)",
        f"{model_recall:.4f}",
        delta=f"{model_recall - popularity_recall:+.4f} vs popularity",
    )
    _ = popularity_col.metric("Popularity baseline, test", f"{popularity_recall:.4f}")
    _ = random_col.metric(
        "Random baseline",
        f"{random_recall:.4f}",
        help="Expected score of ten random unseen movies.",
    )
    _ = st.caption(
        f"Measured once on {test_scores.model_metrics.viewers:,} test viewers "
        "after λ was chosen; never used for training or selection."
    )


def _show_model_selection(
    movie_count: int,
    total_viewers: int,
) -> ModelSelectionResult | None:
    """Render the L2 section. Return the selection result, or None if not run yet.

    Results are kept in ``st.session_state`` so they survive the reruns that
    every later click triggers; otherwise the models would retrain each time.
    """
    _ = st.header("1. Train and compare models")
    _ = st.write(
        " ".join(
            (
                "Each candidate λ (lambda) sets how strongly large weights are",
                "penalised: a low λ fits the training viewers tightly, a high λ",
                "is more cautious. Each candidate model is trained, then tested",
                "on validation viewers who reveal a few likes. The model whose",
                "top ten contains the most of their hidden likes wins.",
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
                result, viewer_counts, test_scores = _train_and_select(
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
        st.session_state["selection_test_scores"] = test_scores

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

    test_scores = cast(
        FinalTestScores | None, st.session_state.get("selection_test_scores")
    )
    random_recall = random_recall_at_k(
        movie_count, input_likes=DEFAULT_INPUT_LIKES, k=DEFAULT_CUTOFF
    )
    if test_scores is not None:
        _show_test_scores(test_scores, random_recall)

    score_rows = _comparison_rows(result, random_recall)
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
                "Recall@10: of the likes a viewer did not reveal, the fraction",
                "that appear in their top ten recommendations (averaged over",
                "validation viewers; 1.0 is perfect).",
                "Random picks ten unseen movies: the floor any model must beat.",
                "Popularity recommends the same most-liked movies to everyone,",
                "so beating it shows the model personalises.",
            )
        )
    )
    return result


def _show_precomputed_l1(movie_count: int) -> tuple[ItemLinearModel, float] | None:
    """Render the L1 section. Return (model, λ), or None if it cannot be used.

    Fails soft with an on-page error instead of crashing the app, e.g. when
    the file is missing or was built for a different movie catalog.
    """
    _ = st.header("1. Use the precomputed L1 model")
    try:
        (
            model,
            regularization,
            recall_at_10,
            training_viewers,
            validation_viewers,
            test_scores,
        ) = _load_precomputed_l1(
            str(L1_WEIGHTS_PATH), str(MATRIX_PATH), DEFAULT_SPLIT_SEED
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
    _show_test_scores(
        test_scores,
        random_recall_at_k(
            movie_count, input_likes=DEFAULT_INPUT_LIKES, k=DEFAULT_CUTOFF
        ),
    )
    return model, regularization


def _show_implementation_details() -> None:
    with st.expander("About the models"):
        _ = st.markdown(
            """
Both models are an item-to-item table: for each movie, a weight for how much
liking every other movie should raise its score. A movie never predicts itself.

- **L2 (ridge)** penalises the *square* of each weight. Weights end up small
  but mostly non-zero. It is solved exactly and fitted live here.
- **L1 (lasso)** penalises the *absolute value* of each weight. Many weights
  become exactly zero, so each movie depends on only a few others. It is much
  slower to fit, so a precomputed model is loaded.

Both pick their λ using validation viewers. The separate test viewers are not
used to choose either model; they are scored once, afterwards, as a final
check. The data is synthetic.
"""
        )


def main() -> None:
    """Build the whole page. Streamlit calls this again on every interaction."""
    st.set_page_config(page_title="Movie favorites", page_icon="🎬")
    _ = st.title("Movie recommendations")
    _ = st.write("Select favorite movies in the picker below.")
    _ = st.caption(
        "The viewers are synthetic, generated from movie genres, directors and actors."
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
