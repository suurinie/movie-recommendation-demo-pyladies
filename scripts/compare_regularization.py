import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csc_array, load_npz

from movie_demo.demo import MovieChoice, load_movie_index
from movie_demo.model_selection import DEFAULT_LAMBDAS, select_regularization
from movie_demo.splitting import DEFAULT_SPLIT_SEED, split_viewers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "seed-42-50000-v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "slides" / "assets" / "l1-l2-comparison.png"
DEFAULT_DETAILS_OUTPUT = PROJECT_ROOT / "slides" / "assets" / "l1-l2-details.png"
DEFAULT_WEIGHTS_OUTPUT = PROJECT_ROOT / "slides" / "assets" / "l1-l2-weights.npz"
FULL_DATA_OUTPUT = PROJECT_ROOT / "slides" / "assets" / "l1-l2-comparison-full.png"
FULL_DATA_DETAILS_OUTPUT = (
    PROJECT_ROOT / "slides" / "assets" / "l1-l2-signed-links-full.png"
)
FULL_DATA_WEIGHTS_OUTPUT = PROJECT_ROOT / "slides" / "assets" / "l1-l2-weights-full.npz"
DEFAULT_L1_LAMBDAS = (1.0, 10.0, 100.0)


def _sample_viewers(
    interactions: csc_array,
    count: int,
    seed: int,
) -> csc_array:
    if count < 1 or count > interactions.shape[1]:
        raise ValueError(f"Viewer sample size must be in [1, {interactions.shape[1]}].")
    rng = np.random.default_rng(seed)
    columns = np.sort(rng.choice(interactions.shape[1], size=count, replace=False))
    return csc_array(interactions[:, columns])


def _off_diagonal_values(weights: np.ndarray) -> np.ndarray:
    off_diagonal = ~np.eye(weights.shape[0], dtype=bool)
    return weights[off_diagonal]


def _zero_fraction(weights: np.ndarray) -> float:
    values = _off_diagonal_values(weights)
    return float(np.count_nonzero(values == 0) / values.size)


def _default_outputs(full_data: bool) -> tuple[Path, Path, Path]:
    if full_data:
        return FULL_DATA_OUTPUT, FULL_DATA_DETAILS_OUTPUT, FULL_DATA_WEIGHTS_OUTPUT
    return DEFAULT_OUTPUT, DEFAULT_DETAILS_OUTPUT, DEFAULT_WEIGHTS_OUTPUT


def _plot_matrices(
    l2_weights: np.ndarray,
    l1_weights: np.ndarray,
    movies: tuple[MovieChoice, ...],
    training_popularity: np.ndarray,
    l2_lambda: float,
    l1_lambda: float,
    l2_recall: float,
    l1_recall: float,
    training_viewers: int,
    validation_viewers: int,
    output: Path,
) -> None:
    movie_order = np.argsort(-training_popularity, kind="stable")
    ordered_l2 = l2_weights[np.ix_(movie_order, movie_order)]
    ordered_l1 = l1_weights[np.ix_(movie_order, movie_order)]

    off_diagonal = ~np.eye(l2_weights.shape[0], dtype=bool)
    l2_values = l2_weights[off_diagonal]
    l1_values = l1_weights[off_diagonal]
    l2_magnitudes = np.abs(l2_values[l2_values != 0])
    l1_magnitudes = np.abs(l1_values[l1_values != 0])
    all_magnitudes = np.concatenate((l2_magnitudes, l1_magnitudes))
    if not len(all_magnitudes):
        raise ValueError("Both selected models have only zero connections.")

    color_map = plt.get_cmap("coolwarm").copy()
    color_map.set_bad("white")
    color_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    figure, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    figure.suptitle(
        "L2 vs L1: learned movie-to-movie connections\n"
        f"Same {training_viewers:,} training and {validation_viewers:,} validation "
        "viewers; each lambda selected by Recall@10",
        fontsize=15,
    )

    heatmaps = []
    for axis, weights, ordered_weights, title in (
        (
            axes[0],
            l2_weights,
            ordered_l2,
            f"L2  |  lambda={l2_lambda:g}  |  Recall@10={l2_recall:.3f}",
        ),
        (
            axes[1],
            l1_weights,
            ordered_l1,
            f"L1  |  lambda={l1_lambda:g}  |  Recall@10={l1_recall:.3f}",
        ),
    ):
        nonzero_magnitudes = np.abs(weights[weights != 0])
        scale = (
            float(np.quantile(nonzero_magnitudes, 0.995))
            if len(nonzero_magnitudes)
            else 1.0
        )
        scale = max(scale, np.finfo(np.float64).eps)
        image = axis.imshow(
            np.ma.masked_where(ordered_weights == 0, ordered_weights / scale),
            cmap=color_map,
            norm=color_norm,
            interpolation="nearest",
            aspect="equal",
        )
        heatmaps.append(image)
        axis.set_title(f"{title}  |  off-diagonal zeros {_zero_fraction(weights):.1%}")
        axis.set_xlabel("Input movies, ordered by training likes")
        axis.set_ylabel("Target movies, same order")
        axis.set_xticks([])
        axis.set_yticks([])

    figure.colorbar(
        heatmaps[0],
        ax=axes,
        orientation="horizontal",
        label=(
            "Relative signed strength (each matrix scaled to its own 99.5th percentile)"
        ),
        shrink=0.7,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _plot_details(
    l2_weights: np.ndarray,
    l1_weights: np.ndarray,
    movies: tuple[MovieChoice, ...],
    output: Path,
    preferred_target: str = "Tonari no Totoro",
) -> None:
    off_diagonal = ~np.eye(l2_weights.shape[0], dtype=bool)
    l2_values = l2_weights[off_diagonal]
    l1_values = l1_weights[off_diagonal]
    l2_magnitudes = np.abs(l2_values[l2_values != 0])
    l1_magnitudes = np.abs(l1_values[l1_values != 0])
    all_magnitudes = np.concatenate((l2_magnitudes, l1_magnitudes))
    if not len(all_magnitudes):
        raise ValueError("Both selected models have only zero connections.")

    figure, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)
    figure.suptitle("A closer look at connection strengths", fontsize=16)
    histogram_axis = axes[0]
    nonzero_magnitudes = all_magnitudes[all_magnitudes > 0]
    lower = float(nonzero_magnitudes.min())
    upper = float(nonzero_magnitudes.max())
    if lower == upper:
        bins = np.geomspace(lower / 1.1, upper * 1.1, 20)
    else:
        bins = np.geomspace(lower, upper, 36)
    if len(l2_magnitudes):
        histogram_axis.hist(
            l2_magnitudes,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color="#3274A1",
            label=f"L2 nonzero values ({1 - _zero_fraction(l2_weights):.1%})",
        )
    if len(l1_magnitudes):
        histogram_axis.hist(
            l1_magnitudes,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color="#E1812C",
            label=f"L1 nonzero values ({1 - _zero_fraction(l1_weights):.1%})",
        )
    histogram_axis.set_xscale("log")
    histogram_axis.set_yscale("log")
    histogram_axis.set_title("Connection magnitudes")
    histogram_axis.set_xlabel("Absolute weight, log scale")
    histogram_axis.set_ylabel("Number of connections")
    histogram_axis.legend(frameon=False)

    connection_count = 5
    positive_counts = np.count_nonzero(l1_weights > 0, axis=1)
    negative_counts = np.count_nonzero(l1_weights < 0, axis=1)
    preferred_ids = [
        movie.movie_id
        for movie in movies
        if movie.title.casefold() == preferred_target.casefold()
    ]
    target_id = preferred_ids[0] if preferred_ids else -1
    if (
        target_id < 0
        or not positive_counts[target_id]
        or not negative_counts[target_id]
    ):
        candidate_rows = np.flatnonzero((positive_counts > 0) & (negative_counts > 0))
        if len(candidate_rows):
            negative_strength = np.abs(np.minimum(l1_weights[candidate_rows], 0)).sum(
                axis=1
            )
            target_id = int(candidate_rows[np.argmax(negative_strength)])
        else:
            target_id = int(np.argmax(negative_counts))

    target_l1_weights = l1_weights[target_id]
    positive_ids = np.flatnonzero(target_l1_weights > 0)
    positive_ids = positive_ids[
        np.argsort(-target_l1_weights[positive_ids])[:connection_count]
    ]
    negative_ids = np.flatnonzero(target_l1_weights < 0)
    negative_ids = negative_ids[
        np.argsort(target_l1_weights[negative_ids])[:connection_count]
    ]
    connection_ids = np.concatenate((positive_ids, negative_ids))

    connection_axis = axes[1]
    y_positions = np.arange(len(connection_ids))
    bar_height = 0.38
    connection_axis.barh(
        y_positions - bar_height / 2,
        l2_weights[target_id, connection_ids]
        / max(float(np.max(np.abs(l2_weights[target_id]))), np.finfo(float).eps),
        height=bar_height,
        color="#3274A1",
        label="L2",
    )
    connection_axis.barh(
        y_positions + bar_height / 2,
        l1_weights[target_id, connection_ids]
        / max(float(np.max(np.abs(l1_weights[target_id]))), np.finfo(float).eps),
        height=bar_height,
        color="#E1812C",
        label="L1",
    )
    labels = [textwrap.fill(movies[movie_id].title, 24) for movie_id in connection_ids]
    connection_axis.set_yticks(y_positions, labels)
    connection_axis.invert_yaxis()
    connection_axis.axvline(0, color="#555555", linewidth=0.8)
    connection_axis.set_title(
        f"Strongest positive and negative L1 links into {movies[target_id].title}"
    )
    connection_axis.set_xlabel("Weight relative to each model's row maximum")
    connection_axis.legend(frameon=False)
    connection_axis.grid(axis="x", alpha=0.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an L2/L1 movie-weight comparison figure."
    )
    parser.add_argument("--training-viewers", type=int, default=5_000)
    parser.add_argument("--validation-viewers", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--l1-lambdas", type=float, nargs="+", default=DEFAULT_L1_LAMBDAS
    )
    parser.add_argument("--l2-lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help=(
            "Use all training and validation viewers; save to separate *-full "
            "files so preview assets are preserved."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--weights-output", type=Path)
    parser.add_argument("--detail-movie", default="Tonari no Totoro")
    args = parser.parse_args()
    default_output, default_details_output, default_weights_output = _default_outputs(
        args.full_data
    )
    output = args.output or default_output
    details_output = args.details_output or default_details_output
    weights_output = args.weights_output or default_weights_output

    interactions = csc_array(load_npz(DATA_DIR / "interactions.npz"))
    movies = load_movie_index(DATA_DIR / "movies.csv")
    if interactions.shape[0] != len(movies):
        raise ValueError("Movie index and interaction matrix dimensions do not match.")

    split = split_viewers(interactions, seed=args.seed)
    if args.full_data:
        training = split.train
        validation = split.validation
    else:
        training = _sample_viewers(split.train, args.training_viewers, args.seed + 1)
        validation = _sample_viewers(
            split.validation, args.validation_viewers, args.seed + 2
        )

    l2_result = select_regularization(
        training,
        validation,
        candidates=tuple(args.l2_lambdas),
        seed=args.seed,
        solver="l2",
    )
    l1_result = select_regularization(
        training,
        validation,
        candidates=tuple(args.l1_lambdas),
        seed=args.seed,
        solver="l1",
    )

    l2_recall = next(
        score.metrics.recall_at_k
        for score in l2_result.scores
        if score.regularization == l2_result.best_regularization
    )
    l1_recall = next(
        score.metrics.recall_at_k
        for score in l1_result.scores
        if score.regularization == l1_result.best_regularization
    )
    training_popularity = np.asarray(training.sum(axis=1)).ravel()
    _plot_matrices(
        l2_result.best_model.weights,
        l1_result.best_model.weights,
        movies,
        training_popularity,
        l2_result.best_regularization,
        l1_result.best_regularization,
        l2_recall,
        l1_recall,
        training.shape[1],
        validation.shape[1],
        output,
    )
    _plot_details(
        l2_result.best_model.weights,
        l1_result.best_model.weights,
        movies,
        details_output,
        preferred_target=args.detail_movie,
    )
    weights_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        weights_output,
        movie_ids=np.arange(len(movies), dtype=np.int32),
        l2_weights=l2_result.best_model.weights,
        l1_weights=l1_result.best_model.weights,
        l2_lambda=np.float64(l2_result.best_regularization),
        l1_lambda=np.float64(l1_result.best_regularization),
        l2_validation_recall_at_10=np.float64(l2_recall),
        l1_validation_recall_at_10=np.float64(l1_recall),
        training_viewers=np.int32(training.shape[1]),
        validation_viewers=np.int32(validation.shape[1]),
    )
    print(f"L2 lambda: {l2_result.best_regularization:g}")
    print(f"L2 validation Recall@10: {l2_recall:.5f}")
    print(f"L1 lambda: {l1_result.best_regularization:g}")
    print(f"L1 validation Recall@10: {l1_recall:.5f}")
    print(f"L2 off-diagonal zeros: {_zero_fraction(l2_result.best_model.weights):.1%}")
    print(f"L1 off-diagonal zeros: {_zero_fraction(l1_result.best_model.weights):.1%}")
    print(f"Saved figure to {output}")
    print(f"Saved detail figure to {details_output}")
    print(f"Saved matrices and metadata to {weights_output}")


if __name__ == "__main__":
    main()
