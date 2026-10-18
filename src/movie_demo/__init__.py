import argparse
import json
import sys
from pathlib import Path

from scipy.sparse import load_npz

from movie_demo.catalog import CatalogError, load_catalog
from movie_demo.dataset_io import save_generated_dataset
from movie_demo.interactions import generate_interactions
from movie_demo.model_selection import (
    DEFAULT_CUTOFF,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_INPUT_LIKES,
    DEFAULT_LAMBDAS,
    select_regularization,
)
from movie_demo.profiles import DEFAULT_SEED, DEFAULT_VIEWERS
from movie_demo.splitting import DEFAULT_SPLIT_SEED, split_viewers


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="movie-demo",
        description="Synthetic movie preferences and collaborative filtering demo.",
    )
    commands = parser.add_subparsers(dest="command")
    catalog_parser = commands.add_parser(
        "catalog", help="Validate and inspect a catalog"
    )
    catalog_parser.add_argument("--catalog", type=Path, required=True)
    generate_parser = commands.add_parser(
        "generate", help="Generate and save a sparse synthetic interaction matrix"
    )
    generate_parser.add_argument("--catalog", type=Path, required=True)
    generate_parser.add_argument(
        "--viewers", type=int, default=DEFAULT_VIEWERS, help="Number of viewers"
    )
    generate_parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed"
    )
    generate_parser.add_argument(
        "--batch-size", type=int, default=128, help="Viewers processed per batch"
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic"),
        help="New output directory (must not already exist)",
    )
    selection_parser = commands.add_parser(
        "select-lambda",
        help="Choose regularization using held-out validation likes",
    )
    selection_parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("data/synthetic/seed-42-50000/interactions.npz"),
    )
    selection_parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    selection_parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=DEFAULT_EVALUATION_SEED,
        help="Seed for selecting validation input likes",
    )
    selection_parser.add_argument(
        "--input-likes", type=int, default=DEFAULT_INPUT_LIKES
    )
    selection_parser.add_argument("--k", type=int, default=DEFAULT_CUTOFF)
    selection_parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="Positive ridge-regularization candidates",
    )
    selection_parser.add_argument(
        "--metric",
        choices=("ndcg_at_k", "recall_at_k"),
        default="ndcg_at_k",
    )
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "catalog":
            catalog = load_catalog(args.catalog)
            for issue in catalog.issues:
                print(f"Warning: {issue}", file=sys.stderr)
            print(json.dumps(catalog.summary(), indent=2))
            return
        if args.command == "select-lambda":
            matrix = load_npz(args.matrix).tocsc()
            split = split_viewers(matrix, seed=args.split_seed)
            result = select_regularization(
                split.train,
                split.validation,
                candidates=tuple(args.lambdas),
                input_likes=args.input_likes,
                k=args.k,
                seed=args.evaluation_seed,
                metric=args.metric,
            )
            print(
                json.dumps(
                    {
                        "training_viewers": split.train.shape[1],
                        "validation_viewers": split.validation.shape[1],
                        "test_viewers": split.test.shape[1],
                        "best_lambda": result.best_regularization,
                        "selection_metric": result.metric,
                        "validation_scores": [
                            {
                                "lambda": score.regularization,
                                "recall_at_k": score.metrics.recall_at_k,
                                "ndcg_at_k": score.metrics.ndcg_at_k,
                            }
                            for score in result.scores
                        ],
                        "popularity_baseline": {
                            "recall_at_k": result.popularity_metrics.recall_at_k,
                            "ndcg_at_k": result.popularity_metrics.ndcg_at_k,
                        },
                    },
                    indent=2,
                )
            )
            return
        catalog = load_catalog(args.catalog)
        if args.output.exists():
            raise FileExistsError(f"Output directory already exists: {args.output}")
        dataset = generate_interactions(
            catalog,
            num_viewers=args.viewers,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        matrix_path, movies_path = save_generated_dataset(catalog, dataset, args.output)
    except (OSError, ValueError, CatalogError) as exc:
        parser.error(str(exc))
    for issue in catalog.issues:
        print(f"Warning: {issue}", file=sys.stderr)
    print(
        json.dumps(
            {
                "movies": len(catalog.movies),
                "viewers": len(dataset.profiles),
                "shape": list(dataset.matrix.shape),
                "likes": dataset.matrix.nnz,
                "density": dataset.matrix.nnz
                / (dataset.matrix.shape[0] * dataset.matrix.shape[1]),
                "matrix": str(matrix_path),
                "movie_index": str(movies_path),
            },
            indent=2,
        )
    )
