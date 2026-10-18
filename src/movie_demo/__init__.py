import argparse
import json
import sys
from pathlib import Path

from movie_demo.catalog import CatalogError, load_catalog
from movie_demo.dataset_io import save_generated_dataset
from movie_demo.interactions import generate_interactions
from movie_demo.profiles import DEFAULT_SEED, DEFAULT_VIEWERS


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
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        catalog = load_catalog(args.catalog)
        if args.command == "catalog":
            for issue in catalog.issues:
                print(f"Warning: {issue}", file=sys.stderr)
            print(json.dumps(catalog.summary(), indent=2))
            return
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
