import argparse
import json
import sys
from pathlib import Path

from movie_demo.catalog import CatalogError, load_catalog


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
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        catalog = load_catalog(args.catalog)
    except (OSError, CatalogError) as exc:
        parser.error(str(exc))
    for issue in catalog.issues:
        print(f"Warning: {issue}", file=sys.stderr)
    print(json.dumps(catalog.summary(), indent=2))
