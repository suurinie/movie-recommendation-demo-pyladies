import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="movie-demo",
        description="Synthetic movie preferences and collaborative filtering demo.",
    )
    parser.parse_args()
    parser.print_help()
