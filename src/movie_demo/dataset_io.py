import csv
from pathlib import Path

from scipy.sparse import save_npz

from movie_demo.catalog import Catalog
from movie_demo.interactions import InteractionDataset


def save_generated_dataset(
    catalog: Catalog,
    dataset: InteractionDataset,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save sparse interactions and the row-indexed compact movie mapping."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if dataset.matrix.shape[0] != len(catalog.movies):
        raise ValueError("Interaction matrix rows must match catalog movie IDs.")
    if dataset.matrix.shape[1] != len(dataset.profiles):
        raise ValueError("Interaction matrix columns must match viewer profiles.")
    if [movie.movie_id for movie in catalog.movies] != list(range(len(catalog.movies))):
        raise ValueError("Catalog movie IDs must be contiguous and row-aligned.")

    output.mkdir(parents=True)
    matrix_path = output / "interactions.npz"
    movies_path = output / "movies.csv"
    save_npz(matrix_path, dataset.matrix, compressed=True)
    with movies_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("movie_id", "title", "director", "year"))
        writer.writerows(
            (
                movie.movie_id,
                movie.title,
                movie.director or "",
                movie.year if movie.year is not None else "",
            )
            for movie in catalog.movies
        )
    return matrix_path, movies_path
