import csv

import pytest

from movie_demo.catalog import REQUIRED_COLUMNS


@pytest.fixture
def write_catalog(tmp_path):
    def write(rows, columns=REQUIRED_COLUMNS):
        path = tmp_path / "movies.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    return write


@pytest.fixture
def movie_row():
    return {
        "Series_Title": "Example",
        "Released_Year": "2001",
        "Genre": "Drama, Comedy",
        "Director": "Director A",
        "Star1": "Actor A",
        "Star2": "Actor B",
        "Star3": "Actor A",
        "Star4": "",
        "No_of_Votes": "100",
    }
