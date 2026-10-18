# Movie recommendation demo

A Python demo for generating sparse movie preferences for 50,000 synthetic
viewers and training a zero-diagonal linear collaborative filtering model.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone git@github.com:suurinie/movie-recommendation-demo-pyladies.git
cd movie-recommendation-demo-pyladies
uv sync --locked
uv run movie-demo --help
```

The project uses Python 3.12 and commits its dependency lockfile.

## Dataset

Place `imdb_top_1000.csv` in the project root. The expected catalog includes
movie titles, genres, directors, cast, release years, and vote counts.
The CSV is intentionally ignored by Git until its source and redistribution
terms are confirmed. Generated data and model artifacts are also ignored.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Implementation status

This first commit provides project setup and a CLI help screen only.
Generation, training, evaluation, and the optional UI are not implemented yet.
See [PLAN.md](./PLAN.md) for the design and commit sequence.