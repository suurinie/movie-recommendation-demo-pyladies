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

Validate the catalog and print a JSON summary:

```bash
uv run movie-demo catalog --catalog imdb_top_1000.csv
```

All nine source columns are required: `Series_Title`, `Released_Year`, `Genre`,
`Director`, `Star1` through `Star4`, and `No_of_Votes`. Titles must be nonblank.
Other cells may be blank: missing years, directors, and votes become `None`,
and missing genres/cast become empty tuples. Missing metadata counts appear
in the summary. Malformed vote counts and CSV structure are rejected.
Malformed years (including the source's `PG` value) become unavailable and
are recorded in the summary and reported as warnings on stderr.

The loader assigns zero-based movie IDs in source-row order, hashes the exact
source bytes with SHA-256, and builds sorted genre/director/actor vocabularies.
Feature matrices have movies as rows and vocabulary entries as columns.
Duplicate genre/cast labels are removed; each nonempty row sums to one so
movies with more genres or actors do not automatically receive larger scores.
Empty feature rows remain zero. Movie IDs are tied to the source hash, not
intended to stay unchanged if the source is reordered.

The repository's software license does not establish redistribution rights
for the externally sourced CSV.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Implementation status

Catalog validation, metadata parsing, and sparse feature preparation are available.
Generation, training, evaluation, and the optional UI are not implemented yet.
See [PLAN.md](./PLAN.md) for the design and commit sequence.