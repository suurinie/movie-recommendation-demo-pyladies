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
Malformed years (such as a misplaced `PG` value) become unavailable and
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

## Synthetic viewer profiles

Reproducible hidden taste profiles are available as a Python iterator:

```python
from itertools import islice

from movie_demo.catalog import load_catalog
from movie_demo.profiles import iter_viewer_profiles

catalog = load_catalog("imdb_top_1000.csv")
profiles = iter_viewer_profiles(catalog, num_viewers=50_000, seed=42)
for profile in islice(profiles, 3):
    print(profile)
```

Each profile has 1-3 preferred genres, up to 2 directors, up to 4 actors, and
an individual target of 12-60 liked movies (sampled around 20). Preference
strengths range from 0.5 to 1.5. About 70% of selected director and actor
preferences are drawn from movies matching the viewer's preferred genres;
remaining choices add cross-genre variety. A profile's random stream is derived
from the seed and viewer ID. Use `start_viewer_id` to generate independent
chunks; concatenating chunks produces the same profiles as a single call.
This step generates profiles only; it does not yet sample viewer-movie
interactions.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Implementation status

Catalog validation, sparse feature preparation, and reproducible viewer profiles
are available. Interaction sampling, training, evaluation, and the optional UI
are not implemented yet.
See [PLAN.md](./PLAN.md) for the design and commit sequence.