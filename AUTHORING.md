# Author and developer guide

Technical documentation for the movie recommendation demo: data preparation
and synthetic interaction generation, model fitting and selection, Python APIs,
and development commands. For workshop participants, start with
[README.md](./README.md).

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
profiles = iter_viewer_profiles(catalog, seed=42)
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
The current working default is 500 viewers. Pass `num_viewers=50_000` for the
larger eventual dataset. Profile generation does not yet sample movie
interactions.

## Sparse liked-movie interactions

Sample profiles into a sparse movie-by-viewer interaction matrix:

```python
from movie_demo.catalog import load_catalog
from movie_demo.interactions import generate_interactions

catalog = load_catalog("imdb_top_1000.csv")
dataset = generate_interactions(catalog, seed=42)
print(dataset.matrix.shape, dataset.matrix.nnz)
```

The working default is 500 viewers; each profile contributes its target number
of unique binary likes. `dataset.matrix` is a SciPy CSC array with shape
`(movies, viewers)`. It contains only observed positive likes; missing entries
are unknown. Sampling combines a stable softmax of genre/director/actor affinity
(weights 3.0, 2.0, and 1.5) with a 10% exploration component. Sampler `v1`
weights exploration by `log1p(votes)`, while `v2` uses raw vote counts. If no
vote counts are available, exploration is uniform. Sampling is without
replacement, and each viewer's randomness depends only on the seed and viewer
ID, so changing `batch_size` does not change the matrix.

Generation is performed in batches, but the returned sparse matrix and profiles
are held in memory.
On the supplied 1,000-movie catalog with seed 42, the working default produced
10,110 likes (about 2.02% density); exact counts depend on the generated
profiles and seed.

To build the larger static matrix and save just that matrix plus its compact
movie index:

```bash
uv run movie-demo generate \
  --catalog imdb_top_1000.csv \
  --viewers 50000 \
  --seed 42 \
  --sampling-version v1 \
  --output data/synthetic/seed-42-50000
```

To generate the raw-vote experiment without overwriting the existing dataset:

```bash
uv run movie-demo generate \
  --catalog imdb_top_1000.csv \
  --viewers 50000 \
  --seed 42 \
  --sampling-version v2 \
  --output data/synthetic/seed-42-50000-v2
```

The command summary reports the sampling version used. `v1` remains the default
so existing generation commands continue to reproduce the original dataset.

The output directory contains `interactions.npz` and `movies.csv`. The matrix
has shape `(movies, viewers)`; row `i` refers to `movie_id=i` in the CSV.
`movies.csv` stores only `movie_id,title,director,year`. Viewer profiles,
genres, cast, and the original catalog are not copied into the output.
Generation refuses to overwrite an existing output directory. The 50,000-viewer
seed-42 dataset and its movie index are committed in `data/synthetic/`.
The input CSV remains local and Git-ignored.
The checked-in matrix contains 1,013,210 positive likes (2.02642% density),
and the compact index contains 1,000 rows.

## Interactive movie picker

Start the Streamlit app from the project root:

```bash
uv run streamlit run src/movie_demo/app.py
```

Enter part of a title (for example, `Totoro`) and select **Search** to show
matches; the app does not search on each keystroke. At most 25 matching titles
are shown at a time, so refine broad searches. Add a match to your favorites,
or expand the alphabetical catalog to browse titles, directors, and years.
Selected favorites can be removed from their chips.

Before recommendations are enabled, select candidate λ values and click
**Train and compare**. The app fits each analytical model on the 40,000
training viewers, chooses λ using 5,000 validation viewers, and leaves 5,000
test viewers untouched. It compares validation NDCG@10 and Recall@10 with a
popularity baseline, then uses the selected training-only model for favorites.
The comparison shows one popularity-baseline result alongside each candidate,
with separate charts for NDCG@10 and Recall@10 and KPI deltas versus popularity.
The **Under the hood** section explains the tools and shows the actual Python
implementations for splitting, fitting, and model selection. This is an
analytical linear-algebra solve, not an epoch-based training animation.
Recommendation scores are rankings, not probabilities.

## Viewer-level dataset splits

Use a seeded NumPy permutation to hold out entire viewers before fitting. This
keeps every interaction from a validation or test viewer out of training, which
matches the task of recommending for a new viewer. Scikit-learn is not needed.

```python
from scipy.sparse import load_npz

from movie_demo.splitting import split_viewers

matrix = load_npz("data/synthetic/seed-42-50000/interactions.npz").tocsc()
split = split_viewers(matrix, seed=42)
print(split.train.shape, split.validation.shape, split.test.shape)
```

The default partition is 80% training, 10% validation, and 10% test. Use the
training matrix to fit the model and validation viewers to choose settings; keep
test viewers untouched until final evaluation. Each returned group includes its
sparse matrix and sorted original viewer IDs, so viewer identity remains
traceable. Splitting preserves movie rows and interaction values. For held-out
viewers, model selection divides their likes into known inputs and held-out
targets for ranking evaluation.

## Zero-diagonal item model

The predictor `X` uses rows as target movies and columns as input movies. With
`H` shaped `(movies, viewers)`, train it by minimizing:

```text
||H - XH||_F^2 + lambda * ||X||_F^2
subject to diag(X) = 0
```

Each movie row is an independent ridge regression against all the other movie
rows. Let `G = H H.T` and `P = inverse(G + lambda I)`. The constrained
analytical solution is `X[i, j] = -P[i, j] / P[i, i]` for `i != j`, with a
zero diagonal. The implementation uses a positive-definite linear solve, not
an explicit matrix inverse. The solve is over movies, not viewers.

```python
from scipy.sparse import load_npz

from movie_demo.model import fit_analytical, fit_sgd
from movie_demo.splitting import split_viewers

matrix = load_npz("data/synthetic/seed-42-50000/interactions.npz").tocsc()
split = split_viewers(matrix, seed=42)
model = fit_analytical(split.train, regularization=100.0)
sgd_model = fit_sgd(split.train, regularization=100.0, max_iter=1_000, seed=42)

# One validation viewer's scores; selected movies are masked to -inf.
liked = split.validation[:, 0].toarray().ravel()
scores = model.predict(liked)
```

`fit_sgd` uses scikit-learn's `SGDRegressor` once per target movie. Each
regressor receives the training viewers as samples and all movie features
except its target movie, enforcing the zero diagonal structurally. It uses
`fit_intercept=False`, squared-error loss, and L2 penalty. Setting
`alpha=lambda / number_of_training_viewers` matches the regularization scale of
the stated sum-of-squares objective. `max_iter`, `tolerance`, and `seed` control
its familiar sklearn training API. The analytical solve is exact and much
faster for this 1,000-movie catalog; SGD is included to teach iterative fitting
and provides a comparison.

Both implementations fit only training viewers; validation/test viewer columns
remain held out. The sparse binary interactions must be cast to floating point
before forming `H H.T`; integer matrix multiplication can overflow even if the
result is cast to float afterward.

## Select regularization on validation viewers

Choose `lambda` by ranking held-out validation likes, not by training loss.
For each validation viewer, five likes are selected deterministically as
recommendation inputs; all their other known likes are targets. The same
input/target split is reused for every candidate. Already-liked movies are
excluded from ranking. The default candidates are `1, 10, 100, 1000, 10000`,
and `100000`; NDCG@10 is the primary selection metric (Recall@10 breaks exact ties).
A popularity baseline uses only training-viewer like counts. Test viewers are
not used for model selection.

Run the selection on the checked-in dataset:

```bash
uv run movie-demo select-lambda \
  --matrix data/synthetic/seed-42-50000/interactions.npz
```

The JSON output contains the selected lambda, per-candidate validation
Recall@10/NDCG@10, and the popularity baseline. The validation input selection
and viewer split have independent seeds (`--evaluation-seed` and
`--split-seed`). Use `--lambdas 10 100 1000` to provide a custom positive
candidate grid.

The Python API exposes the same workflow:

```python
from scipy.sparse import load_npz

from movie_demo.model_selection import select_regularization
from movie_demo.splitting import split_viewers

matrix = load_npz("data/synthetic/seed-42-50000/interactions.npz").tocsc()
split = split_viewers(matrix, seed=42)
selection = select_regularization(split.train, split.validation)
print(selection.best_regularization, selection.scores)
```

Recall@k divides retrieved target likes by all target likes for that viewer;
NDCG@k uses binary relevance and the ideal ranking for that viewer's target
set. Metrics are averaged across validation viewers. After choosing lambda,
the held-out test group can be used once for final reporting; it must not
influence candidate values or selection.

With the default split and evaluation seeds on the committed matrix, the
selected value is `lambda=10000` (NDCG@10 0.04151, Recall@10 0.02512). The
training-only popularity baseline scores 0.02827 and 0.01766, respectively.
These are validation results for this synthetic dataset, not estimates of
real-world recommendation quality.

On the committed data, the 40,000-viewer training split with `lambda=100`
produced a finite zero-diagonal analytical model in about 0.13 seconds. The
scikit-learn fit with `max_iter=100` took about 30.2 seconds on the same
environment and reached a training objective about 1.0002 times the analytical
objective. These timings are a local reference, not a hardware-independent
performance promise.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Implementation status

Catalog validation, reproducible viewer profiles, sparse interaction sampling,
viewer-level splits, analytical/SGD model fitting, and validation-based
regularization selection, and the interactive movie picker are available.
Final test reporting is not implemented yet.
See [PLAN.md](./PLAN.md) for the design and commit sequence.