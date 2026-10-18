# Movie recommendation demo plan

## Goal

Build a Python demo using [imdb_top_1000.csv](./imdb_top_1000.csv):

1. Generate a sparse dataset of at least 10,000 viewers, targeting 50,000.
2. Give viewers preferences for genres, directors, actors, and optionally eras.
3. Train a simple collaborative filtering model with a zero-diagonal item-to-item predictor.
4. Recommend unseen movies when a new viewer supplies movies they like.

The source contains approximately 1,000 movies and includes genres, director, four actors, release year, rating, and vote count.

## Tools

| Tool | Purpose |
|---|---|
| pandas | Load and clean movie metadata |
| numpy | Generate viewers, compute scores, and rank recommendations |
| scipy.sparse | Store sparse interactions |
| scipy.linalg | Solve the model's linear system |
| pytest | Validate constraints, masking, and reproducibility |
| streamlit | Optional interactive movie-selection demo |

Use ordinary Python modules. Keep the optional Streamlit presentation layer separate from generation, training, and evaluation. No GPU or neural-network framework is required.

## 1. Generate synthetic preferences and interactions

Assign each viewer a hidden taste profile:

- A few preferred genres with varying strengths.
- Preferred directors and actors, sampled partly from movies in those genres.
- Optional era preferences.
- An individual activity level.
- Random variation so viewers are not identical genre buckets.

Construct a hidden viewer-movie affinity:

```text
score(viewer, movie)
    = genre_weight * genre_match
    + director_weight * director_match
    + actor_weight * actor_match
    + era_weight * era_match
    + noise
```

Sample positive interactions without replacement, favoring high-affinity movies while allowing exploration. Include a modest popularity/exposure effect.

### Default scale

- 50,000 viewers.
- Approximately 1,000 movies.
- Around 20 positive interactions per viewer on average.
- Approximately 1 million nonzero entries, or 2% density.

Generate viewers in batches rather than retaining a dense 50,000-by-1,000 affinity matrix.

### Interaction semantics

A value of 1 means "watched and liked," not merely "watched." Missing entries are unknown rather than explicit dislikes. This matches the intended input of movies a person likes.

Keep hidden profiles separately for inspecting the synthetic world. The collaborative model receives only interactions, not movie metadata or hidden preferences.

## 2. Train the constrained linear model

Use the orientation from the requested objective:

- H: movies by viewers.
- X: movies by movies.
- XH: predicted movie preferences.

The proposed regularized objective is:

```text
minimize_X ||H - XH||_F^2 + lambda * ||X||_F^2
subject to diag(X) = 0
```

Squaring the Frobenius norm does not change the unregularized minimizer. Ridge regularization does change the objective; it is recommended for numerical stability and generalization. Preserve this distinction when presenting the demo.

Let:

```text
P = inverse(H H^T + lambda I)
```

Then the solution in this orientation is:

```text
X[i, j] = -P[i, j] / P[i, i]   for i != j
X[i, i] = 0
```

Compute P with a positive-definite linear solve against the identity matrix. The system is approximately 1,000 by 1,000, not 50,000 by 50,000.

For a new viewer's binary like-vector h:

```text
recommendation_scores = X h
```

Mask already-selected movies and return the highest-scoring unseen titles. Scores are ranking signals, not calibrated probabilities or guaranteed favorites.

## 3. Evaluate the intended demo behavior

Split viewers into training, validation, and test groups:

- Fit model weights using training viewers only.
- Choose lambda using validation viewers.
- Report final results using test viewers.

For validation and test viewers, reveal a subset of their liked movies and hold out the remainder. Evaluate with 1, 3, 5, and 10 input likes, ensuring enough interactions remain for evaluation.

### Metrics

- Recall@10.
- NDCG@10.
- Catalog coverage.
- Comparison against a popularity baseline computed from training interactions.

Synthetic-data performance demonstrates recovery of the preferences used to generate the data. It does not establish real-world recommendation quality.

### Focused validation

- Model diagonal is zero.
- Selected movies never appear in recommendations.
- Fixed seeds reproduce generation and evaluation.
- Held-out interactions never enter model fitting.
- Stored interactions are binary and genuinely sparse.
- Matrix dimensions and prediction orientation are correct.
- Metrics and recommendation output shapes match their definitions.

## 4. Interactive demo flow

1. Show the movie catalog and synthetic-data statistics.
2. Inspect sample viewers and their hidden preferences.
3. Train the model and show evaluation against the popularity baseline.
4. Select liked movies using a multiselect.
5. Display the top 10 unseen recommendations with genre, director, and cast metadata.
6. Optionally show which selected movies contribute most to each recommendation score.

## 5. Implementation deliverables

Keep the implementation small:

- One pipeline module for loading, generation, training, evaluation, and recommendation.
- One optional Streamlit app using the pipeline functions.
- Focused tests.
- Dependency configuration.
- Short setup and run instructions.

Persist:

- Sparse interaction data.
- Model weights.
- Stable movie-ID mapping.
- Random seed and generation configuration.
- Training configuration, including lambda.
- Evaluation results.

## Implementation order

1. Inspect and clean metadata; establish stable movie IDs.
2. Implement seeded, batched synthetic generation for 50,000 viewers.
3. Implement viewer splits, the popularity baseline, and ranking metrics.
4. Fit and validate the zero-diagonal linear model.
5. Run the reproducible command-line pipeline and measure results.
6. Add the optional interactive UI.

The first milestone is a reproducible command-line pipeline with 50,000 viewers and measured evaluation results. Add the UI after the underlying pipeline works.

## Commit-by-commit delivery

Use uv for dependency management and commit uv.lock. Use Python 3.12,
standard-library argparse for the CLI, and pytest and Ruff for validation.
Add Streamlit only when implementing the UI.

### 1. chore: bootstrap uv project

- Add the package skeleton and movie-demo CLI help screen.
- Commit dependency configuration, lockfile, setup instructions, and this plan.
- Ignore the source CSV, environments, caches, generated data, and model artifacts.
- Verify CLI help, tests, lint, and formatting.

### 2. feat: load and validate movie catalog

- Validate required columns and reject empty catalogs.
- Parse genres and deduplicate cast from the four actor columns.
- Handle missing optional metadata explicitly.
- Assign contiguous movie IDs in source-row order and record the source SHA-256.
- Build deterministic feature vocabularies and normalized genre/actor matches.
- Test valid, missing, and malformed metadata.

### 3. feat: generate reproducible viewer taste profiles

- Use 500 viewers as the working default; retain seed 42.
- Scale up to 50,000 only after the working-size pipeline is validated.
- Choose 1-3 preferred genres, up to 2 directors, and up to 4 actors per viewer.
- Draw 70% of director and actor choices from movies matching preferred genres.
- Store explicit preference strengths sampled from 0.5 to 1.5.
- Sample activity targets from a normal distribution centered at 20, clipped to 12-60.
- Derive per-viewer random generators from the master seed and viewer ID.
- Test determinism, preference validity, and activity bounds.

### 4. feat: sample sparse liked-movie interactions

- Use 500 viewers as the working default while validating the generator.
- Start with affinity weights of 3.0 for genres, 2.0 for directors, and 1.5 for actors.
- Use stable softmax probabilities with 10% popularity-weighted exploration.
- Derive modest popularity weights from log-transformed vote counts.
- Sample without replacement and process viewers in batches.
- Initially omit era preferences to keep the generator explainable.
- Produce binary H in CSC format with shape movies by viewers.
- Retain sparse interactions and process viewers in batches; do not allocate a
  dense viewer-by-movie matrix.
- Test uniqueness, shape, exact activity, taste alignment, and independence
  from batch size.
- Treat these parameters as starting values, not validated results.

### 5. feat: save generated datasets and expose generation CLI

Target command for the committed 50,000-viewer matrix:

```bash
uv run movie-demo generate \
  --catalog imdb_top_1000.csv \
  --viewers 50000 \
  --seed 42 \
  --output data/synthetic/seed-42-50000
```

- Save the sparse movie-by-viewer NPZ matrix and a CSV containing only
  `movie_id,title,director,year`.
- Do not persist hidden profiles or other catalog metadata in this output.
- Refuse to overwrite an existing output directory.
- Test save/load round-trips, CLI execution, invalid inputs, and write errors.
- Commit the seed-42 50,000-viewer matrix and index; keep the input CSV ignored.

### 6. test: validate synthetic taste signal at demo scale

- Validate the committed 50,000-viewer matrix using the real catalog.
- Report shape, nonzeros, density, activity distribution, coverage, and runtime.
- Report popularity concentration and preferred-genre enrichment.
- Use a controlled fixture to test that stronger preferences increase matching selections.
- Document measured results and the reproduction command; do not commit extra generated data.

### Subsequent commits

7. Add per-held-out-viewer input/target likes, ranking metrics, and the
   training-only popularity baseline.
8. Add the regularized zero-diagonal model and numerical tests.
9. Select regularization using validation viewers and report test metrics.
10. Add the interactive Streamlit demo.

## Viewer-level split implementation

- Use NumPy's seeded random generator; do not add scikit-learn for splitting.
- Partition viewer IDs 80/10/10 by default, keeping each viewer's full column
  in one partition.
- Return each partition's CSC matrix and sorted original viewer IDs.
- Fit only on training viewers. Later split validation/test viewer columns into
  input likes and target likes for recommendation evaluation.
- Verify disjoint, exhaustive, reproducible partitions and exact matrix-column
  correspondence.
