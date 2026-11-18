# Movie recommendation demo

An interactive teaching demo for movie recommendations. Choose example movies
you like, train and compare collaborative filtering models, then see which
unseen movies the selected model recommends.

## Get started

We use [uv](https://docs.astral.sh/uv/), the Python package manager. If you
have it, run from a terminal:

```bash
git clone https://github.com/suurinie/movie-recommendation-demo-pyladies.git
cd movie-recommendation-demo-pyladies
uv sync --locked
uv run streamlit run src/movie_demo/app.py
```

Open the local URL printed in the terminal. uv downloads Python 3.12 by itself
if you don't have it.

On the first run you may see two one-time prompts:

- Streamlit asks for your email in the terminal: just press Enter.
- Windows Firewall asks whether to allow `python.exe`: allow it.

Don't have uv, or prefer not to install it? See
[Other ways to install](#other-ways-to-install) below.

The demo uses the included dataset, so you do not need to download a movie
catalog or generate data. The viewers are synthetic, generated from movie
genres, directors and actors; see the [author and developer guide](./AUTHORING.md)
for details.

## Try the demo

1. Choose **L2 — train and compare** to fit candidate models locally, or
   **L1 — precomputed** to use the full-data Lasso model without waiting for a
   fit. The saved L1 model's λ was selected using validation Recall@10.
2. In L2 mode, adjust the candidate λ values and select **Train and compare**.
   Compare validation Recall@10 against the popularity and random baselines.
3. In **Try your favorites**, select movies from the alphabetical
   **Your favorites** picker.
4. Explore the recommendations. They are ranked suggestions, not probabilities.

L2 training uses 40,000 synthetic training viewers and selects λ with 5,000
validation viewers. The precomputed L1 model was fitted on the same training
split and selected using the same validation split. Neither model uses the
5,000 test viewers for selection; they are scored once, after selection, as a
final check.

The **About the models** section summarizes model provenance and evaluation.
Recommendations use movie-to-movie connections learned from sparse likes,
not the catalog's genres, directors, or actors.

For data generation, command-line usage, model details, and development
instructions, see the [author and developer guide](./AUTHORING.md).

## License

The code is released into the public domain under the [Unlicense](./LICENSE).

The movie list (titles, directors, years) comes from the Kaggle dataset
[IMDB Movies Dataset: Top 1000 movies and TV shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows),
published there under CC0 (public domain). The data was originally collected
from IMDb. The synthetic likes were generated from that dataset's genres, cast
and vote counts.

## Other ways to install

### Install uv

macOS with [Homebrew](https://brew.sh):

```bash
brew install uv
```

macOS / Linux, without Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

or, with winget: `winget install --id=astral-sh.uv -e`

Linux: your distribution may package uv (for example Arch, Fedora); otherwise
use the script above, or `pipx install uv`.

Then **open a new terminal** so the `uv` command is found, and follow
[Get started](#get-started).

### Without uv

Use pip with Python 3.12 or newer. After cloning, from the project
folder (no virtual environment activation needed):

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m streamlit run src/movie_demo/app.py
```

Windows (PowerShell or cmd):

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m streamlit run src/movie_demo/app.py
```

pip does not use the lock file, so it may install newer package versions than
the ones the demo was tested with.
