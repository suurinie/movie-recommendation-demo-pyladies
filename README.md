# Movie recommendation demo

An interactive teaching demo for movie recommendations. Choose example movies
you like, train and compare collaborative filtering models, then see which
unseen movies the selected model recommends.

## Get started

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
You do not need to install Python yourself: uv downloads the right version.

### 1. Install uv (once)

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then **open a new terminal** so the `uv` command is found.

### 2. Get and run the demo

```bash
git clone https://github.com/suurinie/movie-recommendation-demo-pyladies.git
cd movie-recommendation-demo-pyladies
uv sync --locked
uv run streamlit run src/movie_demo/app.py
```

Open the local URL printed in the terminal.

On the first run you may see two one-time prompts:

- Streamlit asks for your email in the terminal: just press Enter.
- Windows Firewall asks whether to allow `python.exe`: allow it.

No uv? In a Python 3.12 virtual environment, run `pip install -e .` and then
`streamlit run src/movie_demo/app.py`.

The demo uses the included dataset, so you do not need to download a movie
catalog or generate data. The viewers are synthetic, generated from movie
genres, directors and actors; see the [author and developer guide](./AUTHORING.md)
for details.

## Try the demo

1. Choose **L2 — train and compare** to fit candidate models locally, or
   **L1 — precomputed** to use the full-data Lasso model without waiting for a
   fit. The saved L1 model's λ was selected using validation Recall@10.
2. In L2 mode, adjust the candidate λ values and select **Train and compare**.
   Compare validation Recall@10 against the popularity baseline.
3. In **Try your favorites**, select movies from the alphabetical
   **Your favorites** picker.
4. Explore the recommendations. They are ranked suggestions, not probabilities.

L2 training uses 40,000 synthetic training viewers and selects λ with 5,000
validation viewers. The precomputed L1 model was fitted on the same training
split and selected using the same validation split. Neither model uses the
5,000 test viewers for selection; they are scored once, after selection, as a
final check.

The **About the models** section summarizes model provenance and evaluation.
Recommendations use movie-to-movie relationships learned from sparse likes,
not the catalog's genres, directors, or actors.

For data generation, command-line usage, model details, and development
instructions, see the [author and developer guide](./AUTHORING.md).
