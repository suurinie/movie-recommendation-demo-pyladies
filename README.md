# Movie recommendation demo

An interactive teaching demo for movie recommendations. Choose example movies
you like, train and compare collaborative filtering models, then see which
unseen movies the selected model recommends.

## Get started

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From a terminal:

```bash
git clone https://github.com/suurinie/movie-recommendation-demo-pyladies.git
cd movie-recommendation-demo-pyladies
uv sync --locked
uv run streamlit run src/movie_demo/app.py
```

Open the local URL printed in the terminal. The demo uses the included
synthetic dataset; you do not need to download a movie catalog or generate data.

## Try the demo

1. In **Train and compare models**, keep or adjust the candidate λ values and
   select **Train and compare**.
2. Compare validation NDCG@10 and Recall@10 for each model against the
   popularity baseline. The selected λ is chosen using validation data.
3. In **Try your favorites**, search for a movie (try `Totoro`) or choose one
   from the alphabetical browse list, then add a few favorites.
4. Explore the recommendations. They are ranked suggestions, not probabilities.

Training uses synthetic likes from 40,000 viewers, validation uses 5,000
viewers to compare settings, and a separate 5,000-viewer test set is kept
untouched. The recommendations use the selected model trained on the training
viewers.

The **Under the hood** section in the app shows the Python and tools used for
the data split, model fit, and λ selection. The model learns movie-to-movie
relationships from sparse likes; it does not use the catalog's genres,
directors, or actors to make recommendations.

For data generation, command-line usage, model details, and development
instructions, see the [author and developer guide](./AUTHORING.md).
