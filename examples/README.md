# Examples

Runnable notebooks for Horizon Forecaster. Each one is self-contained: open it in Colab,
run the cells top to bottom, and nothing needs downloading by hand. Each is also saved with
its outputs, so you can read the results on GitHub without running anything.

Their purpose is to show how the library works end to end on external datasets from Kaggle.
Both datasets are simulated or synthetic, so treat the numbers as a demonstration of the machinery rather
than as evidence about how well these models forecast real data.

| Notebook | Frequency | Shows |
|---|---|---|
| [`daily_website_traffic.ipynb`](daily_website_traffic.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lifesight-Software-Pvt-Ltd/horizon/blob/main/examples/daily_website_traffic.ipynb) | daily | The full run on three years of daily data — the three-way split, several combination rules side by side, supplying known future drivers while the library generates the rest, and the chart and CSV outputs. |
| [`weekly_marketing_mix.ipynb`](weekly_marketing_mix.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lifesight-Software-Pvt-Ltd/horizon/blob/main/examples/weekly_marketing_mix.ipynb) | weekly | Three years of weekly marketing-mix data — preparing a week × region × product panel into one series, the weekly split defaults, and forecasting a quarter against a supplied media plan while the seasonal drivers are copied from the same week last year (`last_year`). |

## The paper

[`paper/Robust-Ensemble-Forecasting.pdf`](paper/Robust-Ensemble-Forecasting.pdf) is
*Robust Ensemble Forecasting in Practice: Model Selection and Combination Across
Heterogeneous Time Series* (Anil Kumar Singh, Sanvi Bhardwaj, Rajeev Aravindakshan Nair;
Lifesight).

- **Research paper:** https://github.com/lifesight/robust-ensemble-forecasting — the paper
  with its data, out-of-fold forecasts and the code that regenerates every table and figure in it.
- **White paper:** https://lifesight.io/resources/ensemble-forecasting/

The paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), not under
this repository's Apache License; see the research paper repository for its licence terms.

The paper evaluates the approach behind this library (screen candidate models on a validation
window, combine the survivors with a trimmed mean) on 99 real daily commercial series. These
notebooks only show the library's mechanics, on two small simulated datasets; for evidence about
accuracy, read the paper.

## Running them

**In Colab** — use the badges above. The notebook installs the library from this repository
and loads its data by URL, so there is nothing to set up.

One Colab-specific note: the default runtime is Python 3.13, and `neuralprophet` pins
`numpy<2`, which publishes no 3.13 wheel — so pip compiles numpy from source and the install
takes several extra minutes. Setting **Runtime → Change runtime type → Runtime version →
2026.07** (Python 3.12) avoids that. See [`docs/installation.md`](../docs/installation.md).

**Locally** — from the repository root, install the library and open a notebook:

```bash
pip install -e ".[dev]"
pip install jupyter
jupyter notebook examples/daily_website_traffic.ipynb   # or weekly_marketing_mix.ipynb
```

The notebooks read their data from a URL by default so that Colab works unchanged. Point
`DATA_URL` at the local copy under `examples/data/` if you would rather run offline.

## A word on runtime

These notebooks fit real models — Prophet and PyMC compile on first use, and Bayesian PyMC
draws a full MCMC sample per grid candidate. Each notebook has a `FAST` switch. It is off as
committed, so the saved outputs show all five models; on a free Colab CPU expect about half
an hour for the daily notebook and about 15 minutes for the weekly one, plus the install.
Set `FAST = True` to run three models (Prophet, CatBoost, SARIMAX)
in a few minutes, with their grid searches intact, so the real selection machinery still runs.

## Data

Each dataset is committed here **exactly as downloaded** — no columns removed, no values
changed — so it can be checked against its source. All preparation happens in the notebook,
where you can see it and disagree with it.

### `data/final_website_stats.csv`

- **Personal Ecommerce Website Ad cost & viewer count**
- Author: Kaggle user [`michealknight`](https://www.kaggle.com/michealknight)
- Source: https://www.kaggle.com/datasets/michealknight/personal-ecommerce-website-ad-cost-and-viewer-count
- Licence: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Retrieved: 2026-10-06
- 1096 daily rows, 2022-01-01 to 2024-12-31, no missing values
- **Simulated data**, as the dataset's own Kaggle subtitle states ("Simulated Daily Web
  Analytics with Ad Spend, Holiday, and Weekend Impact on Page")

The notebook drops a saved index column and the `day_of_week` string, and
casts two boolean columns to integers; the committed file itself is untouched.

One property worth knowing before drawing conclusions from the numbers: in this dataset
`ad_spend` barely correlates with `page_views` at any lag from 0 to 7 days (|r| < 0.25 at
every lag, and −0.003 at lag 0). It is kept as a driver because it exercises the exogenous path through
every model, not because it improves the forecast. The drivers that carry signal here are
the calendar flags.

### `data/synthetic_mmm_weekly_india.csv`

- **MMM Weekly Data - Geo:India** — "Synthetic MMM Training Data: Indian FMCG Sales, Media & Pricing"
- Author: Kaggle user [`subhagatoadak`](https://www.kaggle.com/subhagatoadak)
- Source: https://www.kaggle.com/datasets/subhagatoadak/mmm-weekly-data-geoindia
- Licence: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Retrieved: 2026-10-13
- 11,232 rows: 156 weeks (2022-07-04 to 2025-06-23) × 8 regions × 9 products, 28 columns,
  no missing values
- **Synthetic data**, generated by its author to train marketing-mix models

The notebook totals the panel to one national weekly series: sales value and the five media
channels are summed across every row; trade spend is recorded once per region (repeated on each
product row), so it is taken once per region and then summed; the festival and rainfall indices are
averaged across regions. It leaves out Instagram impressions (derived from YouTube and Facebook,
per the dataset's own documentation), CPI and GDP growth (they trend with sales over the whole
period), and the promotion, price, distribution and creative-score columns, to keep the example
readable.

One property worth knowing: the dataset's description calls `Week` a week-ending Saturday, but
every date in the file is a Monday. The notebook uses the dates as they are.
