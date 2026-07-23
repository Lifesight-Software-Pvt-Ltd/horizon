# Quickstart

A minimal end-to-end run: evaluate every model on a held-out split, build a
forecast, and export the results. For the ensemble specifics see
[`ensemble_methods.md`](ensemble_methods.md); for per-model behaviour see
[`supported_models.md`](supported_models.md).

## 1. Your data

`HorizonForecaster` expects a single tidy DataFrame with a date column, a
numeric target column, and (optionally) one or more numeric independent
columns. Rows should be one-per-period at a regular `daily` or `weekly`
spacing.

```python
import pandas as pd

df = pd.DataFrame({
    "date":  pd.date_range("2022-01-01", periods=400, freq="D"),
    "orders": [...],      # the target you want to forecast
    "spend":  [...],      # optional independent driver(s)
})
```

## 2. Construct the forecaster

```python
from horizon_forecaster.forecaster import HorizonForecaster

forecaster = HorizonForecaster(
    df,
    date_column="date",
    target_column="orders",
    frequency="daily",                 # "daily" or "weekly"
    independent_columns=["spend"],     # omit if you have none
    training_size=80,                  # % of rows used for training (rest is the test split)
    models=None,                       # None = every registered model; or e.g. ["sarimax", "catboost"]
    mape_threshold=40.0,               # a model is included in the ensemble only if its MAPE <= this
    ensemble_weighting="inverse_mape", # or "equal"
)
```

`models=None` runs all registered models: `prophet`, `catboost`, `sarimax`,
`neuralprophet`, and `bayesianpymc`. Pass an explicit list to run a subset —
worth doing to exclude `bayesianpymc` when you want a fast run, since its
MCMC sampling is much slower than the others.

## 3. Evaluate on the held-out split

`evaluate()` splits chronologically, fits each model on the training portion,
predicts the test portion, and scores MAPE. It returns a dict keyed by model
name (always a dict, even for one model).

```python
results = forecaster.evaluate()

for name, res in results.items():
    print(f"{name}: MAPE {res.mape.value:.2f}%  (train {res.n_train}, test {res.n_test})")

# The ensemble is populated as a side effect, scored against the real held-out actuals:
ens = forecaster.evaluate_ensemble          # EnsembleResult, or None if nothing qualified
if ens and ens.combined is not None:
    print("ensemble MAPE:", ens.mape.value)
    print("weights:", ens.weights)
    print("rejected:", list(ens.rejected_models))   # fit fine, but above mape_threshold

# Models that failed to fit/predict at all are recorded separately:
print("errors:", forecaster.evaluate_errors)
```

A model in `evaluate_ensemble.rejected_models` fit successfully but scored
above `mape_threshold`; a model in `evaluate_errors` failed outright. The two
are never conflated, and neither raises unless *every* model fails.

## 4. Forecast the future

```python
forecasts = forecaster.forecast(horizon=30)   # 30 future periods

sarimax_future = forecasts["sarimax"].forecast   # DataFrame ["ds", "yhat"] (+ bounds where a model provides them)
future_ensemble = forecaster.forecast_ensemble   # weights reused from evaluate()
```

`forecast()` refits each model on the full dataset (once, cached) and reuses
`evaluate()`'s per-model MAPE for the ensemble weights, since a true future
forecast has no actuals to score against. If you call `forecast()` before
`evaluate()`, it runs `evaluate()` internally once and warns about the added
cost.

If your forecast depends on independent columns, supply their planned future
values; otherwise they are auto-generated from history and a warning is
raised, since an exog-driven forecast is only as good as those assumed inputs:

```python
future_spend = pd.DataFrame({"spend": [...]})   # len == horizon
forecasts = forecaster.forecast(horizon=30, future_independent_data=future_spend)
```

## 5. Export CSVs and plots

```python
from horizon_forecaster.exporters import export_forecast

paths = export_forecast(
    forecasts,
    ensemble=forecaster.forecast_ensemble,
    per_model_mape={m: r.mape.value for m, r in results.items()},
    ensemble_mape=forecaster.evaluate_ensemble.mape.value if forecaster.evaluate_ensemble else None,
    history=df, date_column="date", target_column="orders",
    output_dir="output", prefix="forecast",
    save_plots=True,     # set False for CSVs only
)
print(paths)
```

This writes five CSVs — `final_ensemble`, `final_best_model`, `per_model`,
`rejected`, `model_summary` — and (unless `save_plots=False`) one forecast
PNG per model plus an overlay.

## Plotting directly

If you want figures without exporting, the functions in
`horizon_forecaster.plotting` each draw onto and return a matplotlib `Axes`
(or a `{label: Figure}` dict for the per-model variants) without calling
`plt.show()`:

```python
from horizon_forecaster.plotting import plot_evaluation, plot_model_comparison

plot_evaluation(results)                                   # actuals vs each model's test predictions
plot_model_comparison(results, forecaster.evaluate_ensemble)   # bar chart of per-model + ensemble MAPE
```

## Important: the ensemble is not guaranteed to win

A weighted ensemble can score worse than the single best model — this is a
real property of static weighting, not a bug. Always compare
`evaluate_ensemble.mape` against each model's own `results[name].mape` before
choosing which output to trust. Both a `final_ensemble` and a
`final_best_model` CSV are exported for exactly this reason.
