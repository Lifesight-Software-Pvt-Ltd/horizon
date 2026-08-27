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

**The data must be complete.** A single missing or non-numeric cell in any
column used for training/testing — the date, the target, or an independent
column — raises `InvalidConfigError` at construction. Missing values are never
silently dropped or imputed: clean or impute your data first. (Columns in the
DataFrame that you don't use are ignored, and a fully-constant column is
dropped automatically with a report in `dropped_zero_variance_columns`.)

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
from horizon_forecaster import HorizonForecaster

forecaster = HorizonForecaster(
    df,
    date_column="date",
    target_column="orders",
    frequency="daily",              # "daily" or "weekly"
    independent_columns=["spend"],  # omit if you have none
    validation_size=None,           # selection window; None = frequency default (56 daily / 26 weekly)
    test_size=None,                 # reported window; None = frequency default (28 daily / 13 weekly)
    models=None,                    # None = every registered model; or e.g. ["sarimax", "catboost"]
    mape_threshold=40.0,            # garbage filter on the VALIDATION MAPE; None admits every model
    ensemble_method="trimmed",      # or "inverse_mape", "equal", a list of them, or None for all three
)
```

`models=None` runs all registered models: `prophet`, `catboost`, `sarimax`,
`neuralprophet`, and `bayesianpymc`. Pass an explicit list to run a subset —
worth doing to exclude `bayesianpymc` when you want a fast run, since its
MCMC sampling is much slower than the others.

### Sizing the split

Windows are sized in **periods** by default. For `validation_size` and
`test_size`, an `int` is a number of periods, a `float` strictly between 0 and
1 is a fraction of the data, and `None` takes the frequency default. Training
takes every row the two windows leave.

If you would rather think in percentages, `training_size` is the alternative
route — and it is **percentages only**: `training_size=80` means 80%, never 80
periods. The remainder is divided between validation and test, two-thirds to
validation. It is mutually exclusive with `validation_size`/`test_size`;
passing both raises `InvalidConfigError` rather than letting one silently win.

On a short series the defaults are scaled down (with a warning) rather than
leaving nothing to train on.

## 3. Evaluate on the held-out split

`evaluate()` splits chronologically into **train / validation / test**. Each
model's hyperparameters are chosen by grid search on the **validation**
window, the winner is refit on train + validation, and it is then scored once
on the **test** window — which no selection path is ever allowed to read. So
every result carries two MAPEs, and they are not interchangeable:

| field | window | meaning |
|---|---|---|
| `res.validation_mape` | validation | the score every DECISION was made on. Optimistic by construction, since candidates were chosen to minimise it. |
| `res.mape` | test | the score REPORTED — the honest out-of-sample estimate. |

A large gap between them means the grid search found a candidate that suited
the validation window specifically rather than the series generally.

`evaluate()` returns a dict keyed by model name (always a dict, even for one
model).

```python
results = forecaster.evaluate()

for name, res in results.items():
    print(f"{name}: test MAPE {res.mape.value:.2f}%  "
          f"(validation {res.validation_mape.value:.2f}%; "
          f"train {res.n_train}, validation {res.n_validation}, test {res.n_test})")

# The ensembles are populated as a side effect, scored against the real held-out
# actuals. evaluate_ensemble is a Dict[str, EnsembleResult] keyed by combination
# method -- always a dict, even when a single method was requested.
for method, ens in forecaster.evaluate_ensemble.items():
    if ens.combined is None:
        continue
    print(f"[{method}] rule applied: {ens.rule_applied}"
          f"{' (degraded)' if ens.degraded else ''}, members: {ens.members}")
    print(f"[{method}] ensemble MAPE:", ens.mape.value)
    print(f"[{method}] weights:", ens.weights)          # None for weightless rules such as trimmed
    print(f"[{method}] trim shares:", ens.trim_shares)  # trimming only; None for the weighted rules
    print(f"[{method}] rejected:", list(ens.rejected_models))

# Models that failed to fit/predict at all are recorded separately:
print("errors:", forecaster.evaluate_errors)
```

A model in an ensemble's `rejected_models` fit successfully but wasn't
admitted to the combination; a model in `evaluate_errors` failed outright. The
two are never conflated, and neither raises unless *every* model fails.

## 4. Forecast the future

```python
forecasts = forecaster.forecast(horizon=30)   # 30 future periods

sarimax_future = forecasts["sarimax"].forecast   # DataFrame ["ds", "yhat"] (+ bounds where a model provides them)
future_ensembles = forecaster.forecast_ensemble  # Dict[str, EnsembleResult], same keys as above
```

`forecast()` refits each model on the full dataset (once, cached). Its
ensembles screen and weight on the **validation** MAPE from hyperparameter
selection — which `forecast()` has to perform anyway in order to know which
parameters to fit with, so those scores come for free and are cached. It does
**not** run `evaluate()` internally, so calling `forecast()` first is no longer
more expensive than calling it second.

If your forecast depends on independent columns, supply their planned future
values; otherwise they are auto-generated from history and a warning is
raised, since an exog-driven forecast is only as good as those assumed inputs:

```python
future_spend = pd.DataFrame({"spend": [...]})   # len == horizon
forecasts = forecaster.forecast(horizon=30, future_independent_data=future_spend)
```

## 4b. Future values for your drivers

When you *don't* supply future values for a declared independent column,
they are generated by a **future-exogenous-data strategy** (set once on the
constructor). The default is `carry_forward` — hold the last observed value
flat:

```python
forecaster = HorizonForecaster(
    df, date_column="date", target_column="orders", frequency="daily",
    independent_columns=["spend", "promo"],
    future_exog_strategy="seasonal_naive",   # carry_forward | recent_average |
                                             # historical_average | seasonal_naive |
                                             # linear_trend | cycle
    event_columns=["promo"],                 # 0/1 flags -> default to 0, not carried
    future_exog_aggregator="median",         # for the *_average strategies
    recent_average_window=28,                # N for recent_average
)
```

Two things worth knowing:

- **Type-aware flags.** Binary 0/1 event columns (promos, holidays) default
  to `0` ("no event unless you supply one") instead of being carried forward
  or averaged into a fractional flag — except under `seasonal_naive`, where
  they follow the cycle. Name them with `event_columns`, or they're
  auto-detected (≤2 distinct non-negative values). Naming them explicitly
  disables auto-detection, which is worth doing if one of your continuous
  drivers happens to take only two distinct values over the history — it
  would otherwise be treated as a flag and defaulted to 0.
- **Partial fill.** You can supply future values for *some* columns and let
  the rest be generated — pass a DataFrame with only the columns you know.

After a forecast, inspect exactly what was assumed:

```python
forecasts = forecaster.forecast(horizon=30)     # or with a partial future_independent_data
print(forecaster.future_exog_used)              # ds + every driver actually used
```

See [`future_exog.md`](future_exog.md) for what each strategy does and when
to use it.

## 5. Export CSVs and plots

```python
from horizon_forecaster import DEFAULT_ENSEMBLE_METHOD
from horizon_forecaster.exporters import export_forecast

default_eval_ensemble = (forecaster.evaluate_ensemble or {}).get(DEFAULT_ENSEMBLE_METHOD)

paths = export_forecast(
    forecasts,
    forecaster.forecast_ensemble,   # the whole {method: EnsembleResult} dict
    per_model_mape={m: r.mape.value for m, r in results.items()},
    per_model_validation_mape={m: r.validation_mape.value for m, r in results.items()},
    ensemble_mape=(
        default_eval_ensemble.mape.value
        if default_eval_ensemble is not None and default_eval_ensemble.mape is not None
        else None
    ),
    history=df, date_column="date", target_column="orders",
    output_dir="output", prefix="forecast",
    save_plots=True,     # set False for CSVs only
)
print(paths)
```

This writes `final_ensemble_<method>` — **one per combination rule**, so
asking for all three writes three of them — plus `final_best_model`,
`per_model`, `rejected` and `model_summary`, and (unless `save_plots=False`)
one forecast PNG per model plus an overlay.

`model_summary` carries `validation_mape` and `mape` side by side, so the
selection-vs-reported gap is visible per model. Its `ensemble_weight` column
is deliberately blank for weightless rules such as the default trimmed mean —
there is no weight, and filling in an equal share would imply an averaging
that did not happen. Note `ensemble_mape` is only a fallback for the combined
row and is used only when a single method was requested; with several methods
the forecast-context rows are left blank rather than repeating one score
against three different series.

## Plotting directly

If you want figures without exporting, the functions in
`horizon_forecaster.plotting` each draw onto and return a matplotlib `Axes`
(or a `{label: Figure}` dict for the per-model variants) without calling
`plt.show()`. They all accept the `{method: EnsembleResult}` dict directly:

```python
from horizon_forecaster.plotting import plot_evaluation, plot_model_comparison

plot_evaluation(results, forecaster.evaluate_ensemble)         # actuals vs each model's test predictions
plot_model_comparison(results, forecaster.evaluate_ensemble)   # bar chart of per-model + ensemble MAPE
```

The line charts draw only the **first** requested method — overlaying several
combination rules on one time axis buries the per-model lines the chart exists
to show. `plot_model_comparison` is the exception: comparing rules is exactly
what a bar chart is for, so it draws one bar per method. Its bars are
**test-window** MAPEs, not the validation scores the models were selected on.

## Important: the ensemble is not guaranteed to win

A combined ensemble can score worse than the single best model — this is a
real property of combination, not a bug. Always compare each ensemble's
`.mape` against each model's own `results[name].mape` before choosing which
output to trust. Both a `final_ensemble_<method>` and a `final_best_model` CSV
are exported for exactly this reason.
