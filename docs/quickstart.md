# Quickstart

A minimal end-to-end run: evaluate every model on held-out data, build a
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
from horizon_forecaster.forecaster import HorizonForecaster

forecaster = HorizonForecaster(
    df,
    date_column="date",
    target_column="orders",
    frequency="daily",              # "daily" or "weekly"
    independent_columns=["spend"],  # omit if you have none
    validation_size=None,           # None = the frequency default (56 daily / 26 weekly)
    test_size=None,                 # None = the frequency default (28 daily / 13 weekly)
    models=None,                    # None = every registered model; or e.g. ["sarimax", "catboost"]
    mape_threshold=40.0,            # validation-MAPE cutoff for ensemble membership; None disables it
    ensemble_method="trimmed",      # or "inverse_mape", "equal", a list of those, or None for all three
)
```

`models=None` runs all registered models: `prophet`, `catboost`, `sarimax`,
`neuralprophet`, and `bayesianpymc`. Pass an explicit list to run a subset —
worth doing to exclude `bayesianpymc` when you want a fast run, since its
MCMC sampling is much slower than the others.

### Choosing the split

Data is split chronologically into **three** windows, and they do different
jobs:

- **train** — what models are fitted on.
- **validation** — where every *decision* is made: the hyperparameter grid
  search, `mape_threshold` screening, and inverse-MAPE weights.
- **test** — what gets *reported*, and which no selection path is ever allowed
  to read.

`validation_size` and `test_size` each accept an **int** (a number of periods),
a **float strictly between 0 and 1** (a fraction of the data), or **`None`**
(the per-frequency default: 56 validation / 28 test for daily, 26 / 13 for
weekly). Training takes whatever is left. A float of `1.0` or more is rejected
rather than guessed at, since `28` and `28.0` would otherwise mean the same
thing while `0.28` meant something completely different.

The defaults are absolute period counts rather than percentages on purpose. The
test window is meant to estimate the error you will actually experience, so it
should be about as long as a horizon somebody would really forecast and act on
— 4 weeks of daily data, a quarter of weekly. The validation window has to rank
3–18 grid candidates against each other, so it wants stability, which means
more data; hence twice the test window.

If you would rather think in percentages, `training_size` still works exactly
as it always did — a percentage (50–100) or fraction (0.5–1.0), never a number
of periods — and the remainder is divided between validation and test, two
thirds to validation:

```python
HorizonForecaster(df, ..., training_size=80)   # 80% train, ~13% validation, ~7% test
```

The two routes are **mutually exclusive**: passing `training_size` together
with `validation_size` or `test_size` raises `InvalidConfigError` rather than
silently letting one win.

On a short series, the defaults scale down proportionally (they are capped at
40% of the rows between them, preserving the 2:1 ratio) and you get a warning.
Scores from windows that small are noisy and are not comparable to scores from
a longer series, which uses the full windows. A size you asked for *explicitly*
is honoured exactly, or raises — a window you deliberately chose is never
silently shrunk.

## 3. Evaluate: tuned on validation, reported on test

`evaluate()` selects each model's hyperparameters on the validation window,
refits the winner on train + validation, then predicts and scores the test
window. It returns a dict keyed by model name (always a dict, even for one
model).

```python
results = forecaster.evaluate()

for name, res in results.items():
    print(f"{name}: test MAPE {res.mape.value:.2f}%  "
          f"(validation {res.validation_mape.value:.2f}%; "
          f"train {res.n_train}, validation {res.n_validation}, test {res.n_test})")
```

The two MAPEs are not interchangeable. `validation_mape` is the score every
decision was made on, and is optimistic by construction — candidates were
chosen precisely to minimise it. `mape` is the honest out-of-sample estimate on
the test window. A large gap between them is itself a useful diagnostic: it
means the grid search found a candidate that suited the validation window
specifically rather than the series generally.

The refit before scoring test is deliberate. Fitting on train alone and
predicting straight through both windows would be one fit cheaper per model,
but it would measure a model trained on less data than would ever be deployed;
refitting on everything available makes the reported MAPE an estimate of the
same procedure that produces your real forecast.

```python
# The ensembles are populated as a side effect, scored against the real test actuals.
# Always a dict keyed by combination method, even for a single method.
ens = forecaster.evaluate_ensemble["trimmed"]
if ens.combined is not None:
    print("ensemble MAPE:", ens.mape.value)
    print("rule applied:", ens.rule_applied, "| members:", ens.members)
    print("weights:", ens.weights)                  # None for trimmed — it has no weights
    print("trim shares:\n", ens.trim_shares)        # who got discarded, and how often
    print("rejected:", list(ens.rejected_models))   # fit fine, but above mape_threshold

# Models that failed to fit/predict at all are recorded separately:
print("errors:", forecaster.evaluate_errors)
```

A model in `rejected_models` fit successfully but scored above
`mape_threshold` on validation; a model in `evaluate_errors` failed outright.
The two are never conflated, and neither raises unless *every* model fails.

## 4. Forecast the future

```python
forecasts = forecaster.forecast(horizon=30)   # 30 future periods

sarimax_future = forecasts["sarimax"].forecast   # DataFrame ["ds", "yhat"] (+ bounds where a model provides them)
future_ensemble = forecaster.forecast_ensemble["trimmed"]   # dict keyed by method, as in evaluate()
```

`forecast()` refits each model on the full dataset (once, cached). A true
future forecast has no actuals to score against, so any screening or weighting
the ensemble needs comes from the **validation** MAPE established when each
model's hyperparameters were selected. That selection has to happen anyway —
`forecast()` cannot fit without knowing which parameters won — so the scores
come for free and are cached, whether or not you ever call `evaluate()`.
Calling `forecast()` without `evaluate()` still pays for the grid search (there
is no way to know which parameters to fit with otherwise), but it no longer
runs the whole of `evaluate()` internally and no longer pays for a test-window
prediction it would throw away — so the `UserWarning` that used to flag that
cost is gone.

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
  auto-detected (≤2 distinct non-negative values).
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
from horizon_forecaster.exporters import export_forecast

paths = export_forecast(
    forecasts,
    ensemble=forecaster.forecast_ensemble,     # the whole {method: EnsembleResult} dict
    per_model_mape={m: r.mape.value for m, r in results.items()},
    per_model_validation_mape={m: r.validation_mape.value for m, r in results.items()},
    # A forecast ensemble has no score of its own (no future actuals), so the
    # summary borrows one from evaluate() if you pass it:
    ensemble_mape=(ens.mape.value if ens.mape is not None else None),
    history=df, date_column="date", target_column="orders",
    output_dir="output", prefix="forecast",
    save_plots=True,     # set False for CSVs only
)
print(paths)
```

This writes:

- `forecast_final_ensemble_<method>.csv` — **one per combination method**, so a
  run with several rules produces several files. The method is always in the
  name, even for a single rule: a filename that silently changed shape
  depending on how many rules you asked for would be worse than a slightly
  longer one, and downstream jobs can glob a stable pattern.
- `forecast_final_best_model.csv` — the single best-scoring model's forecast.
- `forecast_per_model.csv` — every model's forecast, stacked.
- `forecast_rejected_models.csv` — one row per (model, method) exclusion; the
  `ensemble_method` column matters because exclusion is not necessarily the
  same across rules.
- `forecast_model_summary.csv` — `model`, `validation_mape`, `mape`,
  `ensemble_method`, `rule_applied`, `ensemble_weight`, `status`.
  `ensemble_weight` is empty for weightless rules such as the default trimmed
  mean; filling in an equal share would imply an averaging that did not happen.

Unless `save_plots=False`, one forecast PNG per model plus an overlay are
written alongside them.

## Plotting directly

If you want figures without exporting, the functions in
`horizon_forecaster.plotting` each draw onto and return a matplotlib `Axes`
(or a `{label: Figure}` dict for the per-model variants) without calling
`plt.show()`:

```python
from horizon_forecaster.plotting import plot_evaluation, plot_model_comparison

plot_evaluation(results)                                       # actuals vs each model's test predictions
plot_model_comparison(results, forecaster.evaluate_ensemble)   # bar chart of per-model + ensemble MAPE
```

Both accept the whole `{method: EnsembleResult}` dict. The line charts draw a
single ensemble (the first method requested, labelled with the rule that was
actually applied); the comparison bar chart shows one bar per method that was
built and scored.

## Important: the ensemble is not guaranteed to win

A combined ensemble can score worse than the single best model — this is a real
property of the method, not a bug. Trimming is more robust to it than static
weights, since it discards a badly-behaved member on the specific periods where
it misbehaves rather than applying one number across the whole horizon, but it
is still not a guarantee. Always compare the ensemble's `mape` against each
model's own `results[name].mape` before choosing which output to trust. Both a
`final_ensemble_<method>` and a `final_best_model` CSV are exported for exactly
this reason.
