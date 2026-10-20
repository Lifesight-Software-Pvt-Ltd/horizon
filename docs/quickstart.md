# Quickstart

A minimal end-to-end run: evaluate every model on a held-out split, build a
forecast, and export the results. For the ensemble specifics see
[`ensemble_methods.md`](ensemble_methods.md); for per-model behaviour see
[`supported_models.md`](supported_models.md). For two complete runs with their
outputs saved, one daily and one weekly, see the
[example notebooks](../examples/).

## 1. Your data

`HorizonForecaster` expects a single tidy DataFrame with a date column, a
numeric target column, and (optionally) one or more numeric independent
columns. Rows should be one-per-period at a regular `daily` or `weekly`
spacing.

**The data must be complete.** A single missing, non-numeric or infinite
(`inf`, `-inf`) cell in any column used for training/testing — the date, the
target, or an independent column — raises `InvalidConfigError` at
construction, naming the column and the dates. (Numbers stored as text are
read as numbers, and driver columns may also use on/off words such as
`yes/no`; see 4b.) Missing values are never silently dropped or imputed:
clean or impute your data first. Columns in the DataFrame that you do not use
are ignored. A declared driver with the same value in every row cannot be
learned, so it is dropped with a warning and listed in
`dropped_zero_variance_columns`.

Each column plays one role: the target and the date column cannot also be
drivers, and each name may appear once in `independent_columns` and once in
the DataFrame.

A driver that changes over the full history but never changes inside one fit
window — a price change or a new promotion that first appears in the
validation or test window — is left out of that window's fit for every model,
since no model can learn an effect from it there. A warning names the driver,
its value and the date it first changes, and says which score includes an
effect the models could not see: the validation score (MAPE or WAPE, which
drives hyperparameter selection, `mape_threshold` screening and ensemble
weights) or the test score. `forecast()` refits on the full series, where the
driver does change, and uses it.

**What data is accepted.** The constructor checks the date column against the
rules below. Anything outside them raises an error naming the rule and the fix,
rather than running on data that would give a confident but wrong result.

| # | Rule | Otherwise |
|---|---|---|
| 1 | `frequency` is `"daily"` or `"weekly"` | `UnsupportedFrequencyError` |
| 2 | Dates are datetimes, or text in one format throughout: `YYYY-MM-DD`, `YYYY/MM/DD`, `DD-MM-YYYY`, `DD/MM/YYYY`, `MM-DD-YYYY`, `MM/DD/YYYY` (four-digit year), or a full ISO timestamp such as `2024-01-15 00:00:00` | `InvalidConfigError` |
| 3 | No empty date cells | `InvalidConfigError` |
| 4 | Each date appears once | `InvalidConfigError` naming the dates |
| 5 | Every step between dates is a whole number of periods — weekly dates all on the same weekday | `InvalidConfigError` |
| 6 | More than half of the steps are exactly one period — the data really is daily or weekly | `InvalidConfigError` if it looks like the other frequency (e.g. weekly data declared daily); `UnsupportedFrequencyError` for any other spacing (every 2 days, fortnightly, monthly) |
| 7 | No date missing between the first and the last — every day (daily) or every week (weekly) has a row | `InvalidConfigError` naming the missing dates |

Dates are parsed before anything is sorted or split, and your DataFrame is
never modified. If every value could be read either way — `03/04/2024` is
3 April or 4 March — the constructor raises rather than guess; pass
`date_format="%d/%m/%Y"` (or `"%m/%d/%Y"`) to say which. `date_format` also
accepts any other strftime pattern, e.g. `"%d.%m.%Y"`.

**Missing dates are rejected, however few.** The models count rows, not
dates, so one missing day shifts every weekly cycle after it, and the
train/validation/test windows cover more time than their size says. The error
lists the first five missing dates and the longest gap. Horizon never fills them
itself, because only you know what they were: add each one with 0 if the
business was closed that day, or the real value if the data was lost. Data
from a business closed on weekends needs the weekend rows too, with 0; the
error names any weekday that never appears.

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
    validation_size=None,           # selection window; None = frequency default (84 daily / 39 weekly)
    test_size=None,                 # reported window; None = frequency default (28 daily / 13 weekly)
    models=None,                    # None = every registered model; or e.g. ["sarimax", "catboost"]
    mape_threshold=40.0,            # degenerate-fit filter on the VALIDATION score; None admits every model
    ensemble_method="trimmed",      # or "inverse_mape", "equal", a list of them, or None for all three
)
```

`models=None` runs all registered models: `prophet`, `catboost`, `sarimax`,
`neuralprophet`, and `bayesianpymc`. Pass an explicit list to run a subset —
worth doing to exclude `bayesianpymc` when you want a fast run, since its
MCMC sampling is much slower than the others.

### Skipping the grid search

By default each model is tuned by a grid search scored on the validation
window, which is where most of the runtime goes. `model_params` overrides that
per model with hyperparameters of your own, and the search is skipped entirely
for any model you name:

```python
forecaster = HorizonForecaster(
    df,
    date_column="date",
    target_column="orders",
    frequency="daily",
    model_params={
        "prophet": {"seasonality_mode": "additive"},
        "catboost": {"iterations": 300, "depth": 6},
    },
)
```

Models you do not name are still tuned normally. This is the fast path when you
already know what works for your series — and it is what the test suite uses to
keep runs short. A model name that does not exist, or a setting SARIMAX or
Bayesian PyMC does not use, raises `InvalidConfigError` with the closest valid
name, since a typo would otherwise switch the grid search off and change the
result without a word.

### Sizing the split

Windows are sized in **periods** by default. For `validation_size` and
`test_size`, an `int` is a number of periods, a `float` strictly between 0 and
1 is a fraction of the data, and `None` takes the frequency default. Training
takes every row the two windows leave.

If you would rather think in percentages, `training_size` is the alternative
route — and it is **percentages only**: `training_size=80` means 80%, never 80
periods. The remainder is divided between validation and test, three-quarters
to validation. It is mutually exclusive with `validation_size`/`test_size`;
passing both raises `InvalidConfigError` rather than letting one silently win.

On a short series the defaults are scaled down (with a warning) rather than
leaving nothing to train on.

### Which score drives the decisions: MAPE or WAPE

Every decision — the grid search, the `mape_threshold` screen, inverse-MAPE
weights and the exported best model — is made on one validation score, set by
`selection_metric`:

- `"auto"` (the default) uses **MAPE**, unless MAPE stops meaning anything on
  your target: when 10% or more of its values are 0 (MAPE leaves out every
  zero actual, so it scores a model on the days it finds easiest), or when any
  value is negative (MAPE explodes as actuals cross zero). Then it uses
  **WAPE**, `sum|actual - forecast| / sum|actual| × 100`, which counts every
  period, and warns that it did.
- `"mape"` or `"wape"` fixes the choice. Any case works (`"MAPE"`,
  `"Wape"`); the resolved one is `forecaster.selection_metric`, always
  lowercase.

Both scores are reported either way, and `mape_threshold` applies to
whichever is selected. On 42 test series the two picked the same models on
35; where they disagreed, WAPE's pick did better on test in 6 of 7.

A target that is 0 in **50% or more** of its periods is refused with
`InvalidConfigError`, whatever `selection_metric` says: on the series tested
above that share, every model did worse than forecasting 0 every period. For
daily data the error says how often the weekly totals are 0, since summing to
weeks and running with `frequency="weekly"` is usually the fix.

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
| `res.validation_wape`, `res.wape` | validation, test | the same two windows scored as WAPE, which counts the zero actuals MAPE leaves out (`res.mape.excluded_zero_actual_count`). |

A large gap between them means the grid search found a candidate that suited
the validation window specifically rather than the series generally. When
`selection_metric` is WAPE, read the WAPE pair the same way.

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

# The bar every forecast has to clear, and how alike the models' mistakes are:
print("seasonal naive test WAPE:", forecaster.naive_baseline.wape)
print(forecaster.test_error_correlation)
```

**The seasonal-naive baseline.** `evaluate()` also scores the simplest
possible forecast on the same windows: the same weekday last week for daily
data, the same week last year for weekly, matched by date.
`forecaster.naive_baseline` is an `EvaluationResult` like the models'. Any
model or ensemble rule whose test WAPE is worse than the baseline's is named
in a warning — on 42 test series the default trimmed ensemble lost to it in 13
of the 35 where it formed, which no score on its own reveals.

**Error correlation.** `forecaster.test_error_correlation` is the correlation
of each pair of models' signed test errors. An ensemble gains only from
members that are wrong in different ways: two models correlating near 1 make
the same mistakes, so averaging them adds little.

A model in an ensemble's `rejected_models` fit successfully but was not
admitted to the combination; a model in `evaluate_errors` failed outright. The
two are never conflated, and neither raises unless *every* model fails. Both
are warned about: `evaluate()` and `forecast()` name any model that failed
(with the cause), any rule that lost a requested model, and any rule that
formed no ensemble at all. A model that returns NaN or infinite predictions
counts as failed.

A model that cannot handle the data is left out before anything runs, with a
warning, and listed in `forecaster.skipped_models`: SARIMAX fits in log space,
so it is skipped when the target has a negative value, or on weekly data when
more than 10% of weeks are 0 while the series is normally above 0 (and refused
if it is the only model requested). Below that, and on daily data at any
share, SARIMAX leaves such zeros out of its own fit, with a warning: a day
counts when its weekday normally has values, so the zeros of a business closed
every Sunday are kept and modelled as usual.

## 4. Forecast the future

```python
forecasts = forecaster.forecast(horizon=30)   # 30 future periods

sarimax_future = forecasts["sarimax"].forecast   # DataFrame ["ds", "yhat"] (+ bounds where a model provides them)
future_ensembles = forecaster.forecast_ensemble  # Dict[str, EnsembleResult], same keys as above
```

`forecast()` refits each model on the full dataset (once, cached). Its
ensembles screen and weight on the **validation** score (MAPE or WAPE, per
`selection_metric`) from hyperparameter selection — which `forecast()` has to
perform anyway in order to know which parameters to fit with, so those scores
come for free and are cached. It does
**not** run `evaluate()` internally, so calling `forecast()` first costs no more
than calling it second.

If your forecast depends on independent columns, supply their planned future
values; otherwise they are auto-generated from history and a warning is
raised, since an exog-driven forecast is only as good as those assumed inputs:

```python
future_spend = pd.DataFrame({"spend": [...]})   # len == horizon
forecasts = forecaster.forecast(horizon=30, future_independent_data=future_spend)
```

The table is matched to forecast periods **by position**: row 1 is the first
period after your last date, and so on. It is checked before any model runs,
and anything below raises `InvalidConfigError` with the reason:

| Check | Accepted | Raises |
|---|---|---|
| Type | a pandas DataFrame | a dict, a Series, anything else |
| Drivers declared | `independent_columns` was set on the constructor | a table when no drivers were declared |
| Columns | declared drivers, all or some (the rest are generated); the date and target columns are allowed and ignored | a column that is not a declared driver (a misspelling gets a "did you mean" hint), the same column name twice |
| Rows | exactly `horizon` rows | any other number, including an empty table |
| Dates (optional) | none (rows taken in order), or a date column / `DatetimeIndex` equal to the forecast dates, in order | blank dates, dates out of order, dates that are not the forecast dates |
| Values | numbers; booleans and on/off words (`yes/no`, `y/n`, `t/f`, `true/false`, `on/off`) read as 1/0 | blanks, other text, infinity |
| Sign | negative values for a driver that goes negative in your data | negative values for a driver that never does |
| Event flags | values the flag takes in your data (0 or 1) | any other value, e.g. 2 |

Two cases warn instead: a value outside a driver's historical range (the
models disagree about values they have never seen: Prophet and NeuralProphet
extend the effect in a straight line, SARIMAX and Bayesian PyMC cap it at the
historical range, CatBoost treats it like the most extreme value it has
seen), and a column for a declared driver that was dropped for never
changing in your data (its future values are ignored).

Each `forecast()` call checks its table against its own horizon, so a table
built for `horizon=12` is rejected by `forecast(horizon=26)`. A call that
raises leaves `forecast_ensemble`, `forecast_errors` and `future_exog_used`
empty rather than holding the previous call's results.

## 4b. Future values for your drivers

When you do *not* supply future values for a declared independent column,
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
    future_exog_aggregator="median",         # for the *_average strategies and seasonal_naive
    recent_average_window=28,                # N for recent_average
)
```

Worth knowing:

- **Type-aware flags.** Binary 0/1 event columns (promos, holidays) default
  to `0` ("no event unless you supply one") instead of being carried forward
  or averaged into a fractional flag — except under `seasonal_naive`, where
  they follow the cycle. Name them with `event_columns`, or they're
  auto-detected: a column whose values are only 0 and 1. Any other column,
  even one with two values (TV spend of 0 or 5,000, prices of 9.99 and
  12.99), is a quantity and follows the strategy.
- **On/off text.** A driver may use `true/false`, `yes/no`, `y/n`, `t/f` or
  `on/off` (any case, mixed freely) instead of 1/0 — in your data and in the
  future table. Other text raises an error that says what is unclear: two
  other words such as "Promo"/"None" (which one means "on"?), categories such
  as "Low"/"Mid"/"High", or text among numbers such as "1,200".
- **Supply your known events.** A generated flag is 0 for the whole forecast,
  so Black Friday, a launch or a sale you already have planned is left out
  unless you pass it in `future_independent_data`. Horizon warns when a flag
  was 1 at the same time last year, and when `carry_forward` would hold a
  last period that looks like a promotion (a driver at 1.5× or more, or 0.5×
  or less, of its recent median, or an event flag of 1).
- **Partial fill.** You can supply future values for *some* columns and let
  the rest be generated — pass a DataFrame with only the columns you know.
- **Spend what-ifs have two limits.** Spend above the highest value in your
  data cannot be estimated reliably: in testing, the default ensemble
  understated a linear effect by about a third at 1.5× the highest past spend
  and about half at 2×, and overstated one with diminishing returns. And
  carry-over is not modelled, so a sustained change in spend is understated
  even within the historical range. See
  [`future_exog.md`](future_exog.md#spend-what-ifs) for the numbers.

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
    forecaster=forecaster,   # after evaluate() and forecast()
    history=df, date_column="date", target_column="orders",
    output_dir="output", prefix="forecast",
    save_plots=True,     # set False for CSVs only
)
print(paths)
```

`forecaster=` fills in everything the files need: the ensembles, each
model's and each rule's scores, `mape_threshold`, the selection metric and
the seasonal-naive baseline. Any of `ensemble`, `per_model_mape`,
`per_model_validation_mape`, `ensemble_mape` and `mape_threshold` passed
explicitly replaces the forecaster's value; an explicit
`per_model_validation_mape` also makes the best model be chosen on those
MAPEs, even on a run that selected on WAPE. The export works without
`forecaster=` as before, just without the WAPE columns, the baseline row and
the error correlation.

This writes `final_ensemble_<method>` — **one per combination rule**, so
asking for all three writes three of them — plus `final_best_model`,
`per_model`, `rejected_models` and `model_summary`, and, with `forecaster=`
and two or more models, `error_correlation`.

**How the best single model is chosen.** `final_best_model` is the model with
the lowest **validation** score among those within `mape_threshold` on
validation *and* on test. Test is only a check that rules a model out, never
what it is ranked on: ranking on test was measured to pick worse, and the
second check stops a model that looked fine on validation but failed on
test. Because test helped decide, that model's test score is slightly
optimistic, and `model_summary`'s `best_model_check` column says so, along
with why every other model was not chosen. If no model passes, the file is
written empty with a warning naming each model's scores; the ensemble files
are unaffected. When the forecaster selected on WAPE, the best model is
chosen on WAPE too, so it agrees with the models the ensemble screened.
Without `forecaster=`, pass `per_model_validation_mape` (required for the
choice) and your forecaster's `mape_threshold`.

Unless `save_plots=False`, it also writes, **per combination rule**, an
overlay (`forecast_overlay_<method>.png`: every model, the history, and that
rule) and a dedicated ensemble figure (`forecast_ensemble_<method>.png`),
alongside one figure per model (`forecast_<model>.png`). Five models and three
rules therefore give three overlays and eight individual figures. The method
suffix is present even for a single rule, so a downstream job can glob a
stable pattern.

`model_summary` carries `validation_mape` and `mape` side by side, so the
selection-vs-reported gap is visible per model, plus `best_model_check`.
After those, in this order: `validation_wape` and `wape`;
`zero_actuals_excluded`, the test rows MAPE left out; `beats_naive`, whether
the row's test WAPE is lower than the seasonal-naive baseline's; and
`selection_metric`. The baseline itself is the last row, `seasonal_naive`,
with status `baseline`. In `rejected_models`, the `mape` column holds the
score the model was screened on, so its WAPE when `selection_metric` is
`"wape"`; the added `selection_metric` column and the `reason` text say
which. Its `ensemble_weight` column
is deliberately blank for weightless rules such as the default trimmed mean —
there is no weight, and filling in an equal share would imply an averaging
that did not happen. The combined rows take each rule's test scores from
`evaluate()`. Without `forecaster=`, `ensemble_mape` is the fallback for the
combined row and is used only when a single method was requested; with
several methods those rows are left blank rather than repeating one score
against three different series.

## Plotting directly

If you want figures without exporting, the functions in
`horizon_forecaster.plotting` draw and return without calling `plt.show()`,
and all accept the `{method: EnsembleResult}` dict directly:

```python
from horizon_forecaster.plotting import (
    plot_evaluation,
    plot_evaluation_per_model,
    plot_model_comparison,
)

# One chart per combination rule: {method: Axes}, always a dict.
axes = plot_evaluation(results, forecaster.evaluate_ensemble)
axes["trimmed"]        # every model + the actuals + the trimmed ensemble
axes["inverse_mape"]   # the same models, against the inverse-MAPE ensemble

# One figure per model, plus one per rule that formed: {label: Figure}.
figures = plot_evaluation_per_model(results, forecaster.evaluate_ensemble)
figures["sarimax"]
figures["ensemble_trimmed"]

# All rules on one Axes -- comparing them is what a bar chart is for. Drawn on the
# metric the run selected with, next to the seasonal-naive baseline.
plot_model_comparison(
    results,
    forecaster.evaluate_ensemble,
    metric=forecaster.selection_metric,
    baseline=forecaster.naive_baseline,
)
```

`plot_evaluation` and `plot_forecast` draw **one chart per combination rule**,
each against the same models and actuals, and return a dict keyed by method —
always a dict, even for a single rule, and keyed `"models"` when no ensemble
was given. Overlaying several rules on one time axis would bury the per-model
lines the chart exists to show, which is why they are separate charts rather
than one crowded one. Passing `ax=` is therefore only valid when a single rule
will be drawn; supplying it for several raises `InvalidConfigError` rather than
silently dropping the rest.

`plot_model_comparison` is the exception: comparing rules head-to-head is
exactly what a bar chart is for, so every rule shares one Axes. Its bars are
**test-window** scores, not the validation scores the models were selected
on: MAPE by default, or WAPE with `metric="wape"`. Pass
`forecaster.selection_metric` so the bars rank the models on the score the run
used, and `baseline=forecaster.naive_baseline` for a grey seasonal-naive bar.
`save_evaluation_plots()` takes the same two arguments. The other charts label
each line with both MAPE and WAPE.

## Important: the ensemble is not guaranteed to win

A combined ensemble can score worse than the single best model — this is a
real property of combination, not a bug. Always compare each ensemble's
`.mape` against each model's own `results[name].mape` (or `.wape` against
`.wape` on a run that selected on WAPE) before choosing which output to trust. Both a `final_ensemble_<method>` and a `final_best_model` CSV
are exported for exactly this reason. Prefer the ensemble when in doubt: in
testing on 19 real series, no score picked the model that then did best on
test more than about 40% of the time, which is the case the trimmed mean
exists for — it does not need to know which model that is.
