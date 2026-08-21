# Ensemble methods

How `HorizonForecaster` combines multiple models' predictions into one series.
This is a reference for how to use and configure it; for the design decisions
and how they were verified, see the build logs in `docs/`.

## What it does

`evaluate()` and `forecast()` run every requested model independently and
return a `Dict[str, EvaluationResult]` / `Dict[str, ForecastResult]` keyed by
model name — that behaviour is unchanged. On top of that, each call also
combines whichever models clear the garbage filter, and exposes the result as a
side effect on the `HorizonForecaster` instance:

```python
forecaster = HorizonForecaster(df, date_column="date", target_column="revenue", frequency="daily")

results = forecaster.evaluate()
ensembles = forecaster.evaluate_ensemble        # Dict[str, EnsembleResult], keyed by method
trimmed = ensembles["trimmed"]

forecast_results = forecaster.forecast(horizon=30)
future = forecaster.forecast_ensemble["trimmed"]   # same shape, for the future horizon
```

`evaluate_ensemble` / `forecast_ensemble` are **always a dict**, even when a
single method was requested — matching the convention the per-model returns
already follow, so the shape of what comes back never depends on how many
things you asked for. They are `None` only when there was nothing to combine at
all (every model failed outright).

## Configuring it

Two constructor parameters, both optional:

```python
HorizonForecaster(
    df, date_column="date", target_column="revenue", frequency="daily",
    mape_threshold=40.0,        # default; None disables screening entirely
    ensemble_method="trimmed",  # default; also "inverse_mape", "equal",
                                # a list of those, or None for all three
)
```

### `ensemble_method`

| Method | What it does | Weights? |
|---|---|---|
| `trimmed` *(default)* | Per period, sort the members, discard the highest and the lowest, and average the rest. | No |
| `inverse_mape` | Weight each member by `1 / validation MAPE`, normalised to sum to 1. | Yes |
| `equal` | Every surviving member gets the same share. | Yes |

Pass a list (`["trimmed", "equal"]`) to build several, or `None` to build all
three. That is cheaper than it sounds: by the time any of this runs the models
have already been fitted, so each additional rule costs one pass of arithmetic
over predictions that are already in memory. If you want to know how much the
choice of rule actually matters on your series, you should not have to re-run
the fits to find out.

**Why trimming is the default**, when the two weighted rules came first:
estimated weights have to be estimated *from* something, and everything
available to estimate them from is a finite sample of past error. A trimmed
mean consumes no such quantity at all — it cannot leak information from a
scoring window by construction, and it has no weights to get wrong. The
weighted rules are kept because they remain reasonable choices and some callers
want them, not because anything recommends them as a default.

### `mape_threshold`

`0.01`–`100`, default `40`, applied to each model's **validation** MAPE (never
the reported test MAPE — see below). A model scoring above it is excluded and
reported in `rejected_models`. This is an upper bound only; an unusually low
MAPE is never rejected.

Treat it as a **garbage filter** — a guard against degenerate fits, a model at
300% MAPE that should not get a vote — rather than an accuracy screen. The
trimmed rule already discards the worst member on every period, so screening
hard *and* trimming spends the same robustness budget twice: five members
screened down to three leaves you with the median. That is a legitimate choice,
but it should be a deliberate one.

Pass `None` to disable screening entirely and admit every model. Note that
`None` is not the same as `0`, which remains invalid: `0` would admit only a
literally perfect model, which is almost never what anybody means. The `0.01`
and `100` bounds are fixed internally and only constrain the value you may
pass, not which models qualify.

## Which MAPE is used where

This is the single most important thing to understand about the ensemble.
Horizon splits data into **three** chronological windows (see
[`quickstart.md`](quickstart.md)):

- **Screening and weighting read the VALIDATION MAPE** — the same window the
  hyperparameter grid search was scored on. That window is allowed to influence
  decisions.
- **The ensemble's own reported MAPE is measured on the TEST window**, against
  test actuals, which no selection path ever reads.

Collapsing those into one number is what allowed the pre-1.0 library to weight
an ensemble using the very scores it then reported. `build_ensemble()` takes
them as two separate arguments (`per_model_selection_mape` and `actuals`) so
they cannot be confused.

One consequence worth knowing: a model whose validation MAPE is `NaN`
(typically because every actual in the validation window was zero — sparse
conversions, installs) is dropped by the **weighted** rules, which cannot
compute a weight from it, but is **kept** by the trimmed rule, which needs no
MAPE at all. It genuinely learned the series; only the metric broke. This is
why `rejected_models` can differ between methods, and why the exported
rejected-models CSV has an `ensemble_method` column.

## `EnsembleResult`

| Field | Meaning |
|---|---|
| `combined` | `DataFrame["ds", "yhat"]` — the combined series, genuinely new (not any single model's own `yhat`). `None` if no member survived. |
| `method` | The rule that was **requested**. |
| `rule_applied` | The rule that actually **ran**: `"trimmed"`, `"median"`, `"mean"`, `"single"` or `"none"`. See the degradation ladder below. |
| `members` | The models that fed the combination, in the order they were combined. |
| `degraded` | `True` when `rule_applied` is a fallback rather than the rule you asked for. A two-member plain mean is a materially weaker output than a five-member trimmed mean, and should not look identical in a result object. |
| `weights` | Normalised weights per member, e.g. `{"prophet": 0.55, "catboost": 0.45}`. **`None` for weightless rules such as `trimmed`** — there is no weight, and reporting an equal split would imply an averaging that did not happen. |
| `trim_shares` | Trimming only: a per-member table of how often each member was discarded as the period's high or low (`trimmed_low`, `trimmed_high`, `trimmed_total`, `n_periods`, `trimmed_fraction`). `None` for the weighted rules. |
| `rejected_models` | `{"catboost": ThresholdRejectionError(...)}` — one entry per excluded model, with its validation MAPE and the threshold it exceeded. |
| `mape` | The ensemble's own `MapeResult`, scored against real actuals on the test window. Set only in the `evaluate()` context; always `None` for `forecast_ensemble`, since there is no ground truth for the future yet. |

`trim_shares` is usually more informative than weights ever were: it tells you
*which* model misbehaves and on how much of the horizon, rather than giving it
one static number for the whole span.

## The trimmed rule degrades, and says so

Trimming needs at least three members — remove the highest and the lowest of
two values and nothing is left to average. That is an arithmetic boundary, not
a tuned threshold, so rather than failing, the rule steps down and records what
it did:

| Members | `rule_applied` | `degraded` |
|---|---|---|
| 5 or more | `trimmed` (mean of the middle members) | `False` |
| exactly 4 | `median` — trimming the high and low of four leaves two values whose mean *is* the median | `False` |
| 3 | `trimmed` (the middle value) | `False` |
| 2 | `mean` — a plain average | `True` |
| 1 | `single` — that model's own forecast, unchanged | `True` |
| 0 | `none`, `combined is None` | `False` |

Naming the four-member case `median` is deliberate: someone comparing runs
across series should not have to recount members to know which statistic they
are looking at. The weighted rules have no such boundary — a weighted average
of two, or of one, is perfectly well defined — so they never report `degraded`.

## Rejection is not failure

A model in `rejected_models` is **not the same thing** as a model in
`evaluate_errors` / `forecast_errors`. A rejected model fit and predicted
successfully — it just was not admitted to the combination. A model in
`evaluate_errors` failed to fit or predict at all. These are reported
completely separately, and threshold rejection never raises, even if every
model is excluded: `evaluate()` still returns everyone's individual results
normally, and only `combined` comes back `None`.

## `forecast()` reuses the selection scores

`forecast()` has no held-out actuals of its own — a true future forecast cannot
be checked against anything yet. So any screening or weighting it needs comes
from the **validation** MAPE established when each model's hyperparameters were
selected.

That selection has to happen regardless (`forecast()` cannot fit without
knowing which parameters won), so the scores come for free and are cached,
whether or not you ever call `evaluate()`. Before 1.0 this method ran the whole
of `evaluate()` internally to obtain them — additionally paying for a
test-window prediction it then discarded — and warned about the cost. That
warning is gone, because the cost is.

## Uncertainty bounds are never combined

The module operates on the `yhat` column only. Averaging confidence/credible
intervals across statistically different model types (Prophet's trend
simulation vs. CatBoost's point estimate vs. PyMC's posterior) has no coherent
interpretation, so `yhat_lower`/`yhat_upper` are left on each model's own
result for individual-model plots and CSVs and never blended.

## Known limitation: the ensemble is not automatically better

Confirmed by direct testing, not assumed: a combined ensemble can score
**worse** than the single best individual model. On a synthetic test, Prophet
alone scored 1.31% MAPE, CatBoost alone scored 12.77%, and the inverse-MAPE
ensemble of the two (weighted ~91%/9% toward Prophet) scored 1.58% — worse than
Prophet alone. Weights are static across the whole horizon while each model's
real per-period error is not, so an otherwise-strong ensemble gets dragged down
on exactly the periods where the weaker model is wrong, in a way one fixed
weight per model cannot correct for.

Trimming is considerably more robust to *this specific* failure, because it
discards the offending member on precisely the periods where it offends rather
than applying one number to the whole span. It is still **not a guarantee** —
trimming can equally discard a good member that happens to be the period's
extreme, and on a small membership it degrades toward a plain average.

This is exactly why the ensemble's own MAPE is computed and exposed rather than
assumed. Always compare it against each individual model's own test MAPE
(`results[model].mape`) before choosing the combined output over the best
single model — and note that the exporter writes both a
`final_ensemble_<method>` and a `final_best_model` CSV so the choice can be
made per run rather than baked in.

## Scope

This covers the *combination* logic only. The output side lives in
`plotting.py` and `exporters.py`, which consume the `EnsembleResult` objects
described here. The combination logic itself is written generically against
whatever is registered in `forecaster.py`'s `MODEL_REGISTRY`; all five current
models were added without any change to this module.
