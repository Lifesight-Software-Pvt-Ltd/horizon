# Ensemble Methods

How `HorizonForecaster` combines multiple models' predictions into one
series. This is a reference for how to use and configure it; for the design
decisions and how they were verified, see [`design_notes.md`](design_notes.md) §3.

## What it does

`evaluate()` and `forecast()` already run every requested model
independently and return a `Dict[str, EvaluationResult]`/
`Dict[str, ForecastResult]` keyed by model name. On top of that, each call also builds a combined **ensemble**
prediction from whichever models are admitted, and exposes it as a side
effect on the `HorizonForecaster` instance:

```python
forecaster = HorizonForecaster(df, date_column="date", target_column="revenue", frequency="daily")

results = forecaster.evaluate()
ensembles = forecaster.evaluate_ensemble        # Dict[str, EnsembleResult] keyed by method
ensemble = ensembles["trimmed"]                 # the default rule

forecast_results = forecaster.forecast(horizon=30)
forecast_ensembles = forecaster.forecast_ensemble   # same shape, for the future horizon
```

Both `evaluate_ensemble` and `forecast_ensemble` are a
`Dict[str, EnsembleResult]` keyed by combination method — **always a dict**,
even when a single method was requested, matching the convention the
per-model returns already follow.

## Configuring it

Two constructor parameters, both optional:

```python
HorizonForecaster(
    df, date_column="date", target_column="revenue", frequency="daily",
    mape_threshold=40.0,          # default
    ensemble_method="trimmed",    # default; also "inverse_mape", "equal",
                                  # a list of names, or None to build all three
)
```

- **`mape_threshold`** (0.01–100, default `40`, or `None`): a
  **degenerate-fit filter** applied to each model's **validation** MAPE — the score it was
  selected on, not the test score it reports. A model scoring above it is
  excluded from the combination and reported in `rejected_models`. This is an
  **upper bound only** — an unusually low MAPE is never rejected. Pass `None`
  to admit every model.

  Treat it as a guard against degenerate fits rather than an accuracy screen:
  the default combination rule already discards the worst member on every
  period, so screening hard *and* trimming spends the same robustness budget
  twice — five members screened down to three leaves the median. The `0.01`
  and `100` limits are fixed internally and only constrain the value you are
  allowed to pass; the `0.01` floor rejects a threshold of `0`, which would
  admit only a literally perfect model.

- **`ensemble_method`**: which combination rule to apply.
  - `"trimmed"` (**the default**) sorts the members each period, discards the
    highest and the lowest, and averages the rest. It has no weights at all.
  - `"inverse_mape"` weights by `1 / validation MAPE`, normalized to sum to 1,
    so a lower-MAPE model gets proportionally more say.
  - `"equal"` gives every admitted model the same share.

  Pass a list to build several, or `None` to build all three. Combination is
  arithmetic on predictions that already exist — the expensive part (fitting)
  has long since happened — so extra rules are close to free.

### The trimmed rule's degradation ladder

Three members is where the arithmetic stops working, not a tuned threshold:
remove the highest and the lowest of two values and nothing is left.

| members | what runs | `rule_applied` | `degraded` |
|---|---|---|---|
| 5+ | trim the high and the low, average the rest | `"trimmed"` | `False` |
| 4 | same — and the mean of the middle two **is** the median, so it is named as such | `"median"` | `False` |
| 3 | same; only the middle value survives | `"trimmed"` | `False` |
| 2 | plain mean — trimming is undefined | `"mean"` | `True` |
| 1 | that member's own forecast | `"single"` | `True` |
| 0 | no combination at all (`combined is None`) | `"none"` | `False` |

The weighted rules have no such boundary — a weighted average of two, or of
one, is perfectly well defined — so they never report `degraded`.

## `EnsembleResult`

Each value in the dict is an `EnsembleResult` (`evaluation/ensemble.py`):

| Field | Meaning |
|---|---|
| `combined` | `DataFrame["ds", "yhat"]` — the combined prediction, a genuinely new series. `None` if no member was available to combine. |
| `method` | The combination rule that was **requested**. |
| `rule_applied` | The rule that actually **ran** — see the ladder above. Branch on this, not on `method`, when what matters is the statistic produced. |
| `members` | The models that fed the combination, in the order they were combined. |
| `degraded` | `True` when `rule_applied` is a fallback because too few members survived for trimming to be defined. Always `False` for the weighted rules. |
| `weights` | `{"prophet": 0.7, "catboost": 0.3}` — normalized weights per member. **`None`** for weightless rules such as the default `trimmed`: there is no weight, and reporting an equal split would imply an averaging that did not happen. |
| `trim_shares` | Trimming only: a per-member DataFrame of how often each member was discarded as the period's high or low. More diagnostically useful than weights — it identifies which model misbehaves, and on how much of the horizon. `None` for the weighted rules and for the degraded fallbacks. |
| `rejected_models` | `{"catboost": ThresholdRejectionError(...)}` — one entry per model excluded before combination, with its MAPE and the threshold it exceeded (if any). |
| `mape` | The ensemble's own `MapeResult`, scored against real actuals. Only set in the `evaluate()` context (held-out actuals exist); always `None` for `forecast_ensemble`, since there is no ground truth for the future yet. |

A model in `rejected_models` is **not the same thing** as a model in
`evaluate_errors`/`forecast_errors`. A rejected model fit and predicted
successfully; it was simply not admitted to the combination. A model in
`evaluate_errors` failed to fit or predict at all. These are reported
completely separately, and rejection never raises an exception, even if every
model gets excluded — `evaluate()` still returns everyone's individual results
normally; only that method's `combined` comes back `None`.

Both are also warned about, once per `evaluate()` or `forecast()` call: models
that failed (with the cause), each rule that lost requested models (and why:
failed, above `mape_threshold`, or no validation MAPE), and, separately, any
rule that ended up with no members and so no ensemble forecast.

### Exclusion is per method

Two independent things can exclude a model, and they do not apply to every
rule:

1. **`mape_threshold`**, when set. Applies to every rule.
2. **A NaN validation MAPE** — typically because every actual in the
   validation window was zero (sparse conversions, installs), or because the
   model could not be scored on that window at all. (When the forecaster
   selects on WAPE, read "MAPE" here and below as WAPE: everything that
   screens, weights or ranks uses `selection_metric`, and WAPE is NaN only
   when the whole window is zero.) This excludes the model
   from both **weighted** rules, `inverse_mape` and `equal`. `inverse_mape`
   literally cannot compute `1 / NaN`; `equal` could, but a model with no
   validation MAPE was never screened against `mape_threshold` either, and the
   weighted rules give every member a full, untrimmed share of the result — so
   admitting an unvetted model there would let it move the number without any
   check on it at all. `trimmed` keeps it, because trimming itself bounds how
   far any single member can move the result: the model genuinely learned the
   series, only the metric broke.

That is why the rejected-models export carries an `ensemble_method` column —
the same model can be dropped by one rule and kept by another in the same run.

## `forecast()` reuses the selection MAPE

`forecast()` has no held-out actuals of its own to score accuracy against — a
true future forecast cannot be checked against anything yet. So its screening
and weighting reuse the **validation** MAPE established when each model's
hyperparameters were selected. That selection has to happen anyway —
`forecast()` cannot fit without knowing which parameters won — so the scores
come for free and are cached, whether or not `evaluate()` was ever called.

`forecast()` does **not** run `evaluate()` internally — it needs only the
cached selection, not a test-window prediction it would discard. The
test-window MAPE that `evaluate()` reports is never used for screening or
weighting.

## Known limitation: the ensemble is not automatically better

Confirmed by direct testing, not assumed: a combined ensemble can score
**worse** than the single best individual model. On a synthetic test, Prophet
alone scored 1.31% MAPE, CatBoost alone scored 12.77%, and the
inverse-MAPE-weighted ensemble of the two (weighted ~91%/9% toward Prophet)
scored 1.58% — worse than Prophet alone. Weights are static across the whole
forecast horizon, while each model's real per-period error is not; an
otherwise-strong ensemble can still be pulled off on exactly the periods
where the weaker model is wrong, in a way one fixed weight per model cannot
correct for.

Trimming is considerably more robust to this specific failure — it discards
the offending member on exactly the periods where it offends, rather than
carrying one fixed weight across the horizon — which is why it is the default.
It is still not a guarantee.

This is exactly why each ensemble's `mape` is computed and exposed directly,
rather than the ensemble being assumed better. Always compare it against each
individual model's own test MAPE (`results[model].mape`) before deciding to
use a combined output over the best single model.

## Scope

This covers the ensemble *combination* logic only. The "plots + CSV" output
side lives in `plotting.py` and `exporters.py`, which consume the
`EnsembleResult` this doc describes. The ensemble logic itself is written
generically against whatever is registered in `forecaster.py`'s
`MODEL_REGISTRY`: all five registered models feed it with **no model-specific
code in this module**.

Uncertainty bounds are deliberately an individual-model concept and are never
propagated into a combined output: averaging confidence/credible intervals
across statistically different models (trend simulation vs. posterior vs.
Gaussian approximation) has no coherent interpretation, so this module
operates on the `yhat` column only.
