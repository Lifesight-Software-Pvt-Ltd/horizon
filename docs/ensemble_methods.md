# Ensemble Methods

How `HorizonForecaster` combines multiple models' predictions into one
weighted ensemble (Phase 4). This is a reference for how to use and
configure it; for the design decisions and how it was verified, see
`docs/build_log_phase3-4-5.md`.

## What it does

`evaluate()` and `forecast()` already run every requested model
independently and return a `Dict[str, EvaluationResult]`/
`Dict[str, ForecastResult]` keyed by model name — that behavior is
unchanged. On top of that, each call also builds a combined, weighted
**ensemble** prediction from whichever models pass an accuracy check, and
exposes it as a side effect on the `HorizonForecaster` instance:

```python
forecaster = HorizonForecaster(df, date_column="date", target_column="revenue", frequency="daily")

results = forecaster.evaluate()
ensemble = forecaster.evaluate_ensemble        # EnsembleResult, or None if evaluate() itself found nothing

forecast_results = forecaster.forecast(horizon=30)
forecast_ensemble = forecaster.forecast_ensemble   # EnsembleResult for the future horizon
```

## Configuring it

Two constructor parameters, both optional:

```python
HorizonForecaster(
    df, date_column="date", target_column="revenue", frequency="daily",
    mape_threshold=40.0,                # default
    ensemble_weighting="inverse_mape",  # default; "equal" is the other option
)
```

- **`mape_threshold`** (0.01–100, default `40`): the accuracy cutoff a
  model's MAPE must be **at or below** to be included in the ensemble.
  A model scoring above it is excluded and reported in
  `rejected_models`. This is an **upper bound only** — an unusually low
  MAPE is never rejected. Lower it to filter more aggressively; raise it
  (up to 100) to be more permissive. The `0.01` and `100` limits are
  fixed internally and only constrain the value you're allowed to pass,
  not which models qualify (the `0.01` floor rejects a threshold of `0`,
  which would admit only a literally perfect model).
- **`ensemble_weighting`**: `"inverse_mape"` (default) gives lower-MAPE
  models proportionally more weight (`weight ∝ 1 / MAPE`, normalized to
  sum to 1). `"equal"` gives every surviving model an equal share
  regardless of its MAPE.

## `EnsembleResult`

Both `evaluate_ensemble` and `forecast_ensemble` are an `EnsembleResult`
(`evaluation/ensemble.py`):

| Field | Meaning |
|---|---|
| `combined` | `DataFrame["ds", "yhat"]` — the weighted combination. `None` if no model survived the threshold. |
| `weights` | `{"prophet": 0.7, "catboost": 0.3}` — normalized weights for surviving models. Empty if `combined` is `None`. |
| `rejected_models` | `{"catboost": ThresholdRejectionError(...)}` — one entry per excluded model, with its actual MAPE and the threshold it exceeded. |
| `weighting` | Which strategy was used. |
| `mape` | The ensemble's own `MapeResult`, scored against real actuals. Only set in the `evaluate()` context (held-out actuals exist); always `None` for `forecast_ensemble`, since there's no ground truth for the future yet. |

A model in `rejected_models` is **not the same thing** as a model in
`evaluate_errors`/`forecast_errors`. A rejected model fit and predicted
successfully — it just wasn't accurate enough to trust in the combined
output. A model in `evaluate_errors` failed to fit or predict at all.
These are reported completely separately, and threshold rejection never
raises an exception, even if every model gets excluded — `evaluate()`
still returns everyone's individual results normally; only
`evaluate_ensemble.combined` comes back `None`.

## `forecast()` reuses `evaluate()`'s weights

`forecast()` has no held-out actuals of its own to score accuracy
against — a true future forecast can't be checked against anything yet.
So `forecast_ensemble`'s weights (and which models pass the threshold)
always come from `evaluate()`'s MAPE. If `evaluate()` hasn't been called
yet this session, `forecast()` runs it internally once — you'll see a
`UserWarning` naming the added cost — and caches the result, so it only
happens once no matter how many times `forecast()` is called afterward.
Call `evaluate()` yourself first if you want that cost to happen
explicitly rather than as a side effect of the first `forecast()` call.

## Known limitation: the ensemble is not automatically better

Confirmed by direct testing, not assumed: a weighted ensemble can score
**worse** than the single best individual model, even with
`inverse_mape` weighting. On a synthetic test, Prophet alone scored
1.31% MAPE, CatBoost alone scored 12.77%, and the ensemble of the two
(weighted ~91%/9% toward Prophet) scored 1.58% — worse than Prophet
alone. Weights are static across the whole forecast horizon, while each
model's real per-period error isn't; an otherwise-strong ensemble can
still get dragged down on exactly the periods where the weaker model is
wrong, in a way one fixed weight per model can't correct for.

This is exactly why `evaluate_ensemble.mape` is computed and exposed
directly, rather than the ensemble being assumed better. Always compare
it against each individual model's own MAPE (`results[model].mape`)
before deciding to use the ensemble's output over the best single model.

## Current scope

This covers the ensemble *combination* logic only. The "plots + CSV"
output side of the original project spec now lives in `plotting.py` and
`exporters.py` (Phase 5), which consume the `EnsembleResult` this doc
describes. The ensemble logic itself is written generically against
whatever's registered in `forecaster.py`'s `MODEL_REGISTRY` — and that
generality has now been confirmed in practice: SARIMAX (Phase 6 Part 1)
and NeuralProphet (Part 2) were both added with **no changes to this
module**, and the ensemble has been exercised end-to-end on a real
4-model run. PyMC (Part 3) is the one model still to come.
