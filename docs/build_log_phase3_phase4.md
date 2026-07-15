# Horizon Forecaster — Build Log: Phase 3 & Phase 4

This document records both phases in one file (matching the Phase 0/1
precedent) since Phase 4 builds directly on top of Phase 3's multi-model
orchestrator and reuses several of its design decisions — reading them
together tells one continuous story rather than two disconnected ones.

---

## Phase 3 — Multi-Model Orchestration & CatBoost

### Goal

Extend the Phase 2 single-hardcoded-Prophet orchestrator into a
registry-driven system that can run any number of models in one
`evaluate()`/`forecast()` call, and add the second real model
(CatBoost) to prove the architecture actually generalizes beyond
Prophet — not just in theory, but against real data.

### What was done

- `CatBoostAdapter` (`data_adapter/catboost_adapter.py`) and
  `CatBoostWrapper` (`models/catboost_wrapper.py`), implementing the
  same `BaseAdapter`/`BaseModel` contracts Prophet already used.
- `catboost:` block in `config/default_params.yaml`, an 18-candidate
  hyperparameter grid (16 tuned + 2 safe low-capacity fallbacks),
  mirroring Prophet's config schema exactly.
- `HorizonForecaster` generalized from a single hardcoded Prophet model
  into a `MODEL_REGISTRY`-driven orchestrator. New `models=` and
  `model_params=` constructor parameters. Adding a future model should
  require exactly one new `MODEL_REGISTRY` entry, nothing else.
- Partial-failure handling: one requested model failing no longer
  aborts the call for the others. Failures land in
  `evaluate_errors`/`forecast_errors` (reset fresh per call);
  `AllModelsRejectedError` only fires when every requested model fails.
- `BaseAdapter.__init__` gained a uniform `frequency` parameter so the
  orchestrator can construct every adapter identically, with zero
  per-model special-casing.

### The real-data diagnostic: why CatBoost's first design was wrong

CatBoost's original feature set was lag(1/2/3/7/14) + rolling(7/14)
target-history features, frozen at train-end and broadcast unchanged
across the whole forecast horizon — a deliberate choice, since
`CatBoostWrapper.predict()` is one batched call, not a per-step
recursive loop, so there was no natural point to update lag features
mid-horizon.

Tested against a real revenue dataset, this produced a dramatically
worse MAPE than Prophet (39.95% vs. 18.68%). Two hypotheses were formed
and directly disproven with the same dataset before landing on the real
cause:

1. **Tree-extrapolation ceiling** (a genuine, separate CatBoost
   limitation — see `catboost_wrapper.py`'s docstring) — disproven: 0
   of 144 test rows exceeded the training-set max, so the model wasn't
   extrapolating past anything it hadn't seen.
2. **Monotonic staleness growing across the horizon** — disproven: the
   second half of the 144-day test window actually scored *better*
   (27.3% MAPE) than the first half (52.6%), the opposite of what a
   simple "staleness compounds over time" theory would predict.

The real cause, confirmed by breaking the test set down by known
promotional/demand-shift flags: **December_Dip** scored 67–90% MAPE
across two separate runs, vastly worse than the 34–40% baseline on
ordinary days. A frozen snapshot of "recent history" has no way to see
a demand dip or promotional spike coming — it keeps predicting the
pre-shift baseline for the entire horizon once the snapshot is taken
right before a known volatile period.

This was independently confirmed against a working reference notebook
(a separate, dedicated CatBoost pipeline on the same dataset, ~16%
MAPE, no such failure mode) that used a **trend index + cyclical
seasonality** feature set instead of lag/rolling features.

### The redesign

`CatBoostAdapter` was rebuilt from scratch around:

- `time_idx`: an explicit linear trend index (absolute day/week offset
  from a fixed `_reference_date`), not a row-position counter — stays
  continuous across the train/future boundary and is gap-robust.
- Cyclical yearly/half-yearly sin/cos encodings (daily data also gets
  weekly/monthly), plus plain calendar fields.
- Independent/exogenous columns passed through as-is.

Every feature is computed identically by `transform_train()` and
`transform_future()`, purely from dates — none depend on the target's
own recent history, so nothing needs "freezing" for a multi-step
horizon. `CatBoostWrapper` was updated to match: no categorical
features declared at all (the working reference notebook used none
either), and data-driven non-negative prediction clipping (only
applied when the training target was itself always non-negative).

### How this was verified

- Sandbox-verified against real Prophet + CatBoost (mirroring the real
  repo into an isolated environment) before touching the real repo, as
  established practice.
- A synthetic "December Dip" simulation: dip-flagged MAPE dropped from
  a catastrophic-blowout range under the old design to 22.59% under
  the new one.
- Full local pytest run (not just sandbox): **209 passed**, confirming
  every touched file (adapter, wrapper, config, three test files) plus
  every untouched Prophet file still worked together.
- **Confirmed on the real dataset the failure was diagnosed on**, at
  90% training size: overall MAPE 18.86% (vs. Prophet's 18.68%,
  essentially matched, down from a pre-fix 39.95–45.47%); flag-segment
  breakdown showed December_Dip at 14.59% MAPE — down from 67–90%
  pre-fix, now the *best*-performing segment rather than the worst —
  with BFCM at 21.97% and all other days at 19.82%, all in a normal,
  tight range.

---

## Phase 4 — Ensemble Weighting

### Goal

Combine each registered model's predictions into one weighted ensemble
output, per the project's original spec (README: "Automatic model
selection via MAPE threshold-based ensemble weighting", "Transparent
reporting of rejected models with their MAPE scores"). Most of the
spec was already implied by earlier phases before this work started:
`utils.validate_mape_threshold_range()` (Phase 1, unused until now),
`exceptions.ThresholdRejectionError` (Phase 1, unused until now),
`evaluation/ensemble.py` (empty stub file), and an explicit design
constraint already documented in `models/base.py` — ensemble
combination operates on the `yhat` column only, uncertainty bounds are
never propagated into it.

### What was done

- `evaluation/ensemble.py`: `build_ensemble()`, a pure function (no
  dependency on `forecaster.py`/`HorizonForecaster`/any model or
  adapter class, mirroring `evaluation/metrics.py`'s standalone
  `mape()`), plus `EnsembleResult`. Filters out any model whose MAPE
  falls outside `[min_mape, max_mape]` — on *either* side, not just
  above the max, per `validate_mape_threshold_range`'s existing
  symmetric-range design — then combines the survivors' `yhat` values
  using one of two weighting strategies: `inverse_mape` (lower-MAPE
  models get proportionally more weight) or `equal` (every surviving
  model weighted equally).
- `HorizonForecaster` gained `min_mape`/`max_mape` (default `1.0`/
  `100.0`, deliberately permissive) and `ensemble_weighting` (default
  `"inverse_mape"`) constructor parameters. `evaluate()`/`forecast()`
  keep their exact Phase 3 return shape (`Dict[str, EvaluationResult]`/
  `Dict[str, ForecastResult]`, unchanged), and additionally populate
  `self.evaluate_ensemble`/`self.forecast_ensemble` as a side effect —
  same pattern as `evaluate_errors`/`forecast_errors`.
- `forecast()` always reuses `evaluate()`'s per-model MAPE for its own
  ensemble weights (it has no held-out actuals of its own to score
  against). If `evaluate()` hasn't run yet this session, `forecast()`
  now runs it internally once, with a `UserWarning` naming the added
  cost, and caches the result (`self._evaluate_mape_cache`) so it never
  reruns on subsequent `forecast()` calls.

### Design decisions made during implementation

- **Reused `ThresholdRejectionError`/`AllModelsRejectedError` instead
  of adding a new exception.** The original plan proposed a new
  `EnsembleRejectedError`; reading `exceptions.py` before writing any
  code turned up that `ThresholdRejectionError` (per-model MAPE +
  allowed range) already existed from Phase 1, unused, built for
  exactly this. No new exception type was needed.
- **Threshold rejection never raises, even when every model is
  rejected.** `build_ensemble()` returns an `EnsembleResult` with
  `combined=None` and empty `weights` rather than raising
  `AllModelsRejectedError` — a model that's too inaccurate for the
  ensemble is a different situation from a model that failed to fit at
  all (Phase 3's `AllModelsRejectedError`). The per-model results are
  still valid and still returned by `evaluate()`/`forecast()`; only
  the combination step has nothing to combine. Discarding valid
  per-model results just because the *ensemble* couldn't be formed
  would be a strictly worse outcome for the caller.
- **`evaluate()`'s ensemble gets its own real MAPE, scored against
  actuals** — not assumed to automatically beat the best individual
  model. This surfaced a genuine, confirmed-not-theoretical finding
  (see below).
- **`forecast()` triggers an internal `evaluate()` call, not a hard
  requirement to call it first**, per explicit project-owner decision
  — but always with a visible warning, consistent with Phase 2's
  established principle that cost should be surfaced transparently,
  never silently absorbed.

### Known limitation, confirmed by direct testing

A weighted ensemble is **not** guaranteed to beat the single best
individual model's MAPE, even with inverse-MAPE weighting. On a
synthetic test dataset: Prophet alone scored 1.31% MAPE, CatBoost
alone scored 12.77%, and the ~91%/9%-weighted ensemble of the two
scored **1.58%** — worse than Prophet alone, despite Prophet carrying
the overwhelming majority of the weight. Root cause: weights are
static across the whole horizon, while each model's real per-period
error is not — an otherwise-strong ensemble can still get dragged down
on the specific periods where the weaker model happens to be wrong, in
a way a single fixed weight per model can't correct for. This is
documented directly in `evaluation/ensemble.py`'s docstring (not
silently absorbed), and is exactly why `EnsembleResult.mape` exists —
always check it against each individual model's own MAPE before
trusting the ensemble over the best single model.

### How this was verified

- Sandbox-verified against real Prophet + CatBoost (not mocks): both
  weighting strategies, threshold rejection on both sides of the
  range, the zero-survivor no-raise path, NaN-MAPE handling, and the
  `forecast()`-reuses-`evaluate()`-weights caching/warning behavior
  (including that a *second* `forecast()` call does not re-warn or
  re-fit).
- Confirmed the "refit exactly once, even across repeated `forecast()`
  calls" guarantee from Phase 2 still holds with the new internal
  `evaluate()` call added in front of it.
- 22 new tests: 14 in `tests/test_ensemble.py` (pure `build_ensemble()`
  unit tests — weighting math, combination correctness, threshold
  rejection on both sides of the range, NaN handling, misaligned-`ds`
  errors, ensemble MAPE scoring) and 8 in `tests/test_forecaster.py`
  (constructor validation, `evaluate_ensemble`/`forecast_ensemble`
  population, weight-reuse and caching behavior). All 34 pre-existing
  constructor/config/partial-failure tests in `test_forecaster.py`
  re-run and confirmed still passing alongside the new ones.
- The full existing grid-search-heavy test suite (52 tests in
  `test_forecaster.py`, mostly multi-candidate real Prophet/CatBoost
  fits) exceeds this sandbox's execution ceiling to run in full, same
  constraint as Phase 3 — a full local `pytest` run is the way to get
  definitive, complete confirmation, same as Phase 3's 209-passed run.

### Deliberately out of scope for Phase 4

`plotting.py`/`exporters.py` remain empty stub files — this phase only
builds the ensemble *combination* logic, not the "plots + CSV" output
side of the original README spec. SARIMAX/PyMC/NeuralProphet are still
unregistered stub adapters/wrappers; `build_ensemble()` is written
generically against whatever's in `MODEL_REGISTRY`, so no ensemble
code changes should be needed when they're added, but this is untested
beyond the current 2-model (Prophet/CatBoost) case.
