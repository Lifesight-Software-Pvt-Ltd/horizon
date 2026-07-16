# Horizon Forecaster — Build Log: Phase 3, Phase 4 & Phase 5

This document records these phases in one file (matching the Phase 0/1
precedent) since Phase 4 builds directly on top of Phase 3's multi-model
orchestrator and reuses several of its design decisions — reading them
together tells one continuous story rather than two disconnected ones.
Phase 5 (the output layer) is small and consumes Phase 4's result shapes
directly. 

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
`utils.validate_mape_threshold()` (Phase 1, unused until now),
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
  exceeds `mape_threshold` — an upper cutoff only, so an unusually low
  MAPE is never rejected — then combines the survivors' `yhat` values
  using one of two weighting strategies: `inverse_mape` (lower-MAPE
  models get proportionally more weight) or `equal` (every surviving
  model weighted equally).
- `HorizonForecaster` gained `mape_threshold` (default `40`; `0.01` and
  `100` are fixed internal bounds on the value the user can pass) and
  `ensemble_weighting` (default `"inverse_mape"`) constructor
  parameters. `evaluate()`/`forecast()`
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

`plotting.py`/`exporters.py` remain empty stub files at the end of
Phase 4 — that phase only builds the ensemble *combination* logic, not
the "plots + CSV" output side of the original README spec (that output
side is Phase 5, below). SARIMAX/PyMC/NeuralProphet are still
unregistered stub adapters/wrappers; `build_ensemble()` is written
generically against whatever's in `MODEL_REGISTRY`, so no ensemble
code changes should be needed when they're added, but this is untested
beyond the current 2-model (Prophet/CatBoost) case.

---

## Phase 5 — Output Layer (plotting + exporters)

### Goal

Turn the shapes Phase 4 stabilized — the per-model `EvaluationResult`/
`ForecastResult` dicts and the `EnsembleResult` (combined series,
weights, rejected-models map, optional ensemble MAPE) — into the two
consumer-facing outputs from the README spec: charts and CSVs. Nothing
about the forecaster's own logic changes; this layer only *reads*
results that already exist.

### Why now, and why so small

Phase 4 is the reason this could be built quickly and safely: the
result contracts it locked in (documented column sets on
`test_predictions`/`forecast`/`combined`, and the
`ThresholdRejectionError` fields on `rejected_models`) are exactly what
the output layer needs. Because those shapes were already settled and
exercised by real Prophet+CatBoost runs, `plotting.py` and
`exporters.py` are thin, pure translation layers with no new domain
logic — hence folding this log into the Phase 3&4 file rather than
starting a new one.

### What was built

- **`plotting.py`** — two chart families at each stage. Overlay charts
  `plot_evaluation()` (actuals once + one line per model with its MAPE in
  the legend + shaded bands where a model has them + optional red-dashed
  ensemble overlay carrying its own MAPE) and `plot_forecast()`
  (per-model forecast, optional history behind it, optional ensemble
  overlay) are compact but crowd as models are added or when models
  agree and their lines coincide. Faceted grids
  `plot_evaluation_grid()`/`plot_forecast_grid()` (added after real-use
  feedback that the overlay was already busy at 2 models) give each model
  and the ensemble its own panel via shared-axis small multiples, staying
  legible at any model count; they return a `Figure` rather than a single
  `Axes`. `plot_model_comparison()` is a MAPE bar chart making the
  "ensemble is not automatically better than the best single model" fact
  from Phase 4 visible at a glance. All of them flag the "ensemble
  requested but none formed" case (every model above `mape_threshold`)
  with a visible red note, while still drawing the individual-model
  results -- rejection from the blend is a trust signal, not a reason to
  hide a real forecast, so the fix is loud labeling, not suppression.
- **`exporters.py`** — `export_forecast_csvs()` writing three files
  (final/blended, per-model long table, rejected-models table) plus the
  three pure `build_*_table()` functions behind them. `plotting.py` also
  gained `save_evaluation_plots()`/`save_forecast_plots()` (render the
  charts and write them as PNGs, closing each figure), and
  `exporters.py` a one-call `export_forecast()` bundle that writes the
  CSVs and the forecast plots into one folder. The bundle imports
  matplotlib lazily, so `export_forecast_csvs()` on its own stays
  matplotlib-free (verified: importing `exporters` does not load
  matplotlib).

### Key design decisions (carried from the Phase 1/4 house style)

- **Pure and import-light, like `evaluation/metrics.py`/`ensemble.py`.**
  Neither module imports `forecaster.py` or any model/adapter class;
  result objects are duck-typed and their type hints are
  `TYPE_CHECKING`-only. Verified directly: importing `plotting` +
  `exporters` pulls in neither Prophet, CatBoost, nor
  `forecaster` (checked via `sys.modules`). So drawing a chart or
  writing a CSV never pays the heavy-model import cost.
- **Plotting draws-and-returns an `Axes`; it never calls `plt.show()`.**
  Showing/saving is the caller's decision — this is what makes the
  functions unit-testable under a headless `Agg` backend and keeps them
  non-blocking in scripts.
- **Colors auto-cycle; only the ensemble has a fixed style.** No
  hardcoded `{"prophet": ..., "catboost": ...}` map, so a registered
  Phase 6 model (SARIMAX/PyMC/NeuralProphet) charts correctly with zero
  changes here. This mirrors `build_ensemble()`'s
  registry-generic stance.
- **Exporters never raise on empty/None input.** Each `build_*_table()`
  returns a correctly-columned empty DataFrame when there's no ensemble
  or nothing was rejected, and `export_forecast_csvs()` always writes
  all three files. A run where every model was threshold-rejected still
  produces predictable, header-only CSVs rather than a crash or a
  missing file — the same "don't punish the caller for an empty
  ensemble" philosophy as `build_ensemble()`'s no-raise path.

### How this was verified

- 43 new tests: 29 in `tests/test_plotting.py` (line/bar counts, legend
  labels incl. MAPE, bounds-shading only when present, ensemble overlay
  presence/absence, the "no ensemble formed" red-note flag appearing only
  when an ensemble was requested but none formed, the faceted grids'
  one-visible-panel-per-model-plus-ensemble counts, the `save_*_plots`
  PNG-writing + nested-dir + empty-input cases, empty-input `ValueError`,
  drawing onto a supplied `Axes`) and 14 in `tests/test_exporters.py`
  (table columns/contents, bounds preserved with NaN-fill for models
  lacking them, NaN-MAPE rejected rows, `tmp_path` CSV round-trips,
  nested-dir creation, custom prefix, the empty-input header-only-files
  case, and the `export_forecast` bundle writing CSVs+PNGs / CSV-only /
  empty-skips-plots paths). All 43 pass in the sandbox (Prophet-free,
  ~2s), alongside the confirmed import
  decoupling check above.
- These modules don't fit or predict, so unlike the Prophet-dependent
  suites they run fully in-sandbox; no local-only gap for Phase 5.

### Deliberately out of scope for Phase 5

No top-level `__init__.py` re-exports were added — callers import from
`horizon_forecaster.plotting` / `horizon_forecaster.exporters` directly,
consistent with how `HorizonForecaster` is imported today. Evaluation-side
CSV export (test predictions to disk) wasn't added since the README spec
names forecast/per-model/rejected specifically; the per-model table
builder is general enough to add it later if wanted. SARIMAX/PyMC/
NeuralProphet remain Phase 6.
