# Horizon Forecaster — Build Log: Phase 8 (Polish, docs, and testing-driven hardening)

Phase 8 is the pre-release phase: final user-facing docs, worked examples, and
dependency pinning, with the actual PyPI publish deferred. In practice the more
valuable part of Phase 8 turned out to be a **dedicated testing round** — a
synthetic edge-case sweep plus real datasets — which surfaced two genuine bugs
and a model-drift problem that the existing suite hadn't caught. Those fixes,
not the docs, are the substance of this log.

## Documentation & polish

- User-facing docs brought up to the locked Phase-7 API: a new
  [`future_exog.md`](future_exog.md) strategy reference, a future-driver
  section in [`quickstart.md`](quickstart.md), a completeness-requirement note
  (see below), and README/CONTRIBUTING status refreshes.
- **Examples kept as empty placeholders on purpose.** A runnable example is a
  Colab that opens with `pip install horizon_forecaster` + an "Open in Colab"
  badge — which only works once the package is public. Building notebooks now
  would hard-code a private-repo token install to be ripped out at publish
  time, so `examples/*.md` stay empty until the release (the walkthrough
  *content* is drafted and ready to drop in).
- `requirements.txt`/`pyproject.toml` confirmed to pin the exact tested
  versions. PyPI-facing metadata (`authors`, `classifiers`, `[project.urls]`)
  and the top-level `__init__` re-exports are recorded on the publishing
  checklist in `CHANGELOG.md`, deferred with the release.

## Testing infrastructure built

To test "as deeply as possible" rather than by hand, several harnesses were
built (kept outside the repo, alongside the private testing notebook):

- a **synthetic dataset generator** emitting ~20 edge-case datasets with known
  ground truth (short/long history, flat/regime-shift/strong-seasonal shapes,
  collinear/trend/negative/test-window-shift drivers, zero-inflated/spiky/
  small-noisy targets, gaps/NaN/unsorted/zero-variance data, multi-KPI files);
- a **pass/fail checker** asserting each dataset's expected behaviour;
- an **invariant fuzzer** (random valid datasets → assert forecast length,
  bound ordering, weight-sum, non-negativity, MAPE ≥ 0);
- **reproducibility** (evaluate ×2 identical; save/load round-trip) and
  **scale/perf** harnesses.

## What the testing surfaced, and the fixes

### 1. A model's unexpected error could abort the whole run

The `missing_values` dataset (NaN in the target and a driver) crashed the
entire batch run with a raw `numpy.linalg.LinAlgError`. Root cause:
`BayesianPyMCWrapper.fit()` estimated its priors with `np.polyfit` on the NaN
data **before** the sampler's `try/except`, and because a `LinAlgError` is not
a `HorizonForecasterError`, `evaluate()`'s per-model `except
HorizonForecasterError` didn't catch it — so it propagated and took the other
four models' results down with it.

Fixed on two levels: (1) `evaluate()`/`forecast()` now isolate **any**
per-model exception, recording a non-typed failure as `ModelConvergenceError`
(with the original exception preserved via `original_exception`) in
`evaluate_errors`/`forecast_errors`, so one model can never abort the rest;
(2) PyMC's prior estimation is now NaN-safe (`_ols_slope` drops non-finite
pairs). The isolation is the general guarantee — it holds for any future model
that throws something unexpected, not just this case.

### 2. Missing data was handled inconsistently — now a hard fail

Rather than let missing values reach the models at all, a **complete-data
requirement** was added at construction (`utils.validate_no_missing_values`):
a single missing or non-numeric cell in the date, target, or a used
independent column raises `InvalidConfigError`. This matches the team's prior
convention (fail loudly rather than silently forecast on a quietly-shrunk or
gap-filled series) — deliberately stricter than an earlier drop-and-warn
threshold cut, which was replaced on review. A follow-up run then showed a
*missing target column* throwing a raw `KeyError` (the validator indexed the
column before checking it existed), so the validator now checks column
existence first and reports a clear `InvalidConfigError` listing the available
columns. Note the check operates on the columns actually **used** — unused
columns are ignored, and a fully-constant column is still dropped as
zero-variance.

### 3. NeuralProphet drifted far from the other models — narrowed

On the synthetic sweep NeuralProphet was the consistent outlier: 18.7% MAPE on
a mid-series level shift and ~13% on strong seasonality, versus ~2% for the
structural models. The datasets are clean parametric signals (trend + Fourier
+ linear regression), which the structural models fit near-perfectly, so a
neural model has no built-in advantage and — with the v1 config — was
**under-fitting**: too few epochs to converge, too little trend flexibility to
track a level shift.

The config was widened (v1, deliberately, not deferred): a middle learning
rate (`0.03`), higher epochs (`150`/`300`, up from `50`/`150`, so 6 candidates
instead of 4), and more trend changepoints (`n_changepoints` 20 daily / 10
weekly). `regime_shift` dropped from 18.7% to ~1.9%. An interim try with
`changepoints_range = 0.9` **regressed** a smooth-trend series
(`negative_effect_driver`, 5.6% → 11.8%): the late changepoints left a short,
noisy final trend segment that extrapolated badly. Reverting to the `0.8`
default kept the level-shift win while recovering most of that regression
(→ 6.5%). `n_lags` stays 0 (the fit-once/predict-any-horizon comparability
constraint from Part 2).

The residual gap (NeuralProphet a couple of points behind on clean synthetic
data) is expected model behaviour, not a config defect — on real, messier data
it sits in the pack — so tuning was stopped there rather than chased into a
see-saw.

## How this was verified

- Full local suite: **407 passed** (up from 371 at the end of Phase 6), across
  Python 3.10–3.12 as the gate. New tests cover the complete-data policy,
  failure isolation (a model raising a raw error is isolated with its cause
  preserved), and the missing-column error path.
- The synthetic sweep re-run confirmed each fix in the real pipeline:
  `missing_values` and `broken_target` now report clean `InvalidConfigError`s
  and the batch continues; NeuralProphet's level-shift MAPE aligned with the
  others.
- The invariant fuzzer passed 60/60 random datasets; reproducibility and
  save/load round-trips hold.

## Deferred (the publishing checklist)

Recorded in `CHANGELOG.md`'s "Known deferred issues": wire `publish.yml` to a
PyPI Trusted Publisher, add PyPI metadata to `pyproject.toml`, re-export the
public API from the top-level `__init__.py`, and replace the empty
`examples/*.md` with runnable Colab notebooks — all of which only make sense
once the package goes public.
