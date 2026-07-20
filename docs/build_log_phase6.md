# Horizon Forecaster — Build Log: Phase 6 (Remaining Models)

Phase 6 adds the three remaining models to the orchestrator, each in its
own self-contained part, gated: a part is only "done" once it passes the
full local test suite and flows through the existing
forecaster/ensemble/export/plot pipeline **without** forcing any change to
those layers. If a model ever required such a change, that would signal a
gap in the Phase 1–2 contracts (`BaseModel`/`BaseAdapter`) to fix in the
contract, not to special-case in the orchestrator. So far no such change
has been needed.

Order: **SARIMAX → (Part 2) → (Part 3)**, simplest/fastest first to set the
template, slowest/most complex last.

---

## Part 1 — SARIMAX (statsmodels)

### Goal

Add SARIMAX as the third registered model, porting the project owner's
proven notebook approach into the adapter/wrapper/config contract, and
improving on it where the notebook had known weaknesses (occasional
very-high MAPE).

### What was built

- `data_adapter/sarimax_adapter.py` — a **stateful** adapter (like the
  CatBoost one): raw `y`, plus an exog matrix of standardised independent
  columns and annual Fourier terms. State fitted on train (log mask,
  standardisation mean/std, the `DeterministicProcess`) is reused by
  `transform_future()`.
- `models/sarimax_wrapper.py` — the model-specific logic: log-space fit,
  ADF differencing, validation-MAPE order selection, stationarity-enforced
  final refit, bias-corrected + clipped back-transform, native prediction
  intervals.
- `config/default_params.yaml` `sarimax:` block (daily/weekly), and one
  `MODEL_REGISTRY` line in `forecaster.py`.

### Key design decisions

- **Log1p target, raw-unit output.** The wrapper fits `log1p(y)` (variance
  stabilisation for a strictly-positive metric) and inverts with `expm1`
  on predict, so `yhat` is returned in RAW units — `forecaster.py` scores
  MAPE against raw `y`, and "y" keeps one consistent meaning across every
  model. The transform lives in the wrapper, not the adapter, for exactly
  that reason.

- **First model with prediction intervals — as a constant-variance band.**
  `predict()` returns `yhat_lower`/`yhat_upper`, the first model to
  exercise the bounds path end-to-end (per-model plots/CSVs; the ensemble
  still ignores bounds by design). The interval is `mean ± z·σ` in log
  space using the model's CONSTANT innovation variance `σ²`, NOT
  `get_forecast().conf_int()`'s horizon-growing forecast variance.
  Discovered on real data: with the growing variance, a 144-step-ahead
  forecast produced a `[8k, 700k]` raw-scale band around an accurate ~80k
  point forecast (the log-space variance balloons over the horizon and
  `expm1` amplifies the upper bound, then the clip mangles it). The
  constant band gives a readable, roughly-multiplicative typical-error
  interval whose width stays stable across the horizon — the same
  constant-variance philosophy as the bias correction. Trade-off: it does
  not widen formally with horizon; it is a typical-error band, not a
  textbook expanding predictive interval.

- **No forecaster-level grid search.** SARIMAX's config carries no
  `hyperparameter_grid`, so the orchestrator (already supporting this —
  `use_grid_search = override is None and "hyperparameter_grid" in config`)
  fits it once and lets the wrapper self-tune. `grid_search_summary` is
  `None` for SARIMAX. Zero orchestrator changes.

- **Order selection by validation MAPE, not AIC.** ADF picks `d`, then a
  small exhaustive `(p,d,q)(P,D,Q,s)` space (p,q ∈ {0,1}; daily uses a
  weekly `s=7` seasonal order, weekly disables the SARIMAX seasonal order
  and leans on Fourier) is scored by MAPE on a validation tail carved out
  of the TRAINING data — not the reported test split, so no leakage. AIC
  is only a fallback if no candidate produces a scorable forecast. This
  targets forecast accuracy directly and was the fix for the notebook's
  occasional very-high MAPE (AIC rewards in-sample fit, not out-of-sample
  accuracy).

- **Explosion control at the source.** The notebook relied on a hard clip
  (`train_max_log + 1.5`) to stop exponential blow-ups after `expm1`. The
  real cause is explosive AR roots admitted by
  `enforce_stationarity=False`. So the winning order is refit with
  `enforce_stationarity=True` (falling back to `False` only if that fit
  fails), making the clip a rare last-resort guard rather than a routine
  crutch. **Divergence guard (added after a real-data edge case).** Even
  with the above, a fit can still blow up in log space on messy real data
  — a near-explosive AR root or an exogenous coefficient extrapolating on
  out-of-range future values — so the log forecast runs far past the clip
  and `expm1` emits a degenerate flat line pinned at the cap (observed on
  real spiky weekly data: a ~3.8e6 flat forecast, ~1400% MAPE, while
  Prophet/CatBoost were ~20-30%). The clip prevents infinities but not
  garbage. So when the clip binds on a majority of the horizon, `predict()`
  now raises `ModelPredictionError`; the orchestrator catches it per model
  and records SARIMAX in `evaluate_errors`/`forecast_errors`, dropping it
  entirely rather than surfacing a misleading flat-line result. This is
  deliberately distinct from a merely-inaccurate model (a real result the
  ensemble's MAPE threshold excludes) — a clip-saturated forecast is a
  genuine divergence, not a weak prediction.

- **Bias correction with the right variance.** `expm1` of the log-space
  mean returns the lognormal median, biasing forecasts low. The point
  forecast adds `+σ²/2` using the model's CONSTANT innovation variance
  (`sigma2`), NOT `get_forecast`'s horizon-growing forecast variance —
  which would inflate later periods and feed the very explosion the clip
  fights. Documented consequence: the bias-corrected point can sit
  slightly above the (uncorrected) interval centre — correct lognormal
  behaviour, since mean > median.

- **Annual Fourier only with enough history.** Annual Fourier terms are
  added for both frequencies, but only when training length covers ≥ half
  the annual period (`_MIN_ANNUAL_COVERAGE = 0.5`). Below that they span
  less than half a cycle and become unconstrained trend-like regressors
  that extrapolate wildly — the same failure mode Prophet's `auto`
  candidates guard against. **Confirmed during build:** on a ~2.5-month
  daily series, including annual Fourier drove synthetic test MAPE from
  ~6% to ~65%; the coverage guard restores ~6%.

- **Auto-detected exog logging.** The library's `independent_columns` is a
  flat list with no spend/flag metadata, so the adapter log1p's a column
  iff it is non-negative and non-binary (spend-like), leaving 0/1 flags
  linear, then standardises all exog. Standardisation is done manually
  (store train mean/std) to avoid adding a scikit-learn dependency.

- **Future exog clamped to the training range (added after the same
  edge case as the divergence guard).** The Section-5b diagnostic isolated
  the real-data divergence to the exogenous variables: the identical series
  fit fine at ~52% MAPE with `independent_columns=None` but diverged with
  them. Cause: a test-window spend spike (far above any training value)
  standardises to a large number and, multiplied by SARIMAX's fitted
  coefficient, blows up the log-linear predictor. So `transform_train()`
  records the standardised min/max per column and `transform_future()`
  clamps future exog to that range — SARIMAX is linear in the exog, so
  extrapolating a coefficient beyond the trained range is unreliable
  anyway. Verified: a synthetic series with a 40x spend spike across the
  test window (which diverged before) now fits at ~17% MAPE. The divergence
  guard stays as defense-in-depth for any other cause.

- **Default `save`/`load`.** statsmodels pickles reliably, so unlike
  Prophet (and unlike the upcoming torch/PyMC parts) SARIMAX needs no
  serialization override.

### How this was verified

- 27 new tests: 12 in `tests/test_data_adapter_sarimax.py` (exog/Fourier
  shapes, auto-log detection, train-stat reuse into the future, future-exog
  clamping to the training range, the short-history Fourier-omission guard,
  missing-exog and pre-train errors) and 15 in `tests/test_sarimax.py`
  (fit/predict shapes + bounds +
  non-negativity, a reasonable-MAPE sanity floor, the full error contract,
  the all-candidates-fail convergence path, and unit tests for the bias
  correction, the clip, and ADF `d` — plus an end-to-end integration test
  running SARIMAX + CatBoost through `evaluate()`/`forecast()` and the
  ensemble). All pass in-sandbox alongside the existing pure suites
  (adapters/utils/ensemble/metrics/exceptions).
- Existing `test_forecaster.py` assertions that assumed a two-model
  default registry were updated for the now-three-model default: the
  "every registered model" / "all models fail" checks are now
  registry-driven (`set(MODEL_REGISTRY.keys())`), the 2-model ensemble
  weighting and partial-failure tests are explicitly pinned to
  `models=["prophet", "catboost"]`, and the unknown-model test now uses a
  genuinely unregistered name (SARIMAX used to be the example of an
  *invalid* model).
- Unlike SARIMAX itself, the Prophet-dependent `test_forecaster.py` tests
  need cmdstan and exceed the sandbox ceiling — a full local `pytest` run
  is the definitive gate before this part is pushed.

### Post-real-data hardening (orchestrator + exporters)

Running Part 1 against the real dataset surfaced problems that were not
SARIMAX bugs but genuine gaps in the surrounding layers. Fixed in the same
pass, all model-agnostic:

- **Horizon warning was far too lenient.** It only fired above *twice the
  entire history*, so forecasting 100 periods from 121 passed silently —
  and produced an implausible forecast. Now fires once the horizon exceeds
  half the available history.

- **Auto-generated future exog was invisible.** With no
  `future_independent_data` supplied, the forecaster silently invents ~2
  years of spend by cycling history, and for an exogenous-driven model
  that can dominate the result. It now warns, naming the columns and
  horizon, and says to treat the output as a scenario rather than a
  prediction.

- **No plausibility check on forecasts.** A model can produce a
  syntactically valid forecast wildly detached from anything ever
  observed. `_warn_on_forecasts_far_outside_history()` flags a forecast
  peaking >50% above (or bottoming >50% below) the observed range. The
  margin matters: an early draft used "entirely outside the historical
  range", which both **missed** the real case (the tail decayed back
  inside) and would **false-positive on every healthy trend** — a good
  reminder to check a rule against the actual data before shipping it.

- **The "final" export baked in a choice it shouldn't.** On the real run
  the ensemble scored 22.40% while CatBoost alone scored 21.32%, yet
  `final.csv` was always the ensemble — i.e. the headline deliverable was
  the worse forecast. Now both are exported (`final_ensemble.csv`,
  `final_best_model.csv`) alongside `model_summary.csv` carrying each
  model's MAPE, ensemble weight and status, so the call is made per run
  with the evidence in hand.

### Deliberately out of scope for Part 1

Parts 2 and 3 (the other two models) are untouched: their adapter/wrapper
stub files remain empty and unregistered. Phase 7 (future-exogenous-data
strategies) is still deferred until all five adapters exist.
