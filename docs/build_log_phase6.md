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

- **First model with native prediction intervals.** `predict()` returns
  `yhat_lower`/`yhat_upper` from `get_forecast().conf_int()`. This is the
  first model to exercise the bounds path end-to-end (per-model
  plots/CSVs); the ensemble still ignores bounds by design.

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
  crutch.

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

- **Default `save`/`load`.** statsmodels pickles reliably, so unlike
  Prophet (and unlike the upcoming torch/PyMC parts) SARIMAX needs no
  serialization override.

### How this was verified

- 24 new tests: 11 in `tests/test_data_adapter_sarimax.py` (exog/Fourier
  shapes, auto-log detection, train-stat reuse into the future, the
  short-history Fourier-omission guard, missing-exog and pre-train
  errors) and 13 in `tests/test_sarimax.py` (fit/predict shapes + bounds +
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

### Deliberately out of scope for Part 1

Parts 2 and 3 (the other two models) are untouched: their adapter/wrapper
stub files remain empty and unregistered. Phase 7 (future-exogenous-data
strategies) is still deferred until all five adapters exist.
