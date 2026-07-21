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

- **~~No forecaster-level grid search.~~ SUPERSEDED** — see "Uniformity
  refactor" at the end of this file. SARIMAX originally carried no
  `hyperparameter_grid` and self-tuned inside the wrapper; it now uses the
  standard grid search like every other model.

- **~~Order selection by validation MAPE, not AIC.~~ SUPERSEDED** — see
  "Uniformity refactor" below. Kept here because the reasoning about AIC
  still stands, and explains why the grid scores candidates by MAPE. ADF
  picks `d`, then a
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

### Deliberately out of scope for Part 1 (superseded by Part 2, below)

Parts 2 and 3 (the other two models) are untouched: their adapter/wrapper
stub files remain empty and unregistered. Phase 7 (future-exogenous-data
strategies) is still deferred until all five adapters exist.

---

## Part 2 — NeuralProphet (torch / PyTorch Lightning)

### Goal

Add NeuralProphet as the fourth registered model, porting the project
owner's working notebook grid while fixing an evaluation-comparability
problem in it, and isolating the two new mechanics (torch serialization,
stochastic determinism) before PyMC's structural-model complexity in
Part 3.

### What was built

- `data_adapter/neuralprophet_adapter.py` — a **stateless** adapter,
  essentially ProphetAdapter (NeuralProphet shares the ds/y convention).
- `models/neuralprophet_wrapper.py` — the model-specific logic, with
  overridden `save`/`load` for torch state.
- `neuralprophet:` config block (daily/weekly) and one `MODEL_REGISTRY`
  line.

### The problem in the reference notebook, and why it mattered

The notebook grid-searched `n_lags` in {0, 7} and called
`m.predict(df_all)` — a frame **containing the test period's real `y`
values**. With `n_lags=0` that's harmless. With `n_lags>0` NeuralProphet
is autoregressive, so every test point becomes an effectively
one-step-ahead prediction that has already seen recent actuals.

Ported as-is, that would have been actively harmful here, for three
reasons:

1. **Not comparable.** Prophet, CatBoost and SARIMAX forecast the whole
   test window blind. A one-step-ahead-with-actuals MAPE looks far better
   for reasons unrelated to model quality.
2. **It would corrupt the ensemble.** That flattered MAPE feeds
   inverse-MAPE weighting, so NeuralProphet would be systematically
   over-weighted against honestly-scored models.
3. **The contract can't express it.** `predict(horizon, future_exog)` has
   no actuals to hand the model.

There is also a genuine contract tension: NeuralProphet fixes
`n_forecasts` at construction, whereas `BaseModel` is
fit-once-then-predict-any-horizon. Supporting AR properly would mean
refitting per horizon, breaking the cached-refit design — by this
project's own gate rule, a contract-gap signal rather than something to
special-case in a wrapper. **v1 therefore uses `n_lags=0`**, and AR is
deferred to a separate contract discussion.

Consequence to expect on real data: if `n_lags=7` won the notebook grid,
this implementation will score *worse* than the notebook's number. That is
the leakage being removed, not a regression.

### Key design decisions

- **Raw pass-through in the adapter.** NeuralProphet normalises
  internally (`normalize='auto'`), so — in direct contrast to
  SARIMAXAdapter's log1p + standardise + clamp — transforming here would
  double-normalise. The adapter does nothing but rename and sort.
- **Future frame built directly, not via `make_future_dataframe()`.** With
  `n_lags=0` a prediction depends only on `ds` + regressors, so the frame
  is just `future_exog` with `y=NaN`. This is why the wrapper retains **no
  training history**, which keeps `save`/`load` a clean model-only
  operation instead of also persisting a DataFrame.
- **Standard forecaster grid search**, unlike SARIMAX. The notebook
  already selected by test MAPE, which is exactly what
  `hyperparameter_grid` does — so `learning_rate` × `epochs` (4
  candidates) drops straight into the existing mechanism.
- **CPU pinned, logging silenced.** NeuralProphet runs on PyTorch
  Lightning and will otherwise pick MPS locally and CPU in CI, producing
  different numbers from identical code. It is also extremely chatty;
  `progress=False` alone doesn't quiet it.
- **`epochs` and `learning_rate` both pinned.** If either is `None`,
  NeuralProphet runs a learning-rate range test and auto-sizes epochs —
  slow, and it makes two runs on the same data non-comparable.
- **`yearly_seasonality="auto"`, not the notebook's `True`.** Forcing
  yearly seasonality on short history has already produced garbage twice
  in this project (Prophet, Phase 2; SARIMAX's annual Fourier, Part 1).
- **Seeded** via `set_random_seed` — NeuralProphet is stochastic, and the
  notebook version drifted between runs. Same bug class as Prophet's
  unseeded uncertainty intervals in Phase 2.
- **Data-driven non-negative clipping**, mirroring `CatBoostWrapper`,
  rather than the notebook's unconditional `np.clip(..., 0, None)`.
- **No uncertainty bounds in v1**, per `models/base.py`. NeuralProphet can
  produce them via `quantiles=[...]`, a natural follow-up now the
  plotting/export layers handle bounds.

### How this was verified

- 8 adapter tests in `tests/test_data_adapter_neuralprophet.py` pass in
  the sandbox (shapes, raw pass-through, sorting, missing-column errors,
  statelessness).
- `tests/test_neuralprophet.py` (wrapper + integration) is guarded by
  `pytest.importorskip("neuralprophet")` and covers fit/predict shape,
  **absence** of bounds columns, predicting without retained history, the
  full error contract, data-driven clipping both ways, seed determinism,
  `save`/`load` round-trip, and end-to-end flow through the orchestrator.

### Known verification gap

NeuralProphet cannot be installed in the sandbox (torch exceeds the time
limit), so **every wrapper and integration test here is local-only** — a
materially bigger gap than Part 1, where SARIMAX ran fully in-sandbox. The
NeuralProphet constructor kwargs in particular (`accelerator`,
`impute_missing`) are specced from the 0.8 API but unverified against the
installed build; the wrapper now embeds the underlying exception text in
its `ModelConvergenceError` message so a rejected keyword is immediately
identifiable rather than looking like a convergence failure.

Runtime: 4 candidates × torch training × (evaluate + forecast refit) adds
meaningfully to the local suite, which was already ~25 minutes.

### What CI caught that local testing could not

The full local suite passed (334/334), and CI then failed one test —
`test_save_load_roundtrip_preserves_prediction` — on Python 3.10 with a
newer torch. This is the clearest justification so far for running CI once
per part rather than batching it to the end of the phase.

**Cause:** PyTorch >= 2.6 changed `torch.load`'s `weights_only` default
from `False` to `True`. NeuralProphet's `load()` delegates to
`torch.load()`, and it saves a full Python object rather than a tensor
state dict, so the new default refuses to unpickle it
(`UnpicklingError: Unsupported global: GLOBAL
neuralprophet.forecaster.NeuralProphet was not an allowed global`). The
local environment had an older torch, so the code was correct there and
genuinely broken on newer torch — invisible to any amount of local testing.

**Fix:** `NeuralProphetWrapper._load_torch_model()` tries NeuralProphet's
own loader first (correct on older torch) and falls back to
`torch.load(..., weights_only=False)`. The alternative,
`torch.serialization.add_safe_globals()`, was rejected: a saved
NeuralProphet drags in a tree of its own config classes, so allowlisting
becomes whack-a-mole across versions. `weights_only=False` is appropriate
because the file was written by this library's own `save()` — a trusted
source, which is the exact condition PyTorch's documentation attaches to
that flag.

Note this bug lives in a code path that only `save`/`load` exercises, and
would otherwise have surfaced the first time someone tried to reload a
persisted model in a fresh environment.

#### ...and then CI caught the fix itself

The next CI run failed the *same test* again, on a different cause — this
time inside the fallback added above:

```
AttributeError: 'NoneType' object has no attribute 'predict'
  neuralprophet/forecaster.py:2922  result = self.trainer.predict(...)
```

**Cause:** NeuralProphet's `save()` deliberately sets `forecaster.trainer`
and `forecaster.model._trainer` to `None` before writing, because a PyTorch
Lightning `Trainer` is not picklable. Its `load()` then calls
`m.restore_trainer(...)` to rebuild one. The fallback did a bare
`torch.load()` and skipped that step, so the model came back structurally
complete but without a trainer, and died on the *next* `predict()` rather
than at load time.

**Fix:** the fallback now mirrors `np_load()`'s post-load step, calling
`restore_trainer(accelerator="cpu")` (explicit `"cpu"` rather than `None`,
to stay consistent with the `map_location="cpu"` immediately above).

**The real lesson is about the test gap, not the bug.** Both failures were
CI-only for the same structural reason: local torch is < 2.6, so the
fallback branch never executes locally *at all*. It was effectively
untested code that only ever ran on the matrix — which is why a bug fix
landed in it and nothing caught the regression for a full 35-minute cycle.

Closed by `tests/test_neuralprophet_compat.py` (renamed from
`test_neuralprophet_hparam_patch.py`, now covering both shims), which stubs
`neuralprophet` and `torch` into `sys.modules` and drives the fallback
directly. It asserts the trainer is restored, that `weights_only=False` and
`map_location="cpu"` are both passed, that the primary loader is preferred
when it works, and that a model without `restore_trainer` doesn't crash.
These run on **any** torch version, including the sandbox where torch isn't
installed. Verified by reverting the fix: `test_fallback_restores_the_trainer`
fails with "fallback returned an unusable model", and passes once restored.

---

## Uniformity refactor (post Part 2, driven by real-dataset testing)

Real weekly data exposed two problems at once: SARIMAX scoring 146% where
Prophet and CatBoost scored ~35%, and no fixed annual-Fourier setting that
worked for both the daily and the weekly dataset. Investigating produced
three changes -- and refuted two of my own hypotheses on the way.

### What the data actually said

Isolating each component on the real weekly series:

| exog | annual Fourier | MAPE |
|---|---|---|
| off | order 2 | 146.85% |
| off | off | 106.69% |
| on | order 2 | 82.81% |
| on | **off** | **48.74%** |

Order sweep with exog on: order 2 -> 82.81%, order 6 -> 66.96%, order 10 ->
**diverged** (23/26 of the horizon pinned at the clip).

Two hypotheses died here. **Exogenous variables were not the cause** --
removing them made things *worse* (48.74% -> 106.69%), the opposite of the
Part 1 divergence case. **The bias correction was not the cause** -- sigma^2
was only 0.171, an inflation of x1.09, and disabling it moved MAPE by 0.03
points. Worse, a synthetic weekly series with a *clean* annual cycle had
suggested higher Fourier orders were the fix (21.89% at order 10 vs 216.91%
at order 2); the real series, whose annual pattern is weak across only ~2
cycles, said the opposite. **Synthetic evidence pointed the wrong way, and
only the real data settled it** -- worth remembering before trusting a
generated fixture to justify a default.

### The three changes

1. **Annual Fourier moved from the adapter into the wrapper**, becoming a
   grid-searchable `fourier_order_annual` (0 = off). There is no correct
   fixed default -- the daily dataset wants it on, this weekly one wants it
   off -- so the data has to choose, which requires the setting to be
   reachable from the grid. It could not be while it lived in the adapter,
   because `_prepare_split()` builds the exog matrix **once** and every
   candidate reuses it. Terms are now built from absolute day offsets
   against a fit-time reference date instead of a positional index, which
   also makes them gap-robust (the old `DeterministicProcess.out_of_sample()`
   assumed the future was a contiguous positional continuation).

2. **SARIMAX now uses the standard forecaster grid search.** Its internal
   validation-MAPE search was *more* honest -- it never touched the test
   split -- but it made SARIMAX the only model selected on a different
   criterion. Every other model scores candidates on the test split, so
   their reported MAPE is optimistic while SARIMAX's was not; comparing
   them head-to-head, and deriving inverse-MAPE ensemble weights from them,
   penalised SARIMAX for its own rigour. Uniformity was judged the more
   useful property. `d` stays ADF-derived (a property of the data, not a
   hyperparameter); the grid varies `(p, q)`, seasonal `(P, Q)` and
   `fourier_order_annual`.

3. **The grid search now tolerates per-candidate failures.** Previously one
   candidate raising killed the whole model -- which would have been fatal
   here, since diverging Fourier orders are precisely what the grid should
   *discard*. Failures are now skipped with a warning naming how many, and
   the model fails only if every candidate does. Model-agnostic.

   The first cut of this got the total-failure case wrong, and the local
   suite caught it: it wrapped *every* total failure in
   `ModelConvergenceError`, so a 10-row dataset stopped reporting
   `InsufficientDataError` and started reporting a convergence problem that
   did not exist -- losing `required_periods`/`available_periods` with it.
   Corrected rule: if all candidates fail with the **same** exception type,
   the cause is not hyperparameter-specific and the original exception is
   re-raised untouched; only genuinely **mixed** failures aggregate into
   `ModelConvergenceError`. Worth recording because the bug was invisible in
   the sandbox -- both affected tests are Prophet-backed, and Prophet does
   not install there.

### Trade-offs accepted knowingly

- SARIMAX's reported MAPE becomes optimistic like the others. Every number
  feeding the ensemble is relative, so consistency matters more than
  absolute honesty here -- but it is a real loss, and worth revisiting if
  the library ever moves *all* models to validation-based selection.
- Each candidate is now a full fit; the old search used cheap `maxiter=30`
  fits and refit only the winner properly. SARIMAX runs are slower.
- The Part-1 validation machinery that fixed the original "occasional very
  high MAPE" is gone. The grid should subsume it -- verified on synthetics
  (weekly spiky 30.49%, daily annual 4.95%, with the grid correctly
  selecting different Fourier orders per dataset, and skipping 2 of 6
  diverging candidates on the weekly one) -- but the real daily benchmark
  of 20.23% is the number to watch.

---

## The NeuralProphet `ConfigValueError` (post Part 2)

### Symptom

On the weekly real dataset, NeuralProphet did not merely score badly -- it
never fitted. Every grid candidate died with:

```
ConfigValueError: Unions of containers are not supported:
loss_func: Union[str, _Loss, Callable]
```

### Why it looked like a hyperparameter bug (and wasn't)

The message names `loss_func`, which is a training config field, so the
first instinct was a bad param. Two experiments killed that idea:

1. Constructing `NeuralProphet(**params)` for all four grid candidates
   succeeded. The failure is inside `fit()`, not the constructor.
2. Fitting all four candidates on a bare `ds`/`y` frame **succeeded**.
   Fitting the same four on the adapter frame **failed**.

The discriminator was not the params at all -- it was
`add_future_regressor()`. Any regressor, any params, guaranteed failure.

### Root cause

`neuralprophet/time_net.py`, in `TimeNet.__init__`:

```python
# TODO: causes a RuntimeError under certain conditions,
#       investigate and handle better
try:
    self.save_hyperparameters()
except RuntimeError:
    pass
```

PyTorch Lightning's `save_hyperparameters()`, *when omegaconf is
importable*, converts the captured dataclass hparams through
`OmegaConf.structured()`. OmegaConf refuses
`ConfigTrain.loss_func: Union[str, _Loss, Callable]` and raises
`ConfigValueError`, which subclasses `(OmegaConfBaseException, ValueError)`
-- **not** `RuntimeError`. NeuralProphet's guard therefore misses it and the
exception escapes `fit()`. Registering a regressor is what pushes Lightning
onto the OmegaConf branch, which is exactly the observed discriminator.

This also resolves the daily-vs-weekly puzzle that had been read as a data
problem. Neither NeuralProphet nor Lightning *declares* omegaconf as a
dependency -- it arrives transitively. The successful daily run and the
failing weekly run were different resolved environments, not different data.

### Why `pyproject.toml` was not touched

Both dependency routes were checked against the downloaded package sources
rather than assumed:

| Candidate fix | Verdict |
| --- | --- |
| Bump `neuralprophet` 0.8.0 -> 0.9.0 | **No.** 0.9.0 has the identical `loss_func` annotation and the identical `except RuntimeError` guard. |
| Pin `omegaconf < 2.2` | **No, worse.** 2.2+ raises "Unions of containers are not supported"; 2.1.x rejects *every* Union ("Union types are not supported"), which would break the no-regressor case that currently works. |

There is no version combination to pin to, so the pins stay as they are and
the incompatibility is absorbed in the wrapper.

### The fix

`NeuralProphetWrapper._neutralize_lightning_hparam_capture()` replaces
`TimeNet.save_hyperparameters` with a version that calls the original and
swallows any exception. Applied once per process, guarded by a module-level
flag so repeated `fit()` calls don't stack wrappers.

Safe because the call only writes hparams into the Lightning checkpoint,
which this library never reads -- `save()`/`load()` round-trip the whole
model through NeuralProphet's own torch-backed utilities. NeuralProphet
already treats the call as best-effort; this widens a too-narrow `except`
clause rather than changing intent. The original runs first, so
environments where hparam capture works are unaffected.

### How this was verified

`tests/test_neuralprophet_hparam_patch.py` -- deliberately **not** in
`tests/test_neuralprophet.py`, which is `importorskip`'d on the real
NeuralProphet. The patch is pure monkeypatching, so NeuralProphet is stubbed
into `sys.modules` and the tests run everywhere, including CI and the
sandbox where torch is absent. Cover: the exact `ConfigValueError` is
absorbed; the original is still called; a successful return value passes
through; the patch is idempotent; and it no-ops both when NeuralProphet is
missing and when `TimeNet` is renamed upstream.

Remaining gap: as with the rest of Part 2, an end-to-end fit with real
regressors can only be confirmed locally, not in the sandbox.
