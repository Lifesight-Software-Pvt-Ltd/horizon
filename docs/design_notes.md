# Design notes

Why Horizon Forecaster is built the way it is. This is the reference for
anyone changing the library rather than using it: the contracts that hold the
five models together, the reasoning behind the choices that are easy to
mistake for arbitrary, and the failure modes that shaped them.

For usage, start with [`quickstart.md`](quickstart.md). For the release
record, see [`CHANGELOG.md`](../CHANGELOG.md).

---

## 1. The shape of the library

Everything follows from one decision: a model is an **(adapter, wrapper)
pair**, and the orchestrator knows nothing else about it.

- A **data adapter** (`data_adapter/`, subclasses `BaseAdapter`) does all
  model-specific reshaping. `transform_train()` turns raw user data into that
  model's training input; `transform_future()` turns future dates plus
  resolved future driver values into its prediction input.
- A **model wrapper** (`models/`, subclasses `BaseModel`) does
  `fit()` / `predict()`, plus `save()` / `load()` where the backend needs
  something better than pickle.
- `HorizonForecaster` (`forecaster.py`) wires
  `adapter.transform_train() → wrapper.fit()` and
  `adapter.transform_future() → wrapper.predict()` identically for every
  entry in `MODEL_REGISTRY`.

A wrapper never calls an adapter, and an adapter never sees a model. Each
side is testable in isolation — the adapter against plain DataFrames, the
wrapper against pre-shaped input.

**The gate rule.** Adding a model should mean adding one `MODEL_REGISTRY`
entry and nothing else. If a new model forces a change to the orchestrator's
core logic, that is evidence of a gap in the `BaseAdapter`/`BaseModel`
contract — fix the contract, do not special-case the orchestrator. All five
current models were absorbed under this rule without an orchestrator change,
including the two that came closest to breaking it: NeuralProphet's
autoregression and PyMC's whole-series posterior (see §5).

**Pure modules stay pure.** `evaluation/metrics.py`,
`evaluation/ensemble.py`, `future_exog.py`, `plotting.py` and `exporters.py`
import no model, adapter, or `forecaster.py`. Result objects are duck-typed
and type hints are `TYPE_CHECKING`-only. Verified rather than assumed:
importing `plotting` and `exporters` pulls in neither Prophet, CatBoost, nor
the forecaster. Drawing a chart or writing a CSV never pays the heavy-import
cost.

**Errors are structured.** Every raised error is a typed subclass of
`HorizonForecasterError` (`exceptions.py`) carrying machine-readable
attributes (`model_name`, `mape_value`, `required_periods`, …), not just a
message. That is what lets `evaluate_errors`, `forecast_errors` and
`EnsembleResult.rejected_models` be built programmatically instead of by
parsing exception text. One base class also lets the orchestrator catch
library failures with a single `except` while genuine bugs still surface as
raw Python exceptions.

**Failures are isolated, per model.** Any exception from one model — typed or
not — is recorded against that model and the others continue. A non-typed
failure is wrapped as `ModelConvergenceError` with the original preserved on
`original_exception`. This exists because a raw `numpy.linalg.LinAlgError`
from PyMC's prior estimation once took four healthy models down with it; the
isolation is the general guarantee, not a patch for that one case.

---

## 2. Evaluation protocol

### Three windows, not two

Data splits chronologically into **train / validation / test**.

- **Validation** carries every decision: hyperparameter grid search,
  `mape_threshold` screening, `inverse_mape` weighting.
- **Test** is reported and read by no selection path.
- The winning candidate is refit on train + validation before the test window
  is scored.

The reported `mape` is therefore an out-of-sample number rather than the
score a model was chosen for; the selection score stays alongside it as
`validation_mape`.

The refit costs one extra fit per model and is deliberate. Fitting on train
alone and predicting across both windows is cheaper but measures a model
trained on less data than would ever be deployed. Refitting on everything
available makes the reported MAPE an estimate of the same procedure that
produces the real forecast. `EvaluationResult.fitted_model` is consequently
that refit — the model that actually produced `test_predictions`.

### Window sizes are absolute, not percentages

Defaults are **56 validation / 28 test** for daily, **26 / 13** for weekly,
with training taking the remainder. Percentages would make both windows scale
with however much history the caller happens to have, which is the wrong
driver for either: a test window estimates the error you will experience when
you forecast, so it should be about the length of a horizon somebody would
really act on, regardless of whether you hold two years of history or ten.

`validation_size`/`test_size` accept an `int` (periods), a `float` strictly
between 0 and 1 (a fraction), or `None` for the default. `training_size`
survives as a percentage-only alternative route (50–100, or 0.5–1.0) with the
remainder split two-thirds to validation, and is **mutually exclusive** with
the other two — passing both raises `InvalidConfigError` rather than letting
one silently win.

On a series too short for the defaults they scale down proportionally, capped
at 40% of rows between them, preserving the 2:1 ratio and warning with the
resulting sizes. Explicitly requested sizes are never silently shrunk.

### Selection by MAPE, and its limits

Candidates are scored by validation MAPE rather than an information
criterion. AIC rewards in-sample fit, not out-of-sample accuracy, and
selecting SARIMAX orders by AIC was a direct cause of occasional very high
MAPE in the pre-library version of that model.

MAPE's own weaknesses are known and unaddressed in 1.0: it is undefined
against zero actuals (sparse conversion or install series were observed at
43–68% "MAPE" purely from zero handling), and it says nothing about whether a
model beats a seasonal-naive baseline. `mape()` returns a `MapeResult`
carrying how many rows were excluded for zero actuals, so the data-quality
problem is at least visible. A benchmark-relative metric (RelMAE) and MASE
are the intended fix; they reach into selection, the threshold and the
weighting at once, so they belong in their own design cycle rather than a
late change to a release being frozen.

### Uncertainty bounds are per-model only

`yhat_lower`/`yhat_upper` are optional in the `BaseModel` contract, provided
by Prophet, SARIMAX and Bayesian PyMC, and flow into per-model plots and
CSVs. They are **never combined into the ensemble**. Averaging intervals
produced by a trend simulation, a Gaussian innovation band and an MCMC
posterior has no coherent interpretation. This is enforced by a test, not
only documented.

---

## 3. The ensemble

`build_ensembles()` combines the point forecasts of every model whose
validation MAPE is at or below `mape_threshold`, under one or more rules,
returning one `EnsembleResult` per rule.

### Why a trimmed mean is the default

Estimated weights have to be estimated from something, and everything
available is a finite sample of past error. A trimmed mean — per period,
discard the highest and lowest member, average the rest — spends no
estimation budget at all and is robust exactly where static weights are
fragile.

Static weights fail in a specific, confirmed way. On a synthetic test Prophet
alone scored 1.31% MAPE, CatBoost 12.77%, and their ~91%/9% inverse-MAPE
blend scored **1.58%** — worse than Prophet alone despite Prophet carrying
almost all the weight. Weights are fixed across the horizon while each
model's per-period error is not, so a strong ensemble can still be dragged
off on precisely the periods where a weak member is wrong. Trimming drops
the offending member on the periods where it offends.

**A combined forecast is still not guaranteed to beat the best single
model.** That is a real property of combination, not a bug, which is why
`evaluate()` scores every ensemble against actuals and the exporter writes
both a `final_ensemble_<method>` and a `final_best_model` CSV. The choice is
made per run, with the evidence in hand. This is not hypothetical: on one
real run the ensemble scored 22.40% against CatBoost's 21.32%. On another —
weekly, three comparably-accurate models with partly independent errors — the
ensemble at 34.41% beat every member. Both outcomes are normal.

### Member counts and degradation

For the trimmed rule: 3+ members trim normally; 2 fall back to a plain mean;
1 returns that model's own forecast; 0 gives `combined=None`. Three is where
the arithmetic stops working, not a tuned threshold — remove the highest and
lowest of two values and nothing remains. Degradation is reported as a
first-class `degraded` field rather than a warning, and `rule_applied` names
what actually ran: over exactly four members, trimming leaves two values
whose mean *is* the median, and it says `"median"`.

### `mape_threshold` filters degenerate fits

It defaults to 40 and accepts `None` to admit everything. It is there to drop
degenerate fits, not to screen for accuracy: sweeping the equivalent cutoff
from 0.20 to 0.80 on an external evaluation moved the screened rule's mean
error by 0.018 total with no monotone trend, and it trailed a plain trimmed
mean at every value. The trimmed rule already discards the worst member every
period, so screening hard *and* trimming spends the same robustness budget
twice.

Threshold rejection never raises, even when every model is rejected —
`combined` is `None` and the per-model results are still returned. A model
too inaccurate for the blend is a different situation from a model that
failed to fit, and discarding valid per-model results because the combination
step had nothing to do would be strictly worse for the caller.

### Weights are optional

`EnsembleResult.weights` is `Optional` and is **`None`** for weightless
rules. This is not cosmetic: `weights={}` already meant "no member survived",
so a trimmed ensemble returning an empty dict would have been
indistinguishable from total rejection. Code that assumes a populated dict
will break.

A model whose validation MAPE is NaN is kept by the trimmed rule and excluded
by the weighted ones. For `inverse_mape` that is arithmetic — there is no
`1 / NaN`. `equal` reads only the member count and could weight such a model,
but a model with no validation MAPE was never screened against
`mape_threshold` either, and the weighted rules are where an unscreened
member does the most damage.

Asking for several rules at once is close to free: the fits happened long
before, and combination is arithmetic over predictions already in memory.

---

## 4. Future driver values

When `forecast()` is not given future values for the declared independent
columns, they are generated **once, model-agnostically**, in the
orchestrator, and the same values feed every model's `transform_future()`.
`future_exog.py` holds a `FUTURE_EXOG_STRATEGIES` registry of pure functions
behind a single `resolve_future_exog()` entry point. Strategies:
`carry_forward` (default), `recent_average`, `historical_average`,
`seasonal_naive`, `linear_trend`, and `cycle`.

- **The default is `carry_forward`, not replay.** The original behaviour
  replayed the last `horizon` periods of history, which injects the recent
  pattern — including any one-off spike — into the future. Holding the last
  level flat is the least-surprising neutral baseline. `cycle` keeps the old
  behaviour for anyone who wants it.
- **Flags are type-aware.** Carrying a binary promo flag forward assumes a
  permanent promo; averaging one yields a nonsensical fraction. Continuous
  drivers use the chosen strategy; binary 0/1 event flags default to 0 ("no
  event unless supplied") — except under `seasonal_naive`, where they follow
  the cycle so recurring events still repeat. Flags come from an explicit
  `event_columns` list, else are auto-detected (≤2 distinct, non-negative
  values). `cycle` is exempt from all of this and tiles every column as-is,
  reproducing the old output exactly.
- **`seasonal_naive` carries drift.** Pure seasonal naive repeats last year's
  level, which is wrong for a growing driver over a long horizon. The
  implementation fits a trend, adds the phase-wise seasonal residual, and
  projects both. A flat driver degrades to pure seasonal naive; under one
  full cycle of history it falls back to `carry_forward`.
- **Partial fill.** A supplied column is used as-is and only the missing
  columns are generated — matching the common case where next quarter's
  planned spend is known but the promotional calendar is not.
- **Transparency.** The resolved values (supplied + generated, with `ds`) are
  exposed as `forecaster.future_exog_used` and on each `ForecastResult`, so
  the assumptions behind a scenario can be inspected rather than trusted
  blind. A warning names the generated columns and the strategy used.
- Continuous output for a non-negative driver is clipped at 0, so
  `linear_trend` cannot project spend negative; a driver that genuinely goes
  negative in history is left alone. Out-of-range *clamping* stays in the
  adapters that already do it.

---

## 5. Per-model notes

### Prophet

The reference implementation of the contract — no lag features, no exog
matrix construction, no sampling to tune — which is why it was built first.
Independent columns keep their original names, since Prophet's
`add_regressor()` references columns by name.

- **Seeded, because it was not reproducible.** `yhat_lower`/`yhat_upper`
  varied between repeated `predict()` calls on the same fitted model —
  Prophet's uncertainty simulation draws from numpy's global random state.
  `predict()` now sets the seed (default 42, overridable via
  `params={"random_seed": ...}`) immediately before calling Prophet. The seed
  is filtered out before Prophet's constructor sees it, being a wrapper-level
  concept. Verified across separate processes, not just repeated calls.
- **`save()`/`load()` use `model_to_json()`, not pickle**, per Prophet's own
  documentation — the cmdstanpy backend does not deserialize reliably across
  environments. The JSON is wrapped in an envelope carrying wrapper state
  (regressor columns, frequency, params) so a reload reconstructs the
  wrapper, not merely the Prophet object.
- **`yearly_seasonality` fallbacks in the grid.** Tuned candidates hard-code
  a yearly Fourier order that assumes enough history to constrain a yearly
  cycle; on short history Prophet fits one anyway and extrapolates wildly
  (MAPE up to 802%, including negative predictions on a strictly positive
  metric). Rather than hand-code a data-span rule, the grid carries two safe
  `'auto'` candidates and lets selection favour them on short series.
- Prophet can still emit negative point forecasts on a low-level series. No
  output clamp is applied.

### CatBoost

Features are a **trend index plus cyclical seasonality**, not target lags.
`time_idx` is an absolute day/week offset from a fixed reference date (not a
row counter), so it stays continuous across the train/future boundary and is
gap-robust; sin/cos encodings cover yearly and half-yearly cycles, with
weekly and monthly added for daily data; calendar fields and independent
columns pass through.

This replaced a lag(1/2/3/7/14) + rolling(7/14) design, and the reason it
was replaced still applies. Because `predict()` is one batched call rather than a
per-step recursive loop, those features had to be frozen at train-end and
broadcast across the whole horizon. On real revenue data that scored 39.95%
MAPE against Prophet's 18.68%. Two plausible explanations were tested and
disproven — the tree-extrapolation ceiling (0 of 144 test rows exceeded the
training max) and monotonic staleness (the second half of the window scored
*better* than the first). Breaking the error down by known demand flags
found it: a December demand dip scored 67–90% MAPE against a 34–40%
baseline. A frozen snapshot of recent history cannot see a demand shift
coming; it keeps predicting the pre-shift baseline for the entire horizon.
After the redesign, that same segment scored 14.59% — the best segment
rather than the worst — with overall MAPE at 18.86%, matching Prophet.

Every feature is computed identically by `transform_train()` and
`transform_future()`, purely from dates, so nothing needs freezing. No
feature is declared categorical. Non-negative clipping is data-driven, and
applied only when the training target was itself always non-negative.

### SARIMAX

Fits `log1p(y)` and returns raw units via `expm1`, so `y` keeps one meaning
across every model and MAPE is scored consistently. The transform lives in
the wrapper for that reason; the adapter supplies raw `y` plus a
standardised exog matrix. `d` comes from an ADF test on the data; the grid
varies `(p, q)`, the seasonal `(P, Q)` at daily frequency, and
`fourier_order_annual`. Weekly disables the SARIMAX seasonal order — an
`s=52` term is prohibitively slow — and leans on the Fourier terms instead.

- **`trend` must be passed explicitly.** statsmodels defaults to no constant
  term, which is correct for a differenced series but means fitting a
  zero-mean process when `d=0` and `D=0`. This produced degenerate forecasts
  on every series the model did not difference.
- **Annual Fourier is a grid hyperparameter, not a fixed setting.** There is
  no correct default: on a ~3-year daily series annual Fourier is a clear
  win, and on a ~2-year weekly series it was actively harmful (82.81% at
  order 2 versus 48.74% with it off). It could not be grid-searched while it
  lived in the adapter, because the exog matrix is built once and reused by
  every candidate — so it moved into the wrapper, built from absolute day
  offsets against a fit-time reference date (gap-robust, phase-aligned).
  A related warning: synthetic data with a *clean* annual cycle pointed the
  opposite way (higher orders better); only the real series, whose annual
  pattern is weak across ~2 cycles, settled it. Worth remembering before
  trusting a generated fixture to justify a default.
- **Bias correction and intervals both use the constant innovation
  variance**, not `get_forecast()`'s horizon-growing forecast variance.
  `expm1` of the log-space mean returns the lognormal median, biasing low, so
  the point forecast adds `+σ²/2`. Using the growing variance would inflate
  later periods and feed the very explosion the clip guards against. The
  interval is `mean ± z·σ` in log space: a readable typical-error band whose
  width is stable across the horizon. The trade-off is real — it does not
  widen formally with horizon. With the growing variance, a 144-step forecast
  produced an `[8k, 700k]` band around an accurate ~80k point forecast. A
  documented consequence of the bias correction is that the corrected point
  can sit slightly above the interval centre; mean > median is correct
  lognormal behaviour.
- **Explosion is controlled at the source, with a divergence guard behind
  it.** The winning order is refit with `enforce_stationarity=True` (falling
  back to `False` only if that fit fails), which makes the hard clip a rare
  last resort rather than a routine crutch. It is still reachable: a
  near-explosive root or an exog coefficient extrapolating on out-of-range
  values can run the log forecast past the cap, and `expm1` then emits a
  degenerate flat line (observed: a ~3.8e6 flat forecast at ~1400% MAPE while
  other models sat at 20–30%). When the clip binds on a majority of the
  horizon, `predict()` raises `ModelPredictionError` and the model is dropped
  from the run. A clip-saturated forecast is a divergence, not a weak
  prediction, and is deliberately distinct from a model the threshold
  excludes.
- **Future exog is clamped to the training range.** A test-window spend spike
  far above any training value standardises to a large number and, times a
  fitted coefficient, blows up the log-linear predictor. `transform_train()`
  records the standardised min/max per column and `transform_future()` clamps
  to it. SARIMAX is linear in the exog, so extrapolating a coefficient beyond
  the trained range is unreliable anyway. Verified: a synthetic series with a
  40× spend spike across the test window, which diverged before, fits at ~17%.
- **The collinearity warning is diagnostic, not a fix.**
  `_warn_on_annual_collinearity()` flags any independent column whose
  variance is ≥30% explained by an annual sin/cos basis. It exists because
  adding drivers once made SARIMAX *worse* on a weekly series, and the cause
  turned out to be a stockout-weighting column that was itself seasonal
  (`corr(y) = -0.584`, `corr(annual sin) = +0.576`, `corr(trend) = -0.766`).
  On ~100 rows the fit cannot attribute that shared variance to both the
  column and six Fourier harmonics. That is an identifiability limit of the
  data, not a defect — the grid already resolves it by picking a low Fourier
  order once exog is present — so the right response is to make the cause
  visible, not to chase the number. The 0.30 threshold is calibrated against
  a frame mirroring the real data: it flags the seasonal stockout column at
  R²=0.95 and stays silent on spend columns and a sparse promo flag. Nothing
  is dropped.
- statsmodels pickles reliably, so SARIMAX needs no `save`/`load` override.

Known and unresolved: SARIMAX's daily MAPE has drifted between runs on
identical data with the identical winning candidate (17.73% vs 19.62%) while
the other four reproduce to the digit. Likely statsmodels MLE or BLAS
threading non-determinism — a hypothesis, not a finding, and left open rather
than "fixed" on a guess.

### NeuralProphet

A stateless adapter that renames and sorts, and nothing else: NeuralProphet
normalises internally, so the SARIMAX-style log/standardise/clamp treatment
would double-normalise. The future frame is built directly (`future_exog`
with `y=NaN`) rather than via `make_future_dataframe()`, which is why the
wrapper retains no training history and `save`/`load` stays a model-only
operation.

- **`n_lags=0` in 1.0, deliberately.** A reference implementation grid-searched
  `n_lags` in {0, 7} and predicted on a frame containing the test period's
  real `y`. With `n_lags>0` NeuralProphet is autoregressive, so every test
  point becomes an effectively one-step-ahead prediction that has already
  seen recent actuals. Ported as-is that would be (a) not comparable with
  four models forecasting the window blind, (b) corrupting — the flattered
  MAPE feeds weighting and screening, and (c) inexpressible: `predict(horizon,
  future_exog)` has no actuals to hand the model. There is also a genuine
  contract tension, since NeuralProphet fixes `n_forecasts` at construction
  while `BaseModel` is fit-once/predict-any-horizon; supporting AR properly
  means refitting per horizon. By the gate rule that is a contract discussion,
  not a wrapper special case.
- **Seeded and CPU-pinned.** It is stochastic, and Lightning otherwise picks
  MPS on Apple silicon and CPU elsewhere, producing different numbers from
  identical code.
  `epochs` and `learning_rate` are both pinned — leaving either `None` runs a
  learning-rate range test and auto-sizes epochs, which is slow and makes two
  runs on the same data non-comparable.
- **`collect_metrics: False` is load-bearing.** With metrics enabled
  NeuralProphet attaches a Lightning logger, and Lightning then runs
  `_log_hyperparams()` → `save_hparams_to_yaml()` → `OmegaConf.save()`, which
  rejects `ConfigTrain`'s `loss_func: Union[str, _Loss, Callable]` and kills
  `fit()` before a single epoch. On newer stacks the same logger fails
  differently, via TensorBoard → TensorFlow → jax → `np.dtypes.StringDType`.
  The config sets it false for both frequencies, and both the config and the
  test defaults assert it — the first attempt at this fix landed in the
  `daily` block only, and `weekly` was the frequency actually failing.
- **`weights_only=False` in the torch load fallback, with the trainer
  restored.** PyTorch ≥2.6 flipped `torch.load`'s `weights_only` default to
  `True`, and NeuralProphet saves a full Python object rather than a state
  dict, so the new default refuses to unpickle it. The wrapper tries
  NeuralProphet's own loader first and falls back. Allowlisting via
  `add_safe_globals()` was rejected — a saved NeuralProphet drags in a tree of
  its own config classes, so the allowlist would need revising with every
  version —
  and `weights_only=False` is appropriate for a file written by this
  library's own `save()`. The fallback must also call
  `restore_trainer(accelerator="cpu")`: `save()` deliberately nulls the
  Lightning trainer (it is not picklable), so a bare `torch.load` returns a
  structurally complete model that fails on the *next* `predict()`.
  `test_neuralprophet_compat.py` stubs `neuralprophet` and `torch` to drive
  this path on any torch version, because it never executes on an older one.
- **Config widened to stop under-fitting.** On a synthetic sweep
  NeuralProphet was the consistent outlier — 18.7% MAPE on a mid-series level
  shift against ~2% for the structural models, which fit clean parametric
  signals near-perfectly. A middle learning rate (0.03), higher epochs
  (150/300) and more changepoints (20 daily / 10 weekly) brought the level
  shift to ~1.9%. An interim `changepoints_range = 0.9` regressed a
  smooth-trend series (5.6% → 11.8%) by leaving a short, noisy final segment
  to extrapolate from; reverting to the 0.8 default kept the win.
- `yearly_seasonality="auto"`, not forced — forcing it on short history has
  produced unusable forecasts twice in this project. Non-negative clipping is
  data-driven, mirroring CatBoost. No uncertainty bounds in 1.0, though
  `quantiles=[...]` is a natural follow-up now the output layer handles them.

### Bayesian PyMC

A structural model — linear trend + annual Fourier (+ weekly for daily) + a
linear regression on standardised exog — observed through a Student-T
likelihood and sampled with NUTS. The adapter is close to SARIMAX's minus the
log1p and the collinearity warning; target standardisation and Fourier live
in the wrapper, mirroring that split.

- **`predict()` reconstructs `mu` from stored posterior draws, in pure
  numpy.** A reference implementation built one `Deterministic('mu')` over
  train + test together and sliced the test rows out of the posterior, which
  only works when both are sampled at once. Under `BaseModel`, `fit()` sees
  only training rows and `predict()` only future dates — so the wrapper
  samples on train once, keeps the flattened posterior draws of every
  coefficient, and reconstructs `mu` at future dates as a linear combination
  of them. Same model, same numbers, and `predict()` never imports pymc.
- **Priors are keyed off column shape, because there is no metadata
  channel.** The library takes arbitrary `independent_columns` and the
  adapter and wrapper are constructed independently, so nothing can tell the
  wrapper which column is spend. A continuous driver (>2 distinct values)
  gets an informative prior centred on its OLS slope against the standardised
  target; a binary flag (≤2 distinct) gets a zero-centred prior. Prior
  estimation drops non-finite pairs.
- **Bounds sample the likelihood, not just `mu`.** Banding the latent mean
  captures parameter uncertainty but omits observation noise, and
  under-covered badly — only ~34% of real daily actuals fell inside a nominal
  94% band. Retaining the posterior `sigma`/`nu` draws and drawing Student-T
  noise restores coverage. The point forecast is still `mu.mean` (the
  predictive mean for `nu > 1`), so MAPE is unaffected; only the band widened.
- **No unconditional zero clamp.** The back-transform once applied
  `np.clip(..., 0.0, None)` regardless of the data, so any target that
  legitimately goes negative — net revenue after refunds, inventory deltas,
  week-on-week change — had every negative prediction silently rewritten to
  0. Clipping is now data-driven like CatBoost's.
- **`save`/`load` via a NetCDF sidecar.** `InferenceData` is not pickled
  directly: the trace is written to `<path>.nc` via arviz and the rest of the
  wrapper is pickled. `predict()` depends only on the pickled numpy state, so
  a reload reproduces predictions exactly even if the sidecar is absent.
- **The daily config is tuned for sampling cost. The constraint is speed,
  not validity.** Weekly (≈100 rows, one annual cycle) samples cleanly
  at 2000 tune / 2000 draws. Daily (≈1144 rows) took about an hour on a first
  full-grid run: an annual basis, a weekly basis, and drivers that themselves
  track the annual cycle make a poorly-conditioned ridge, and NUTS answers
  with very deep leapfrog trees. Three changes fixed it — halve draws/tune to
  1000 (a sweep showed point MAPE flat across a 4× range: 500 → 11.46%,
  1000 → 11.56%, 2000 → 11.65%, pure Monte-Carlo jitter, while time scaled
  linearly), keep the Fourier order low (2–4, never 6, which was slowest and
  never won), and grid-select `beta_fourier_sigma` (0.3 tight vs 1.0 loose)
  rather than hardcoding it. A tight Fourier prior regularises the ridge — on
  the collinear synthetic it halved sampling time *and* improved MAPE (1.0 →
  14.55%, 0.3 → 11.56%) — but it underfits strong, well-supported
  seasonality, so which wins is data-dependent and belongs in the grid.
  One correction is worth recording: an early diagnostic reported ~40%
  divergences, and the daily posterior was briefly judged untrustworthy. That
  figure was an artifact of an 80-step warmup in a throwaway test. With
  adequate warmup the posterior is clean (0 divergences, R-hat ≈ 1.0).
- **A macOS linker workaround runs before the first model is built.**
  PyTensor appends `-ld64` to compile commands on macOS 15+, and current
  Apple toolchains no longer have that linker, so every compile failed with
  `ld: library 'd64' not found` and Bayesian PyMC dropped out of every run on
  those machines — silently, because failure isolation kept the other four
  going. The PyTensor pin cannot move to pick up a fix (see §7), so
  `models/_pytensor_compat.py` probes the configured compiler once per
  process and drops that one flag only when the probe fails with exactly that
  error. Probe-then-patch rather than patch-always because PyTensor added the
  flag deliberately, to dodge bugs in Xcode 15's newer linker; a toolchain
  that still accepts it keeps it.
- It remains by far the slowest model — a full NUTS sample per grid
  candidate. That is inherent to the approach. `nutpie` was considered as a
  further speedup and not adopted in 1.0: it addresses speed only, and the
  grid-selected regularisation already brought daily back to a practical
  runtime.

---

## 6. Output layer

`plotting.py` offers two chart families at each stage: overlays
(`plot_evaluation()`, `plot_forecast()`) and dedicated per-model figures
(`plot_evaluation_per_model()`, `plot_forecast_per_model()`), plus
`plot_model_comparison()`, a MAPE bar chart that makes "the ensemble is not
automatically better" visible at a glance. The per-model figures exist
because the overlay crowds at even two models when their lines coincide, and
a first faceted-grid attempt made each panel too small to read.

- **One chart per combination rule, not one chart for the first rule.** Both
  overlays return `{method: Axes}` — always a dict, matching the convention
  `evaluate()` and `evaluate_ensemble` already follow — and the per-model
  functions emit an `ensemble_<method>` figure for every rule that formed. An
  earlier cut collapsed to the first requested method on the grounds that
  several rules on one time axis bury the per-model lines. The premise was
  right and the conclusion was wrong: with `ensemble_method=None` a caller
  pays to build all three rules and could then see only one of them, with no
  indication the other two existed. Separate charts keep the axis readable
  *and* keep every rule visible, each against the same models and actuals,
  which is what makes the rules comparable against the truth rather than only
  against each other. A caller-supplied `ax` is consequently refused for more
  than one rule — silently ignoring the extras is the defect this shape
  exists to prevent.
- **The bar chart is the deliberate exception**: every rule shares one Axes,
  because comparing rules head-to-head is what a bar chart is for. Its labels
  are rotated, since an "ensemble (inverse_mape)" label is several times
  longer than a model name and collides with its neighbours at that bar count
  when drawn horizontally.

- **Charts draw and return; they never call `plt.show()`.** Rendering is the
  caller's decision, which keeps the functions testable under a headless
  backend and non-blocking in scripts.
- **Colors auto-cycle and only the ensemble has a fixed style.** No hardcoded
  per-model color map, so a newly registered model charts correctly with no
  change here — the same registry-generic stance as the ensemble.
- **The combined line is labelled by the rule that actually ran**, so a
  degraded two-member plain mean is distinguishable from a five-member
  trimmed mean.
- **"Ensemble requested but none formed" is labelled loudly, in red, while
  the individual model results are still drawn.** Rejection from the blend is
  a trust signal, not a reason to hide a real forecast.
- **Exporters never raise on empty input.** Each `build_*_table()` returns a
  correctly-columned empty DataFrame, and `export_forecast_csvs()` always
  writes every file. A run where everything was rejected produces predictable
  header-only CSVs rather than a crash or a missing file.
- **Both a `final_ensemble_<method>` and a `final_best_model` CSV are
  written**, always, with `model_summary` carrying validation MAPE, test
  MAPE, method, rule applied, weight and status. The export deliberately does
  not bake in the choice between them.
- The `export_forecast()` bundle imports matplotlib lazily, so
  `export_forecast_csvs()` alone stays matplotlib-free.

---

## 7. Packaging

Every runtime dependency is pinned to an exact tested version. Five pins and
declarations are load-bearing rather than tidy, and all are commented inline
in `pyproject.toml`:

- **`numpy==1.26.4`** — required by `neuralprophet==0.8.0`'s `numpy<2`
  constraint, not chosen. numpy ships no 3.13 wheel for its 1.x line, so on
  Python 3.13 pip compiles it from source. See
  [`installation.md`](installation.md).
- **`pymc==5.25.1`** — held back by the numpy pin, since later PyMC releases
  require numpy 2. It also fixes PyTensor below 2.32, which is why the macOS
  `-ld64` workaround (§5) is carried in this library rather than taken from
  an upgrade.
- **`matplotlib==3.10.9`** — held on the 3.10.x line: 3.11.x dropped Python
  3.10 support and broke installs on 3.10.
- **`setuptools>=68,<82`** — pytorch-lightning (via neuralprophet) imports
  `pkg_resources` at runtime, and setuptools 82 removed it entirely. A loose
  `>=68` was tried first and failed for exactly this reason: pip satisfied it
  with 82+.
- **`requires = ["setuptools>=77.0"]`** in `[build-system]` — the PEP 639
  bare-string `license = "Apache-2.0"` is only understood from 77.0.0.
  A plain `pip install .` hides this because pip resolves a current
  setuptools in its isolated build environment; it surfaces on a
  `--no-isolation` build.

Two more declarations exist because their absence was confirmed, not
theorised:

- `[tool.setuptools.packages.find] include` — without it, any stray
  top-level folder (a venv named `test_env`, a `catboost_info` artifact) is
  mistaken for a second package by flat-layout discovery and the build fails.
- `[tool.setuptools.package-data]` — without it, `config/*.yaml` is silently
  absent from a built wheel. Editable installs symlink the source tree and
  mask this completely; only inspecting a real wheel reveals it.

`requirements.txt` mirrors the runtime pins for tools that expect a plain
requirements file; there is deliberately no `requirements-dev.txt`, since the
`dev` extra is the single dev dependency list.

**Known and open:** 24 CVEs in `torch`/`pytorch-lightning`, forced by
`neuralprophet==0.8.0`'s own upper bounds. Adopting a NeuralProphet
pre-release would resolve them but carries breaking API changes and no
testing against this code; the pin stands until a vetted release is adopted.

---

## 8. How this library is tested

The suite fits real models rather than mocking them. Mocking is used only
where it is the only way to hit a path deterministically — CatBoost's
prediction-clipping tests monkeypatch `_fitted_model.predict` to return a
fixed array, the NeuralProphet torch-compat tests stub `torch` because
the fallback branch never runs on an older version, and the PyTensor
linker-workaround tests stub the platform and compiler because the patch
branch only runs on an affected Mac.

Beyond unit tests, three things repeatedly found what unit tests did not:

- **A synthetic edge-case sweep** with known ground truth — short and long
  history, flat, regime-shift and strong-seasonal shapes, collinear, trending
  and negative drivers, zero-inflated and spiky targets, gaps, NaNs, unsorted
  and zero-variance data. This is what surfaced the failure-isolation gap and
  NeuralProphet's under-fitting.
- **An invariant fuzzer** over random valid datasets — forecast length, bound
  ordering, weight sums, non-negativity, MAPE ≥ 0.
- **Real datasets.** Several decisions in this document exist only because
  synthetic data pointed the wrong way: SARIMAX's Fourier order, CatBoost's
  entire feature design, the plausibility-warning margin. Synthetic fixtures
  are clean parametric signals, which is exactly the regime where structural
  models look perfect and everything else looks broken.

Two practices are recommended for anyone working on this library, both
established after a defect that could have been avoided by them.

**Obtain the traceback before writing the fix.** The NeuralProphet
`ConfigValueError` was first diagnosed by reading library source rather than
by confirming against a stack trace. The resulting fix was plausible,
internally consistent, and wrong, and contrary evidence that was already
visible was rationalised rather than investigated.

**Check a proposed rule against real data before shipping it.** The first
version of the plausibility check used "entirely outside the historical
range", which both missed the case that motivated it (the forecast tail
decayed back inside the range) and would have raised false positives on every
healthy trend.

---

## 9. Deliberate gaps

None of these block a stable API; all are additive.

- **Rolling-origin backtesting.** Every score comes from a single test
  window, so it reflects one stretch of history rather than an average over
  several origins.
- **Benchmark-relative metrics (RelMAE) and MASE.** MAPE alone says nothing
  about whether a model beats a seasonal-naive baseline and is unusable on
  series with zero actuals. These reach into selection, the threshold and
  the weighting together, and should be designed as one change. Sketch for
  whoever implements it: a `mase()` + `MaseResult` in `evaluation/metrics.py`
  (unlike `mape()` it needs the training series and the seasonal period), an
  optional `mase` field on `EvaluationResult`, and metric-aware grid sorting,
  rejection and weighting.
- **Family de-duplication.** Prophet and NeuralProphet share an
  additive-decomposition methodology (signed-error correlation 0.787, the
  highest of any pair involving either), so a blend containing both is less
  diverse than its member count suggests. Nothing accounts for that.
- **An error-diversity diagnostic.** The value of combining depends on
  members making *different* errors — pairwise signed-error correlation
  across members would let users check whether there is anything to exploit
  on their own series.
- **Non-negativity is not enforced library-wide.** Prophet can emit negative
  point forecasts; SARIMAX is non-negative structurally via its `log1p` fit;
  CatBoost, NeuralProphet and Bayesian PyMC clamp only when the training
  target was itself non-negative.
- **NeuralProphet autoregression** (`n_lags > 0`), which needs the contract
  discussion described in §5.
