# Supported models

`HorizonForecaster` is a `MODEL_REGISTRY`-driven orchestrator: every model is
an `(adapter, wrapper)` pair registered in `forecaster.py`, and the same
`evaluate()`/`forecast()`/ensemble/export/plot pipeline runs each one through
a shared contract (`BaseModel`/`BaseAdapter`). Adding a model is a registry
entry, not a change to the orchestrator.

All five models are implemented and registered today. Select a subset with
the `models=` constructor argument, or pass `models=None` to run them all.

## Registered models

| Key | Backend | Notes |
|---|---|---|
| `prophet` | Prophet (cmdstanpy) | Additive/multiplicative seasonality. |
| `catboost` | CatBoost | Gradient-boosted trees over trend-index + cyclical-seasonality features (no target-history lags), so it forecasts any horizon in one batched call. |
| `sarimax` | statsmodels | Log-space (S)ARIMA with an exogenous regression matrix and grid-chosen annual Fourier terms. |
| `neuralprophet` | NeuralProphet (torch / PyTorch Lightning) | Decomposition model, `n_lags=0` (no autoregression in v1), CPU-pinned for reproducibility. |
| `bayesianpymc` | PyMC (NUTS / MCMC) | Bayesian structural model: linear trend + Fourier seasonality + exog regression under a Student-T likelihood. Slowest by far. |

Every model, and every ensemble rule, gets the same kind of 95% band
(`yhat_lower`/`yhat_upper`), calibrated on its own validation errors; see
[Uncertainty bands](#uncertainty-bands) below.

All five are selected the same way — a hyperparameter grid search scored on
the **validation** window by `selection_metric` (MAPE, or WAPE when the target
has many zeros or goes negative) — and all five are then refit on
train + validation and reported on the test window, which no selection path
reads. So their reported numbers are directly comparable and feed the
ensemble on equal footing.

### prophet

Fits Prophet with regressors from your independent columns. Grid varies
`seasonality_mode` and the yearly/weekly seasonality strength. Its own
interval is replaced by the calibrated band, so Horizon's config sets
`uncertainty_samples: 0` and Prophet skips its interval simulation
altogether. A wrapper used on its own, with Prophet's defaults, still returns
`yhat_lower`/`yhat_upper`, seeded so they are the same on every call.
Prophet-safe `save()`/`load()` (not generic pickle).

### catboost

The adapter builds features from a fixed-reference trend index plus cyclical
(sin/cos) seasonal encodings and calendar fields — every feature is computable
from dates alone, with no dependence on the target's recent history. That is
deliberate: it lets one batched `predict()` cover any horizon without a
recursive per-step loop. The alternative — lag and rolling features — degrades
badly across sudden level shifts on real data (see [`design_notes.md`](design_notes.md) §5). Non-negative clipping is data-driven (only when the training target
was itself never negative). CatBoost-native `save()`/`load()`; `load()` reads the model
file from next to the saved envelope, so the pair can be moved together. Training logs
are not written to disk (`allow_writing_files=False`, overridable through `model_params`).

### sarimax

Fits `log1p(y)` as a (seasonal) ARIMA process with a standardised exogenous
matrix, and returns raw-unit `yhat`. Differencing order `d` is ADF-derived
from the data; `(p, q)`, seasonal `(P, Q)`, and `fourier_order_annual` are
grid-searched. Design points worth knowing:

- **Its own interval is not used.** The wrapper's constant-variance band
  (`mean ± z·σ` in log space) carries only the fitted model's innovation
  variance, none of the parameter or misspecification uncertainty, so it
  under-covered: 76% of actuals inside its 95% band on 80 real and synthetic
  test windows. The calibrated band covered 87% on the same windows.
- **Annual Fourier order is a hyperparameter** (`0` = off), because no fixed
  setting suited both daily and weekly real data.
- **Divergence guard**: if a fit saturates the log-space explosion clip across
  most of the horizon, `predict()` raises and the orchestrator drops the model
  into `evaluate_errors` rather than emitting a degenerate flat line.
- **Negative targets**: the log-space fit cannot represent a value below -1
  (it becomes a gap) and never forecasts below zero, so SARIMAX is left out,
  with a warning, of any run whose target has a negative value, and listed in
  `forecaster.skipped_models`.
- **Unexpected zeros are left out of the fit.** In log space a 0 among values
  in the thousands is an outlier of about 7 units; one or two such days took
  its test WAPE from 4 to 84 in testing. A 0 on a weekday that normally has
  values (daily), or any 0 when the series is normally above 0 (weekly), is
  treated as missing in SARIMAX's own fit, with a warning; the other models
  still see it as 0, and zeros on closed weekdays are kept. Weekly data holds
  up less well: above 10% zero weeks SARIMAX is left out of the run.
- **Exog is clamped to the training range** so an out-of-range future value
  cannot extrapolate a fitted coefficient into a divergence.
- **Annual-collinearity warning**: if an independent column strongly tracks
  the annual cycle (it explains the same variance as the Fourier terms), a
  warning is raised — on short histories the two compete and the higher-Fourier
  candidates can destabilise. Diagnostic only.

Uses the default pickle `save()`/`load()` (statsmodels serialises reliably).

### neuralprophet

Decomposition model via NeuralProphet/PyTorch Lightning. Deliberate v1 choices:
`n_lags=0` (no autoregression, so it stays comparable to the other models and
fits the fit-once/predict-any-horizon contract); CPU pinned and logging
silenced so Apple-silicon (MPS) and CPU-only runs agree; `epochs` and `learning_rate`
pinned via the grid for reproducibility (a `model_params` override must set both: with
either missing NeuralProphet runs its learning-rate finder, which fails on torch 2.6+); `collect_metrics: false` (required —
enabling metrics attaches a Lightning logger that crashes `fit()` via an
OmegaConf serialisation of the loss-function type union). No interval of its
own; the calibrated band applies as for every model. Torch-backed `save()`/`load()`, with a load fallback that
restores the (unpicklable) Lightning trainer.

### bayesianpymc

A Bayesian structural time-series model sampled with NUTS: a linear trend, a
Fourier seasonal basis (annual for both frequencies once the data spans 6
months, plus a weekly cycle for daily data), and a linear regression on your standardised independent columns,
all observed through a Student-T likelihood (heavy tails absorb spikes a
Gaussian fit would chase). Design points worth knowing:

- **`y` is z-scored inside the wrapper** (sampled in standardised space,
  back-transformed to raw units on predict), and the Fourier basis is built in
  the wrapper from absolute day offsets — same split as SARIMAX, so `y` keeps
  one raw meaning and the Fourier order is grid-searchable.
- **`predict()` reconstructs the forecast from the stored posterior draws** in
  pure numpy (a fit-once/predict-any-horizon linear combination of the sampled
  coefficients), so prediction never re-runs the sampler and never imports
  pymc.
- **Coefficient priors are chosen per column**: a continuous driver (spend,
  impressions) gets an informative prior centred on its OLS slope against the
  target; a binary flag gets a plain zero-centred prior.
- **Its own credible band is not used by Horizon.** Used on its own, the
  wrapper returns a posterior-**predictive** interval (the 3rd/97th
  percentiles of draws from `y ~ StudentT(nu, mu, sigma)`, a 94% band).
  HorizonForecaster replaces it with the calibrated band like every other
  model's; on the 12 test windows PyMC ran on, both covered about 95%.
- **Daily is tuned for scale.** On ~1000+ daily rows, seasonal-collinear exog
  makes NUTS take deep, slow trajectories. The daily config therefore halves
  draws/tune to 1000 (which barely moves the point forecast), keeps the
  Fourier order at 3–4, and **grid-selects `beta_fourier_sigma` (0.3 vs 1.0)**
  so the data chooses how tightly to regularise the Fourier basis — a tight
  prior helps on weak/collinear seasonality, a loose one preserves strong
  seasonality, and neither is universally better. Weekly samples cleanly at
  the full 2000/2000.
- **The weekly cycle stops at order 3.** `n_fourier` sets the annual order;
  on whole-day dates a weekly order of 4 only repeats order 3, so it is never
  used.
- **No annual terms from under 6 months of data.** A fit whose dates span
  fewer than 183 days leaves the annual Fourier terms out: from less than half
  a year the model has to guess the rest of the yearly cycle, and on real data
  that guess swung test WAPE from 11.5 to 73.9 between two windows a week
  apart. Each fit decides from its own dates, so with the default windows (84
  validation and 28 test days) a 7–9 month series has its validation fit
  run without them while the test fit and `forecast()` include them.
  `evaluate()` and `forecast()` warn once when this happens.
- **Convergence is checked.** `evaluate()` (test fit) and `forecast()` warn
  when the largest R-hat is above 1.05 or the smallest bulk effective sample
  size is below 100; raise `draws` and `tune` if you see it. The values are on
  the fitted wrapper as `sample_stats` (`max_rhat`, `min_ess_bulk`,
  `num_divergences`).
- **`save()`/`load()` write the PyMC `InferenceData` to a NetCDF sidecar** and
  pickle the rest; predictions reload exactly from the pickled posterior draws.

Expect this model to be substantially slower than the other four — it runs a
full MCMC per grid candidate.

On macOS, PyTensor (PyMC's compiler) adds a linker flag that recent Apple
toolchains reject; Horizon detects and works around this automatically. See
[`installation.md`](installation.md#macos-bayesian-pymc-and-the--ld64-linker-flag).

## Uncertainty bands

Every forecast Horizon returns, from each model and each ensemble rule, has a
95% band in `yhat_lower`/`yhat_upper`, built the same way from that forecast's
own errors on the validation window (split conformal prediction). The
models' own intervals are not used: they mean different things, two of
the five models (CatBoost and NeuralProphet) and every ensemble have
none, and the ones that exist cover too little.

**How it is built.** For a forecast made h periods after the last fitted
date, with L validation periods:

1. Take the absolute errors |actual − forecast| on the validation window,
   leaving out days that are 0 where the series normally has a value
   (outages; see below).
2. The width q is the ⌈0.95 × (n + 1)⌉-th smallest of the n errors kept: the
   81st of 84 by default for daily data, the 38th of 39 for weekly.
3. The band is forecast ± q × √(max(1, h ÷ L)): flat for as far ahead as the
   validation window reaches, then widening with the square root of the
   distance. When the target is never negative, the lower edge is floored
   at 0 wherever the forecast is at or above 0, so the band always contains
   its forecast.

An ensemble rule's band comes from its own validation forecast (the members'
validation forecasts combined by the same rule), never from the members'
bands. The models' validation forecasts already exist from selection, so the
bands cost no extra fits. How each band was calibrated is on
`EvaluationResult.band`, `ForecastResult.band` and `EnsembleResult.band`.

**How well it covers.** On 80 distinct test windows across 44 real and
synthetic series:

| | First 28 days / 13 weeks | Daily, 3–16 months ahead | Weekly, 9 months – 2 years ahead |
|---|---|---|---|
| Calibrated band | 87–95% by model; trimmed ensemble 91% | 82–88% | 57–61% |
| Models' own 95% band (Prophet, SARIMAX, PyMC) | 76–95% | 70–74% | 38–47% |

Near-term coverage sits a little under 95% partly because each band is
calibrated on the errors of the grid candidate that won for having the
smallest validation errors, which makes those errors slightly optimistic. In
the study, the grid-searched models covered less over the first four weeks
(Prophet 91%, SARIMAX 87%) than the two run with fixed settings (CatBoost and
NeuralProphet 93%), consistent with this. CatBoost's default 18-candidate grid
was not part of the study, so its coverage in a default run may be a little
lower than measured.
`model_summary` reports each band's coverage on the run's own test window
(`band_coverage`, `band_test_points`), so the 95% can be checked on every
run.

**Warnings and limits.**

- **Horizons past the validation window.** `forecast()` warns once when the
  horizon runs past L, with the coverage measured that far ahead. Long weekly
  horizons are hard for every method.
- **Generated drivers.** The band is calibrated with the real driver values;
  when `forecast()` generates them, coverage fell by about 9 points in
  testing, and the generated-drivers warning says so.
- **Outages.** Unexpected zero days are left out of the calibration, so a few
  lost days do not widen the band for every other day; the band does not
  cover such days, and a warning names how many were left out.
- **Fewer than 19 validation errors.** No 95% band can be guaranteed, so that
  output has no band columns and a warning says why. The default windows are
  far above this; a very short `validation_size` is not.
- **No validation forecast.** A model that could not be scored on the
  validation window (its training window below the model's row floor, for
  example) has no errors to calibrate from, so it has no band, and neither
  has an ensemble that includes it. The warning says so; a longer
  `validation_size` would not help, since it takes rows from training.
- **Settings for the models' own intervals are refused**
  (`interval_width` and `uncertainty_samples` for Prophet, `alpha` for SARIMAX,
  `ci_lower_pct`/`ci_upper_pct` for Bayesian PyMC), since they would have no
  effect. Prophet's `uncertainty_samples` is set to 0 by Horizon's own
  config, so it does not compute an interval that would be thrown away.

## See also

- [`ensemble_methods.md`](ensemble_methods.md) — how the per-model forecasts
  are combined and weighted (and why the ensemble is not guaranteed to win).
- [`future_exog.md`](future_exog.md) — how `forecast()` fills in future values
  for your independent columns when you do not supply them.
