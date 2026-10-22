# Supported models

`HorizonForecaster` is a `MODEL_REGISTRY`-driven orchestrator: every model is
an `(adapter, wrapper)` pair registered in `forecaster.py`, and the same
`evaluate()`/`forecast()`/ensemble/export/plot pipeline runs each one through
a shared contract (`BaseModel`/`BaseAdapter`). Adding a model is a registry
entry, not a change to the orchestrator.

All five models are implemented and registered today. Select a subset with
the `models=` constructor argument, or pass `models=None` to run them all.

## Registered models

| Key | Backend | Uncertainty bounds | Notes |
|---|---|---|---|
| `prophet` | Prophet (cmdstanpy) | yes (`yhat_lower`/`yhat_upper`) | Additive/multiplicative seasonality; seeded so intervals are reproducible. |
| `catboost` | CatBoost | no | Gradient-boosted trees over trend-index + cyclical-seasonality features (no target-history lags), so it forecasts any horizon in one batched call. |
| `sarimax` | statsmodels | yes (constant-variance band) | Log-space (S)ARIMA with an exogenous regression matrix and grid-chosen annual Fourier terms. |
| `neuralprophet` | NeuralProphet (torch / PyTorch Lightning) | no (v1) | Decomposition model, `n_lags=0` (no autoregression in v1), CPU-pinned for reproducibility. |
| `bayesianpymc` | PyMC (NUTS / MCMC) | yes (posterior credible band) | Bayesian structural model: linear trend + Fourier seasonality + exog regression under a Student-T likelihood. Slowest by far. |

All five are selected the same way — a hyperparameter grid search scored on
the **validation** window by `selection_metric` (MAPE, or WAPE when the target
has many zeros or goes negative) — and all five are then refit on
train + validation and reported on the test window, which no selection path
reads. So their reported numbers are directly comparable and feed the
ensemble on equal footing.

### prophet

Fits Prophet with regressors from your independent columns. Grid varies
`seasonality_mode` and the yearly/weekly seasonality strength. Emits genuine
prediction intervals; its uncertainty simulation is explicitly seeded so
`yhat_lower`/`yhat_upper` do not drift between calls on the same fitted model.
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

- **Prediction interval is a constant-variance band** (`mean ± z·σ` in log
  space), not `conf_int()`'s horizon-growing variance, which blows the raw
  upper bound up over long horizons.
- **The band can under-cover when the model is misspecified — by design, not
  a bug.** Like all analytic ARIMA-family intervals, it propagates only the
  fitted model's *innovation* variance; it carries no parameter-estimation or
  model-misspecification uncertainty. So whenever the fit slightly misses the
  true structure, real errors exceed a band that is only "correct" under the
  idealised model. This bites hardest on **low-noise, strongly-structured
  series** (e.g. a large deterministic seasonal swing with little residual
  noise): the band is sized by the small noise while the errors are dominated
  by structural misfit. A calibration sweep on such synthetic data measured
  ~75% empirical coverage against a 95% nominal band (and, tellingly, raising
  the Fourier order did *not* fix it, confirming it is the interval
  assumption, not seasonal resolution). On noisier real-world data the analytic
  band is relatively wider and coverage is usually much closer to nominal. The
  point forecast (`yhat`/MAPE) is unaffected — this is purely about interval
  width. **When calibrated uncertainty matters more than the point estimate,
  prefer `bayesianpymc`'s posterior-predictive band** (it integrates parameter
  uncertainty by sampling and stayed ~94%-calibrated on the same series) **or
  read the spread across the ensemble.**
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
OmegaConf serialisation of the loss-function type union). No uncertainty
bounds in v1. Torch-backed `save()`/`load()`, with a load fallback that
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
- **Credible band**: `yhat_lower`/`yhat_upper` are a posterior-**predictive**
  interval — the 3rd/97th percentiles of draws from `y ~ StudentT(nu, mu,
  sigma)`, i.e. the mean-function uncertainty plus the observation noise (a
  94% band). Percentiles of `mu` alone (the latent mean) would omit the noise
  and badly under-cover.
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
  apart. Each fit decides from its own dates, so on a 7–9 month series the
  validation fit can run without them while the test fit and `forecast()`
  include them. `evaluate()` and `forecast()` warn once when this happens.
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

## Uncertainty bounds

Bounds (`yhat_lower`/`yhat_upper`) are optional and per-model — Prophet,
SARIMAX and Bayesian PyMC provide them, CatBoost and NeuralProphet (v1) do
not. They flow into per-model plots and CSVs but are **never combined into the
ensemble**, which operates on the `yhat` point forecasts alone.

## See also

- [`ensemble_methods.md`](ensemble_methods.md) — how the per-model forecasts
  are combined and weighted (and why the ensemble is not guaranteed to win).
- [`future_exog.md`](future_exog.md) — how `forecast()` fills in future values
  for your independent columns when you do not supply them.
