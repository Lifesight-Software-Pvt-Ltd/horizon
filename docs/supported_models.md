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

All five are selected the same way — a hyperparameter grid search scored by
MAPE on the **validation** window — and all five are then refit on
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
was itself never negative). CatBoost-native `save()`/`load()`.

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
pinned via the grid for reproducibility; `collect_metrics: false` (required —
enabling metrics attaches a Lightning logger that crashes `fit()` via an
OmegaConf serialisation of the loss-function type union). No uncertainty
bounds in v1. Torch-backed `save()`/`load()`, with a load fallback that
restores the (unpicklable) Lightning trainer.

### bayesianpymc

A Bayesian structural time-series model sampled with NUTS: a linear trend, a
Fourier seasonal basis (annual for both frequencies, plus a weekly cycle for
daily data), and a linear regression on your standardised independent columns,
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
  draws/tune to 1000 (which barely moves the point forecast), caps the Fourier
  order at 4, and **grid-selects `beta_fourier_sigma` (0.3 vs 1.0)** so the
  data chooses how tightly to regularise the Fourier basis — a tight prior
  helps on weak/collinear seasonality, a loose one preserves strong
  seasonality, and neither is universally better. Weekly samples cleanly at
  the full 2000/2000.
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
