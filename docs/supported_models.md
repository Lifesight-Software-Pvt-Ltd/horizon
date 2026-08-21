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
MAPE on the **validation** window — and all five are then refit on train +
validation and reported on the **test** window, so their numbers are directly
comparable and feed the ensemble on equal footing. SARIMAX included: it used to
be the odd one out, self-tuning through a private search over a tail it carved
out of train itself, which is no longer the case (see below).

Non-negative output is **not** guaranteed library-wide. CatBoost, NeuralProphet
and Bayesian PyMC clamp negative predictions at 0 only when the training target
was itself never negative — a data-driven decision, so a series that
legitimately goes negative is returned unaltered. SARIMAX is non-negative
because its back-transform clips the log-space forecast at 0 before
`expm1` — fitting in `log1p` space alone would not guarantee it, since
`expm1` of a negative log value is itself negative. Prophet applies no clamp at all and can
emit negative point forecasts on a low-level series. If your downstream use
cannot tolerate that, clip the output yourself; nothing in the library does it
for you.

### prophet

Fits Prophet with regressors from your independent columns. Grid varies
`seasonality_mode` and the yearly/weekly seasonality strength. Emits genuine
prediction intervals; its uncertainty simulation is explicitly seeded so
`yhat_lower`/`yhat_upper` don't drift between calls on the same fitted model.
Prophet-safe `save()`/`load()` (not generic pickle).

### catboost

The adapter builds features from a fixed-reference trend index plus cyclical
(sin/cos) seasonal encodings and calendar fields — every feature is computable
from dates alone, with no dependence on the target's recent history. That is
deliberate: it lets one batched `predict()` cover any horizon without a
recursive per-step loop, and it replaced an earlier lag/rolling-feature design
that failed badly across sudden level shifts on real data (see the Phase 3
build log). Non-negative clipping is data-driven (only when the training target
was itself never negative). CatBoost-native `save()`/`load()`.

### sarimax

Fits `log1p(y)` as a (seasonal) ARIMA process with a standardised exogenous
matrix, and returns raw-unit `yhat`.

**Selection works exactly like every other model's.** `(p, q)`, seasonal
`(P, Q)` and `fourier_order_annual` come from the shared, forecaster-level
hyperparameter grid, scored on the validation window. Only the differencing
order `d` is decided inside the wrapper, by an ADF test on the log target —
that is a property of the data, not a hyperparameter to tune. (SARIMAX
previously ran its own internal search over a validation tail it carved out of
train itself, on the grounds that shared selection meant selecting on the
window that got reported. The three-way split removed that concern for
everybody, so the private search went with it. Any older documentation saying
SARIMAX "self-tunes internally and has no grid" is wrong.)

Design points worth knowing:

- **A deterministic trend term is passed when nothing is differenced.** With
  `d=0` and `D=0`, statsmodels' default of no constant means a *zero-mean*
  process — and this wrapper fits in `log1p` space, where a typical business
  series sits around 7. Asked to fit a zero-mean model to data centred at 7, the
  optimiser pushes the AR root to ~1 so the level decays as slowly as possible;
  the fit looks respectable in-sample while the forecast mean-reverts toward
  zero. On synthetic data this scored ~98% MAPE on a trending daily series and
  ~98% on a flat one; with the term in place, ~6–7% and ~4% respectively. The
  term used is `"ct"` (constant *and* linear trend), not just `"c"`: ADF's
  default constant-only regression will happily call a trending series
  stationary, so "not differenced" cannot be read as "has no trend", and a
  constant alone leaves the forecast reverting to the training mean. On a
  genuinely flat series the slope is estimated at essentially zero and costs
  nothing. Fits with `d>=1` are completely unaffected — differenced series
  behave exactly as they always did.
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
  the Fourier order did *not* fix it — confirming it's the interval
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
  can't extrapolate a fitted coefficient into a blow-up.
- **Annual-collinearity warning**: if an independent column strongly tracks
  the annual cycle (it explains the same variance as the Fourier terms), a
  warning is raised — on short histories the two compete and the higher-Fourier
  candidates can destabilise. Diagnostic only.

Uses the default pickle `save()`/`load()` (statsmodels serialises reliably).

### neuralprophet

Decomposition model via NeuralProphet/PyTorch Lightning. Deliberate v1 choices:
`n_lags=0` (no autoregression, so it stays comparable to the other models and
fits the fit-once/predict-any-horizon contract); CPU pinned and logging
silenced so local (MPS) and CI (CPU) runs agree; `epochs` and `learning_rate`
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
- **Non-negative clipping is data-driven**, matching CatBoost and
  NeuralProphet: the floor at 0 is applied only when the training target was
  itself never negative. It used to be unconditional, which silently rewrote
  every negative prediction on a legitimately-negative target (profit, margin,
  net change).
- **`save()`/`load()` write the PyMC `InferenceData` to a NetCDF sidecar** and
  pickle the rest; predictions reload exactly from the pickled posterior draws.

Expect this model to be substantially slower than the other four — it runs a
full MCMC per grid candidate.

## Uncertainty bounds

Bounds (`yhat_lower`/`yhat_upper`) are optional and per-model — Prophet,
SARIMAX and Bayesian PyMC provide them, CatBoost and NeuralProphet (v1) do
not. They flow into per-model plots and CSVs but are **never combined into the
ensemble**, which operates on the `yhat` point forecasts alone.

## See also

- [`ensemble_methods.md`](ensemble_methods.md) — how the per-model forecasts
  are combined (trimmed mean by default) and why the ensemble isn't guaranteed
  to win.
- [`future_exog.md`](future_exog.md) — how `forecast()` fills in future values
  for your independent columns when you don't supply them.
