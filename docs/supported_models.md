# Supported models

`HorizonForecaster` is a `MODEL_REGISTRY`-driven orchestrator: every model is
an `(adapter, wrapper)` pair registered in `forecaster.py`, and the same
`evaluate()`/`forecast()`/ensemble/export/plot pipeline runs each one through
a shared contract (`BaseModel`/`BaseAdapter`). Adding a model is a registry
entry, not a change to the orchestrator.

Four models are implemented and registered today; a fifth (Bayesian PyMC) is
planned. Select a subset with the `models=` constructor argument, or pass
`models=None` to run them all.

## Registered models

| Key | Backend | Uncertainty bounds | Notes |
|---|---|---|---|
| `prophet` | Prophet (cmdstanpy) | yes (`yhat_lower`/`yhat_upper`) | Additive/multiplicative seasonality; seeded so intervals are reproducible. |
| `catboost` | CatBoost | no | Gradient-boosted trees over trend-index + cyclical-seasonality features (no target-history lags), so it forecasts any horizon in one batched call. |
| `sarimax` | statsmodels | yes (constant-variance band) | Log-space (S)ARIMA with an exogenous regression matrix and grid-chosen annual Fourier terms. |
| `neuralprophet` | NeuralProphet (torch / PyTorch Lightning) | no (v1) | Decomposition model, `n_lags=0` (no autoregression in v1), CPU-pinned for reproducibility. |

All four are selected the same way — a hyperparameter grid search scored by
MAPE on the test split — so their reported numbers are directly comparable
and feed the ensemble on equal footing.

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
matrix, and returns raw-unit `yhat`. Differencing order `d` is ADF-derived
from the data; `(p, q)`, seasonal `(P, Q)`, and `fourier_order_annual` are
grid-searched. Design points worth knowing:

- **Prediction interval is a constant-variance band** (`mean ± z·σ` in log
  space), not `conf_int()`'s horizon-growing variance, which blows the raw
  upper bound up over long horizons.
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

## Planned

### bayesianpymc

Empty stub files exist (`data_adapter/bayesianpymc_adapter.py`,
`models/bayesianpymc_wrapper.py`) and PyMC is already a declared dependency,
but the model is **not registered yet** — it is Phase 6, Part 3. Until then
it will not run, and `models=["bayesianpymc"]` is not valid.

## Uncertainty bounds

Bounds (`yhat_lower`/`yhat_upper`) are optional and per-model — Prophet and
SARIMAX provide them, CatBoost and NeuralProphet (v1) do not. They flow into
per-model plots and CSVs but are **never combined into the ensemble**, which
operates on the `yhat` point forecasts alone.
