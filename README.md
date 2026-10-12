# Horizon Forecaster

An open-source ensemble time-series forecasting library that runs several
strong forecasting models through one shared interface and combines them into
a single forecast — with transparent, per-model accuracy reporting so you
always know *which* forecast you should trust.

It is built for business time series such as revenue, orders, installs,
conversions, or sign-ups at **daily** or **weekly** granularity, optionally
driven by independent variables (ad spend, impressions, promos, holidays,
stockouts, and so on).

Rather than betting on one model, Horizon Forecaster fits Prophet, CatBoost,
SARIMAX, NeuralProphet, and a Bayesian PyMC model on the same data, tunes and
screens each one on a **validation** window, reports each one's accuracy on a
separate **test** window that no selection step is ever allowed to read, and
combines the survivors — while telling you the per-model MAPE, which models
were excluded and why, and whether the combined forecast actually beats the
best individual model (it is not guaranteed to).

---

## Supported models

Every model is an `(adapter, wrapper)` pair registered in one `MODEL_REGISTRY`,
and all of them run through the same `evaluate()` / `forecast()` / ensemble /
export / plot pipeline. They are selected the same way — a hyperparameter grid
search scored by MAPE on the validation window — so their numbers are directly
comparable.

| Model | Backend | Uncertainty bounds | Notes |
|---|---|---|---|
| `prophet` | Prophet (cmdstanpy) | yes | Additive/multiplicative seasonality; seeded so intervals are reproducible. Can emit negative point forecasts on a low-level series — no output clamp is applied. |
| `catboost` | CatBoost | no | Gradient-boosted trees over trend-index + cyclical-seasonality features (no target lags), so it forecasts any horizon in one batched call. |
| `sarimax` | statsmodels | yes | Log-space (S)ARIMA with a standardised exogenous regression matrix. Grid-searched like every other model — `(p, q)` and the annual Fourier order come from the shared grid, joined by seasonal `(P, Q)` at daily frequency (weekly disables the seasonal order, since an s=52 term is prohibitively slow, and leans on the Fourier terms instead); only `d` is derived from the data by an ADF test inside the wrapper. |
| `neuralprophet` | NeuralProphet (torch) | no (v1) | Decomposition model, `n_lags=0` (no autoregression in v1), CPU-pinned for reproducibility. |
| `bayesianpymc` | PyMC (NUTS/MCMC) | yes | Bayesian structural model: linear trend + Fourier seasonality + exog regression under a Student-T likelihood. Slowest by far. |

Uncertainty bounds (`yhat_lower` / `yhat_upper`) are per-model and optional —
Prophet, SARIMAX, and Bayesian PyMC provide them; CatBoost and NeuralProphet
(v1) do not. They flow into per-model plots and CSVs but are **never combined
into the ensemble**, which operates on the point forecasts alone. See
[`docs/supported_models.md`](docs/supported_models.md) for full per-model
behaviour.

> **Note on speed:** Bayesian PyMC runs a full MCMC sample per grid candidate
> and is dramatically slower than the other four. Pass `models=` to exclude it,
> or give it a small-draw override, when you want a fast run.

---

## Key features

- **Five models, one interface.** Fit and compare Prophet, CatBoost, SARIMAX,
  NeuralProphet, and Bayesian PyMC with identical calls; add a model by
  registering one `(adapter, wrapper)` pair, with no change to the orchestrator.
- **Honest, held-out evaluation.** `evaluate()` splits chronologically into
  **train / validation / test**. Every decision — the hyperparameter grid
  search, `mape_threshold` screening, inverse-MAPE weights — is scored on the
  validation window; the winner is then refit on train + validation and scored
  once on the test window, which no selection path ever reads. The reported
  `mape` is therefore an out-of-sample number rather than the score a model was
  chosen for, and the selection score is kept alongside it as
  `validation_mape` (with explicit zero-actual handling in both).
- **Robust combination rules.** Models at or below a configurable
  `mape_threshold` are combined by a **trimmed mean** (the default: discard the
  highest and lowest member each period, average the rest), or by
  `inverse_mape` (lower validation MAPE → more weight) or `equal` weighting.
  Ask for several at once, or all three. The combined result's own accuracy is
  scored against real actuals, so you can check whether it beats the best
  single model.
- **Transparent failure & rejection reporting.** A model that fits but scores
  above the threshold is *rejected* from the ensemble (reported with its MAPE);
  a model that fails to fit at all is recorded separately. One model failing
  never aborts the others.
- **Daily and weekly granularity**, flexible target variables, and optional
  independent drivers.
- **Future-driver strategies.** When you forecast without supplying future
  values for your drivers, they're generated by a selectable strategy
  (`carry_forward`, `recent_average`, season-aware `historical_average`,
  `seasonal_naive` with drift, `linear_trend`, or `cycle`), with type-aware
  handling of event flags and partial fill for the columns you *do* know.
- **Ready-to-use outputs.** Overlay and per-model charts, a MAPE comparison bar
  chart, and a one-call export of the run's CSVs plus PNGs.
- **Built-in safety warnings** for over-long horizons, auto-generated future
  driver values, and forecasts that land implausibly far outside history.

---

## Installation

Requires **Python 3.10–3.13** and a C/C++ toolchain (Prophet and PyMC compile
at install/first use; on 3.13 numpy is built from source too — see
[`docs/installation.md`](docs/installation.md)). A PyPI release is coming; for
now, install from a clone:

```bash
git clone https://github.com/Lifesight-Software-Pvt-Ltd/horizon.git
cd horizon
pip install -e .            # editable install; runtime deps are pinned in pyproject.toml
pip install -e ".[dev]"     # add pytest, pytest-cov, black, flake8, bandit
```

See [`docs/installation.md`](docs/installation.md) for platform notes and the
rationale behind the load-bearing dependency pins.

---

## Quickstart

```python
import pandas as pd
from horizon_forecaster.forecaster import HorizonForecaster

# A tidy DataFrame: a date column, a numeric target, optional driver columns.
df = pd.DataFrame({
    "date":   pd.date_range("2022-01-01", periods=400, freq="D"),
    "orders": [...],
    "spend":  [...],   # optional independent driver
})

forecaster = HorizonForecaster(
    df,
    date_column="date",
    target_column="orders",
    frequency="daily",              # "daily" or "weekly"
    independent_columns=["spend"],  # omit if you have none
    validation_size=56,             # periods used for selection; None = frequency default
    test_size=28,                   # periods reported; None = frequency default
    models=None,                    # None = all five; or e.g. ["sarimax", "catboost"]
    mape_threshold=40.0,            # degenerate-fit filter on the validation MAPE; None disables it
    ensemble_method="trimmed",      # or "inverse_mape", "equal", a list, or None for all three
)

# 1) Evaluate every model: tuned on validation, reported on test
results = forecaster.evaluate()
for name, res in results.items():
    print(f"{name}: test MAPE {res.mape.value:.2f}% "
          f"(validation {res.validation_mape.value:.2f}%)")

ens = forecaster.evaluate_ensemble["trimmed"]   # dict keyed by method
if ens.combined is not None:
    print("ensemble MAPE:", ens.mape.value, "rule applied:", ens.rule_applied)

# 2) Forecast the future (refits on all data, once, cached)
forecasts = forecaster.forecast(horizon=30)
future = forecasts["sarimax"].forecast      # DataFrame ["ds", "yhat"] (+ bounds where available)
```

Training takes whatever the two evaluation windows leave. Leave
`validation_size`/`test_size` as `None` to use the per-frequency defaults (56
validation / 28 test for daily, 26 / 13 for weekly), pass a float strictly
between 0 and 1 for a fraction of the data, or use `training_size=80` for the
percentage route instead — the two routes are mutually exclusive.

`evaluate()` and `forecast()` always return a dict keyed by model name (even
for a single model). Full walkthrough — including exporting CSVs/plots and
supplying planned future driver values — in
[`docs/quickstart.md`](docs/quickstart.md).

---

## How the ensemble works

`build_ensembles()` combines the point forecasts of every model whose
**validation** MAPE is at or below `mape_threshold`, using one or more
combination rules. The default is a **trimmed mean**: per period, discard the
highest and lowest member and average what remains. `inverse_mape` and `equal`
weighting are also available and can be built alongside it — the models are
already fitted by then, so extra rules are close to free.
`evaluate_ensemble` / `forecast_ensemble` are a `Dict[str, EnsembleResult]`
keyed by method, always a dict even for a single method. Excluded models are
reported in `rejected_models` with their MAPE and the threshold they exceeded —
distinct from models that failed to fit, which land in `evaluate_errors` /
`forecast_errors`.

`mape_threshold` (default 40) is best understood as a filter for
**degenerate fits** rather than an accuracy screen: the trimmed rule already
discards the worst member on every period, so screening hard *and* trimming
spends the same robustness budget twice. Pass `None` to admit every model.

**A combined ensemble is not guaranteed to beat the best single model.** With
static weights it can be dragged down on exactly the periods where a weaker
member is wrong. Trimming is considerably more robust to that specific failure
(it drops the offending member on the periods where it offends) but is still
not a guarantee. This is a real property of the method, not a bug, which is why
`evaluate()` scores each ensemble against actuals and the exporter writes *both*
a `final_ensemble_<method>` and a `final_best_model` CSV so you can choose per
run. See [`docs/ensemble_methods.md`](docs/ensemble_methods.md).

---

## Outputs

`export_forecast()` writes, into one folder:

- **CSVs** — one `final_ensemble_<method>` per combination rule, plus
  `final_best_model`, `per_model`, `rejected_models` (per model *and* method), and
  `model_summary` (validation MAPE, test MAPE, method, rule applied, weight and
  status).
- **Plots** (unless disabled) — one overlay per combination rule (every
  model against that rule), one dedicated figure per model, and one per rule.
  Five models and three rules give three overlays and eight individual
  figures.

The plotting functions in `horizon_forecaster.plotting` can also be called
directly; each returns a matplotlib `Axes`/`Figure` without calling
`plt.show()`, so you control rendering.

---

## Architecture

`HorizonForecaster` is a registry-driven orchestrator. Each model implements a
shared contract:

- a **data adapter** (`BaseAdapter`) that reshapes raw user data into that
  model's expected input, and
- a **model wrapper** (`BaseModel`) with `fit()` / `predict()` plus optional
  native `save()` / `load()`.

Adding a model means writing that pair and adding one `MODEL_REGISTRY` entry —
nothing in the orchestrator changes. The rule that keeps it that way: if a new
model forces a change to the orchestrator's core logic, that signals a gap in
the shared contract, to be fixed in the contract rather than special-cased in
the orchestrator. All five registered models satisfy it.

---

## Project status & roadmap

**Version 1.0.0 — stable and ready to use.** The model set, the three-way
split, the combination rules and the export/plot layer are implemented and
tested across daily and weekly data, on Python 3.10 through 3.13. The API
described here is the one you can build on; anything that changes it will come
with a version bump and a changelog entry.

Install from a clone today (see [Installation](#installation)); a PyPI release
is next.

Known gaps, deliberately **not** in 1.0 and all candidates for future work:

- **Rolling-origin backtesting.** Every score here comes from a single test
  window, so it reflects one particular stretch of history rather than an
  average over several origins.
- **Benchmark-relative metrics** (RelMAE / MASE). MAPE alone says nothing about
  whether a model beats a seasonal naive baseline, and is unusable on series
  with zero actuals.
- **Family de-duplication.** Prophet and NeuralProphet are close relatives, so a
  combination containing both is less diverse than its member count suggests;
  nothing currently accounts for that.
- **An error-diversity diagnostic.** The value of combining depends on members
  making *different* errors, which the library does not yet measure or report.
- **Runnable example notebooks.** A clean `pip install` plus an Open-in-Colab
  badge is the form these should take, so they follow the PyPI release.

The reasoning behind the design — the contracts, the per-model decisions, and
the failure modes that produced them — is in
[`docs/design_notes.md`](docs/design_notes.md); the release record is in
[`CHANGELOG.md`](CHANGELOG.md).

---

## Documentation

- [`docs/installation.md`](docs/installation.md) — install & dependencies
- [`docs/quickstart.md`](docs/quickstart.md) — end-to-end usage
- [`docs/supported_models.md`](docs/supported_models.md) — per-model behaviour
- [`docs/ensemble_methods.md`](docs/ensemble_methods.md) — combination rules & caveats
- [`docs/future_exog.md`](docs/future_exog.md) — future-driver strategies
- [`docs/design_notes.md`](docs/design_notes.md) — why the library is built this way
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to contribute, and what is in scope
- [`CHANGELOG.md`](CHANGELOG.md) — full change history

---

## Contributing

Contributions are welcome — bug fixes, new models, new combination rules or
future-driver strategies. Please open an issue before a pull request for
anything beyond a typo; [`CONTRIBUTING.md`](CONTRIBUTING.md) covers the setup,
the conventions, what is in scope and what is likely to be turned down, and
[`docs/design_notes.md`](docs/design_notes.md) explains why the load-bearing
decisions are what they are. In short: every model follows the same
adapter → wrapper → config → tests pattern, and a new model must pass through
the existing pipeline without modifying the orchestrator.

Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through
[`SECURITY.md`](SECURITY.md), not the public issue tracker.

---

## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
