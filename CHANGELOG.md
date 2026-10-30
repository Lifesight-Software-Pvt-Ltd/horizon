# Changelog

All notable changes to `horizon_forecaster` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-10-26

The first public release. The reasoning behind each choice below is in
[`docs/design_notes.md`](docs/design_notes.md).

### Models

- Five models behind one interface: Prophet, CatBoost, SARIMAX,
  NeuralProphet and a Bayesian structural model in PyMC. Each is an
  `(adapter, wrapper)` pair in `MODEL_REGISTRY`, so adding a model needs no
  change to the orchestrator. See [`docs/supported_models.md`](docs/supported_models.md).
- Every model is grid-searched on the validation window from
  `config/default_params.yaml`; `model_params` fixes a model's settings and
  skips its grid. Unknown model names, unknown settings for SARIMAX and
  PyMC, and settings that only shape a model's own interval are refused.
- CatBoost forecasts from a trend index and cyclical seasonality, with no
  target lags, so one batched call covers any horizon.
- SARIMAX fits in `log1p` space with an ADF-chosen `d`, a trend term when
  nothing is differenced, a divergence guard, and future drivers clamped to
  their training range. It is left out of a run whose target goes negative,
  or of a weekly run with more than 10% zero weeks (`skipped_models`); zeros
  where the series normally has a value are left out of its fit.
- NeuralProphet runs without autoregression (`n_lags=0`), pinned to the CPU,
  with `epochs` and `learning_rate` both required in an override.
- Bayesian PyMC samples with NUTS under a Student-T likelihood, rebuilds
  forecasts from stored posterior draws, adds annual terms only to fits
  spanning 183 days or more, caps the weekly cycle at order 3, and warns when
  sampling did not converge (R-hat above 1.05 or bulk ESS below 100). A macOS
  linker workaround for PyTensor runs automatically where it is needed.
- Every wrapper can be saved and loaded.

### Evaluation

- Data splits chronologically into train, validation and test: 84 / 28
  periods for daily data and 39 / 13 for weekly by default, scaled down on a
  short series to at most 45% of the rows. `validation_size` and `test_size`
  take periods or a fraction; `training_size` is a percentage-only
  alternative.
- Every decision (the grid search, the `mape_threshold` screen and the
  inverse-MAPE weights) reads the validation window. The winner is refit on
  train + validation and scored once on the test window, so `mape` is an
  out-of-sample number and `validation_mape` the score it was chosen by.
- MAPE and WAPE are both reported. `selection_metric="auto"` selects on WAPE
  when the target is 0 in 10% or more of its periods or goes negative;
  `"mape"` or `"wape"` fixes the choice. A target that is 0 in half its
  periods or more is refused.
- A seasonal-naive baseline is scored on the same windows
  (`naive_baseline`), with a warning for every model and rule that does
  worse, and `test_error_correlation` shows how alike the models' mistakes
  are.

### Combination rules

- A per-period trimmed mean is the default; `inverse_mape` and `equal` are
  available, and `ensemble_method=None` builds all three. See
  [`docs/ensemble_methods.md`](docs/ensemble_methods.md).
- `mape_threshold` (default 40, or `None`) drops degenerate fits before
  combining. Each `EnsembleResult` names the rule that actually ran
  (`rule_applied`), whether it fell back for lack of members (`degraded`),
  how often trimming dropped each member (`trim_shares`), and why each
  excluded model was left out (`rejected_models`).

### Uncertainty bands

- Every model and every ensemble rule carries a 95% band in
  `yhat_lower` / `yhat_upper`, calibrated on that forecast's own validation
  errors (split conformal prediction). It is flat for as far ahead as the
  validation window reaches and widens with the square root of the distance
  after that. Unexpected zero days are left out of the calibration, and when
  the target is never negative the lower edge is floored at 0 wherever the
  forecast is at or above 0.
- On 80 test windows across 44 real and synthetic series the bands held
  87–95% of actuals over the first four weeks (the trimmed ensemble 91%),
  and 82–88% from 3 to 16 months ahead on daily data.
- How each band was calibrated is on `.band`; `model_summary` reports each
  band's coverage of the test window. See
  [`docs/supported_models.md`](docs/supported_models.md#uncertainty-bands).

### Future driver values

- Drivers not supplied for the forecast are generated once, for every model.
  The default, `"auto"`, is `last_year` with a year or more of history: each
  driver is copied from the same weekday 364 days earlier, or the same week
  52 weeks earlier, so the latest year repeats, and a driver live now but
  absent a year ago holds its last value. With less history it is
  `carry_forward`. The others are `recent_average`, `historical_average`,
  `seasonal_naive` with drift, `linear_trend` and `cycle`. Event flags are 0
  except under `seasonal_naive` and `last_year`, where they repeat. Supplied
  columns are used as given and only the rest are generated. See
  [`docs/future_exog.md`](docs/future_exog.md).
- A supplied future table must match the forecast exactly (rows, columns,
  dates, values); values outside a driver's historical range warn.
- `future_exog_used` shows every value the forecast was given.

### Data checks

- Dates are parsed before anything is sorted: one of six text formats or a
  datetime, each date once, weekly dates on one weekday, and no date missing.
  Ambiguous day/month dates raise unless `date_format` is given.
- The date, the target and every driver must be complete and finite. On/off
  text (`yes/no`, `true/false`, `y/n`, `t/f`, `on/off`) is read as 1/0; other
  text is refused.
- Each column plays one role, and each driver is named once. A driver that
  never changes is dropped with a warning; one that never changes inside a
  fit window is left out of that fit for every model.

### Warnings

- One warning per call for each problem it finds, among them: models that
  failed, were screened out or left an ensemble short; rules that formed no
  ensemble; models worse than the seasonal-naive baseline; generated future
  drivers; a last period that `carry_forward` would hold; events from last
  year missing from, or copied into, a generated future; drivers that
  started or stopped since the year `last_year` copies; future values
  outside the historical range; over-long horizons and horizons past the
  bands' validation window; outputs with no band; zero days left out of the
  calibration; PyMC fits without annual terms or with poor convergence; and
  forecasts far outside history.

### Charts and exports

- `plot_evaluation()` and `plot_forecast()` draw one overlay per combination
  rule, `plot_evaluation_per_model()` and `plot_forecast_per_model()` one
  figure per model and rule, each with its 95% band, and
  `plot_model_comparison()` a bar chart of test MAPE or WAPE with the
  seasonal-naive baseline.
- `export_forecast()` writes one `final_ensemble_<method>` CSV per rule,
  `final_best_model` (the best validation score among models within
  `mape_threshold` on validation and test), `per_model`, `rejected_models`,
  `model_summary` and, with `forecaster=`, `error_correlation`, plus the
  charts as PNGs.

### Platforms and packaging

- Python 3.10–3.13, with every runtime dependency pinned to a tested version.
  `neuralprophet==0.8.0` requires `numpy<2`, so on Python 3.13 pip builds
  numpy from source; see [`docs/installation.md`](docs/installation.md).
- Importing the package loads no model backend; each wrapper imports its own
  inside `fit()`.
- A lint workflow (black, flake8, bandit) on every pull request and push to
  `main`, Dependabot, issue and pull-request templates, `CODEOWNERS`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md` and a `NOTICE` file. The test suite
  fits real models and is run by hand; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

### Known gaps

- Rolling-origin backtesting: every score comes from a single test window.
- Per-step band calibration for long horizons.
- Benchmark-relative metrics (RelMAE, MASE).
- Family de-duplication of Prophet and NeuralProphet.
- Acting on error diversity: the error correlation is reported, not used.
- Non-negativity is not enforced library-wide; Prophet can forecast below 0.
- NeuralProphet autoregression (`n_lags > 0`).
- 24 CVEs in `torch` / `pytorch-lightning`, forced by `neuralprophet==0.8.0`'s
  own version constraints.
- A PyPI release: the package installs from a clone.
