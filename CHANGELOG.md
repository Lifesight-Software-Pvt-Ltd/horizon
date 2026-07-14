# Changelog

All notable changes to `horizon_forecaster` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/); this project has not yet made a tagged release.

## [Unreleased]

### Added

- Project scaffolding: package layout, `pyproject.toml`/`setup.py`, CI (`tests.yml`, gated `publish.yml`), dependency spike confirming Prophet, NeuralProphet, SARIMAX (statsmodels), CatBoost, and PyMC import together cleanly. (Phase 0)
- Exception hierarchy (`exceptions.py`): `HorizonForecasterError` base plus 7 typed subclasses with structured attributes. (Phase 1)
- Shared utilities (`utils.py`): zero-variance column cleaning, frequency inference/validation, training-size and MAPE-threshold validators. (Phase 1)
- Evaluation metrics (`evaluation/metrics.py`): MAPE with explicit zero-actual handling (returns `MapeResult`, not a bare float), plus WAPE, RMSE, MAE. (Phase 1)
- Abstract `BaseModel` contract (`models/base.py`), stress-tested against stub versions of all 5 target models; uncertainty bounds (`yhat_lower`/`yhat_upper`) locked in as optional, individual-model-only, never propagated into an ensemble. (Phase 1)
- Data adapters (`data_adapter/base_adapter.py`, `data_adapter/prophet_adapter.py`) reshaping raw data for Prophet, with hard errors (not silent skips) on declared-but-missing columns. (Phase 2)
- `ProphetWrapper`: real fit/predict, regressor handling, Prophet-safe save/load (not generic pickle). (Phase 2)
- Frequency-aware `config/default_params.yaml` (`min_rows_required`: 14 daily / 8 weekly). (Phase 2)
- `HorizonForecaster.evaluate()`: chronological train/test split, fit, predict, MAPE. (Phase 2)
- `HorizonForecaster.forecast()`: full-data refit (once, cached), default future-exogenous-variable strategy (reuse/cycle historical values), horizon-vs-history warning. (Phase 2)
- Default hyperparameter grid search (6 candidates per frequency: 4 tuned + 2 safe `auto` fallbacks) in `evaluate()`/`forecast()`, auto-selecting the lowest-MAPE candidate; exposed via `EvaluationResult.grid_search_summary`. (Phase 2, real-dataset testing)
- Declared-vs-actual frequency mismatch warning in `HorizonForecaster.__init__()`. (Phase 2, real-dataset testing)
- `CatBoostAdapter` (`data_adapter/catboost_adapter.py`): stateful adapter engineering calendar features plus lag (1/2/3/7/14) and rolling (7/14) target-history features; a frozen "direct" strategy anchors the test/forecast horizon's features to the end of training history and broadcasts them across the whole horizon, avoiding recursive multi-step prediction and its compounding error. (Phase 3)
- `CatBoostWrapper` (`models/catboost_wrapper.py`): fit/predict via CatBoost's native `Pool` with explicit categorical-feature handling for calendar columns; CatBoost-native `save()`/`load()` (not generic pickle). (Phase 3)
- `catboost:` block in `config/default_params.yaml`: 18-candidate hyperparameter grid per frequency (16 tuned combinations + 2 safe low-capacity fallbacks), mirroring Prophet's config schema. (Phase 3)
- `HorizonForecaster` generalized from a single hardcoded Prophet model into a `MODEL_REGISTRY`-driven multi-model orchestrator: new `models=` (which registered models to run) and `model_params=` (per-model hyperparameter overrides that skip grid search) constructor parameters. Adding a future model should require only one new `MODEL_REGISTRY` entry. (Phase 3)
- Partial-failure handling in `evaluate()`/`forecast()`: one requested model failing no longer aborts the call for the others. Failures are captured per-call in `evaluate_errors`/`forecast_errors` (reset fresh on each call, not accumulated); `AllModelsRejectedError` is raised only when every requested model fails. (Phase 3)

### Fixed

- Non-reproducible Prophet uncertainty intervals (`yhat_lower`/`yhat_upper` varied across calls on an identical fitted model) — root-caused to Prophet's unseeded uncertainty simulation, fixed with an explicit, overridable random seed. (Phase 2)
- `config/default_params.yaml` missing from the built wheel despite working in editable installs. (Phase 2, packaging hardening)
- Fixed hyperparameter combinations producing broken forecasts (MAPE up to 802%, including negative predictions) on short-history datasets — resolved by adding safe `auto`-based fallback candidates to the grid search rather than a hand-coded data-span rule. (Phase 2, real-dataset testing)
- Test-split construction bug: the chronological test split was built by calling `transform_train()` on the held-out test rows in isolation. This worked incidentally for the stateless `ProphetAdapter`, but would have silently corrupted `CatBoostAdapter`'s stateful lag/rolling features (computed from the short, disconnected test split instead of real training history). Fixed by calling `transform_train()` once on the training data, then `transform_future()` (which reuses retained history) to build the test split's features. (Phase 3)
- `training_size` leaving zero evaluation rows now raises `InvalidConfigError` immediately at construction instead of inside per-model evaluation, where the new partial-failure handling would have silently swallowed it and misreported it as `AllModelsRejectedError` instead of the real, dataset-level configuration problem. (Phase 3)
- Missing coverage branch in `CatBoostAdapter`'s insufficient-history check (`rolling_windows` path was never exercised, since the existing test's `lags` check always failed first). (Phase 3)

### Changed

- Pinned all previously-unpinned dependencies; added `requires-python` constraint. (Phase 2, packaging hardening)
- Added a CI test matrix across Python 3.10–3.12, catching a `matplotlib` version incompatibility and a `setuptools 82` regression (`pkg_resources` removed). (Phase 2, packaging hardening)
- Added explicit `[tool.setuptools.packages.find]` include list to prevent stray local folders (e.g. `test_env/`) from being mistaken for a second top-level package. (Phase 2, packaging hardening)
- **Breaking:** `HorizonForecaster.evaluate()`/`.forecast()` now always return a `Dict[str, EvaluationResult]`/`Dict[str, ForecastResult]` keyed by model name, even when only one model is requested — previously returned a single result object directly. (Phase 3)
- `publish.yml` switched from `twine` + long-lived `PYPI_API_TOKEN` secrets to PyPI Trusted Publishing (OIDC via `pypa/gh-action-pypi-publish`) — no stored credentials, short-lived per-run tokens instead. Manual `workflow_dispatch` runs now take a `target` input (`testpypi` default / `pypi`), so a manual run defaults to a safe TestPyPI dry run rather than a real release; tag pushes (`v*`) still always publish to real PyPI.

### Known deferred issues

- 24 CVEs in `torch`/`pytorch-lightning`, forced by `neuralprophet==0.8.0`'s own version constraints — consciously deferred to Phase 6 rather than pinning an untested pre-release NeuralProphet.
- PyMC and NeuralProphet still need their own `save()`/`load()` overrides (Phase 6).
- `examples/real_dataset_testing.ipynb` exists in Colab but is not yet committed to this repository, pending GitHub PAT/org-admin access.
