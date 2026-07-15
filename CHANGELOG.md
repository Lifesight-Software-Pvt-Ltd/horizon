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
- `CatBoostAdapter` (`data_adapter/catboost_adapter.py`): feature engineering built from a trend index (`time_idx`, an absolute day/week offset from a fixed reference date) plus cyclical yearly/half-yearly seasonality (and, for daily data, weekly/monthly seasonality too) via sin/cos encoding, calendar fields, and independent columns passed through as-is. Every feature is computed identically by `transform_train()` and `transform_future()` directly from dates alone — none depend on the target's own recent history, so the adapter needs no "frozen" mechanism for multi-step horizons (see Fixed below for why this replaced an earlier lag/rolling-feature design). (Phase 3, redesigned after real-dataset testing)
- `CatBoostWrapper` (`models/catboost_wrapper.py`): fit/predict via CatBoost's native `Pool`; CatBoost-native `save()`/`load()` (not generic pickle); data-driven non-negative prediction clipping (only applied when the training target was itself always non-negative). (Phase 3)
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
- **CatBoost long-horizon accuracy**: the original `CatBoostAdapter` design (lag(1/2/3/7/14) + rolling(7/14) target-history features, frozen at train-end and broadcast unchanged across the whole forecast horizon — necessary because `CatBoostWrapper.predict()` is one batched call, not a per-step recursive loop) produced catastrophic MAPE on real data whenever the horizon crossed a sudden target level-shift: 52.6% MAPE in the first half of a 144-day test window vs. 27.3% in the second half, and 67-90% MAPE specifically on a real, flagged demand-dip period vs. 34-40% on ordinary days — because a frozen snapshot of "recent history" has no way to see a spike or dip coming and keeps predicting the pre-shift baseline for the entire horizon. Root-caused by comparing against an independently-built, dedicated CatBoost reference notebook against the same dataset (~16% MAPE, no such failure mode) that used a trend index + cyclical seasonality instead of lag/rolling features. Fixed by replacing the feature set entirely (see Added above) rather than tuning around the symptom. (Phase 3, real-dataset testing)
- `CatBoostWrapper._categorical_feature_names` is now always empty — no calendar feature is declared categorical, matching the working reference notebook (which used none) rather than the original, untested categorical-encoding assumption. (Phase 3)
- Fixed `CatBoostWrapper.save()`/`.load()` silently dropping the new non-negative-clipping decision (`_target_min`) on a reload — caught while adding the feature, before it shipped as a real bug. (Phase 3)

### Changed

- Pinned all previously-unpinned dependencies; added `requires-python` constraint. (Phase 2, packaging hardening)
- Added a CI test matrix across Python 3.10–3.12, catching a `matplotlib` version incompatibility and a `setuptools 82` regression (`pkg_resources` removed). (Phase 2, packaging hardening)
- Added explicit `[tool.setuptools.packages.find]` include list to prevent stray local folders (e.g. `test_env/`) from being mistaken for a second top-level package. (Phase 2, packaging hardening)
- **Breaking:** `HorizonForecaster.evaluate()`/`.forecast()` now always return a `Dict[str, EvaluationResult]`/`Dict[str, ForecastResult]` keyed by model name, even when only one model is requested — previously returned a single result object directly. (Phase 3)
- `BaseAdapter.__init__()` now accepts a uniform `frequency` parameter (used by `CatBoostAdapter`, ignored by `ProphetAdapter`) so `forecaster.py`'s `MODEL_REGISTRY`-driven adapter instantiation stays identical across every model, with no per-model special-casing in the orchestrator. (Phase 3)
- `CatBoostWrapper.min_rows_required` is now a plain training-row-count floor, identical in meaning to `ProphetWrapper`'s — previously it applied to rows remaining *after* `CatBoostAdapter`'s lag-warmup drop, a mechanism that no longer exists. (Phase 3)
- `publish.yml` switched from `twine` + long-lived `PYPI_API_TOKEN` secrets to PyPI Trusted Publishing (OIDC via `pypa/gh-action-pypi-publish`) — no stored credentials, short-lived per-run tokens instead. Manual `workflow_dispatch` runs now take a `target` input (`testpypi` default / `pypi`), so a manual run defaults to a safe TestPyPI dry run rather than a real release; tag pushes (`v*`) still always publish to real PyPI.

### Known deferred issues

- 24 CVEs in `torch`/`pytorch-lightning`, forced by `neuralprophet==0.8.0`'s own version constraints — consciously deferred to Phase 6 rather than pinning an untested pre-release NeuralProphet.
- PyMC and NeuralProphet still need their own `save()`/`load()` overrides (Phase 6).
- `examples/real_dataset_testing.ipynb` exists in Colab but is not yet committed to this repository.
