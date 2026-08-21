# Contributing to Horizon Forecaster

This document reflects how the project has actually been built (Phases 0–8,
through the 1.0.0 release), not a generic template — the conventions below
are the ones already in use throughout the codebase. Following them keeps a
contribution consistent with everything already here, rather than adding
a second style alongside the existing one.

## Development setup

```bash
git clone https://github.com/Lifesight-Software-Pvt-Ltd/horizon.git
cd horizon
python -m venv test_env      # any venv name works; test_env/ is already gitignored
source test_env/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for dependencies: runtime
pins under `[project].dependencies`, and the test/lint tools under the
`dev` optional-dependency group that `pip install -e ".[dev]"` pulls in.
`requirements.txt` mirrors the runtime pins for tools that expect a plain
requirements file, so update both if you add a runtime dependency. There is
deliberately no `requirements-dev.txt` — the `dev` extra is the only dev
dependency list, so there is nothing to keep in step with it.

## Running tests

```bash
pytest --cov=horizon_forecaster
```

If you haven't done an editable install, `PYTHONPATH=. pytest tests/`
works too. CI (`.github/workflows/tests.yml`) runs the full suite across
Python 3.10, 3.11, and 3.12, but is currently triggered **manually**
(`on: workflow_dispatch`) rather than on push — so run the suite locally
before opening a PR; nothing will run it for you.

A few tests fit real Prophet/CatBoost models with the default
hyperparameter grid search (6 and 18 candidates respectively), which is
slow. When writing a new test that doesn't need to exercise the grid
search itself, pass an explicit `model_params` override
(e.g. `model_params={"prophet": {"seasonality_mode": "additive"}}`) to
skip it — see almost any test in `test_forecaster.py` for the pattern.
Note that `evaluate()` fits each model once more than the grid alone
implies: the winning candidate is refit on train + validation before the
test window is scored. `forecast()` does *not* run `evaluate()` — it only
needs the cached selection — so a test that just wants a forecast should
not pay for a test-window prediction it throws away.

Tests that construct a `HorizonForecaster` on a short synthetic series
should set `validation_size`/`test_size` explicitly. The defaults are
absolute period counts (56/28 daily, 26/13 weekly) and will either scale
themselves down with a warning or leave too little to train on.
Tests in this project fit real models rather than mocking them; mocking
is only used where it's the only practical way to hit a specific
code path deterministically (e.g. `test_catboost.py`'s prediction-clipping
tests monkeypatch `_fitted_model.predict` to return a fixed array).

## Architecture, in one paragraph

Every model integrates as a matched **adapter** (`data_adapter/`,
subclasses `BaseAdapter`) + **wrapper** (`models/`, subclasses
`BaseModel`) pair. Adapters do ALL model-specific data reshaping — a
wrapper never sees raw user data, only whatever its adapter already
shaped for it. `forecaster.py`'s `HorizonForecaster` is a thin,
registry-driven orchestrator (`MODEL_REGISTRY`) that wires
`adapter.transform_train() → wrapper.fit()` and
`adapter.transform_future() → wrapper.predict()` identically for every
registered model. **Adding a new model should mean adding one
`MODEL_REGISTRY` entry, nothing else in `forecaster.py`.** If it doesn't,
that's a signal the `BaseAdapter`/`BaseModel` contract has a gap — fix
the contract, don't special-case the orchestrator for one model.

## Exceptions

Every raised error is a typed subclass of `HorizonForecasterError`
(`exceptions.py`), carrying structured, machine-readable attributes
(`model_name`, `mape_value`, `required_periods`, etc.) in addition to a
human-readable message — not just a string to parse. This is what lets
`evaluate_errors`/`forecast_errors`/`EnsembleResult.rejected_models`
build a real, programmatic report instead of guessing from exception
text. If you need a new failure mode, check `exceptions.py` first —
several genuinely-fitting exceptions (`ThresholdRejectionError`,
`AllModelsRejectedError`) sat unused for a full phase before a use case
actually needed them; it's worth checking before adding a new one.

## Documentation expectations for a real change

- **`CHANGELOG.md`**: add an `Added`/`Fixed`/`Changed` entry for anything
  a user of the library would notice. Open a new `## [Unreleased]` section
  above the latest released version if one isn't there yet — do not file
  under `## Pre-1.0 history`, which is a closed record of work that has
  already shipped. If you
  found and fixed a real bug or behavior gap (not just added a feature),
  say what the actual problem was — see the CatBoost lag/rolling-feature
  entry for the level of detail expected: what broke, how it was
  diagnosed, what evidence confirmed the fix.
- **`docs/build_log_phase*.md`**: for a substantial phase of work, record
  what was built, the real design decisions made (and any that were
  revised mid-build, and why), and how it was verified. These are meant
  to be read later by someone who wasn't in the room — write for that
  reader, not as a commit-message summary.
- A change that's purely internal refactoring with no user-visible
  behavior difference doesn't need a CHANGELOG entry, but should still
  be covered by existing or updated tests.

## Before opening a PR

- `pytest --cov=horizon_forecaster` passes locally.
- New behavior has real tests (fitting real models, not mocks, unless
  mocking is the only way to hit the path deterministically).
- `black .` (line length 100, see `pyproject.toml`) and `flake8` are
  clean.
- CHANGELOG updated if the change is user-visible (see above).
- If you're touching feature engineering or model behavior specifically,
  verify against a real dataset if you can, not just synthetic test
  data — the CatBoost redesign in Phase 3 exists precisely because a
  synthetic-only test suite didn't catch a real, dataset-shape-dependent
  failure mode. The per-phase build logs in `docs/` record the dataset
  shapes and edge cases each model was checked against.

## Current project status

The library is at **1.0.0**. All five models — Prophet, CatBoost, SARIMAX,
NeuralProphet and Bayesian PyMC — are fully implemented (adapter + wrapper
+ config + tests), registered in `MODEL_REGISTRY` and orchestrated by
`HorizonForecaster`, with `evaluation/ensemble.py` combining them and the
`plotting.py`/`exporters.py` output layer consuming the results. Bayesian
PyMC is much slower than the others (NUTS/MCMC per grid candidate), so its
daily config is tuned for scale. `future_exog.py` (Phase 7) handles how
`forecast()` invents future driver values when they aren't supplied.

1.0.0 reworked the evaluation protocol itself, and two of its decisions are
worth knowing before you change anything in `forecaster.py` or
`evaluation/ensemble.py`:

- **Selection and reporting live in different windows.** Grid search,
  `mape_threshold` screening and inverse-MAPE weighting read the
  **validation** window only; the test window is reported and never read by
  a selection path. Any new feature that wants an accuracy number has to
  choose which of the two it is entitled to — if it influences a decision,
  it gets the validation score.
- **The default combination rule is a trimmed mean**, which has no weights
  at all. Code that assumes `EnsembleResult.weights` is a populated dict
  will break: it is `Optional` and is `None` for weightless rules.

Runnable example notebooks and a gated PyPI release remain deferred to the
publishing stage — a clean `pip install` + Open-in-Colab only makes sense
once the package is published (the `examples/*.md` files are empty
placeholders until then). Rolling-origin backtesting, benchmark-relative
metrics (RelMAE/MASE), family de-duplication of Prophet vs NeuralProphet,
and an error-diversity diagnostic are all acknowledged gaps rather than
completed work; see `CHANGELOG.md`'s "Known deferred issues" section for
the current, honest list.
