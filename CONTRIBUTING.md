# Contributing to Horizon Forecaster

This document reflects how the project has actually been built so far
(Phases 0–4), not a generic template — the conventions below are the ones
already in use throughout the codebase. Following them keeps a
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

`requirements.txt`/`requirements-dev.txt` mirror `pyproject.toml`'s
dependency list exactly — if you add a dependency, update all three
together, not just one.

## Running tests

```bash
pytest --cov=horizon_forecaster
```

If you haven't done an editable install, `PYTHONPATH=. pytest tests/`
works too. CI (`.github/workflows/tests.yml`) runs the full suite across
Python 3.10, 3.11, and 3.12 on every push and PR.

A few tests fit real Prophet/CatBoost models with the default
hyperparameter grid search (6 and 18 candidates respectively), which is
slow. When writing a new test that doesn't need to exercise the grid
search itself, pass an explicit `model_params` override
(e.g. `model_params={"prophet": {"seasonality_mode": "additive"}}`) to
skip it — see almost any test in `test_forecaster.py` for the pattern.
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
`evaluate_errors`/`forecast_errors`/`evaluate_ensemble.rejected_models`
build a real, programmatic report instead of guessing from exception
text. If you need a new failure mode, check `exceptions.py` first —
several genuinely-fitting exceptions (`ThresholdRejectionError`,
`AllModelsRejectedError`) sat unused for a full phase before a use case
actually needed them; it's worth checking before adding a new one.

## Documentation expectations for a real change

- **`CHANGELOG.md`**: update the `Added`/`Fixed`/`Changed` section under
  `[Unreleased]` for anything a user of the library would notice. If you
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
  failure mode. `docs/real_dataset_test_matrix.md` has a working
  checklist of dataset shapes and edge cases worth trying.

## Current project status

Prophet and CatBoost are fully implemented (adapter + wrapper +
config + tests), orchestrated by `HorizonForecaster`, with Phase 4's
ensemble-weighting logic (`evaluation/ensemble.py`) combining them.
SARIMAX, PyMC, and NeuralProphet exist as empty stub files only
(`data_adapter/`, `models/`) — deliberately not registered in
`MODEL_REGISTRY` yet. `plotting.py`/`exporters.py` are also still empty
stubs. See `CHANGELOG.md`'s "Known deferred issues" section for the
current, honest list of what's not done yet.
