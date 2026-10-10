# Contributing to Horizon Forecaster

Contributions are welcome. This guide describes the conventions already in use
throughout the codebase — following them keeps a change consistent with what is
here rather than adding a second style alongside it.

Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Start with an issue

**Open an issue before opening a pull request**, for anything larger than a
typo or a doc fix. A short issue saves you from building something that will
be turned down for a reason that was never written down anywhere — and most
rejections here are about scope, not quality.

- **Bug reports** need the Python version, the OS, the model(s) involved, and a
  minimal reproduction — ideally a small synthetic frame rather than your real
  data. The full traceback matters more than a description of it.
- **Feature proposals** should say what forecasting problem the feature solves
  before saying what code it adds. If it changes a number the library reports,
  say which and by roughly how much.

### What is in scope

- Bug fixes, with a test that fails before the fix and passes after.
- A **new model**, as a matched adapter + wrapper + config block + tests. This
  is the best-supported kind of contribution; see below.
- New future-driver strategies in `future_exog.py`, new combination rules in
  `evaluation/ensemble.py`, new metrics in `evaluation/metrics.py` — all three
  are registries of pure functions and are designed to be extended.
- Documentation fixes, and clearer errors or warnings.
- Platform and packaging fixes (a dependency pin that can be lifted, a build
  that fails on your OS).

### What is likely to be turned down

- Changes to the evaluation protocol — the three-window split, what may read
  the test window, or which score drives selection — without a prior
  discussion. These are the guarantees the library is built on; see
  [`docs/design_notes.md`](docs/design_notes.md) §2.
- Special-casing one model inside `forecaster.py`. If a model needs the
  orchestrator to change, that is a gap in the `BaseAdapter`/`BaseModel`
  contract, and the contract is what gets fixed.
- Dependency bumps without evidence. Several pins are load-bearing and
  commented inline in `pyproject.toml`; a bump needs a manual CI matrix run
  (3.10–3.13) behind it.
- Unpinning `neuralprophet` to clear the torch CVEs. This has been looked at:
  the available alternative is a pre-release with breaking API changes and no
  coverage against this code.
- Mocked tests where a real fit would do (see Running tests).
- Large refactors that touch many files at once. Split them; a reviewer cannot
  evaluate architecture and behaviour in the same diff.

## Development setup

```bash
git clone https://github.com/Lifesight-Software-Pvt-Ltd/horizon.git
cd horizon
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for dependencies: runtime pins
under `[project].dependencies`, and the test/lint tools under the `dev` extra
that `pip install -e ".[dev]"` pulls in. `requirements.txt` mirrors the runtime
pins for tools that expect a plain requirements file, so update both if you add
a runtime dependency. There is deliberately no `requirements-dev.txt` — the
`dev` extra is the only dev dependency list.

Any venv name works; `.venv/` and `test_env/` are both gitignored, and
`pyproject.toml`'s explicit package list means a stray folder in the project
root will not break the build either way.

## Running tests

```bash
pytest --cov=horizon_forecaster
```

If you have not done an editable install, `PYTHONPATH=. pytest tests/` works
too.

**Only linting runs automatically on your PR.** `.github/workflows/lint.yml`
runs `black --check` and `flake8` on every pull request and must pass before
a merge. The test suite in `.github/workflows/tests.yml` covers Python 3.10
through 3.13 but is triggered manually (`on: workflow_dispatch`) by a
maintainer, so run the suite locally before you open a PR and say in the PR
which Python version you ran it on. A maintainer runs the matrix as part of
review.

Expect the full suite to take roughly 25–30 minutes: it fits real models rather
than mocking them. A few tests run the default grid search (6 candidates for
Prophet, 18 for CatBoost), which dominates that time. When writing a test that
does not need to exercise the search itself, pass an explicit `model_params`
override — `model_params={"prophet": {"seasonality_mode": "additive"}}` — to
skip it. Almost any test in `test_forecaster.py` shows the pattern.

Two things about the cost model are worth knowing. `evaluate()` fits each model
once more than the grid implies, because the winning candidate is refit on
train + validation before the test window is scored. And `forecast()` does
*not* run `evaluate()` — it only needs the cached selection — so a test that
just wants a forecast should not pay for a test-window prediction it throws
away.

A test that constructs a `HorizonForecaster` on a short synthetic series should
set `validation_size`/`test_size` explicitly. The defaults are absolute period
counts (56/28 daily, 26/13 weekly) and will either scale themselves down with a
warning or leave too little to train on.

**Fit real models; do not mock them.** Mocking is used in exactly one kind of
place — where it is the only way to hit a path deterministically, such as
`test_catboost.py` monkeypatching `_fitted_model.predict` to return a fixed
array, or `test_neuralprophet_compat.py` stubbing `torch` to drive a fallback
branch that never executes on an older version. A mock that stands in for a fit
proves the code runs, not that the forecast is right.

## Architecture, in one paragraph

Every model integrates as a matched **adapter** (`data_adapter/`, subclasses
`BaseAdapter`) + **wrapper** (`models/`, subclasses `BaseModel`) pair. Adapters
do all model-specific data reshaping — a wrapper never sees raw user data, only
what its adapter shaped for it. `forecaster.py`'s `HorizonForecaster` is a
thin, registry-driven orchestrator (`MODEL_REGISTRY`) that wires
`adapter.transform_train() → wrapper.fit()` and
`adapter.transform_future() → wrapper.predict()` identically for every
registered model. **Adding a model should mean adding one `MODEL_REGISTRY`
entry and nothing else in `forecaster.py`.** If it does not, that is a signal
the contract has a gap — fix the contract, do not special-case the
orchestrator.

[`docs/design_notes.md`](docs/design_notes.md) is the long form of this: the
contracts, the per-model reasoning, and the failure modes behind decisions that
otherwise look arbitrary. Read it before changing behaviour.

### Adding a model

The five existing models are the template — SARIMAX is the most representative
(stateful adapter, wrapper-side transform, grid-searched). A complete
contribution is:

1. `data_adapter/<name>_adapter.py` — all reshaping, both `transform_train()`
   and `transform_future()`, with state fitted on train reused for the future.
2. `models/<name>_wrapper.py` — `fit()`/`predict()`, plus `save()`/`load()` if
   the backend does not pickle reliably. `predict()` returns `ds`/`yhat`, and
   `yhat_lower`/`yhat_upper` only if the model produces them natively.
3. A `<name>:` block in `config/default_params.yaml` with daily and weekly
   `hyperparameters`, `hyperparameter_grid` and `wrapper_config`, mirroring the
   existing blocks.
4. One `MODEL_REGISTRY` line.
5. Tests: adapter tests that need no backend, plus wrapper and integration
   tests guarded by `pytest.importorskip` if the backend is heavy.

Seed anything stochastic, and pin anything that would otherwise auto-tune per
run — two models here have been caught producing different numbers from
identical code.

## Exceptions

Every raised error is a typed subclass of `HorizonForecasterError`
(`exceptions.py`) carrying structured, machine-readable attributes
(`model_name`, `mape_value`, `required_periods`, …) alongside a human-readable
message. That is what lets `evaluate_errors`, `forecast_errors` and
`EnsembleResult.rejected_models` build a real report rather than parsing
exception text. If you need a new failure mode, check `exceptions.py` first —
several fitting exceptions sat unused for a long time before a use case
actually needed them.

## Before opening a PR

- `pytest --cov=horizon_forecaster` passes locally, and you have said which
  Python version you ran.
- New behaviour has real tests; a bug fix has a test that fails without it.
- `black .` and `flake8` are clean. Both are configured to agree: line
  length 100 in `pyproject.toml`'s `[tool.black]`, and the matching
  `.flake8` (which also carries the per-file ignores for the handful of
  deliberate exceptions in `tests/`). The repository is formatted and lint-
  clean as it stands, so `black .` should leave your diff untouched apart
  from the lines you actually changed.
- `CHANGELOG.md` has an `Added`/`Fixed`/`Changed` entry if a user of the
  library would notice the change. Open a new `## [Unreleased]` section above
  the latest release if one is not there; do not file under
  `## Prior development`, which is a closed record. For a real bug, say what
  actually broke, how it was diagnosed, and what confirmed the fix — the
  entries in 1.0.0 set the level of detail. Purely internal refactoring needs
  no entry, but still needs test coverage.
- Substantial design decisions go in `docs/design_notes.md`, in the relevant
  section, written for someone who was not in the room.
- If you touched feature engineering or model behaviour, verify against a real
  dataset if you can. Synthetic fixtures are clean parametric signals, and
  several decisions in this library exist because synthetic evidence pointed
  the wrong way and only real data settled it.

Keep the PR focused on one thing, and describe what you verified rather than
what you intended. Review may be slow; the lint check runs automatically, but
the maintainers run the test matrix by hand.

## Project status

The library is at **1.0.0** with a stable API. All five models — Prophet,
CatBoost, SARIMAX, NeuralProphet and Bayesian PyMC — are implemented, tested
and registered, with `evaluation/ensemble.py` combining them and the
`plotting.py`/`exporters.py` layer consuming the results. Installation is from
a clone; the PyPI release is next.

Two properties of the evaluation protocol are worth knowing before you change
anything in `forecaster.py` or `evaluation/ensemble.py`:

- **Selection and reporting live in different windows.** Grid search,
  `mape_threshold` screening and inverse-MAPE weighting read the **validation**
  window only; the test window is reported and never read by a selection path.
  Any new feature that wants an accuracy number has to choose which of the two
  it is entitled to — if it influences a decision, it gets the validation
  score.
- **The default combination rule is a trimmed mean**, which has no weights at
  all. Code that assumes `EnsembleResult.weights` is a populated dict will
  break: it is `Optional` and is `None` for weightless rules.

Known gaps — rolling-origin backtesting, benchmark-relative metrics
(RelMAE/MASE), family de-duplication of Prophet and NeuralProphet, and an
error-diversity diagnostic — are listed with their reasoning under "Still
deferred" in [`CHANGELOG.md`](CHANGELOG.md) and in
[`docs/design_notes.md`](docs/design_notes.md) §9. They are open to
contribution, but each is a design conversation first.
