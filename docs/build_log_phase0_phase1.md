# Horizon Forecaster — Build Log: Phase 0 & Phase 1

This document records what was built, why it was built that way, and how each
decision was verified — for the foundational phases of the project, before
any model-specific (Prophet/SARIMAX/CatBoost/NeuralProphet/PyMC) code existed.

---

## Phase 0 — Project Scaffolding & Environment Validation

### Goal
Establish a working, version-controlled, CI-tested project skeleton — and,
critically, prove that all 5 target model libraries can coexist in one
environment — **before** any real logic is written. The principle: fail
fast and cheaply, while the repo is still empty, rather than discovering a
dependency conflict after wrappers depend on it.

### What was done

| Item | Detail |
|---|---|
| Repo skeleton | Full package/test/docs/examples folder structure created, matching the agreed architecture |
| `pyproject.toml` + `setup.py` | Modern packaging config (`pyproject.toml`) as source of truth; `setup.py` kept as a minimal backward-compatibility shim |
| `requirements.txt` / `requirements-dev.txt` | Left intentionally **unpinned** at this stage |
| Dependency spike | All 5 libraries (Prophet, NeuralProphet, statsmodels/SARIMAX, CatBoost, PyMC) installed together in one clean virtual environment and imported together in a single script |
| `.gitignore` | Standard Python excludes, plus explicit exclusion of the actual local venv folder (`test_env/`) |
| `LICENSE` | Deferred to the org's existing repo `LICENSE` (Apache-2.0) rather than adding a duplicate |
| `.github/workflows/tests.yml` | CI workflow: installs the package in editable mode with dev deps, runs `pytest --cov` on every push/PR |
| `.github/workflows/publish.yml` | Initially empty (caused a false CI failure — see Verification below); fixed and explicitly gated to `workflow_dispatch` + version-tag triggers only, so it does **not** fire on every commit |
| Git setup | Repo initialized, merged with the org's pre-existing remote history (`Lifesight-Software-Pvt-Ltd/horizon`, which already had a `LICENSE` and placeholder `README.md`), pushed to `main` |

### Why these decisions

- **Dependency spike before any real code**: PyMC (via PyTensor) and
  NeuralProphet (via PyTorch/PyTorch-Lightning) have historically been the
  two libraries most likely to have narrow, conflicting version
  requirements on numpy/pandas/scipy. Discovering a conflict here costs
  minutes; discovering it after 3 wrappers are built costs days.
- **Unpinned requirements initially**: the exact working version set
  wasn't known yet — pinning before the spike would mean re-pinning
  immediately after. Pinning is deferred to a later phase, once a known-good
  combination is proven under real use.
- **`publish.yml` gated to manual/tag triggers**: auto-publishing to PyPI
  on every push risks accidentally shipping a broken release before v1 is
  stable. Gating it removes that risk entirely until it's deliberately
  triggered.
- **CI wired up before any real tests exist**: establishes the "green
  check" habit from commit one, so a broken pipeline is caught in seconds,
  not discovered after hundreds of lines of code have accumulated.

### How this was verified

1. **Local import test**: a script explicitly importing all 5 model
   libraries together in one process — confirmed to run with no
   `ImportError`s (one benign warning from CatBoost's optional plotly
   dependency, unrelated to the 5 core libraries).
2. **`pip install -e ".[dev]"`**: confirmed to succeed cleanly, all
   dependencies resolved without conflict.
3. **Local `pytest` run**: placeholder test passed, confirming pytest
   itself was correctly configured via `pyproject.toml`.
4. **CI on the actual shared GitHub repo** (not just local): the `tests.yml`
   workflow was confirmed **green** on `Lifesight-Software-Pvt-Ltd/horizon`
   after resolving two real-world issues encountered along the way:
   - A git authentication/permissions issue (403 on push) — resolved once
     write access was granted and credentials refreshed.
   - `publish.yml` showing as "Failure" on every push — root-caused to the
     file being **completely empty** (not a code bug), fixed by giving it
     valid content and restricting its trigger.

**Phase 0 exit state**: clean install, all 5 models import together with no
conflicts, CI green on the shared org repository, `tests.yml` running
correctly on every push while `publish.yml` stays dormant until deliberately
triggered.

---

## Phase 1 — Foundational Contracts

### Goal
Build the four files every later model wrapper depends on — `exceptions.py`,
`utils.py`, `evaluation/metrics.py`, `models/base.py` — and get their
interfaces right **before** any model-specific code exists, since changing
these later is expensive (would require touching every wrapper built on top
of them).

### What was done

#### 1. `exceptions.py`
A structured exception hierarchy, all inheriting from a single
`HorizonForecasterError` base:

- `InsufficientDataError` — not enough history to train a given model
- `ModelConvergenceError` — model's internal fit procedure failed/didn't converge
- `ModelPredictionError` — model was fit but failed during predict
- `ThresholdRejectionError` — model trained fine, but MAPE fell outside the allowed range
- `AllModelsRejectedError` — every requested model was rejected
- `InvalidConfigError` — user-supplied config value is invalid (e.g. bad `training_size`)
- `UnsupportedFrequencyError` — frequency other than daily/weekly requested

Each exception carries **structured attributes** (e.g. `mape_value`,
`required_periods`, `rejected_models`), not just a message string.

#### 2. `utils.py`
Generic, model-agnostic helpers:
- `clean_zero_variance()` — drops columns with no variance (a common cause
  of singular-matrix errors in SARIMAX, and generally zero-signal features)
- `infer_frequency()` / `validate_frequency()` — detects/validates
  daily-vs-weekly spacing in a date column
- `validate_training_size()` — enforces the 50–100% range from the project
  spec, accepting either fraction (0.8) or percentage (80) input
- `validate_mape_threshold_range()` — enforces the 1–100% range from the
  project spec for ensemble MAPE thresholds

#### 3. `evaluation/metrics.py`
Pure, model-independent metric functions:
- `mape()` — the canonical metric per project spec, with explicit
  **zero-actual handling** (a known MAPE failure mode on sparse daily
  data). Returns a structured `MapeResult` (value + how many rows were
  excluded due to zero actuals), not a bare float, so `forecaster.py` can
  later warn when too much of a series was excluded.
- `wape()`, `rmse()`, `mae()` — supplementary metrics only; never used for
  threshold/ensemble decisions, per project spec (MAPE-only).

#### 4. `models/base.py`
The abstract `BaseModel` class every wrapper must inherit from:
- Abstract `fit(train_df, target_column) -> self` and
  `predict(horizon, future_exog=None) -> DataFrame`
- Concrete `get_params()` / `set_params()`
- `frequency` and `params` (hyperparameter overrides) threaded through the
  constructor
- Explicit exception contract: `fit()` may raise `InsufficientDataError` /
  `ModelConvergenceError`; `predict()` may raise `ModelPredictionError`
- `predict()`'s output contract: **required** `ds`/`yhat` columns;
  **optional** `yhat_lower`/`yhat_upper` for models that natively support
  uncertainty estimation (Prophet, SARIMAX, PyMC). This is documented as an
  **individual-model-only** concept — uncertainty bounds are never
  propagated into the ensemble output, since averaging confidence/credible
  intervals across statistically different models (trend-simulation vs.
  posterior vs. Gaussian approximation) has no coherent interpretation.

### Why these decisions

- **One base exception class**: lets `forecaster.py` catch any
  library-specific failure with a single `except HorizonForecasterError`,
  distinct from genuine bugs (which should still surface as raw Python
  exceptions, not get silently absorbed).
- **Structured exception attributes over message strings**: enables the
  "rejected models" report to be built programmatically from real data
  (`mape_value`, `required_periods`, etc.), rather than parsing error text.
- **`MapeResult` object instead of a bare float**: makes an otherwise
  invisible data-quality problem (a MAPE computed on a series full of
  zero-actual days) visible and actionable later, without changing the
  function's core behavior.
- **Uncertainty bounds excluded from the ensemble by design**: a
  considered decision, not an oversight — explicitly discussed and decided
  that individual-model plots/CSVs can show `yhat_lower`/`yhat_upper` where
  the underlying model supports it, but the final ensemble output stays
  strictly to `ds`/`yhat`, avoiding a statistically dubious "ensembled
  confidence interval."

### How this was verified

1. **Unit tests for every file**, written alongside the implementation, not
   after:
   - `test_exceptions.py` — 10 tests, including a critical contract test
     confirming every subclass is catchable as the shared base exception.
   - `test_utils.py` — 21 tests, including edge cases (all-NaN columns,
     boundary values on the 50/100 and 1/100 ranges).
   - `test_metrics.py` — 18 tests, with explicit coverage of the zero-actual
     edge case (all-zero actuals, partial zero actuals, both handling
     strategies).
   - `test_base.py` — 10 tests using a dummy concrete subclass, confirming
     the abstract contract is enforced (cannot instantiate `BaseModel`
     directly; `fit()`/`predict()` behave correctly; `is_fitted` state is
     managed correctly on both success and failure paths).
   - **Result: 59/59 tests passing.**

2. **Coverage analysis**, not just pass/fail:
   - 99% line coverage across all four files.
   - The 2 uncovered lines in `models/base.py` were confirmed to be the
     `raise NotImplementedError` bodies of the two `@abstractmethod`
     definitions — structurally unreachable code, since Python's `abc`
     module guarantees any concrete subclass must override them. This is
     the correct, expected shape of 100% *meaningful* coverage, not a real
     gap.

3. **Interface stress test against all 5 real models** (the highest-value
   check in this phase): rather than only proving the interface works for
   the easy case (Prophet), minimal stub subclasses were written for
   **all 5 target models** and instantiated against the real `BaseModel`
   from this phase, to surface any signature mismatch before real
   wrapper code was written on top of it. Findings:
   - Prophet, SARIMAX, CatBoost, NeuralProphet: fit cleanly with no
     signature changes needed.
   - SARIMAX's native `LinAlgError` and PyMC's non-raising convergence
     diagnostics both route correctly into the existing
     `ModelConvergenceError` — confirmed to be a wrapper-level
     responsibility (catch-and-translate), not something requiring a
     `base.py` change.
   - PyMC's natural production of uncertainty intervals surfaced the one
     genuine open design question of this phase (bounds handling),
     which was resolved as described above.
   - **All 5 stubs instantiated successfully against the real `BaseModel`**,
     confirming the interface holds before any real wrapper was built.

4. **CI on the shared GitHub repository**: confirmed green after this
   phase's code was pushed, running the full 59-test suite (plus the
   pre-existing dependency-import and placeholder tests, 61 total) on a
   clean Ubuntu runner — not just passing locally.

**Phase 1 exit state**: all four foundational files built, 100% of tests
passing, 99% code coverage (with the remaining 1% confirmed structurally
unreachable), the interface explicitly stress-tested against all 5 target
models' known quirks rather than assumed to work, one real design decision
(uncertainty bounds) identified and resolved with the reasoning documented
directly in the code's docstring contract, and CI green on the shared repo.

---

## Open item carried forward into later phases

The uncertainty-bounds exclusion rule (bounds are individual-model-only,
never in the ensemble) is currently **documented in `base.py`'s docstring**
but not yet **enforced by a test**, since `ensemble.py` doesn't exist yet.
**Action for Phase 4**: when `evaluation/ensemble.py` is built, include an
explicit test that feeds a DataFrame containing `yhat_lower`/`yhat_upper`
into the ensemble function and asserts the final output contains no trace
of them — turning this documented contract into an enforced one.
