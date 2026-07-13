# Horizon Forecaster — Build Log: Phase 2 (Complete)

This document records the full first vertical slice of the project: the
Prophet data adapter, the Prophet model wrapper, its default configuration,
and the `CustomForecaster` orchestrator that wires them together into
`evaluate()` and `forecast()`. It also covers a substantial environment/
packaging hardening effort that happened partway through this phase,
triggered by real failures on a colleague's machine and in CI. 

---

## Context: why Prophet, and why this order

Phase 2's goal is to validate the entire architecture (adapter → wrapper →
future orchestration) against the **simplest** of the 5 target models before
touching the other 4. Prophet needs no lag features, no exogenous matrix
construction, and no MCMC tuning — so any problem that surfaces here is a
genuine architecture problem, not something confounded by a specific
model's complexity.

---

## Step 1 — `data_adapter/base_adapter.py` + `data_adapter/prophet_adapter.py`

### What was done

- **`BaseAdapter`**: abstract contract with two methods —
  `transform_train()` (raw data → model-ready training data) and
  `transform_future()` (future dates + optional future independent
  variables → model-ready prediction input).
- **`ProphetAdapter`**: concrete implementation reshaping raw data into
  Prophet's required `ds`/`y` column format, keeping independent variables
  under their original names (since Prophet's `add_regressor()` references
  columns by name).

### Key design decision: adapters and wrappers are fully decoupled

Wrappers never call adapters themselves. `forecaster.py` (built later) is
the only place that wires `raw_df → adapter.transform_train() →
wrapper.fit()` together. This means the adapter can be tested in complete
isolation (no Prophet model involved) and the wrapper can be tested in
complete isolation (fed pre-shaped `ds`/`y` data directly, no adapter
involved).

### Design decision, revised mid-build: hard errors over silent gaps

The first version of `transform_train()`/`transform_future()` silently
skipped any independent column that was declared but missing from the
actual data. This was deliberately changed to **raise `InvalidConfigError`
immediately, naming the missing column**, for both training and future
data. Rationale: a silently dropped column (e.g. a typo like `"promoo"`
instead of `"promo"`) would produce a working-but-quietly-weaker model with
no visible signal that anything was wrong. A hard error costs the user a
few seconds to notice and fix; a silent gap could go unnoticed indefinitely.

This also clarified a related contract: by the time `transform_future()` is
called, the caller (`forecaster.py`, in a later phase) must have already
resolved future independent-variable values — either real user-provided
data, or a fallback strategy (carry-forward, seasonal-naive, historical
average). A `None` arriving at the adapter when columns were declared is
now treated as a bug in the caller, not a valid "no data given" case.

### How this was verified

- 12 tests covering: correct column renaming, multiple/zero independent
  columns, chronological sorting, and — critically — the hard-error
  behavior on both training and future data (missing required columns
  raise `InvalidConfigError` naming the exact missing column).
- 95%+ coverage on both files; the only uncovered lines are the two
  abstract method bodies in `BaseAdapter`, which are structurally
  unreachable (Python's `abc` module enforces that subclasses override
  them).

---

## Step 2 — `models/prophet_wrapper.py`

### What was done

`ProphetWrapper` implements the `BaseModel` contract from Phase 1:
- `fit()`: validates required columns are present, enforces a minimum-rows
  floor (`MIN_ROWS_REQUIRED = 10`, a conservative placeholder pending
  frequency-aware tuning once `config/default_params.yaml` exists),
  registers any independent columns as Prophet regressors, and translates
  any genuine underlying Prophet/Stan failure into `ModelConvergenceError`.
- `predict()`: requires a pre-shaped `future_exog` DataFrame (unlike
  `BaseModel`'s generic `Optional` signature — Prophet always needs future
  context, so a `None` here is treated as a caller error), validates
  row-count and required-regressor-column presence before calling
  Prophet's own `predict()`, and returns `ds`/`yhat`/`yhat_lower`/
  `yhat_upper`.

### Design decision: session-level model reuse, refit-once, optional persistence

Working through a forecasting scenario (train on early data, evaluate on a
holdout, then produce a real future forecast) surfaced an important
question: should the model used for the actual delivered forecast be the
one fit only on the training split, or refit on the full dataset once
evaluation is done?

**Decided**: refit once on the full dataset before producing the real
forecast, since the held-out test split is the most recent, most relevant
data available — discarding it when generating the actual forecast would
be strictly worse than using everything available. This refit happens
**once per session** (not once per `forecast()` call), is cached, and — the
part that affects this file directly — must be **persistable to disk**, so
a user can reuse a fitted model in a future session without refitting.

A further principle, explicitly decided: **all 5 models get refit fully and
identically, regardless of how expensive that is for a given model.**
PyMC being slow is not a reason to run it in a degraded/faster mode while
the other 4 run at full quality — that would be an unfair, hidden
compromise on the exact model a user may have chosen because it performed
well. Cost is to be surfaced transparently (a warning before a slow step),
never solved by quietly lowering quality.

### Why `save()`/`load()` needed care, not a generic default

`BaseModel` gained new (non-abstract, additive) `save()`/`load()` methods
with a generic pickle-based default. This is fine for models that pickle
reliably (SARIMAX, later), but **Prophet's own documentation explicitly
warns against raw pickling** of its fitted models, since the underlying
Stan/cmdstanpy backend does not reliably deserialize across environments
or library versions. `ProphetWrapper` overrides `save()`/`load()` to use
Prophet's own `model_to_json()`/`model_from_json()`, wrapped in a small
JSON envelope carrying the wrapper-level state (regressor columns,
frequency, params) needed to fully reconstruct the wrapper, not just the
underlying Prophet object. (PyMC and NeuralProphet will need their own
overrides later, for the same reason, using their own libraries'
recommended serialization approaches.)

### A real bug found and fixed during this step: non-reproducible uncertainty intervals

While proving the save/load roundtrip produced identical predictions, a
genuine, pre-existing bug surfaced: **Prophet's `yhat_lower`/`yhat_upper`
values were not reproducible across repeated `predict()` calls on the same
fitted model** — `yhat` itself was always identical, but the uncertainty
bounds varied slightly every call. Root cause: Prophet's uncertainty
simulation draws from numpy's global random state without a fixed seed by
default. This was **not** a save/load defect — it was confirmed to happen
even calling `predict()` twice in a row on the exact same in-memory model,
with no serialization involved at all.

**Fix**: `predict()` now sets `np.random.seed()` (default `42`, overridable
via `params={"random_seed": ...}`) immediately before calling Prophet's
own `predict()`. The seed value is explicitly filtered out before being
passed to Prophet's constructor in `fit()`, since it is a wrapper-level
concept, not a real Prophet parameter, and would otherwise raise a
`TypeError`.

Verified fixed by: (a) two `predict()` calls in the same process now
return bit-identical `yhat_lower`/`yhat_upper`, and (b) three **separate**
Python process invocations of fit+predict on identical data returned
bit-identical results — confirming the fix is genuinely robust, not a
within-session coincidence.

### How this was verified

- **22 tests**, including:
  - Standard fit/predict correctness and shape checks.
  - Exact boundary tests for `MIN_ROWS_REQUIRED` (9 rows fails, 10 rows
    succeeds), not just an arbitrary small-number case.
  - **Genuine failure-path tests**, not fabricated ones: duplicate `ds`
    timestamps were found to trigger a real Prophet/Stan optimization
    failure (confirmed empirically first), used to test the
    `ModelConvergenceError` translation path; NaN values in a registered
    regressor column were found to trigger a real Prophet `predict()`
    failure, used to test the `ModelPredictionError` translation path.
  - Full save/load roundtrip tests, including a check that a reloaded
    model is a genuinely independent instance (not accidentally sharing
    state with the original in-memory object) and that reloaded
    predictions are bit-identical to the original's.
- **100% line coverage** on `prophet_wrapper.py.` One piece of dead code
  was found and removed during this process: an `except InsufficientDataError:
  raise` clause that could never actually execute, since that exception is
  raised before the `try` block it appeared to guard even starts. Harmless,
  but misleading to a future reader, so it was removed rather than left in
  place to preserve a coverage number.
- **Manual numerical sanity check**: a clean, known +2/day synthetic trend
  was fit and forecast 10 days forward; the forecast continued the exact
  trend precisely (220, 222, 224, ... as expected), confirming the pipeline
  produces numerically sensible output, not just code that runs without
  error.
- Full suite (95 tests across the whole project) confirmed passing
  together, at 99% overall coverage, with all remaining uncovered lines
  confirmed to be structurally-unreachable abstract method bodies.

---

## Step 3 — `config/default_params.yaml` + frequency-aware minimums

### What was done

Created Prophet's daily and weekly default hyperparameters, structured to
explicitly separate two concerns:
- `hyperparameters`: passed straight through to `Prophet(**hyperparameters)`.
- `wrapper_config`: settings consumed by the wrapper itself
  (`min_rows_required`, `random_seed`), deliberately kept in a separate
  block so wrappers never need to filter library-specific keys out of a
  mixed dict (which is exactly the fragile pattern step 2's `random_seed`
  fix had temporarily introduced).

This required refactoring `ProphetWrapper`'s constructor: `min_rows_required`
and `random_seed` became explicit, typed arguments with frequency-aware
defaults, replacing the flat `MIN_ROWS_REQUIRED = 10` class constant flagged
as an open item after step 2.

### Design decision: real, reasoned frequency floors, not a guess

- **Daily: 14 rows** (2 full weeks) — the earliest point a weekly pattern
  can even theoretically repeat once.
- **Weekly: 8 rows** (~2 months) — below this there isn't enough span to
  estimate a trend plus any real pattern, as opposed to fitting noise.

Also corrected `weekly_seasonality` to `false` in the weekly config block
— at weekly granularity, there is no "within-week" pattern to detect,
since each row already represents a full week.


- 4 new tests confirming frequency-awareness specifically (daily default
  ≠ weekly default, explicit override works, `random_seed` defaults and
  overrides work, wrapper-level config survives the save/load roundtrip).
- Re-verified end to end: real YAML loaded, real values passed into a real
  `ProphetWrapper` instance, real `fit()` call succeeded.
- Once the "never actually landed" gap was found and fixed, re-confirmed
  with the actual test run from the real repository (26/26 passing),
  not assumed.

---

## Environment & packaging hardening (a detour, not a planned step)

Between finishing the core Prophet vertical slice and building
`forecaster.py`, a colleague's attempt to run the project surfaced a
chain of real environment issues. Each was root-caused with actual
evidence (error logs, `pip` resolver output, web searches for the exact
failure) rather than guessed at, and each fix was verified before moving
to the next.

1. **Unpinned dependencies resolved differently on a different machine/date.**
   `requirements.txt`/`pyproject.toml` had been left unpinned since Phase 0
   (deliberately, pending a "pin later" plan). A colleague's fresh install
   pulled much newer versions of everything, which didn't have a mutually
   compatible resolution, and `pip`'s resolver eventually backtracked into
   an old `statsmodels` release whose legacy build assumed `pkg_resources`
   was implicitly available. **Fix**: pinned all dependencies to the exact
   versions already proven to work (from the original Phase 0 spike),
   and added `requires-python = ">=3.10,<3.13"` so an unsupported Python
   is rejected outright with a clear message, instead of failing deep
   inside a confusing resolver backtrace.

2. **A CI test matrix (Python 3.10, 3.11, 3.12) was added**, specifically
   so this class of version-compatibility issue is caught automatically by
   CI going forward, not discovered via a colleague's laptop again. It
   caught real issues on its very first run:
   - `matplotlib==3.11.0` (the version that resolved naturally on a
     Python 3.12 machine) had already dropped Python 3.10 support —
     pinning it broke the 3.10 job specifically. Fixed by pinning to
     `3.10.6`, a version confirmed (from the actual pip error log) to
     support 3.10 through 3.12.
   - Python 3.12 failed on the same `pkg_resources` issue as the
     colleague's original failure. Root-caused further than "add
     setuptools": `setuptools>=68` alone wasn't sufficient, because it was
     satisfied by `setuptools 82.0.0` — a version confirmed (via multiple
     independent upstream bug reports) to have **completely removed**
     `pkg_resources` from the package. Fixed with a real upper bound:
     `setuptools>=68,<82`.

3. **A packaging footgun**: a colleague's `pip install -e ".[dev]"` failed
   with "multiple top-level packages discovered," traced to their virtual
   environment folder (`test_env/`) sitting in the project root and being
   mistaken by setuptools' automatic flat-layout discovery for a second
   package. Rather than rely on a naming convention (e.g. "always use a
   dot-prefixed venv name") to avoid this, `pyproject.toml` was given an
   explicit `[tool.setuptools.packages.find] include` list, verified to
   solve the problem by deliberately recreating the exact failure
   condition (stray `test_env/` and `catboost_info/` folders in the
   project root) and confirming a clean build regardless.

4. **A genuinely unverified packaging gap**: inspecting an actual built
   wheel (not just an editable install) revealed `config/default_params.yaml`
   was completely absent from the distributable package — editable installs
   had been masking this the whole time. Fixed with an explicit
   `[tool.setuptools.package-data]` declaration, verified by rebuilding
   the wheel and confirming the YAML file's presence inside it directly
   (`unzip`-level inspection, not just "the build succeeded").

5. **Dependency vulnerability scan** (`pip-audit`): found 24 known CVEs,
   concentrated in `torch==2.2.2` and `pytorch-lightning==1.9.5`. Both
   versions are forced by `neuralprophet==0.8.0`'s own upper-bound
   constraints, not chosen directly. A newer NeuralProphet release
   (`1.0.0rc9`/`rc10` — note: an earlier claim of a clean `0.9.0` release
   was checked against PyPI directly and found to be inaccurate) resolves
   to much newer, non-vulnerable `torch`/`pytorch-lightning` versions, but
   is a pre-release with likely breaking API changes and zero test
   coverage against this project's code today (`NeuralProphetWrapper`
   doesn't exist yet). **Decision**: do not pin a pre-release now; defer
   the real version decision to Phase 6, when `NeuralProphetWrapper` is
   actually built and can be tested directly against whatever the current
   best release is at that time. Documented explicitly here as a known,
   consciously deferred risk — not silently ignored.

6. **Verification chain, not a single check**: throughout this detour,
   fixes were confirmed at multiple independent levels before being
   considered done — local machine, a genuinely independent colleague
   machine (different OS/architecture), GitHub CI across 3 Python
   versions, `pip check` (dependency consistency), `black`/`flake8`
   (code quality — one real unused import was found and removed), and a
   real wheel build/install in a clean venv with a `find`-level check
   that the packaged YAML file physically exists on disk.

---

## Step 5 — `HorizonForecaster.evaluate()`

### What was done

The first place raw user data actually flows through the whole pipeline:
chronological train/test split → `ProphetAdapter.transform_train()` on
both splits → `ProphetWrapper.fit()` on train only → `predict()` on test
→ `MapeResult` computed by comparing predictions to real held-out target
values.

### Design decisions

- **Chronological split, never random** — standard for time series;
  random shuffling would leak future information into training.
- **`training_size=100%` is explicitly guarded**: it would leave zero
  rows for evaluation, which would otherwise fail confusingly deep inside
  `metrics.mape()`. Raises a clear `InvalidConfigError` instead.
- **Zero-variance independent columns are cleaned once, at construction**,
  and — importantly — automatically removed from the user's requested
  `independent_columns` list too. Without this, the adapter's own
  hard-error-on-missing-column (from step 1) would fire against a column
  the forecaster itself had silently dropped, which would look like a
  bug in the user's data when it was actually self-inflicted.
- **`fit()`/`ModelConvergenceError`/etc. are allowed to propagate
  naturally** — with only one model and no ensemble yet, there's no
  "rejected models" report to build around them; that's Phase 4's job.

### How this was verified

- 14 tests, including constructor validation, config loading (confirmed
  correct values loaded per frequency), and structural checks on the
  returned `EvaluationResult`.
- **Numerically verified, not just structurally**: a clean, noise-free
  synthetic trend produced a MAPE of exactly `0.0000%` on held-out test
  data the model had never seen — real proof the full chain (split →
  adapt → fit → predict → MAPE) is wired correctly, not just that it runs
  without raising.
- Confirmed against the real repository (not just the sandbox) after the
  step-3 lesson: real `cat`/`git status`/test-run evidence before
  considering the step done.

---

## Step 6 — `HorizonForecaster.forecast()`

### What was done

The second orchestrator method: a full-data refit (train + test combined)
followed by a real future prediction, implementing the two-stage
`evaluate()`/`forecast()` design settled on earlier in the project (see
reasoning below).

### Design decisions, all settled through earlier discussion before this step was coded

- **Refit exactly once per instance, cached, reused across calls** — even
  across calls with different `horizon` values. Verified with an
  object-identity check (`is`) across two `forecast()` calls, not just
  "the numbers looked plausible."
- **All models refit fully and identically, regardless of cost** — a
  principle carried over from step 2's design discussion, now actually
  implemented: there is no "cheap mode" for any model, ever, in this
  method.
- **Default future-independent-variable strategy**, matching the original
  project spec's own wording precisely ("if 180 days forecast, take data
  from past 180 days... if data is only 3 months but 6 months forecast,
  handle internally"): reuse the actual last-N periods of historical
  independent-variable values; if the horizon exceeds available history,
  cycle (tile) the full historical sequence rather than failing. Verified
  with an exact-value test (reproducing precisely which historical values
  get reused) and a cycling test (a 45-period horizon from only 20 rows of
  history, confirmed to produce no `NaN`s and the correct row count).
- **Horizon-vs-history warning**: an open item flagged after step 2's
  manual stress test (a 26-week forecast from 10 weeks of history ran
  without error but showed suspiciously flat, falsely confident
  uncertainty bounds). Implemented as a non-blocking `warnings.warn()`
  when horizon exceeds 2x available history — verified to fire when
  warranted and confirmed silent otherwise (using
  `warnings.simplefilter("error")` to turn any unexpected warning into a
  hard test failure).
- **`ForecastResult` surfaces `used_default_future_strategy` explicitly**
  — so a caller/UI can distinguish a forecast built on real user-supplied
  future data from one relying on an assumed continuation of history,
  rather than presenting both with equal implied confidence.

### How this was verified

- 11 new tests (25 total), all passing against
  the real repository.
- Manual numerical sanity check: a known +2/period synthetic trend,
  forecast forward, reproduced the exact expected trend values
  (differences between consecutive forecasted points all exactly `2.0`).
- Full project suite: 124 tests passing, 99% coverage, `forecaster.py`
  itself at 100%.

---

## Step 7 — Real-dataset testing surfaces two more gaps, both fixed

The 6 steps above were validated against clean synthetic data. Testing
`evaluate()`/`forecast()` against real datasets (via a reusable Colab
workflow, `evaluate()`/`forecast()` helpers, visualizations, and a
findings-log template) surfaced two genuine gaps neither synthetic data
nor unit tests had exposed.

### Gap 1: fixed default hyperparameters underperformed known-good ones

Prophet's out-of-the-box defaults scored worse on real data than
hyperparameter combinations already proven to work well in practice.
A single fixed default can't be simultaneously right for every dataset
shape, so **`evaluate()`/`forecast()` now run a hyperparameter grid
search by default**: 6 candidates per frequency
(`config/default_params.yaml`'s `hyperparameter_grid`) — 4 tuned
combinations plus 2 safe `yearly_seasonality='auto'` fallbacks — each
fit and scored on the same chronological train/test split, with the
lowest-MAPE candidate selected automatically
(`EvaluationResult.grid_search_summary` exposes the full comparison).
An explicit `model_params` override at construction still skips the
search entirely, unchanged from the original design.

**Found mid-implementation**: forcing the 4 tuned candidates onto
short-history datasets produced badly broken forecasts — MAPE as high
as 802%, including negative predictions for a strictly positive metric.
Root cause: those combos hard-code a `yearly_seasonality` Fourier order
that assumes enough history to constrain a yearly cycle; with too
little data, Prophet still tries to fit one and extrapolates wildly.
Rather than hand-coding a data-span threshold to guard against this,
the 2 `'auto'` fallback candidates were added to the grid — Prophet's
own data-span-aware setting naturally disables yearly seasonality when
unsupported, so the grid search itself favors the safe fallback on
short datasets and the tuned combos on long ones, without an explicit
rule to maintain.

**Accepted trade-off**: grid search makes `evaluate()`/`forecast()`
slower (roughly 1–4 minutes instead of one fit), in exchange for not
silently shipping a badly-fit default. Deliberate, not an oversight.

### Gap 2: declared frequency was never checked against actual data spacing

A user could declare `frequency="daily"` while passing weekly-spaced
data (or vice versa) with no warning — silently producing seasonality
assumptions and future forecast date spacing inconsistent with what the
model was actually trained on. Fixed:
`HorizonForecaster.__init__()` now calls `infer_frequency()` against the
actual date column and warns (non-blocking) if the inferred spacing
doesn't match the declared `frequency`, or if the spacing can't be
confirmed as clean daily/weekly at all.

### How this was verified

- Test suite grew from 124 to 137 tests, all passing, individually
  re-confirmed against the real repository after this change (not
  assumed from a sandbox run).
- Both fixes verified against the specific real-data failure modes that
  surfaced them: the grid search against the 802%-MAPE short-history
  case directly, the frequency check against a deliberately mismatched
  daily/weekly dataset pair.

### Known open thread from this step

The Colab notebook this testing was done in (`examples/real_dataset_testing.ipynb`)
is not yet committed to this repository — it currently lives in Colab
only. Pushing it is blocked on the same GitHub PAT/org-admin access
issue noted below, not forgotten.

---

## Phase 2 exit state

All 6 core steps plus the step-7 real-dataset-testing follow-up are
complete and independently verified against the real repository (not
assumed from a local sandbox). The full vertical slice — raw data in,
real Prophet forecasts out, with honest evaluation metrics and
hyperparameter selection in between — works end to end, with the
environment it depends on hardened against real, encountered failure
modes rather than hypothetical ones.

---

## Open items carried forward into later phases

- **Resolved during this phase, no longer open**: `MIN_ROWS_REQUIRED`
  frequency-awareness (step 3), the horizon-vs-history warning
  (step 6), the fixed-default-hyperparameter gap, and the
  declared-vs-actual frequency check (both step 7) — all flagged as
  open items during Phase 2, all now implemented and tested.
- PyMC and NeuralProphet will still need their own `save()`/`load()`
  overrides later (Phase 6), following the principle established for
  Prophet: use each library's own recommended native serialization,
  never the generic pickle default.
- The uncertainty-bounds-excluded-from-ensemble rule (documented in
  `base.py` since Phase 1) still needs an enforcing test once
  `evaluation/ensemble.py` is built (Phase 4).
- The NeuralProphet CVE question (torch/pytorch-lightning vulnerabilities
  forced by `neuralprophet==0.8.0`'s own constraints) is consciously
  deferred, not resolved — to be revisited in Phase 6 when
  `NeuralProphetWrapper` is actually built and can be tested against
  whatever the current best NeuralProphet release is at that time.
- `forecast()`'s default future-independent-variable strategy currently
  implements exactly one approach (reuse-and-cycle historical values).
  The other strategies discussed earlier in the project (seasonal-naive,
  historical-average, linear-trend) remain a documented future
  enhancement, not yet built — worth revisiting once real usage shows
  whether the current default is sufficient.
- `examples/real_dataset_testing.ipynb` exists in Colab but is not yet
  committed to this repository, and the GitHub PAT/org-admin access
  needed to resolve this is still pending.
