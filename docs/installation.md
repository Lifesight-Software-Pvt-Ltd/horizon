# Installation

## Requirements

- **Python 3.10 through 3.13** (`requires-python = ">=3.10,<3.14"`). All four
  are tested; the test suite is run by hand rather than in CI, so see
  [`CONTRIBUTING.md`](../CONTRIBUTING.md) before relying on it for a change of
  your own.
- A working C/C++ toolchain for the heavier model backends — Prophet
  (cmdstanpy) and PyMC compile at install/first-use time. On most Linux and
  macOS setups this is already present; on Windows the Prophet install is
  smoothest under conda.

### Python 3.13: slower to install, identical to run

3.13 is fully supported — the test suite passes identically to 3.12, and a
seeded five-model run matches it to two decimals on every per-model MAPE. But
the install behaves differently, and it is worth knowing why before you wait
through it.

`neuralprophet==0.8.0` requires `numpy<2`, which is why this project pins
`numpy==1.26.4`. numpy publishes no 3.13 wheel for its 1.x line, so on 3.13
pip falls back to the source distribution and **compiles numpy**. Two
consequences:

- **The C toolchain is required, not merely helpful**, and the build adds
  several minutes. Verified working on Linux x86-64 and macOS arm64. Every
  other dependency installs as an ordinary 3.13 wheel.
- **Linear algebra is slower**, because a source build links whatever BLAS
  the system provides rather than the tuned one numpy bundles into its
  wheels — roughly 3× on raw matmul and SVD, but only about 5% on a real
  run, since the time here goes to Stan sampling, torch training and tree
  building rather than large linear-algebra calls.

Where you have the choice, **3.12 is the faster platform.** This all resolves
if NeuralProphet ships a release accepting numpy 2, at which point 3.13 gets
a normal wheel install like every other version.

### Google Colab

Colab's default runtime is Python 3.13 and works as-is; the install just
spends the extra minutes building numpy.

To skip that wait and get the faster BLAS, pin the runtime instead:
**Runtime → Change runtime type → Runtime version → 2026.07** (Python
3.12.13). A pinned past runtime boots in 2–3 minutes rather than seconds,
which is expected. Google retains each past runtime for one year after
release, so re-pin to the newest one still shipping 3.12 when 2026.07
lapses.

## Install from source

A PyPI release is coming. For now, install from a clone:

```bash
git clone https://github.com/Lifesight-Software-Pvt-Ltd/horizon.git
cd horizon
pip install -e .
```

`pip install -e .` is an *editable* install: it symlinks the source tree, so
your changes take effect without reinstalling. The runtime dependencies
(pandas, numpy, prophet, neuralprophet, statsmodels, catboost, pymc,
matplotlib, pyyaml, setuptools) are pinned in `pyproject.toml` and installed
automatically.

## Development install

To also get the test and lint tools:

```bash
pip install -e ".[dev]"      # adds pytest, pytest-cov, black, flake8
```

There is deliberately no `requirements-dev.txt` — the `dev` extra in
`pyproject.toml` is the single dev dependency list. `requirements.txt` mirrors
the runtime pins only, for tools that expect a plain requirements file.

## Verify the install

```bash
python -c "from horizon_forecaster.forecaster import HorizonForecaster; print('ok')"
pytest                             # full suite
pytest --cov=horizon_forecaster    # with coverage
```

Note that the model backends are heavy: a full `pytest` run fits real
Prophet, CatBoost, SARIMAX, NeuralProphet, and Bayesian PyMC models and
takes a while (~25–30 minutes on a laptop). Bayesian PyMC is the slowest —
it runs a full NUTS/MCMC sample per fit — with NeuralProphet's torch
training next.

## A note on pinned dependencies

Every runtime dependency is pinned to an exact version, and two pins are
load-bearing rather than cosmetic (both documented inline in
`pyproject.toml`):

- `matplotlib==3.10.6` — later 3.11.x dropped Python 3.10 support and broke
  the 3.10 CI job.
- `setuptools>=68,<82` — setuptools 82 removed `pkg_resources`, which a
  transitive dependency (pytorch-lightning, via neuralprophet) imports at
  runtime.

If you bump either, run the full suite on every supported Python version
(3.10–3.13) before trusting it.
