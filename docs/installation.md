# Installation

## Requirements

- **Python 3.10, 3.11, or 3.12** (`requires-python = ">=3.10,<3.13"`). The
  CI matrix runs all three.
- A working C/C++ toolchain for the heavier model backends — Prophet
  (cmdstanpy) and PyMC compile at install/first-use time. On most Linux and
  macOS setups this is already present; on Windows the Prophet install is
  smoothest under conda.

## Install from source

The project is not yet published to PyPI, so install from a clone:

```bash
git clone https://github.com/Lifesight-Software-Pvt-Ltd/horizon.git
cd horizon
pip install -e .
```

`pip install -e .` is an *editable* install: it symlinks the source tree, so
your changes take effect without reinstalling. The runtime dependencies
(pandas, numpy, prophet, neuralprophet, statsmodels, catboost, pymc,
matplotlib, pyyaml) are pinned in `pyproject.toml` and installed
automatically.

## Development install (running the test suite)

The test and lint tools are declared as an optional dependency group in
`pyproject.toml`, so one command gets you everything needed to run the
suite:

```bash
pip install -e ".[dev]"      # adds pytest, pytest-cov, black, flake8
```

That is the only supported way to install them, and it is exactly what CI
does. There is deliberately no `requirements-dev.txt`: a second list of the
same four packages is a second thing to keep in step, and the one time the
two disagree is the time it matters.

`requirements.txt` still exists, and mirrors the runtime pins for tools
that expect a plain requirements file — but `pyproject.toml` is the source
of truth for both runtime and dev dependencies.

## Verify the install

```bash
python -c "from horizon_forecaster import HorizonForecaster; print('ok')"
pytest                             # full suite
pytest --cov=horizon_forecaster    # with coverage
```

A green run is currently **447 passed**.

Note that the model backends are heavy: a full `pytest` run fits real
Prophet, CatBoost, SARIMAX, NeuralProphet, and Bayesian PyMC models and
takes a while (~30 minutes on a laptop). Bayesian PyMC is the slowest — it
runs a full NUTS/MCMC sample per fit — with NeuralProphet's torch training
next. To iterate faster while working on a specific area, deselect the two
slowest backends:

```bash
pytest --ignore=tests/test_bayesianpymc.py --ignore=tests/test_neuralprophet.py
```

Expect warnings during the run rather than silence — several tests
deliberately exercise the short-series and insufficient-data paths, which
warn by design.

## A note on pinned dependencies

Every runtime dependency is pinned to an exact version, and two pins are
load-bearing rather than cosmetic (both documented inline in
`pyproject.toml`):

- `matplotlib==3.10.6` — later 3.11.x dropped Python 3.10 support and broke
  the 3.10 CI job.
- `setuptools>=68,<82` — setuptools 82 removed `pkg_resources`, which a
  transitive dependency (pytorch-lightning, via neuralprophet) imports at
  runtime.

If you bump either, run the full CI matrix (3.10–3.12) before trusting it.
