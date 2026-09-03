# Installation

## Requirements

- **Python 3.10, 3.11, or 3.12** (`requires-python = ">=3.10,<3.13"`). The
  CI matrix runs all three. Attempting 3.13 fails at metadata resolution with
  `requires a different Python: 3.13.x not in '<3.13,>=3.10'`.

  The upper bound is **forced by NeuralProphet**, not chosen. Prophet,
  statsmodels, CatBoost and PyMC all install cleanly on 3.13 (with numpy
  2.x); `neuralprophet==0.8.0` declares `numpy>=1.25.0,<2.0.0`, and the
  lowest numpy published for 3.13 is 2.1.0 — an unsatisfiable constraint, and
  the reason this project pins `numpy==1.26.4` in the first place. Newer
  NeuralProphet releases cap Python explicitly instead (0.9.0 requires
  `<=3.12`; even `1.0.0rc10` requires `<3.13`), so no NeuralProphet release
  currently supports 3.13 at all.

  This bound therefore lifts when NeuralProphet ships a release that accepts
  numpy 2 and declares 3.13 support — not before, and not by anything in this
  repository.
- A working C/C++ toolchain for the heavier model backends — Prophet
  (cmdstanpy) and PyMC compile at install/first-use time. On most Linux and
  macOS setups this is already present; on Windows the Prophet install is
  smoothest under conda.

### Google Colab

Colab's **default** runtime tracks the latest Python and has moved to 3.13, so
a plain `pip install` there now fails with the message above — including in
notebooks that worked previously, since the notebook did not change but the
default runtime underneath it did.

Pin the runtime instead: **Runtime → Change runtime type → Runtime version →
2026.07** (Python 3.12.13), then re-run the install. A pinned past runtime
boots in 2–3 minutes rather than seconds, which is expected. Google retains
each past runtime for one year after its release, so re-pin to the newest
runtime still shipping Python 3.12 when 2026.07 lapses.

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

If you bump either, run the full CI matrix (3.10–3.12) before trusting it.
