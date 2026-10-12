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
- **On macOS the source build links Apple's Accelerate**, and numpy 1.26
  with Accelerate can emit spurious `RuntimeWarning: divide by zero /
  overflow / invalid value encountered in matmul` from ordinary matrix
  products (you will see them from Bayesian PyMC's `predict()` in the test
  suite). They are a numpy/Accelerate floating-point-flag artefact, not bad
  numbers: the results are finite and the tests that check them pass.

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
pip install -e ".[dev]"      # adds pytest, pytest-cov, black, flake8, bandit
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

## macOS: Bayesian PyMC and the `-ld64` linker flag

PyMC compiles each model to C through PyTensor. On macOS 15 and later,
PyTensor adds `-ld64` to every compile command to select Apple's classic
linker. Recent Xcode / Command Line Tools releases have removed that linker,
so clang reads the flag as "link a library called `d64`" and every compile
fails with:

```
ld: library 'd64' not found
```

Left alone, Bayesian PyMC fails on every fit. Horizon handles this for you:
before the first PyMC model is built, it links an empty test program with
`-ld64` using PyTensor's configured compiler. Only if that fails with exactly
this error does it drop the flag from PyTensor's compile arguments, for the
rest of the process. On toolchains that still accept the flag, and on Linux
and Windows, nothing changes. The upstream issue is
[pytensor#2268](https://github.com/pymc-devs/pytensor/issues/2268).

If you previously hit the error, clear PyTensor's compile cache once so no
half-built modules are reused:

```bash
pytensor-cache purge
```

If you use PyMC outside Horizon in the same environment, the same error will
appear there. Two workarounds that need no Horizon code:

- Point PyTensor at a compiler wrapper that strips the flag:
  ```bash
  mkdir -p ~/bin
  cat > ~/bin/clang++-no-ld64 <<'SH'
  #!/bin/sh
  for a in "$@"; do
    shift
    [ "$a" = "-ld64" ] || set -- "$@" "$a"
  done
  exec clang++ "$@"
  SH
  chmod +x ~/bin/clang++-no-ld64
  export PYTENSOR_FLAGS="cxx=$HOME/bin/clang++-no-ld64"
  ```
- Disable C compilation entirely with `PYTENSOR_FLAGS="cxx="`. This always
  works but sampling becomes much slower.

Horizon's check respects both: it probes whichever compiler `cxx` names, and
does nothing when `cxx` is empty.

## A note on pinned dependencies

Every runtime dependency is pinned to an exact version, and several pins are
load-bearing rather than cosmetic (documented inline in `pyproject.toml`):

- `numpy==1.26.4` — required by `neuralprophet==0.8.0`'s `numpy<2`, not
  chosen (see the Python 3.13 section above).
- `pymc==5.25.1` — held back by the numpy pin: later PyMC releases require
  numpy 2. It also fixes PyTensor below 2.32, which is why the macOS `-ld64`
  workaround above lives in Horizon rather than in an upgrade.
- `matplotlib==3.10.9` — held on the 3.10.x line because 3.11.x dropped
  Python 3.10 support and broke installs on 3.10. Patch bumps within 3.10.x
  are fine.
- `setuptools>=68,<82` — setuptools 82 removed `pkg_resources`, which a
  transitive dependency (pytorch-lightning, via neuralprophet) imports at
  runtime.

If you bump any of them, run the full suite on every supported Python version
(3.10–3.13) before trusting it.
