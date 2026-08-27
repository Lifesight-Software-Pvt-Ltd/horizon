"""
Guards on the package's front door: horizon_forecaster/__init__.py.

Two promises live there and nothing else pinned either of them. Both are
cheap to break by accident -- a renamed symbol, or one convenience import
added at the top of a wrapper -- and neither breaks any other test when it
does.
"""

import subprocess
import sys

import horizon_forecaster


# The heavy backends each wrapper imports lazily inside fit(). Importing the
# package must not pull in any of them: a script that only needs the metrics
# or the ensemble maths should pay none of that cost.
_HEAVY_BACKENDS = {
    "prophet",
    "neuralprophet",
    "catboost",
    "statsmodels",
    "pymc",
    "torch",
}


def test_every_name_in_all_is_actually_importable():
    missing = [name for name in horizon_forecaster.__all__ if not hasattr(horizon_forecaster, name)]
    assert missing == [], f"__all__ names that do not resolve: {missing}"


def test_version_is_exported_and_matches_the_installed_metadata():
    from importlib.metadata import PackageNotFoundError, version

    assert isinstance(horizon_forecaster.__version__, str)
    try:
        installed = version("horizon_forecaster")
    except PackageNotFoundError:  # running from a source tree, not installed
        return
    assert horizon_forecaster.__version__ == installed


def test_importing_the_package_does_not_pull_in_the_heavy_stack():
    # A subprocess is required: this pytest session has almost certainly
    # imported the backends already via the per-model test modules, so
    # checking sys.modules in-process would prove nothing.
    probe = (
        "import sys; import horizon_forecaster; "
        f"leaked = sorted({sorted(_HEAVY_BACKENDS)!r} and "
        f"set({sorted(_HEAVY_BACKENDS)!r}) & set(sys.modules)); "
        "assert not leaked, leaked"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
