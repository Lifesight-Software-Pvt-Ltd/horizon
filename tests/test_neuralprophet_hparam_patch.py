"""
Tests for the NeuralProphet ``save_hyperparameters()`` compatibility patch.

Kept separate from tests/test_neuralprophet.py, which is importorskip'd on
the real NeuralProphet. This behaviour is pure Python monkeypatching, so it
can and should be verified everywhere -- including CI and the sandbox, where
torch is not installed. NeuralProphet is stubbed into sys.modules with just
enough surface for the patch to bind to.

Regression cover for: NeuralProphet's TimeNet guards save_hyperparameters()
with `except RuntimeError`, but PyTorch Lightning + OmegaConf raise
ConfigValueError (a ValueError) when future regressors are registered, so
the guard misses it and fit() dies. See the wrapper method's docstring.
"""

import sys
import types

import pytest

import horizon_forecaster.models.neuralprophet_wrapper as npw
from horizon_forecaster.models.neuralprophet_wrapper import NeuralProphetWrapper


class _ConfigValueError(ValueError):
    """Stands in for omegaconf.errors.ConfigValueError (a ValueError)."""


@pytest.fixture
def stub_time_net(monkeypatch):
    """Install a fake `neuralprophet.time_net.TimeNet` and reset the patch flag."""
    calls = {"count": 0}

    class FakeTimeNet:
        def save_hyperparameters(self, *args, **kwargs):
            calls["count"] += 1
            raise _ConfigValueError(
                "Unions of containers are not supported:\n"
                "loss_func: Union[str, _Loss, Callable]"
            )

    module = types.ModuleType("neuralprophet.time_net")
    module.TimeNet = FakeTimeNet

    package = types.ModuleType("neuralprophet")
    package.time_net = module

    monkeypatch.setitem(sys.modules, "neuralprophet", package)
    monkeypatch.setitem(sys.modules, "neuralprophet.time_net", module)
    monkeypatch.setattr(npw, "_HPARAM_CAPTURE_PATCHED", False)
    return FakeTimeNet, calls


def test_patch_swallows_the_omegaconf_union_error(stub_time_net):
    """The exact failure that stopped NeuralProphet fitting must not escape."""
    FakeTimeNet, _ = stub_time_net

    with pytest.raises(_ConfigValueError):
        FakeTimeNet().save_hyperparameters()  # unpatched: raises

    NeuralProphetWrapper._neutralize_lightning_hparam_capture()

    assert FakeTimeNet().save_hyperparameters() is None  # patched: absorbed


def test_patch_still_calls_the_original(stub_time_net):
    """
    The original is invoked first, so environments where hparam capture works
    keep working -- this suppresses the failure, it does not disable the call.
    """
    FakeTimeNet, calls = stub_time_net

    NeuralProphetWrapper._neutralize_lightning_hparam_capture()
    FakeTimeNet().save_hyperparameters()

    assert calls["count"] == 1


def test_patch_preserves_a_successful_return_value(stub_time_net):
    """A working save_hyperparameters() must pass its result through untouched."""
    FakeTimeNet, _ = stub_time_net
    FakeTimeNet.save_hyperparameters = lambda self, *a, **k: "captured"

    NeuralProphetWrapper._neutralize_lightning_hparam_capture()

    assert FakeTimeNet().save_hyperparameters() == "captured"


def test_patch_is_idempotent(stub_time_net):
    """
    Repeated calls must not stack wrappers. Without the module-level guard,
    every fit() would add another layer of try/except around the last.
    """
    FakeTimeNet, _ = stub_time_net

    NeuralProphetWrapper._neutralize_lightning_hparam_capture()
    once = FakeTimeNet.save_hyperparameters
    NeuralProphetWrapper._neutralize_lightning_hparam_capture()

    assert FakeTimeNet.save_hyperparameters is once
    assert npw._HPARAM_CAPTURE_PATCHED is True


def test_patch_is_a_noop_when_neuralprophet_is_absent(monkeypatch):
    """
    fit() calls this before the NeuralProphet import is guaranteed usable, so
    a missing dependency must return quietly rather than raise ImportError.
    """
    monkeypatch.setattr(npw, "_HPARAM_CAPTURE_PATCHED", False)
    monkeypatch.setitem(sys.modules, "neuralprophet", None)

    NeuralProphetWrapper._neutralize_lightning_hparam_capture()  # must not raise

    assert npw._HPARAM_CAPTURE_PATCHED is False


def test_patch_is_a_noop_when_timenet_is_renamed_upstream(monkeypatch):
    """Tolerate an upstream rename instead of dying on a missing attribute."""
    module = types.ModuleType("neuralprophet.time_net")  # no TimeNet attribute
    package = types.ModuleType("neuralprophet")
    package.time_net = module

    monkeypatch.setitem(sys.modules, "neuralprophet", package)
    monkeypatch.setitem(sys.modules, "neuralprophet.time_net", module)
    monkeypatch.setattr(npw, "_HPARAM_CAPTURE_PATCHED", False)

    NeuralProphetWrapper._neutralize_lightning_hparam_capture()  # must not raise

    assert npw._HPARAM_CAPTURE_PATCHED is False
