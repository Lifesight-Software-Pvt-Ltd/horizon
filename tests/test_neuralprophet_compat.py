"""
Tests for the wrapper's NeuralProphet compatibility shims.

Kept separate from tests/test_neuralprophet.py, which is importorskip'd on
the real NeuralProphet. Both shims here are pure Python indirection, so they
can and should be verified everywhere -- including CI and the sandbox, where
torch is not installed. NeuralProphet (and torch) are stubbed into
sys.modules with just enough surface for the code under test to bind to.

Two shims, each covering a real failure:

1. ``_neutralize_lightning_hparam_capture()`` -- NeuralProphet's TimeNet
   guards save_hyperparameters() with `except RuntimeError`, but PyTorch
   Lightning + OmegaConf raise ConfigValueError (a ValueError) once future
   regressors are registered, so the guard misses it and fit() dies.

2. ``_load_torch_model()`` -- on torch >= 2.6 the primary loader fails and
   the fallback must reproduce what NeuralProphet's own load() does,
   including restore_trainer(). Both failures on this method were CI-only,
   because local torch < 2.6 never reaches the fallback at all. These stub
   tests close that gap: they exercise the fallback on any torch version.
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


# ---------- _load_torch_model: the torch >= 2.6 fallback ----------


class _FakeForecaster:
    """Stands in for a loaded NeuralProphet, tracking trainer restoration."""

    def __init__(self):
        self.trainer = None  # exactly how save() leaves it on disk
        self.restored_with = "not called"

    def restore_trainer(self, accelerator=None):
        self.trainer = f"trainer<{accelerator}>"
        self.restored_with = accelerator


def _install_np_and_torch(monkeypatch, np_load, torch_load):
    package = type(sys)("neuralprophet")
    package.load = np_load
    torch_module = type(sys)("torch")
    torch_module.load = torch_load
    monkeypatch.setitem(sys.modules, "neuralprophet", package)
    monkeypatch.setitem(sys.modules, "torch", torch_module)


def test_primary_loader_is_used_when_it_works(monkeypatch):
    """On torch < 2.6, NeuralProphet's own load() handles everything itself."""
    expected = _FakeForecaster()

    def torch_load(*args, **kwargs):
        raise AssertionError("fallback must not run when np_load succeeds")

    _install_np_and_torch(monkeypatch, lambda path: expected, torch_load)

    assert NeuralProphetWrapper._load_torch_model("some/path") is expected
    assert expected.restored_with == "not called"  # np_load already did it


def test_fallback_restores_the_trainer(monkeypatch):
    """
    Regression cover for a CI-only failure: save() nulls the Lightning
    trainer (it isn't picklable) and NeuralProphet's load() rebuilds it via
    restore_trainer(). A bare torch.load() skips that, so the model returns
    with trainer=None and the next predict() dies with a bare
    "'NoneType' object has no attribute 'predict'".
    """
    loaded = _FakeForecaster()

    def failing_np_load(path):
        raise RuntimeError("simulates torch >= 2.6 weights_only refusal")

    _install_np_and_torch(monkeypatch, failing_np_load, lambda *a, **k: loaded)

    result = NeuralProphetWrapper._load_torch_model("some/path")

    assert result is loaded
    assert result.trainer is not None, "fallback returned an unusable model"
    assert result.restored_with == "cpu"  # consistent with map_location


def test_fallback_passes_weights_only_false_and_cpu_map_location(monkeypatch):
    """The two flags that make the fallback work at all."""
    captured = {}

    def torch_load(path, **kwargs):
        captured.update(kwargs)
        return _FakeForecaster()

    _install_np_and_torch(
        monkeypatch, lambda path: (_ for _ in ()).throw(RuntimeError("boom")), torch_load
    )

    NeuralProphetWrapper._load_torch_model("some/path")

    assert captured["weights_only"] is False
    assert captured["map_location"] == "cpu"


def test_fallback_tolerates_a_model_without_restore_trainer(monkeypatch):
    """Guard against an upstream rename rather than dying on getattr."""

    class Bare:
        pass

    bare = Bare()
    _install_np_and_torch(
        monkeypatch, lambda path: (_ for _ in ()).throw(RuntimeError("boom")), lambda *a, **k: bare
    )

    assert NeuralProphetWrapper._load_torch_model("some/path") is bare  # must not raise


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
