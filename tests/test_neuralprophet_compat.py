"""
Tests for ``NeuralProphetWrapper._load_torch_model()``.

Kept separate from tests/test_neuralprophet.py, which is importorskip'd on
the real NeuralProphet. This shim is pure Python indirection, so it can and
should be verified everywhere, including environments where torch is not
installed. NeuralProphet and torch are stubbed into sys.modules with
just enough surface for the code under test to bind to.

Why this file exists: on torch >= 2.6 the primary loader fails and the
fallback must reproduce what NeuralProphet's own load() does, including
restore_trainer(). Both failures this method has had appeared only in CI, for the
same structural reason: where local torch is < 2.6 the fallback branch never
executes at all, making it effectively untested code that ran only on the
matrix. These stub tests exercise the fallback directly on any torch
version, closing that gap.
"""

import sys

from horizon_forecaster.models.neuralprophet_wrapper import NeuralProphetWrapper

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
