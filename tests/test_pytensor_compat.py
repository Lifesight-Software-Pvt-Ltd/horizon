"""
Tests for ``horizon_forecaster.models._pytensor_compat``.

The shim only does anything on macOS with a toolchain that rejects PyTensor's
``-ld64`` flag, so on CI (Linux) and on most Macs its patch branch would never
run. These tests stub the platform check, the PyTensor lookup and the
compiler probe, so every branch is exercised on any OS without PyTensor or a
C++ compiler installed.
"""

import subprocess

import pytest

from horizon_forecaster.models import _pytensor_compat as compat

_FLAGS = ["-O3", "-undefined", "dynamic_lookup", "-ld64", "-fPIC"]


def _make_compiler():
    """A stand-in for pytensor's GCC_compiler with the same staticmethod shape."""

    class FakeCompiler:
        @staticmethod
        def compile_args(march_flags=True):
            return list(_FLAGS) + (["-march=native"] if march_flags else [])

    return FakeCompiler


@pytest.fixture(autouse=True)
def _reset_decision(monkeypatch):
    monkeypatch.setattr(compat, "_decision", None)


@pytest.fixture
def macos(monkeypatch):
    monkeypatch.setattr(compat, "_is_macos", lambda: True)


def _install(monkeypatch, compiler, cxx="clang++", rejects=True):
    probes = []

    def probe(c):
        probes.append(c)
        return rejects

    monkeypatch.setattr(compat, "_load_pytensor", lambda: (cxx, compiler))
    monkeypatch.setattr(compat, "_toolchain_rejects_ld64", probe)
    return probes


# ---------- ensure_macos_linker_compat ----------


def test_flag_dropped_when_toolchain_rejects_it(monkeypatch, macos):
    compiler = _make_compiler()
    _install(monkeypatch, compiler, rejects=True)

    assert compat.ensure_macos_linker_compat() is True

    assert compiler.compile_args() == [
        "-O3",
        "-undefined",
        "dynamic_lookup",
        "-fPIC",
        "-march=native",
    ]
    # Arguments still pass through to the original.
    assert compiler.compile_args(march_flags=False) == [
        "-O3",
        "-undefined",
        "dynamic_lookup",
        "-fPIC",
    ]


def test_patch_survives_access_through_class_and_instance(monkeypatch, macos):
    # PyTensor calls compile_args both as GCC_compiler.compile_args() and
    # cls.compile_args(); both must see the patched staticmethod.
    compiler = _make_compiler()
    _install(monkeypatch, compiler, rejects=True)
    compat.ensure_macos_linker_compat()

    assert "-ld64" not in compiler().compile_args()


def test_untouched_when_toolchain_accepts_flag(monkeypatch, macos):
    compiler = _make_compiler()
    original = compiler.compile_args
    _install(monkeypatch, compiler, rejects=False)

    assert compat.ensure_macos_linker_compat() is False
    assert compiler.compile_args is original
    assert "-ld64" in compiler.compile_args()


def test_untouched_off_macos(monkeypatch):
    monkeypatch.setattr(compat, "_is_macos", lambda: False)
    compiler = _make_compiler()
    original = compiler.compile_args
    probes = _install(monkeypatch, compiler, rejects=True)

    assert compat.ensure_macos_linker_compat() is False
    assert compiler.compile_args is original
    assert probes == []  # never even probes


def test_untouched_when_pytensor_unavailable(monkeypatch, macos):
    monkeypatch.setattr(compat, "_load_pytensor", lambda: None)
    assert compat.ensure_macos_linker_compat() is False


def test_untouched_when_user_disabled_c_compilation(monkeypatch, macos):
    # PYTENSOR_FLAGS="cxx=" -- the user opted out of C compilation entirely.
    compiler = _make_compiler()
    original = compiler.compile_args
    probes = _install(monkeypatch, compiler, cxx="", rejects=True)

    assert compat.ensure_macos_linker_compat() is False
    assert compiler.compile_args is original
    assert probes == []


def test_probes_the_users_configured_compiler(monkeypatch, macos):
    compiler = _make_compiler()
    probes = _install(monkeypatch, compiler, cxx="/Users/me/bin/my-clang++", rejects=False)

    compat.ensure_macos_linker_compat()
    assert probes == ["/Users/me/bin/my-clang++"]


def test_decision_made_once_and_never_double_wrapped(monkeypatch, macos):
    compiler = _make_compiler()
    probes = _install(monkeypatch, compiler, rejects=True)

    assert compat.ensure_macos_linker_compat() is True
    patched = compiler.compile_args
    assert compat.ensure_macos_linker_compat() is True

    assert probes == ["clang++"]
    assert compiler.compile_args is patched
    assert compiler.compile_args.__wrapped__() == list(_FLAGS) + ["-march=native"]


# ---------- _toolchain_rejects_ld64 ----------


def _fake_run(returncode, stderr="", stdout=""):
    def run(cmd, **kwargs):
        assert cmd[1] == "-ld64"
        assert kwargs.get("shell") is not True
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    return run


def test_probe_detects_missing_d64_library(monkeypatch):
    monkeypatch.setattr(
        compat.subprocess,
        "run",
        _fake_run(1, stderr="ld: library 'd64' not found\nclang++: error: linker command failed"),
    )
    assert compat._toolchain_rejects_ld64("clang++") is True


def test_probe_false_when_link_succeeds(monkeypatch):
    monkeypatch.setattr(compat.subprocess, "run", _fake_run(0))
    assert compat._toolchain_rejects_ld64("clang++") is False


def test_probe_false_on_unrelated_failure(monkeypatch):
    # A broken compiler for some other reason is not ours to "fix".
    monkeypatch.setattr(compat.subprocess, "run", _fake_run(1, stderr="error: no such sysroot"))
    assert compat._toolchain_rejects_ld64("clang++") is False


@pytest.mark.parametrize(
    "exc", [FileNotFoundError("no clang++"), subprocess.TimeoutExpired("c", 60)]
)
def test_probe_false_when_compiler_cannot_run(monkeypatch, exc):
    def run(cmd, **kwargs):
        raise exc

    monkeypatch.setattr(compat.subprocess, "run", run)
    assert compat._toolchain_rejects_ld64("clang++") is False
