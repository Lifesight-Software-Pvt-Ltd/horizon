"""
Compatibility shim for PyTensor's macOS linker flag.

PyTensor -- the compiler PyMC uses to turn a model into C code -- appends
``-ld64`` to every compile command on macOS 15 and later, to force Apple's
classic linker. It decides this from the macOS version number alone, as a
proxy for the Xcode version, and its own source notes that the proxy may be
wrong. Current Apple toolchains have removed the classic linker entirely, so
clang reads ``-ld64`` as ``-l d64`` -- "link against a library called d64" --
and every compilation fails with::

    ld: library 'd64' not found

PyMC cannot sample at all when that happens, so Bayesian PyMC fails on every
fit. The library's partial-failure handling keeps the other models running,
but the user silently loses one. Upstream:
https://github.com/pymc-devs/pytensor/issues/2268 (open; no released fix).

The fix here is deliberately narrow:

- It acts only on macOS.
- It first checks whether the toolchain actually rejects the flag, by linking
  an empty program with ``-ld64`` using PyTensor's own configured compiler.
  PyTensor added the flag on purpose, to avoid bugs in Xcode 15's newer
  linker, so on a toolchain that still accepts it nothing is changed.
- When the flag is rejected, it wraps ``GCC_compiler.compile_args`` -- the
  single function every PyTensor compile path draws its flags from -- so that
  exactly that one flag is dropped.
- The decision is made once per process.

This patches a PyTensor internal. That is contained here because PyMC 5.25.1
pins ``pytensor<2.32``, and the pin cannot move until NeuralProphet accepts
numpy 2. If the internal ever changes shape, ``_load_pytensor`` returns None
and the shim does nothing rather than failing.

Nothing in this module imports PyTensor at module level, so importing
horizon_forecaster stays free of the heavy backends.
"""

import logging
import os
import subprocess  # nosec B404 -- runs the user's own configured C++ compiler, no shell
import sys
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_LD64 = "-ld64"
_REJECTION_SIGNATURE = "library 'd64' not found"
_PROBE_TIMEOUT_SECONDS = 60

# None until decided; True once patched; False when not needed or not possible.
_decision: Optional[bool] = None


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _load_pytensor() -> Optional[Tuple[str, type]]:
    """
    Return (configured C++ compiler, GCC_compiler class), or None when
    PyTensor is not importable or no longer has the expected shape.
    """
    try:
        import pytensor
        from pytensor.link.c import cmodule
    except ImportError:
        return None

    compiler = getattr(cmodule, "GCC_compiler", None)
    if compiler is None or not callable(getattr(compiler, "compile_args", None)):
        return None
    return str(getattr(pytensor.config, "cxx", "") or ""), compiler


def _toolchain_rejects_ld64(cxx: str) -> bool:
    """
    True only when linking an empty program with ``-ld64`` fails with the
    specific "library 'd64' not found" error. Any other outcome -- success, a
    different failure, a missing compiler -- returns False, so the shim never
    changes PyTensor's behaviour on a guess.
    """
    with tempfile.TemporaryDirectory() as workdir:
        source = os.path.join(workdir, "probe.cpp")
        binary = os.path.join(workdir, "probe")
        with open(source, "w") as handle:
            handle.write("int main() { return 0; }\n")
        try:
            result = subprocess.run(  # nosec B603 -- fixed argument list, no shell
                [cxx, _LD64, source, "-o", binary],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return result.returncode != 0 and _REJECTION_SIGNATURE in (result.stderr + result.stdout)


def ensure_macos_linker_compat() -> bool:
    """
    Drop PyTensor's ``-ld64`` flag when this machine's toolchain rejects it.

    Safe to call repeatedly: the decision is made once per process. Must run
    before PyTensor compiles anything -- in practice, before ``import pymc``.

    Returns True when the flag is being dropped, False otherwise.
    """
    global _decision
    if _decision is not None:
        return _decision

    if not _is_macos():
        _decision = False
        return _decision

    loaded = _load_pytensor()
    if loaded is None:
        _decision = False
        return _decision

    cxx, compiler = loaded
    if not cxx or not _toolchain_rejects_ld64(cxx):
        # An empty cxx means the user disabled C compilation themselves.
        _decision = False
        return _decision

    original = compiler.compile_args

    def compile_args(*args, **kwargs):
        return [flag for flag in original(*args, **kwargs) if flag != _LD64]

    compile_args.__wrapped__ = original
    compiler.compile_args = staticmethod(compile_args)

    logger.info(
        "This macOS toolchain rejects PyTensor's '%s' linker flag; dropping it so "
        "Bayesian PyMC can compile. See pytensor issue #2268.",
        _LD64,
    )
    _decision = True
    return _decision
