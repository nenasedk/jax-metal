"""pytest plugin: run JAX's tests on the CPU backend with the shim's METAL lowering rules.

    cd re/work/jax-src  # JAX's tests at the installed version (see shim/README.md)
    JAX_PLATFORMS=cpu PYTHONPATH=../../../shim/tests/jax_suite python -m pytest \
        -p cpu_rules_plugin tests/linalg_test.py

Registers the rules of jax_plugins.metal_shim for platform "cpu" in place of XLA's own, so the
algorithms (QR, Jacobi eigh, SVD, triangular solve, FFT, cumulative scans, ...) and their
tracing run against JAX's tests on any machine, without a GPU and without risking it. It
finds tracing errors and wrong algorithms; MPS crashes and MPS accuracy need the GPU.

real/imag/complex keep XLA's CPU lowerings: the METAL rules build them from a complex->real
convert, which MPS accepts (it keeps the real part) but which crashes XLA's CPU compiler.
The package's __init__ (which registers the METAL plugin) is not run.
"""

import os
import sys
import types

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python",
                           "jax_plugins", "metal_shim")

_namespace = sys.modules.setdefault("jax_plugins", types.ModuleType("jax_plugins"))
if not hasattr(_namespace, "__path__"):
    _namespace.__path__ = []
_package = types.ModuleType("jax_plugins.metal_shim")
_package.__path__ = [os.path.normpath(PACKAGE_DIR)]
sys.modules["jax_plugins.metal_shim"] = _package

from jax._src.interpreters import mlir  # noqa: E402
from jax._src.lax import lax as lax_internal  # noqa: E402
from jax_plugins.metal_shim import _complex, _lowerings  # noqa: E402

_CPU_ONLY = (lax_internal.real_p, lax_internal.imag_p, lax_internal.complex_p)
_cpu_rules = mlir._platform_specific_lowerings["cpu"]  # pylint: disable=protected-access
_kept = {primitive: _cpu_rules.get(primitive) for primitive in _CPU_ONLY}
_lowerings.PLATFORM = _complex.PLATFORM = "cpu"
_lowerings.register()
for _primitive, _rule in _kept.items():
    if _rule is None:
        _cpu_rules.pop(_primitive, None)
    else:
        _cpu_rules[_primitive] = _rule
