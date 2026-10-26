"""pytest plugin: run JAX's tests on the CPU backend as if it were METAL, without a GPU.

    cd re/work/jax-src  # JAX's tests at the installed version (see shim/README.md)
    JAX_PLATFORMS=cpu JAX_METAL_CHECK_LOG=/tmp/check.jsonl \
        PYTHONPATH=../../../shim/tests/jax_suite python -m pytest \
        -p cpu_rules_plugin tests/linalg_test.py
    python ../../../shim/tests/jax_suite/metal_check.py /tmp/check.jsonl

CPU then lowers programs the way METAL does: the shim's METAL rules are registered for "cpu",
and XLA's other CPU-specific lowerings (LAPACK custom calls, threefry, ...) are removed, so
primitives without a METAL rule use JAX's generic lowering, or have none, as on METAL. Tests
are also selected as for METAL (jax_test_dut=METAL, one device). This finds tracing errors
and wrong algorithms in the rules.

Every program about to compile also goes through metal_check.metal_issues(), which
predicts what Apple's plugin could not compile (ops without an MPS pattern, composites,
types MPS lacks, ...). With $JAX_METAL_CHECK_LOG set, each test with predicted issues is
appended there as a JSON line with its CPU outcome. MPS crashes and accuracy still need METAL.

real/imag/complex keep XLA's lowerings: the METAL rules build them from a complex->real
convert, which MPS accepts (it keeps the real part) but which crashes XLA's CPU compiler.
The package's __init__ (which registers the METAL plugin) is not run.
"""

import json
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

import jax  # noqa: E402
from jax._src import compiler  # noqa: E402
import jax._src.test_util  # noqa: E402  (defines the jax_test_dut flag)
from jax._src.interpreters import mlir  # noqa: E402
from jax._src.lax import lax as lax_internal  # noqa: E402
from jax_plugins.metal_shim import _complex, _lowerings  # noqa: E402

import metal_check  # noqa: E402  (next to this file)

_cpu_rules = mlir._platform_specific_lowerings["cpu"]  # pylint: disable=protected-access
_xla_cpu_rules = dict(_cpu_rules)
_lowerings.PLATFORM = _complex.PLATFORM = "cpu"
_lowerings.register()
for _primitive, _rule in _xla_cpu_rules.items():
    # No METAL rule: generic lowering, as on METAL. Except threefry2x32: with its generic
    # (unrolled) lowering, XLA's CPU compiler never finishes some sampling programs once
    # log1p has any non-XLA lowering (random.ball with a traced p); the generic form runs on
    # METAL (op probe, PRNG tests), and the CPU form is plain HLO loops.
    if _cpu_rules[_primitive] is _rule and _primitive.name != "threefry2x32":
        del _cpu_rules[_primitive]
for _primitive in (lax_internal.real_p, lax_internal.imag_p, lax_internal.complex_p):
    _cpu_rules.pop(_primitive, None)

_register_lowering = mlir.register_lowering


def _register_lowering_except_cpu(primitive, rule, platform=None, *args, **kwargs):
    # Modules imported later (jax.experimental.sparse, ...) register CPU-specific lowerings
    # too; METAL has none of them.
    if platform == "cpu":
        return None
    return _register_lowering(primitive, rule, platform, *args, **kwargs)


mlir.register_lowering = _register_lowering_except_cpu
if "JAX_USE_SHARDY_PARTITIONER" not in os.environ:  # as the package's initialize() does
    jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_test_dut", "METAL")  # defined by test_util, imported above
# METAL has one device. Multi-device test files ask for 8 CPU devices at import; keep one, so
# their multi-device tests skip as they do on METAL instead of running on a mesh METAL lacks.
jax._src.test_util.request_cpu_devices = lambda nr_devices: None

_LOG = os.environ.get("JAX_METAL_CHECK_LOG")
_current = {"nodeid": None, "issues": set()}
_compile = compiler.compile_or_get_cached


def _checked_compile(backend, computation, *args, **kwargs):
    _current["issues"].update(metal_check.metal_issues(computation))
    return _compile(backend, computation, *args, **kwargs)


compiler.compile_or_get_cached = _checked_compile


def pytest_runtest_logstart(nodeid, location):
    del location
    _current["nodeid"], _current["issues"] = nodeid, set()


def pytest_runtest_logreport(report):
    # One record per test, at its outcome: the call phase, or setup when it did not pass.
    if not (report.when == "call" or (report.when == "setup" and report.outcome != "passed")):
        return
    if _LOG and _current["issues"]:
        record = {"nodeid": report.nodeid, "outcome": report.outcome,
                  "issues": sorted(_current["issues"])}
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
