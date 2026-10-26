"""pytest plugin: stream each test's start and outcome to $JAX_SUITE_LOG as JSON lines.

Used by run_jax_tests.py so that a test which aborts the process (plugin crash) can be
identified from the last "start" record without a matching outcome. Other settings, all
optional:

- $JAX_SUITE_ONLY (one node id per line) keeps only the listed tests;
- $JAX_SUITE_SHARD="I/N" keeps the I-th of N contiguous slices (1-based) of the tests left
  after -k and $JAX_SUITE_ONLY, so a large file can run in batches;
- tests listed in $JAX_SUITE_SKIP (one node id per line) are deselected after sharding, so a
  run can resume where it stopped without changing which tests belong to a shard;
- $JAX_SUITE_COST_LOG receives each test's duration, memory (peak resident memory on Linux,
  physical footprint afterwards on macOS) and generated parameters, from the process that ran
  it;
- when the process footprint passes $JAX_SUITE_SOFT_MEM bytes after a test, the session exits
  with RESTART_EXIT so the runner restarts it with fresh caches instead of letting memory grow
  (a pytest-xdist worker clears JAX's caches instead: its exit would count as a crash);
- once time.time() passes $JAX_SUITE_DEADLINE, the session exits with STOP_EXIT after the
  current test.

Under pytest-xdist, a worker that dies takes its test with it; xdist reports that test as
failed ("worker ... crashed"), recorded here as "crashed".
"""

import json
import os
import sys
import time

import pytest

_LOG = os.environ.get("JAX_SUITE_LOG")
_SKIP = os.environ.get("JAX_SUITE_SKIP")
_ONLY = os.environ.get("JAX_SUITE_ONLY")
_SHARD = os.environ.get("JAX_SUITE_SHARD")
_COST_LOG = os.environ.get("JAX_SUITE_COST_LOG")
_SOFT_MEM = int(os.environ.get("JAX_SUITE_SOFT_MEM", "0"))
_DEADLINE = float(os.environ.get("JAX_SUITE_DEADLINE", "0"))
_MB = 2**20

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import guard  # noqa: E402  (shim/tests/guard.py)
from run_jax_tests import RESTART_EXIT, STOP_EXIT  # noqa: E402  (next to this file)


def _append(path, record):
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _write(record):
    _append(_LOG, record)


def _node_ids(path):
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def shard_slice(count, shard):
    """Bounds of shard "I/N" (1-based) of `count` items: N near-equal contiguous slices."""
    index, total = (int(part) for part in shard.split("/"))
    return (index - 1) * count // total, index * count // total


def _deselect(config, items, keep):
    dropped = [item for item in items if not keep(item)]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [item for item in items if keep(item)]


@pytest.hookimpl(trylast=True)  # after pytest's own -k/-m deselection
def pytest_collection_modifyitems(config, items):
    if _ONLY:
        only = _node_ids(_ONLY)
        _deselect(config, items, lambda item: item.nodeid in only)
    if _SHARD:
        start, stop = shard_slice(len(items), _SHARD)
        in_shard = {id(item) for item in items[start:stop]}
        _deselect(config, items, lambda item: id(item) in in_shard)
    if _SKIP and os.path.exists(_SKIP):
        skip = _node_ids(_SKIP)
        _deselect(config, items, lambda item: item.nodeid not in skip)


# --- cost of each test, measured in the process that runs it ---------------------------

_started = {}


def _reset_peak_memory():
    """Current resident memory in bytes, with Linux's peak (VmHWM) reset to it."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
    except OSError:
        pass
    return guard.footprint(os.getpid())


def _peak_memory():
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return guard.footprint(os.getpid())  # macOS: the footprint after the test


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    del nextitem  # runs only where the test runs (a worker under xdist)
    if _COST_LOG:
        # Generated JAX tests carry their parameters (shapes, dtypes) in __x_params_repr__.
        params = getattr(getattr(item, "obj", None), "__x_params_repr__", "")
        _started[item.nodeid] = (time.time(), _reset_peak_memory(), params)
    yield


def pytest_runtest_logfinish(nodeid, location):
    del location
    if nodeid in _started:
        start, before, params = _started.pop(nodeid)
        record = {"nodeid": nodeid, "seconds": round(time.time() - start, 3),
                  "before_mb": round(before / _MB), "peak_mb": round(_peak_memory() / _MB)}
        if params:
            record["params"] = params[:300]
        _append(_COST_LOG, record)
    if _DEADLINE and time.time() > _DEADLINE:
        _write({"stop": "time budget"})
        os._exit(STOP_EXIT)
    if _SOFT_MEM and guard.footprint(os.getpid()) > _SOFT_MEM:
        if os.environ.get("PYTEST_XDIST_WORKER"):
            import gc  # pylint: disable=import-outside-toplevel
            import jax  # pylint: disable=import-outside-toplevel
            jax.clear_caches()
            gc.collect()
        else:
            _write({"restart": guard.footprint(os.getpid())})
            os._exit(RESTART_EXIT)


def pytest_runtest_logstart(nodeid, location):
    del location
    _write({"start": nodeid})


def pytest_runtest_logreport(report):
    # One outcome per test: the call phase, or setup when it skipped/failed before the call,
    # or xdist's report for a test whose worker died (when="???").
    if (report.when in ("call", "???")
            or (report.when == "setup" and report.outcome != "passed")):
        outcome, message = report.outcome, ""
        if report.outcome != "passed" and report.longrepr is not None:
            crash = getattr(report.longrepr, "reprcrash", None)
            if crash is not None:
                message = crash.message
            elif isinstance(report.longrepr, tuple):  # skips: (path, line, reason)
                message = str(report.longrepr[-1])
            else:
                message = str(report.longrepr).strip().splitlines()[-1]
        if "worker '" in str(report.longrepr) and "crashed while running" in str(report.longrepr):
            outcome, message = "crashed", "pytest-xdist worker crashed"
        _write({"nodeid": report.nodeid, "outcome": outcome, "message": message[:500]})
