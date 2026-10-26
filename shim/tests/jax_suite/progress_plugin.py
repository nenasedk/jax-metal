"""pytest plugin: stream each test's start and outcome to $JAX_SUITE_LOG as JSON lines.

Used by run_jax_tests.py so that a test which aborts the process (plugin crash) can be
identified from the last "start" record without a matching outcome. Other settings, all
optional:

- $JAX_SUITE_SHARD="I/N" keeps the I-th of N contiguous slices (1-based) of the tests left
  after -k, so a large file can run in batches;
- tests listed in $JAX_SUITE_SKIP (one node id per line) are deselected after sharding, so a
  run can resume where it stopped without changing which tests belong to a shard;
- when the process footprint passes $JAX_SUITE_SOFT_MEM bytes after a test, the session exits
  with RESTART_EXIT so the runner restarts it with fresh caches instead of letting memory grow;
- once time.time() passes $JAX_SUITE_DEADLINE, the session exits with STOP_EXIT after the
  current test.
"""

import json
import os
import sys
import time

import pytest

_LOG = os.environ.get("JAX_SUITE_LOG")
_SKIP = os.environ.get("JAX_SUITE_SKIP")
_SHARD = os.environ.get("JAX_SUITE_SHARD")
_SOFT_MEM = int(os.environ.get("JAX_SUITE_SOFT_MEM", "0"))
_DEADLINE = float(os.environ.get("JAX_SUITE_DEADLINE", "0"))
RESTART_EXIT = 75
STOP_EXIT = 76

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import guard  # noqa: E402  (shim/tests/guard.py)


def _write(record):
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


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
    if _SHARD:
        start, stop = shard_slice(len(items), _SHARD)
        in_shard = {id(item) for item in items[start:stop]}
        _deselect(config, items, lambda item: id(item) in in_shard)
    if _SKIP and os.path.exists(_SKIP):
        with open(_SKIP, encoding="utf-8") as handle:
            skip = {line.strip() for line in handle if line.strip()}
        _deselect(config, items, lambda item: item.nodeid not in skip)


def pytest_runtest_logfinish(nodeid, location):
    del nodeid, location
    if _DEADLINE and time.time() > _DEADLINE:
        _write({"stop": "time budget"})
        os._exit(STOP_EXIT)
    if _SOFT_MEM and guard.footprint(os.getpid()) > _SOFT_MEM:
        _write({"restart": guard.footprint(os.getpid())})
        os._exit(RESTART_EXIT)


def pytest_runtest_logstart(nodeid, location):
    del location
    _write({"start": nodeid})


def pytest_runtest_logreport(report):
    # One outcome per test: the call phase, or setup when it skipped/failed before the call.
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        message = ""
        if report.outcome != "passed" and report.longrepr is not None:
            crash = getattr(report.longrepr, "reprcrash", None)
            if crash is not None:
                message = crash.message
            elif isinstance(report.longrepr, tuple):  # skips: (path, line, reason)
                message = str(report.longrepr[-1])
            else:
                message = str(report.longrepr).strip().splitlines()[-1]
        _write({"nodeid": report.nodeid, "outcome": report.outcome, "message": message[:500]})
