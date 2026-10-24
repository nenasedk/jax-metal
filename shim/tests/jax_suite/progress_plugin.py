"""pytest plugin: stream each test's start and outcome to $JAX_SUITE_LOG as JSON lines.

Used by run_jax_tests.py so that a test which aborts the process (plugin crash) can be
identified from the last "start" record without a matching outcome. Tests listed in
$JAX_SUITE_SKIP (one node id per line) are deselected, so a run can resume where it stopped.
When the process footprint passes $JAX_SUITE_SOFT_MEM bytes after a test, the session exits
with RESTART_EXIT so the runner restarts it with fresh caches instead of letting memory grow.
"""

import json
import os
import sys

_LOG = os.environ.get("JAX_SUITE_LOG")
_SKIP = os.environ.get("JAX_SUITE_SKIP")
_SOFT_MEM = int(os.environ.get("JAX_SUITE_SOFT_MEM", "0"))
RESTART_EXIT = 75

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import guard  # noqa: E402  (shim/tests/guard.py)


def _write(record):
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def pytest_collection_modifyitems(config, items):
    if not _SKIP or not os.path.exists(_SKIP):
        return
    with open(_SKIP, encoding="utf-8") as handle:
        skip = {line.strip() for line in handle if line.strip()}
    kept = [item for item in items if item.nodeid not in skip]
    if len(kept) != len(items):
        config.hook.pytest_deselected(items=[item for item in items if item.nodeid in skip])
        items[:] = kept


def pytest_runtest_logfinish(nodeid, location):
    del nodeid, location
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
