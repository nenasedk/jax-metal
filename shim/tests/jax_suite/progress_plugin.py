"""pytest plugin: stream each test's start and outcome to $JAX_SUITE_LOG as JSON lines.

Used by run_jax_tests.py so that a test which aborts the process (plugin crash) can be
identified from the last "start" record without a matching outcome.
"""

import json
import os

_LOG = os.environ.get("JAX_SUITE_LOG")


def _write(record):
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


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
