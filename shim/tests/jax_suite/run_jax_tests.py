"""Run JAX's own test files on the METAL backend, surviving plugin crashes.

    .venv-shim/bin/python shim/tests/jax_suite/run_jax_tests.py JAX_SRC OUT_DIR test_file.py ...

JAX_SRC is a checkout of the jax repository at the installed version (tests/ only is enough).
Each file runs under pytest with progress_plugin.py, inside shim/tests/guard.py: a memory cap
on the pytest process (Metal allocations included), a per-test timeout, an idle timeout for
hangs, lower priority, and one GPU job at a time. If the process dies or is killed, the test
that was running is recorded as "crashed" and the file resumes after it. pytest also restarts
itself when its footprint passes a soft limit, so caches do not grow without bound.
Results go to OUT_DIR/<file>.jsonl, and OUT_DIR/summary.md groups failures by message.
"""

import collections
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import guard  # noqa: E402  (shim/tests/guard.py)
from progress_plugin import RESTART_EXIT  # noqa: E402

TIMEOUT_SECONDS = int(os.environ.get("JAX_SUITE_TIMEOUT", "120"))  # per test; pytest-timeout exits
IDLE_SECONDS = TIMEOUT_SECONDS + 60  # backstop for hangs pytest-timeout cannot break


def read_records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tail(path, lines):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()[-lines:]


def run_file(jax_src, out_dir, test_file):
    log = os.path.join(out_dir, test_file.replace("/", "_") + ".jsonl")
    skip_path, stderr_path = log + ".skip", log + ".stderr"
    for path in (log, skip_path):
        if os.path.exists(path):
            os.remove(path)
    mem_limit = guard.default_mem_limit()
    crashed = []
    while True:
        records = read_records(log)
        with open(skip_path, "w", encoding="utf-8") as handle:
            handle.writelines(r["nodeid"] + "\n" for r in records if "nodeid" in r)
        env = dict(os.environ, JAX_SUITE_LOG=log, JAX_SUITE_SKIP=skip_path,
                   JAX_SUITE_SOFT_MEM=str(int(0.6 * mem_limit)),
                   PYTHONPATH=HERE + os.pathsep + os.path.dirname(HERE),
                   JAX_NUM_GENERATED_CASES=os.environ.get("JAX_NUM_GENERATED_CASES", "10"))
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "-p", "progress_plugin", f"--timeout={TIMEOUT_SECONDS}",
                   "--timeout-method=thread", os.path.join("tests", test_file)]
        start = time.time()
        with open(stderr_path, "w", encoding="utf-8") as stderr:
            code, reason = guard.run(command, cwd=jax_src, env=env, stdout=stderr,
                                     stderr=stderr, mem_limit=mem_limit,
                                     idle_timeout=IDLE_SECONDS, progress_path=log)
        records = read_records(log)
        finished = {r["nodeid"] for r in records if "nodeid" in r}
        started = [r["start"] for r in records if "start" in r]
        dangling = [nodeid for nodeid in started if nodeid not in finished]
        if code == RESTART_EXIT and not reason:
            print(f"  restarting pytest (memory soft limit) after {len(finished)} tests", flush=True)
            continue
        if not dangling and not reason:
            if code not in (0, 1, 5):  # 0 ok, 1 failures, 5 nothing collected
                last = (tail(stderr_path, 1) or [""])[-1]
                print(f"  pytest exited {code}: {last[:200]}", flush=True)
            break
        if not dangling:  # killed between tests (e.g. at collection): nothing to blame
            print(f"  guard killed pytest outside a test: {reason}", flush=True)
            break
        victim = dangling[-1]
        crashed.append(victim)
        stderr_tail = tail(stderr_path, 120)
        with open(f"{log}.crash-{len(crashed)}.txt", "w", encoding="utf-8") as handle:
            handle.write(f"{victim}\nexit {code}\nguard: {reason}\n\n" + "\n".join(stderr_tail))
        message = f"guard: {reason}" if reason else f"exit {code}: {(stderr_tail or [''])[-1][:300]}"
        if not reason and "+ Timeout +" in "\n".join(tail(stderr_path, 100000)):
            message = f"timeout after {TIMEOUT_SECONDS}s"
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"nodeid": victim, "outcome": "crashed", "message": message}) + "\n")
        print(f"  crash in {victim} after {time.time() - start:.0f}s ({message[:80]}); resuming",
              flush=True)
    os.remove(skip_path)
    return [r for r in read_records(log) if "outcome" in r]


def normalize(message):
    """Group similar failure messages: drop numbers, paths, shapes."""
    message = message.splitlines()[0] if message else ""
    message = re.sub(r"/[^\s:]+", "<path>", message)
    message = re.sub(r"\d+(\.\d+)?(e[-+]?\d+)?", "N", message)
    return message[:160]


def summarize(out_dir, results):
    lines = ["# JAX test suite on METAL", "",
             "| File | passed | failed | skipped | crashed |", "|---|---|---|---|---|"]
    groups = collections.Counter()
    examples = {}
    for test_file, records in results.items():
        outcomes = collections.Counter(r["outcome"] for r in records if "outcome" in r)
        lines.append(f"| `{test_file}` | {outcomes['passed']} | {outcomes['failed']} | "
                     f"{outcomes['skipped']} | {outcomes['crashed']} |")
        for record in records:
            if record.get("outcome") in ("failed", "crashed"):
                key = (record["outcome"], normalize(record["message"]))
                groups[key] += 1
                examples.setdefault(key, record["nodeid"])
    lines += ["", "## Failure groups (most common first)", "",
              "| Count | Outcome | Message (normalized) | Example |", "|---|---|---|---|"]
    for (outcome, message), count in groups.most_common():
        cell = message.replace("|", "/")
        lines.append(f"| {count} | {outcome} | {cell} | `{examples[(outcome, message)]}` |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(jax_src, out_dir, test_files):
    jax_src, out_dir = os.path.abspath(jax_src), os.path.abspath(out_dir)  # pytest runs in jax_src
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    with guard.GpuLock():
        for test_file in test_files:
            start = time.time()
            print(f"{test_file}: running", flush=True)
            results[test_file] = run_file(jax_src, out_dir, test_file)
            outcomes = collections.Counter(r["outcome"] for r in results[test_file])
            print(f"{test_file}: {dict(outcomes)} in {time.time() - start:.0f}s", flush=True)
            summarize(out_dir, results)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
