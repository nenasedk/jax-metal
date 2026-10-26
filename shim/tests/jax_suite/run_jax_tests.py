"""Run JAX's own test files on the METAL backend, surviving plugin crashes.

    .venv-shim/bin/python shim/tests/jax_suite/run_jax_tests.py [-k EXPR] [--shard I/N]
        [--resume | --retry-failures] [--budget-minutes M] JAX_SRC OUT_DIR TEST ...

JAX_SRC is a checkout of the jax repository at the installed version (tests/ only is enough).
TEST is a file under JAX_SRC/tests (linalg_test.py) or a node id prefix
(linalg_test.py::NumpyLinalgTest). Each runs under pytest with progress_plugin.py, inside
shim/tests/guard.py: a memory cap on the pytest process (Metal allocations included), a
per-test timeout, an idle timeout for hangs, lower priority, and one GPU job at a time. If the
process dies or is killed, the test that was running is recorded as "crashed" and the run
resumes after it. pytest also restarts itself when its footprint passes a soft limit, so caches
do not grow without bound.

Batches: -k and --shard pick part of each TEST (--shard 2/4 is the second of four contiguous
slices of what -k leaves), --budget-minutes stops between tests after that long, and --resume
skips tests that already have a result, crashed ones included (--retry-failures runs failed and
crashed ones again, for after a fix).

Results: OUT_DIR/<file>.jsonl holds one record per test of that file, merged across runs (a
run replaces the records of the tests it ran and keeps the rest); OUT_DIR/<file>.run.jsonl
is the log of a run in progress. OUT_DIR/summary.md covers every result in OUT_DIR, grouping
failures by message.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import guard  # noqa: E402  (shim/tests/guard.py)
from progress_plugin import RESTART_EXIT, STOP_EXIT  # noqa: E402

TIMEOUT_SECONDS = int(os.environ.get("JAX_SUITE_TIMEOUT", "120"))  # per test; pytest-timeout exits
IDLE_SECONDS = TIMEOUT_SECONDS + 60  # backstop for hangs pytest-timeout cannot break
RUN_SUFFIX = ".run.jsonl"


def read_records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tail(path, lines):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()[-lines:]


def log_path(out_dir, test):
    """Results file for TEST: one per test file, whatever part of it TEST names."""
    return os.path.join(out_dir, test.split("::")[0].replace("/", "_") + ".jsonl")


def merge_run(log):
    """Fold the run log of `log` (if any) into it: this run's outcomes replace older ones."""
    run_log = log[:-len(".jsonl")] + RUN_SUFFIX
    if not os.path.exists(run_log):
        return
    results = {r["nodeid"]: r for r in read_records(log) if "outcome" in r}
    for record in read_records(run_log):
        if "outcome" in record:
            results.pop(record["nodeid"], None)  # re-runs move to the end, in run order
            results[record["nodeid"]] = record
    with open(log + ".tmp", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in results.values())
    os.replace(log + ".tmp", log)
    os.remove(run_log)


def run_file(jax_src, out_dir, test, *, keyword, shard, skip_outcomes, deadline):
    """Run one TEST; returns (this run's outcome records, whether the time budget ran out)."""
    log = log_path(out_dir, test)
    run_log = log[:-len(".jsonl")] + RUN_SUFFIX
    skip_path, stderr_path = log + ".skip", log + ".stderr"
    merge_run(log)  # a run log left by a runner that was itself killed
    done_before = {r["nodeid"] for r in read_records(log) if r.get("outcome") in skip_outcomes}
    mem_limit = guard.default_mem_limit()
    crashed = []
    stopped = False
    try:
        while True:
            records = read_records(run_log)
            with open(skip_path, "w", encoding="utf-8") as handle:
                handle.writelines(nodeid + "\n" for nodeid in sorted(
                    done_before | {r["nodeid"] for r in records if "nodeid" in r}))
            env = dict(os.environ, JAX_SUITE_LOG=run_log, JAX_SUITE_SKIP=skip_path,
                       JAX_SUITE_SOFT_MEM=str(int(0.6 * mem_limit)),
                       PYTHONPATH=HERE + os.pathsep + os.path.dirname(HERE),
                       JAX_NUM_GENERATED_CASES=os.environ.get("JAX_NUM_GENERATED_CASES", "10"))
            if shard:
                env["JAX_SUITE_SHARD"] = shard
            if deadline:
                env["JAX_SUITE_DEADLINE"] = str(deadline)
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "-p", "progress_plugin", f"--timeout={TIMEOUT_SECONDS}",
                       "--timeout-method=thread", os.path.join("tests", test)]
            if keyword:
                command += ["-k", keyword]
            start = time.time()
            with open(stderr_path, "w", encoding="utf-8") as stderr:
                code, reason = guard.run(command, cwd=jax_src, env=env, stdout=stderr,
                                         stderr=stderr, mem_limit=mem_limit,
                                         idle_timeout=IDLE_SECONDS, progress_path=run_log)
            records = read_records(run_log)
            finished = {r["nodeid"] for r in records if "nodeid" in r}
            started = [r["start"] for r in records if "start" in r]
            dangling = [nodeid for nodeid in started if nodeid not in finished]
            if code == RESTART_EXIT and not reason:
                print(f"  restarting pytest (memory soft limit) after {len(finished)} tests",
                      flush=True)
                continue
            if code == STOP_EXIT and not reason:
                stopped = True
                break
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
            crash_number = len(glob.glob(glob.escape(log) + ".crash-*.txt")) + 1
            with open(f"{log}.crash-{crash_number}.txt", "w", encoding="utf-8") as handle:
                handle.write(f"{victim}\nexit {code}\nguard: {reason}\n\n" + "\n".join(stderr_tail))
            message = (f"guard: {reason}" if reason
                       else f"exit {code}: {(stderr_tail or [''])[-1][:300]}")
            if not reason and "+ Timeout +" in "\n".join(tail(stderr_path, 100000)):
                message = f"timeout after {TIMEOUT_SECONDS}s"
            with open(run_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"nodeid": victim, "outcome": "crashed",
                                         "message": message}) + "\n")
            print(f"  crash in {victim} after {time.time() - start:.0f}s ({message[:80]}); "
                  "resuming", flush=True)
            if deadline and time.time() > deadline:
                stopped = True
                break
        ran = [r for r in read_records(run_log) if "outcome" in r]
    finally:
        merge_run(log)
        if os.path.exists(skip_path):
            os.remove(skip_path)
    return ran, stopped


def normalize(message):
    """Group similar failure messages: drop numbers, paths, shapes."""
    message = message.splitlines()[0] if message else ""
    message = re.sub(r"/[^\s:]+", "<path>", message)
    message = re.sub(r"\d+(\.\d+)?(e[-+]?\d+)?", "N", message)
    return message[:160]


def summarize(out_dir):
    """Write OUT_DIR/summary.md from every results file in OUT_DIR."""
    by_file = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(glob.escape(out_dir), "*.jsonl"))):
        if not path.endswith(RUN_SUFFIX):
            for record in read_records(path):
                if "outcome" in record:
                    by_file[record["nodeid"].split("::")[0]].append(record)
    lines = ["# JAX test suite on METAL", "",
             "| File | passed | failed | skipped | crashed |", "|---|---|---|---|---|"]
    groups = collections.Counter()
    examples = {}
    totals = collections.Counter()
    for test_file, records in sorted(by_file.items()):
        outcomes = collections.Counter(r["outcome"] for r in records)
        totals.update(outcomes)
        lines.append(f"| `{test_file}` | {outcomes['passed']} | {outcomes['failed']} | "
                     f"{outcomes['skipped']} | {outcomes['crashed']} |")
        for record in records:
            if record["outcome"] in ("failed", "crashed"):
                key = (record["outcome"], normalize(record["message"]))
                groups[key] += 1
                examples.setdefault(key, record["nodeid"])
    lines.append(f"| **total** | {totals['passed']} | {totals['failed']} | "
                 f"{totals['skipped']} | {totals['crashed']} |")
    lines += ["", "## Failure groups (most common first)", "",
              "| Count | Outcome | Message (normalized) | Example |", "|---|---|---|---|"]
    for (outcome, message), count in groups.most_common():
        cell = message.replace("|", "/")
        lines.append(f"| {count} | {outcome} | {cell} | `{examples[(outcome, message)]}` |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _shard(text):
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if not match or not 1 <= int(match[1]) <= int(match[2]):
        raise argparse.ArgumentTypeError("expected I/N with 1 <= I <= N, e.g. 2/4")
    return text


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("jax_src", metavar="JAX_SRC")
    parser.add_argument("out_dir", metavar="OUT_DIR")
    parser.add_argument("tests", metavar="TEST", nargs="+",
                        help="file under JAX_SRC/tests, or a node id prefix within one")
    parser.add_argument("-k", dest="keyword", metavar="EXPR",
                        help="pytest -k expression applied to every TEST")
    parser.add_argument("--shard", type=_shard, metavar="I/N",
                        help="run only the I-th of N contiguous slices (1-based) of each TEST")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true",
                        help="skip tests that already have a result in OUT_DIR")
    resume.add_argument("--retry-failures", action="store_true",
                        help="like --resume, but run failed and crashed tests again")
    parser.add_argument("--budget-minutes", type=float, metavar="M",
                        help="stop between tests after M minutes (continue with --resume)")
    return parser.parse_intermixed_args(argv)  # options may come between TESTs


def main(argv):
    args = parse_args(argv)
    jax_src, out_dir = os.path.abspath(args.jax_src), os.path.abspath(args.out_dir)  # pytest runs in jax_src
    os.makedirs(out_dir, exist_ok=True)
    deadline = time.time() + 60 * args.budget_minutes if args.budget_minutes else None
    skip_outcomes = (("passed", "skipped") if args.retry_failures
                     else ("passed", "skipped", "failed", "crashed") if args.resume else ())
    with guard.GpuLock():
        for index, test in enumerate(args.tests):
            if deadline and time.time() > deadline:
                stopped = True
            else:
                start = time.time()
                print(f"{test}: running", flush=True)
                ran, stopped = run_file(jax_src, out_dir, test, keyword=args.keyword,
                                        shard=args.shard, skip_outcomes=skip_outcomes,
                                        deadline=deadline)
                outcomes = collections.Counter(r["outcome"] for r in ran)
                print(f"{test}: {dict(outcomes)} in {time.time() - start:.0f}s", flush=True)
                summarize(out_dir)
            if stopped:
                left = " ".join(args.tests[index:])
                print(f"time budget used up; continue with --resume ({left})", flush=True)
                break


if __name__ == "__main__":
    main(sys.argv[1:])
