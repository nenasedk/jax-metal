"""Run JAX's own test files on the METAL backend, surviving plugin crashes.

    .venv-shim/bin/python shim/tests/jax_suite/run_jax_tests.py [-k EXPR] [--shard I/N]
        [--only FILE] [--exclude FILE] [--resume | --retry-failures] [--budget-minutes M]
        [--compare-cpu CPU_DIR] [--cpu [--workers N]] JAX_SRC OUT_DIR [TEST ...]

JAX_SRC is a checkout of the jax repository at the installed version (tests/ only is enough).
TEST is a file under JAX_SRC/tests (linalg_test.py) or a node id prefix
(linalg_test.py::NumpyLinalgTest); without any, the files in DEFAULT_FILES. Each runs under
pytest with progress_plugin.py, inside
shim/tests/guard.py: a memory cap on the pytest process (Metal allocations included), a
per-test timeout, an idle timeout for hangs, lower priority, and one GPU job at a time. If the
process dies or is killed, the test that was running is recorded as "crashed" and the run
resumes after it. pytest also restarts itself when its footprint passes a soft limit, so caches
do not grow without bound.

Batches: -k and --shard pick part of each TEST (--shard 2/4 is the second of four contiguous
slices of what -k leaves), --only/--exclude take files of node ids (one per line),
--budget-minutes stops between tests after that long (exit status 3), and --resume skips tests
that already have a result, crashed ones included (--retry-failures runs failed and crashed
ones again, for after a fix).

--cpu runs the same tests on the CPU backend in METAL emulation (cpu_rules_plugin.py), on any
machine and without the GPU lock; metal_check's predictions go to OUT_DIR/<file>.check.jsonl.
--workers N runs N pytest-xdist workers (for --cpu; the time budget is then checked between
files only).

The run stops (exit status 4) if the GPU still does not answer 30 s after the guard killed a
test: the Mac needs a restart before more GPU work.

Results: OUT_DIR/<file>.jsonl holds one record per test of that file, merged across runs (a
run replaces the records of the tests it ran and keeps the rest); OUT_DIR/<file>.run.jsonl is
the log of a run in progress; OUT_DIR/<file>.cost.jsonl has each test's duration and memory.
OUT_DIR/summary.md covers every result in OUT_DIR, grouping failures by message. With
--compare-cpu, it labels each failure by the same test's outcome in a --cpu run: "METAL only"
(it passed on CPU: MPS behavior or a plugin limit) or "also on CPU" (our rules, or the test).
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

TIMEOUT_SECONDS = int(os.environ.get("JAX_SUITE_TIMEOUT", "120"))  # per test; pytest-timeout exits
IDLE_SECONDS = TIMEOUT_SECONDS + 60  # backstop for hangs pytest-timeout cannot break
RUN_SUFFIX, COST_SUFFIX, CHECK_SUFFIX = ".run.jsonl", ".cost.jsonl", ".check.jsonl"
BUDGET_EXIT = 3
GPU_LOST_EXIT = 4
RESTART_EXIT = 75  # pytest exits with these (progress_plugin.py): restart it, or stop for the budget
STOP_EXIT = 76
# JAX's test files the shim is checked against, the most coverage per minute and least risk
# first (the order gpu_session.py runs them on METAL).
DEFAULT_FILES = (
    "fft_test.py", "random_test.py", "lax_test.py", "lax_numpy_operators_test.py",
    "lax_numpy_reducers_test.py", "lax_numpy_indexing_test.py", "lax_autodiff_test.py",
    "lax_control_flow_test.py", "nn_test.py", "lax_vmap_test.py", "lax_vmap_op_test.py",
    "lax_numpy_einsum_test.py", "random_lax_test.py", "lax_scipy_special_functions_test.py",
    "lax_scipy_test.py", "scipy_signal_test.py", "ann_test.py", "linalg_test.py",
    "lax_numpy_test.py",
)


class GpuLost(Exception):
    """The guard killed a test and the GPU did not recover."""


def read_records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_node_ids(path):
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def tail(path, lines):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()[-lines:]


def log_path(out_dir, test):
    """Results file for TEST: one per test file, whatever part of it TEST names."""
    return os.path.join(out_dir, test.split("::")[0].replace("/", "_") + ".jsonl")


def sibling(log, suffix):
    return log[:-len(".jsonl")] + suffix


def merge_run(log):
    """Fold the run log of `log` (if any) into it: this run's outcomes replace older ones."""
    run_log = sibling(log, RUN_SUFFIX)
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


def run_file(jax_src, out_dir, test, *, keyword, shard, skip_outcomes, deadline, only=None,
             exclude=(), cpu=False, workers=0):
    """Run one TEST; returns (this run's outcome records, whether the time budget ran out)."""
    log = log_path(out_dir, test)
    run_log = sibling(log, RUN_SUFFIX)
    skip_path, stderr_path = log + ".skip", log + ".stderr"
    merge_run(log)  # a run log left by a runner that was itself killed
    done_before = {r["nodeid"] for r in read_records(log) if r.get("outcome") in skip_outcomes}
    mem_limit = guard.default_mem_limit()
    if workers:  # the cap covers the whole group; each worker keeps its own caches
        mem_limit = min(workers * mem_limit, int(0.75 * guard.physical_memory()))
    crashed = []
    stopped = False
    try:
        while True:
            records = read_records(run_log)
            with open(skip_path, "w", encoding="utf-8") as handle:
                handle.writelines(nodeid + "\n" for nodeid in sorted(
                    done_before | set(exclude) | {r["nodeid"] for r in records if "nodeid" in r}))
            env = dict(os.environ, JAX_SUITE_LOG=run_log, JAX_SUITE_SKIP=skip_path,
                       JAX_SUITE_COST_LOG=sibling(log, COST_SUFFIX),
                       JAX_SUITE_SOFT_MEM=str(int(0.6 * mem_limit / max(workers, 1))),
                       PYTHONPATH=HERE + os.pathsep + os.path.dirname(HERE),
                       JAX_NUM_GENERATED_CASES=os.environ.get("JAX_NUM_GENERATED_CASES", "10"))
            if only is not None:
                env["JAX_SUITE_ONLY"] = only
            if shard:
                env["JAX_SUITE_SHARD"] = shard
            if deadline and not workers:  # a worker exiting early would count as a crash
                env["JAX_SUITE_DEADLINE"] = str(deadline)
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "-p", "progress_plugin", f"--timeout={TIMEOUT_SECONDS}",
                       "--timeout-method=thread", os.path.join("tests", test)]
            if cpu:
                env.update(JAX_PLATFORMS="cpu", JAX_METAL_CHECK_LOG=sibling(log, CHECK_SUFFIX))
                command += ["-p", "cpu_rules_plugin"]
            if workers:
                command += ["-n", str(workers)]
            if keyword:
                command += ["-k", keyword]
            start = time.time()
            with open(stderr_path, "w", encoding="utf-8") as stderr:
                code, reason = guard.run(command, cwd=jax_src, env=env, stdout=stderr,
                                         stderr=stderr, mem_limit=mem_limit,
                                         idle_timeout=IDLE_SECONDS, progress_path=run_log,
                                         gpu_stall=0 if cpu else guard.GPU_STALL_SECONDS)
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
            if reason and guard.GPU_LOST in reason:
                raise GpuLost(f"{victim}: {reason}")
            print(f"  crash in {victim} after {time.time() - start:.0f}s ({message[:80]}); "
                  "resuming", flush=True)
            if deadline and time.time() > deadline:
                stopped = True
                break
        # One per test (under xdist, workers and the controller both log each outcome).
        ran = list({r["nodeid"]: r for r in read_records(run_log) if "outcome" in r}.values())
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


def load_results(out_dir):
    """{nodeid: record} for every results file (<file>.py.jsonl) in out_dir."""
    results = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(out_dir), "*.py.jsonl"))):
        for record in read_records(path):
            if "outcome" in record:
                results[record["nodeid"]] = record
    return results


def load_predictions(out_dir):
    """{nodeid: predicted METAL issues} from the metal_check logs of a --cpu run."""
    predictions = {}
    for path in glob.glob(os.path.join(glob.escape(out_dir), "*" + CHECK_SUFFIX)):
        for record in read_records(path):
            predictions[record["nodeid"]] = record["issues"]
    return predictions


def triage(nodeid, cpu_results):
    """Where a METAL failure belongs, by the same test's outcome in a --cpu run."""
    cpu = cpu_results.get(nodeid)
    if cpu is None:
        return "no CPU result"
    if cpu["outcome"] == "passed":
        return "METAL only"
    if cpu["outcome"] == "skipped":
        return "skipped on CPU"
    return "also on CPU"


def summarize(out_dir, cpu_dir=None, title="JAX test suite on METAL"):
    """Write OUT_DIR/summary.md from every results file in OUT_DIR."""
    results = load_results(out_dir)
    cpu_results = load_results(cpu_dir) if cpu_dir else {}
    predictions = load_predictions(cpu_dir) if cpu_dir else {}
    by_file = collections.defaultdict(list)
    for record in results.values():
        by_file[record["nodeid"].split("::")[0]].append(record)
    lines = [f"# {title}", "", "| File | passed | failed | skipped | crashed |",
             "|---|---|---|---|---|"]
    totals = collections.Counter()
    for test_file, records in sorted(by_file.items()):
        outcomes = collections.Counter(r["outcome"] for r in records)
        totals.update(outcomes)
        lines.append(f"| `{test_file}` | {outcomes['passed']} | {outcomes['failed']} | "
                     f"{outcomes['skipped']} | {outcomes['crashed']} |")
    lines.append(f"| **total** | {totals['passed']} | {totals['failed']} | "
                 f"{totals['skipped']} | {totals['crashed']} |")
    failures = [r for r in results.values() if r["outcome"] in ("failed", "crashed")]
    if cpu_dir:
        lines += ["", f"Failures compared with the CPU run in `{os.path.relpath(cpu_dir, out_dir)}`"
                  " (METAL emulation): **METAL only** passed there (MPS behavior or a plugin "
                  "limit), **also on CPU** failed there too (the shim's rules, or the test)."]
        sections = collections.defaultdict(list)
        for record in failures:
            sections[triage(record["nodeid"], cpu_results)].append(record)
        order = ("METAL only", "also on CPU", "skipped on CPU", "no CPU result")
        sections = {label: sections[label] for label in order if sections.get(label)}
        lines += ["", "| Triage | Failures |", "|---|---|"]
        lines += [f"| {label} | {len(records)} |" for label, records in sections.items()]
    else:
        sections = {"all": failures}
    for label, records in sections.items():
        groups, examples, predicted = collections.Counter(), {}, {}
        for record in records:
            key = (record["outcome"], normalize(record["message"]))
            groups[key] += 1
            examples.setdefault(key, record["nodeid"])
            if predictions.get(record["nodeid"]):
                predicted.setdefault(key, "; ".join(predictions[record["nodeid"]]))
        heading = "Failure groups" if label == "all" else f"{label}: failure groups"
        lines += ["", f"## {heading} (most common first)", "",
                  "| Count | Outcome | Message (normalized) | Example |"
                  + (" Predicted on CPU |" if cpu_dir else ""),
                  "|---|---|---|---|" + ("---|" if cpu_dir else "")]
        for (outcome, message), count in groups.most_common():
            row = (f"| {count} | {outcome} | {message.replace('|', '/')} | "
                   f"`{examples[(outcome, message)]}` |")
            if cpu_dir:
                row += f" {predicted.get((outcome, message), '').replace('|', '/')} |"
            lines.append(row)
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
    parser.add_argument("tests", metavar="TEST", nargs="*",
                        help="file under JAX_SRC/tests, or a node id prefix within one "
                             "(default: the 19 files in DEFAULT_FILES)")
    parser.add_argument("-k", dest="keyword", metavar="EXPR",
                        help="pytest -k expression applied to every TEST")
    parser.add_argument("--shard", type=_shard, metavar="I/N",
                        help="run only the I-th of N contiguous slices (1-based) of each TEST")
    parser.add_argument("--only", metavar="FILE", help="run only the node ids listed in FILE")
    parser.add_argument("--exclude", metavar="FILE", help="skip the node ids listed in FILE")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true",
                        help="skip tests that already have a result in OUT_DIR")
    resume.add_argument("--retry-failures", action="store_true",
                        help="like --resume, but run failed and crashed tests again")
    parser.add_argument("--budget-minutes", type=float, metavar="M",
                        help="stop between tests after M minutes (exit status 3; continue "
                             "with --resume)")
    parser.add_argument("--cpu", action="store_true",
                        help="run on the CPU backend in METAL emulation (no GPU)")
    parser.add_argument("--workers", type=int, default=0, metavar="N",
                        help="pytest-xdist workers (for --cpu)")
    parser.add_argument("--compare-cpu", metavar="CPU_DIR",
                        help="label failures in summary.md by a --cpu run's results")
    parser.add_argument("--summary-only", action="store_true",
                        help="only rewrite OUT_DIR/summary.md")
    args = parser.parse_intermixed_args(argv)  # options may come between TESTs
    args.tests = args.tests or list(DEFAULT_FILES)
    return args


def main(argv):
    args = parse_args(argv)
    jax_src, out_dir = os.path.abspath(args.jax_src), os.path.abspath(args.out_dir)  # pytest runs in jax_src
    cpu_dir = os.path.abspath(args.compare_cpu) if args.compare_cpu else None
    title = ("JAX test suite on CPU, emulating METAL" if args.cpu
             else "JAX test suite on METAL")
    os.makedirs(out_dir, exist_ok=True)
    if args.summary_only:
        summarize(out_dir, cpu_dir, title)
        return 0
    deadline = time.time() + 60 * args.budget_minutes if args.budget_minutes else None
    skip_outcomes = (("passed", "skipped") if args.retry_failures
                     else ("passed", "skipped", "failed", "crashed") if args.resume else ())
    only = os.path.abspath(args.only) if args.only else None
    exclude = read_node_ids(args.exclude) if args.exclude else set()
    stopped = False
    with guard.GpuLock(enabled=not args.cpu):
        for index, test in enumerate(args.tests):
            if deadline and time.time() > deadline:
                stopped = True
            else:
                start = time.time()
                print(f"{test}: running", flush=True)
                try:
                    ran, stopped = run_file(jax_src, out_dir, test, keyword=args.keyword,
                                            shard=args.shard, skip_outcomes=skip_outcomes,
                                            deadline=deadline, only=only, exclude=exclude,
                                            cpu=args.cpu, workers=args.workers)
                except GpuLost as error:
                    summarize(out_dir, cpu_dir, title)
                    print(f"stopping: {error}; restart the Mac before more GPU work",
                          flush=True)
                    return GPU_LOST_EXIT
                outcomes = collections.Counter(r["outcome"] for r in ran)
                print(f"{test}: {dict(outcomes)} in {time.time() - start:.0f}s", flush=True)
                summarize(out_dir, cpu_dir, title)
            if stopped:
                left = " ".join(args.tests[index:])
                print(f"time budget used up; continue with --resume ({left})", flush=True)
                return BUDGET_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
