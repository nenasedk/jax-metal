"""Run JAX's own test files on the METAL backend, surviving plugin crashes.

    .venv-shim/bin/python shim/tests/jax_suite/run_jax_tests.py JAX_SRC OUT_DIR test_file.py ...

JAX_SRC is a checkout of the jax repository at the installed version (tests/ only is enough).
Each file runs under pytest with progress_plugin.py; if the process dies, the test that was
running is recorded as "crashed", deselected, and the file is resumed. Results go to
OUT_DIR/<file>.jsonl, and OUT_DIR/summary.md groups failures by message for triage.
"""

import collections
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT_SECONDS = 300


def read_records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_file(jax_src, out_dir, test_file):
    log = os.path.join(out_dir, test_file.replace("/", "_") + ".jsonl")
    if os.path.exists(log):
        os.remove(log)
    crashed = []
    while True:
        env = dict(os.environ, JAX_SUITE_LOG=log, PYTHONPATH=HERE,
                   JAX_NUM_GENERATED_CASES=os.environ.get("JAX_NUM_GENERATED_CASES", "10"))
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "-p", "progress_plugin", f"--timeout={TIMEOUT_SECONDS}",
                   os.path.join("tests", test_file)]
        command += [f"--deselect={nodeid}" for nodeid in crashed]
        start = time.time()
        proc = subprocess.run(command, cwd=jax_src, env=env, capture_output=True, text=True)
        records = read_records(log)
        finished = {r["nodeid"] for r in records if "nodeid" in r}
        started = [r["start"] for r in records if "start" in r]
        dangling = [nodeid for nodeid in started if nodeid not in finished and nodeid not in crashed]
        if proc.returncode in (0, 1, 5) or not dangling:  # 0 ok, 1 failures, 5 nothing collected
            if proc.returncode not in (0, 1, 5):
                tail = (proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines() or [""])[-1]
                print(f"  pytest exited {proc.returncode}: {tail[:200]}", flush=True)
            break
        victim = dangling[-1]
        crashed.append(victim)
        with open(f"{log}.crash-{len(crashed)}.txt", "w", encoding="utf-8") as handle:
            handle.write(f"{victim}\nexit {proc.returncode}\n\n" + "\n".join(proc.stderr.splitlines()[-120:]))
        reason = (proc.stderr.strip().splitlines() or [""])[-1][:300]
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"nodeid": victim, "outcome": "crashed",
                                     "message": f"exit {proc.returncode}: {reason}"}) + "\n")
        print(f"  crash in {victim} after {time.time() - start:.0f}s; resuming", flush=True)
    return read_records(log)


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
    for test_file in test_files:
        start = time.time()
        print(f"{test_file}: running", flush=True)
        results[test_file] = run_file(jax_src, out_dir, test_file)
        outcomes = collections.Counter(r["outcome"] for r in results[test_file] if "outcome" in r)
        print(f"{test_file}: {dict(outcomes)} in {time.time() - start:.0f}s", flush=True)
        summarize(out_dir, results)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
