"""Run the whole GPU validation on an Apple-silicon Mac with one command, resumably.

    make -C shim gpu-session [ARGS="--budget-minutes 90"]
    .venv-shim/bin/python shim/tests/gpu_session.py [--budget-minutes M] [--steps A,B]
        [--skip A,B] [--redo A,B] [--retry-failures] [--bench] [--restart] [--dry-run]

Steps, in order. Every command that uses the GPU runs under guard.py, one GPU job at a time.

  setup              JAX's tests at the installed version (re/work/jax-src), pytest, C++ test data
  install            the package's install check, then a first METAL computation
  cxx                make test: the C++ load tests (Apple's plugin alone, then through the shim)
  e2e, phase3, complex, narrow   the regression tests
  probe              probe_jax_ops.py -> re/findings/jax-op-probe.md
  accuracy           accuracy_sweep.py --strict
  accuracy-extended  accuracy_extended.py -> re/findings/accuracy-extended.md
  jax-suite          JAX's test files without the heaviest tests, within the time budget
  jax-heavy          the heaviest tests (by memory and time in the CPU run), last
  bench              only with --bench: op_bench.py and train_bench.py, at normal priority

A step that has finished, whatever its result, is not run again (--redo NAME runs it again);
setup and install are cheap checks and always run. The JAX steps stop when the time budget
(--budget-minutes, default 120) is used up and continue where they stopped the next time;
after a fix, --retry-failures runs JAX's failed and crashed tests again.
The session stops early if the first METAL computation fails, or if the GPU has not
recovered after the guard killed a job.

Everything goes to shim/build/gpu-session/: summary.md (the steps and the machine), logs/,
and jax-suite/ (summary.md there labels each failure "METAL only" or "also on CPU" from the
CPU run in shim/build/jax-suite-cpu). Afterwards, commit it and re/findings (shim/build is
git-ignored, so `git add -f shim/build/gpu-session`).
"""

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

TESTS = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.dirname(TESTS)
ROOT = os.path.dirname(SHIM)
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "jax_suite"))
import guard  # noqa: E402
import run_jax_tests  # noqa: E402

SESSION = os.path.join(SHIM, "build", "gpu-session")
LOGS = os.path.join(SESSION, "logs")
STATE = os.path.join(SESSION, "state.json")
SUITE_OUT = os.path.join(SESSION, "jax-suite")
HEAVY = os.path.join(SESSION, "heavy-tests.txt")
CPU_DIR = os.path.join(SHIM, "build", "jax-suite-cpu")
JAX_SRC = os.environ.get("JAX_SRC", os.path.join(ROOT, "re", "work", "jax-src"))
FINDINGS = os.path.join(ROOT, "re", "findings")
TEST_DATA = os.path.join(SHIM, "build", "testdata")
PY = sys.executable

JAX_FILES = run_jax_tests.DEFAULT_FILES
# A test is heavy when, in the CPU run, it took longer or its process grew by more than this,
# or when its parameters have an array dimension of at least HEAVY_DIM. METAL's footprint
# differs from the CPU's, but big shapes and long compiles are what went wrong on METAL.
HEAVY_SECONDS = 5
HEAVY_MB = 100
HEAVY_DIM = 1000
ALWAYS = ("setup", "install")


class Abort(Exception):
    """The session cannot usefully go on."""


def log_path(name):
    return os.path.join(LOGS, name + ".log")


def tail(path, lines=8):
    try:
        return run_jax_tests.tail(path, lines)
    except OSError:
        return []


def guarded(name, command, *, timeout, throttle=True, env=None):
    """Run `command` (cwd shim/) under the guard and the GPU lock; returns (code, reason)."""
    with open(log_path(name), "w", encoding="utf-8") as out:
        out.write(f"$ {' '.join(command)}\n")
        out.flush()
        with guard.GpuLock():
            code, reason = guard.run(command, cwd=SHIM, env=env, stdout=out,
                                     stderr=subprocess.STDOUT, timeout=timeout, throttle=throttle)
    if reason:
        with open(log_path(name), "a", encoding="utf-8") as out:
            out.write(f"\nguard: killed: {reason}\n")
        if guard.GPU_LOST in reason:
            raise Abort(f"{name}: {reason}; restart the Mac before going on")
    return code, reason


def plain(name, command, *, append=False):
    """Run a command that does not use the GPU; returns its exit status."""
    with open(log_path(name), "a" if append else "w", encoding="utf-8") as out:
        out.write(f"$ {' '.join(command)}\n")
        out.flush()
        return subprocess.run(command, cwd=SHIM, stdout=out, stderr=subprocess.STDOUT).returncode


# --- steps --------------------------------------------------------------------------------
# Each returns (status, note); status is passed, failed, killed, partial or done (JAX steps).

def _result(code, reason, note=""):
    if reason:
        return "killed", reason
    return ("passed" if code == 0 else "failed"), note or ("" if code == 0 else f"exit {code}")


def step_setup(args):
    del args
    notes = []
    if not os.path.isdir(os.path.join(JAX_SRC, "tests")):
        version = importlib.metadata.version("jax")
        code = plain("setup", ["git", "clone", "--depth", "1", "--branch", f"jax-v{version}",
                               "--filter=blob:none", "--sparse", "https://github.com/jax-ml/jax",
                               JAX_SRC])
        code = code or plain("setup", ["git", "-C", JAX_SRC, "sparse-checkout", "set", "tests"],
                             append=True)
        if code:
            raise Abort(f"could not check out JAX's tests into {JAX_SRC} (see {log_path('setup')})")
        notes.append(f"checked out JAX {version}'s tests")
    missing = [package for module, package in (("pytest", "pytest"),
                                                ("pytest_timeout", "pytest-timeout"),
                                                ("absl", "absl-py"))
               if importlib.util.find_spec(module) is None]
    if missing:
        if plain("setup", [PY, "-m", "pip", "install", *missing], append=True):
            raise Abort(f"could not install {' '.join(missing)} (see {log_path('setup')})")
        notes.append("installed " + " ".join(missing))
    if not os.path.exists(os.path.join(TEST_DATA, "manifest.txt")):
        if plain("setup", [PY, os.path.join(TESTS, "gen_artifact.py"), TEST_DATA], append=True):
            raise Abort(f"could not write the C++ test data (see {log_path('setup')})")
        notes.append("wrote C++ test data")
    return "passed", "; ".join(notes)


SMOKE = ("import jax, jax.numpy as jnp\n"
         "device = jax.devices()[0]\n"
         "assert device.platform.upper() == 'METAL', jax.devices()\n"
         "x = jnp.arange(16.0).reshape(4, 4)\n"
         "y = jax.jit(lambda a: jnp.sin(a) @ a.T + 1)(x)\n"
         "import numpy as np\n"
         "np.testing.assert_allclose(np.asarray(y), np.sin(np.asarray(x)) @ np.asarray(x).T + 1,"
         " rtol=1e-4, atol=1e-4)\n"
         "print('METAL ok:', device, jax.__version__)\n")


def step_install(args):
    del args
    info = [PY, "-m", "jax_plugins.metal_shim", "info"]
    if plain("install", info):
        plain("install", [PY, "-m", "jax_plugins.metal_shim", "fetch"], append=True)
        if plain("install", info, append=True):
            raise Abort(f"the shim is not installed correctly (see {log_path('install')})")
    with open(log_path("install"), "a", encoding="utf-8") as out:
        out.write("\n")
    code, reason = guarded("install-smoke", [PY, "-c", SMOKE], timeout=300)
    if code or reason:
        raise Abort(f"the first METAL computation failed (see {log_path('install-smoke')}): "
                    + (reason or (tail(log_path("install-smoke"), 1) or [""])[-1]))
    return "passed", ""


def apple_plugin():
    found = subprocess.run([PY, "-c", "from jax_plugins.metal_shim import _plugin; "
                            "print(_plugin.find_plugin() or '')"],
                           capture_output=True, text=True, check=False)
    return found.stdout.strip().splitlines()[-1] if found.stdout.strip() else ""


def step_cxx(args):
    del args
    plugin = apple_plugin()
    if not plugin:
        return "failed", "Apple's plugin not found"
    code, reason = guarded("cxx", ["make", "test", f"PLUGIN={plugin}", f"TEST_DATA={TEST_DATA}"],
                           timeout=900)
    return _result(code, reason)


def script_step(name, script, *script_args, timeout=1800, throttle=True):
    def step(args):
        del args
        code, reason = guarded(name, [PY, os.path.join(SHIM, script), *script_args],
                               timeout=timeout, throttle=throttle)
        return _result(code, reason)
    return step


def largest_dimension(params):
    """The largest number inside a tuple in a test's parameter string, e.g. a shape."""
    tuples = re.findall(r"\(([\d,\s]+)\)", params)
    return max((int(n) for t in tuples for n in re.findall(r"\d+", t)), default=0)


def heavy_tests():
    """Node ids of the heaviest tests by the CPU run: slow, memory-hungry, big, or killed."""
    heavy = set()
    for path in sorted(os.listdir(CPU_DIR)) if os.path.isdir(CPU_DIR) else ():
        full = os.path.join(CPU_DIR, path)
        if path.endswith(run_jax_tests.COST_SUFFIX):
            for record in run_jax_tests.read_records(full):
                if (record["seconds"] > HEAVY_SECONDS
                        or record["peak_mb"] - record["before_mb"] > HEAVY_MB
                        or largest_dimension(record.get("params", "")) >= HEAVY_DIM):
                    heavy.add(record["nodeid"])
        elif path.endswith(".py.jsonl"):
            heavy.update(r["nodeid"] for r in run_jax_tests.read_records(full)
                         if r.get("outcome") == "crashed")
    return sorted(heavy)


def write_heavy():
    heavy = heavy_tests()
    with open(HEAVY, "w", encoding="utf-8") as handle:
        handle.writelines(nodeid + "\n" for nodeid in heavy)
    return heavy


def run_suite(name, files, selection, deadline, retry_failures):
    """run_jax_tests.py over `files`; it guards each pytest process and takes the GPU lock."""
    minutes = (deadline - time.time()) / 60
    if minutes < 1:
        return "partial", "no time left in the budget"
    command = [PY, os.path.join(TESTS, "jax_suite", "run_jax_tests.py"), JAX_SRC, SUITE_OUT,
               *files, *selection, "--retry-failures" if retry_failures else "--resume",
               "--budget-minutes", f"{minutes:.1f}"]
    if os.path.isdir(CPU_DIR):
        command += ["--compare-cpu", CPU_DIR]
    with open(log_path(name), "a", encoding="utf-8") as out:
        out.write(f"$ {' '.join(command)}\n")
        out.flush()
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # the runner prints one line per file and per crash
            out.write(line)
            out.flush()
            print("    " + line, end="", flush=True)
        code = proc.wait()
    if code == run_jax_tests.GPU_LOST_EXIT:
        raise Abort(f"{name}: the GPU did not recover after a kill; restart the Mac before "
                    "going on")
    if code == run_jax_tests.BUDGET_EXIT:
        return "partial", "time budget used up"
    # Test failures are results, in jax-suite/summary.md; "failed" is the runner's own.
    return ("done" if code == 0 else "failed"), ("" if code == 0 else f"exit {code}")


def step_jax_suite(args, deadline):
    write_heavy()
    return run_suite("jax-suite", JAX_FILES, ["--exclude", HEAVY], deadline,
                     args.retry_failures)


def step_jax_heavy(args, deadline):
    heavy = write_heavy()
    files = [f for f in JAX_FILES if any(n.startswith(f"tests/{f}::") for n in heavy)]
    if not files:
        return "passed", "no heavy tests"
    return run_suite("jax-heavy", files, ["--only", HEAVY], deadline, args.retry_failures)


def step_bench(args):
    del args
    out = os.path.join(SESSION, "bench")
    os.makedirs(out, exist_ok=True)
    statuses = []
    for name, command in (
            ("bench-op", [PY, "benchmarks/op_bench.py", os.path.join(out, "op_bench.json")]),
            ("bench-train-metal", [PY, "benchmarks/train_bench.py", "METAL",
                                   os.path.join(out, "train_metal.json")]),
            ("bench-train-cpu", [PY, "benchmarks/train_bench.py", "cpu",
                                 os.path.join(out, "train_cpu.json")])):
        code, reason = guarded(name, command, timeout=1800, throttle=False)
        statuses.append(_result(code, reason)[0])
    if importlib.util.find_spec("flax") and importlib.util.find_spec("optax"):
        code, reason = guarded("bench-flax", [PY, "benchmarks/flax_train.py",
                                              os.path.join(out, "flax_train.json")],
                               timeout=1800, throttle=False)
        statuses.append(_result(code, reason)[0])
    worst = next((s for s in ("killed", "failed") if s in statuses), "passed")
    return worst, ""


STEPS = {
    "setup": step_setup,
    "install": step_install,
    "cxx": step_cxx,
    "e2e": script_step("e2e", "tests/test_jax_e2e.py"),
    "phase3": script_step("phase3", "tests/test_phase3.py"),
    "complex": script_step("complex", "tests/test_complex.py"),
    "narrow": script_step("narrow", "tests/test_narrow.py"),
    "probe": script_step("probe", "tests/probe_jax_ops.py",
                         os.path.join(FINDINGS, "jax-op-probe.md"), timeout=3600),
    "accuracy": script_step("accuracy", "tests/accuracy_sweep.py", "--strict"),
    "accuracy-extended": script_step("accuracy-extended", "tests/accuracy_extended.py",
                                     os.path.join(FINDINGS, "accuracy-extended.md")),
    "jax-suite": step_jax_suite,
    "jax-heavy": step_jax_heavy,
    "bench": step_bench,
}
BUDGETED = ("jax-suite", "jax-heavy")


# --- state and report ---------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as handle:
            return json.load(handle)
    return {"steps": {}}


def save_state(state):
    with open(STATE + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=1)
    os.replace(STATE + ".tmp", STATE)


def machine():
    def sysctl(name):
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                                  check=False).stdout.strip()
        except OSError:
            return ""
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=False).stdout.strip()
    versions = {}
    for package in ("jax", "jaxlib", "jax-metal-shim"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "commit": commit + (" (with local changes)" if dirty else ""),
        "machine": sysctl("machdep.cpu.brand_string") or platform.machine(),
        "memory": f"{guard.physical_memory() / 2**30:.0f} GB",
        "os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
        "python": platform.python_version(),
        **versions,
    }


def write_summary(state):
    lines = ["# GPU session", ""]
    lines += [f"- {key}: {value}" for key, value in state.get("machine", {}).items()]
    lines += ["", "| Step | Status | Minutes | Finished | Note | Log |", "|---|---|---|---|---|---|"]
    for name in STEPS:
        record = state["steps"].get(name)
        if record:
            lines.append(f"| {name} | {record['status']} | {record['seconds'] / 60:.1f} | "
                         f"{record['finished']} | {record['note'].replace('|', '/')[:200]} | "
                         f"`logs/{name}.log` |")
    suite_summary = os.path.join(SUITE_OUT, "summary.md")
    if os.path.exists(suite_summary):
        with open(suite_summary, encoding="utf-8") as handle:
            table = [line for line in handle if line.startswith(("| File", "|---", "| **total"))]
        lines += ["", "JAX's tests so far:", "", *(line.rstrip("\n") for line in table[:3])]
    lines += ["", "Details: [jax-suite/summary.md](jax-suite/summary.md) (each failure labeled "
              "METAL only or also on CPU); op probe: `re/findings/jax-op-probe.md`; extended "
              "accuracy: `re/findings/accuracy-extended.md`."]
    with open(os.path.join(SESSION, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def names(text):
    chosen = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in STEPS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step {', '.join(unknown)} "
                                         f"(steps: {', '.join(STEPS)})")
    return chosen


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--budget-minutes", type=float, default=120, metavar="M",
                        help="time for JAX's tests in this run (default 120)")
    parser.add_argument("--steps", type=names, metavar="A,B", help="run only these steps")
    parser.add_argument("--skip", type=names, default=[], metavar="A,B", help="leave these out")
    parser.add_argument("--redo", type=names, default=[], metavar="A,B",
                        help="run these again even if they have finished")
    parser.add_argument("--retry-failures", action="store_true",
                        help="run JAX's failed and crashed tests again (after a fix); implies "
                             "--redo jax-suite,jax-heavy")
    parser.add_argument("--bench", action="store_true", help="also run the benchmarks")
    parser.add_argument("--restart", action="store_true",
                        help="forget earlier progress: delete shim/build/gpu-session first")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would run, without running anything")
    args = parser.parse_args(argv)
    if args.retry_failures:
        args.redo = [*args.redo, *BUDGETED]
    return args


def plan(args, state):
    """[(name, will_run, why)] in order."""
    rows = []
    for name in STEPS:
        record = state["steps"].get(name)
        if args.steps is not None and name not in args.steps:
            rows.append((name, False, "not selected"))
        elif name in args.skip or (name == "bench" and not args.bench
                                   and not (args.steps and "bench" in args.steps)):
            rows.append((name, False, "skipped"))
        elif record and record["status"] != "partial" and name not in ALWAYS + tuple(args.redo):
            rows.append((name, False, f"done earlier: {record['status']}"))
        elif record and name not in ALWAYS:
            rows.append((name, True, "continue" if record["status"] == "partial" else "again"))
        else:
            rows.append((name, True, ""))
    return rows


def main(argv):
    args = parse_args(argv)
    if sys.platform != "darwin" and not args.dry_run:
        print("gpu_session: needs macOS on Apple silicon (try --dry-run)", file=sys.stderr)
        return 2
    if args.restart and not args.dry_run and os.path.isdir(SESSION):
        shutil.rmtree(SESSION)
    state = load_state()
    rows = plan(args, state)
    if args.dry_run:
        heavy = heavy_tests()
        print(f"JAX tests: {JAX_SRC}; CPU comparison: {CPU_DIR} "
              f"({'present' if os.path.isdir(CPU_DIR) else 'missing'}); {len(heavy)} heavy tests")
        for name, will_run, why in rows:
            print(f"  {'run ' if will_run else 'skip'} {name:18} {why}")
        return 0
    os.makedirs(LOGS, exist_ok=True)
    state["machine"] = machine()
    deadline = time.time() + 60 * args.budget_minutes
    code = 0
    try:
        for name, will_run, why in rows:
            if not will_run:
                continue
            print(f"{name}: running{' (' + why + ')' if why else ''}", flush=True)
            start = time.time()
            if name in BUDGETED:
                status, note = STEPS[name](args, deadline)
            else:
                status, note = STEPS[name](args)
            state["steps"][name] = {
                "status": status, "note": note, "seconds": round(time.time() - start),
                "finished": datetime.datetime.now().isoformat(timespec="minutes")}
            save_state(state)
            write_summary(state)
            print(f"{name}: {status}{' - ' + note if note else ''} "
                  f"({(time.time() - start) / 60:.1f} min)", flush=True)
            if status in ("failed", "killed"):
                for line in tail(log_path(name)):
                    print("    " + line, flush=True)
            if status == "partial":
                code = run_jax_tests.BUDGET_EXIT
                break
    except Abort as error:
        print(f"gpu_session: stopping: {error}", file=sys.stderr, flush=True)
        code = 1
    finally:
        if os.path.isdir(LOGS):
            save_state(state)
            write_summary(state)
    print(f"\nSummary: {os.path.relpath(os.path.join(SESSION, 'summary.md'), ROOT)}")
    if code == run_jax_tests.BUDGET_EXIT:
        print("Time budget used up: run `make -C shim gpu-session` again to continue.")
    print("Then commit the results:\n"
          "  git add -f shim/build/gpu-session && git add re/findings\n"
          "  git commit -m 'GPU session results'")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
