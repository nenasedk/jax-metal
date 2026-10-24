"""Run GPU test commands without endangering the machine.

    .venv-shim/bin/python shim/tests/guard.py [--mem-gb N] [--timeout S] [--no-throttle] -- CMD ...

A runaway METAL test once grew to 20 GB on an 8 GB Mac, filled the memory compressor and
kernel-panicked the machine (watchdogd starved). Another kept the GPU busy for minutes, and
WindowServer, waiting on the same GPU, was killed by its watchdog. The guard prevents both:

- memory cap: the command's whole process group is killed when its physical footprint (which
  includes Metal allocations) exceeds the cap, or when the system is nearly out of memory while
  the group holds a sizeable share of it;
- GPU availability: gpu_probe (tests/gpu_probe.m, built on first use) times a trivial
  command buffer every half second; when it gets no answer for --gpu-stall seconds (default
  5; with ~6 s for the GPU to recover after the kill, well under WindowServer's 40 s
  watchdog) the job is killed;
- time limits: an overall timeout, and an idle timeout on a progress file for hung GPU calls
  that pytest-timeout cannot interrupt;
- lower priority (nice + utility QoS) so WindowServer and watchdogd keep running;
- one GPU job at a time, through a lock file shared by all guarded commands.

Exit status is the command's own, or 137 when the guard killed it (the reason goes to stderr).
"""

import argparse
import ctypes
import ctypes.util
import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

LOCK_PATH = os.path.join(tempfile.gettempdir(), "jax-metal-gpu.lock")
KILLED = 137
GPU_STALL_SECONDS = 5.0  # a killed job can hold the GPU ~6 s more; WindowServer gives up at 40 s
SHIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_SOURCE = os.path.join(SHIM_DIR, "tests", "gpu_probe.m")
PROBE_BINARY = os.path.join(SHIM_DIR, "build", "gpu_probe")

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


def _sysctl(name, ctype):
    value, size = ctype(), ctypes.c_size_t(ctypes.sizeof(ctype))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), f"sysctl {name}")
    return value


def physical_memory():
    return _sysctl("hw.memsize", ctypes.c_uint64).value


def free_memory_percent():
    """kern.memorystatus_level: the kernel's estimate of available memory, in percent."""
    return _sysctl("kern.memorystatus_level", ctypes.c_uint32).value


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
        "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
        "proc_exit_abstime")]


def footprint(pid):
    """Physical footprint of one process in bytes (0 if it is gone)."""
    info = _RusageInfoV0()
    if _libproc.proc_pid_rusage(pid, 0, ctypes.byref(info)):
        return 0
    return info.phys_footprint


def group_pids(pgid):
    buffer = (ctypes.c_int * 4096)()
    count = _libproc.proc_listpgrppids(pgid, buffer, ctypes.sizeof(buffer))  # returns a pid count
    return [pid for pid in buffer[:max(count, 0)] if pid > 0]


def group_footprint(pgid):
    return sum(footprint(pid) for pid in group_pids(pgid))


def default_mem_limit():
    """JAX_METAL_TEST_MEM_GB, else 30% of RAM capped at 6 GB (2.4 GB on an 8 GB Mac).

    Metal allocates faster than the guard polls, so a job can overshoot the cap by about a
    gigabyte before it is killed; the default leaves room for that.
    """
    if "JAX_METAL_TEST_MEM_GB" in os.environ:
        return int(float(os.environ["JAX_METAL_TEST_MEM_GB"]) * 2**30)
    return min(int(0.3 * physical_memory()), 6 * 2**30)


class GpuLock:
    """Exclusive lock so that guarded GPU jobs never run concurrently."""

    def __init__(self, wait=True):
        self.wait = wait
        self.handle = None

    def __enter__(self):
        self.handle = open(LOCK_PATH, "a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.seek(0)
            holder = self.handle.read().strip() or "another job"
            if not self.wait:
                raise SystemExit(f"guard: GPU busy ({holder}); not waiting")
            print(f"guard: GPU busy ({holder}); waiting", file=sys.stderr, flush=True)
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid {os.getpid()}: {' '.join(sys.argv)[:200]}\n")
        self.handle.flush()
        return self

    def __exit__(self, *exc):
        self.handle.truncate(0)
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


class GpuProbe:
    """Runs gpu_probe and tracks when the GPU last completed one of its command buffers."""

    def __init__(self):
        if not os.path.exists(PROBE_BINARY) or (
                os.path.getmtime(PROBE_BINARY) < os.path.getmtime(PROBE_SOURCE)):
            subprocess.run(["make", "-C", SHIM_DIR, "probe"], check=True, capture_output=True)
        self.proc = subprocess.Popen([PROBE_BINARY, "0.5"], stdout=subprocess.PIPE, text=True)
        self.last_answer = time.time()
        self.worst_ms = 0.0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.last_answer = time.time()
            self.worst_ms = max(self.worst_ms, float(line))

    def stalled_for(self):
        """Seconds since the GPU last answered (the probe waits at most 0.5 s between asks)."""
        return time.time() - self.last_answer

    def close(self):
        self.proc.kill()
        self.proc.wait()


def _lower_priority():
    os.setpriority(os.PRIO_PROCESS, 0, 10)


def run(command, *, cwd=None, env=None, stdout=None, stderr=None, mem_limit=None,
        timeout=None, idle_timeout=None, progress_path=None, throttle=True,
        min_free_percent=10, gpu_stall=GPU_STALL_SECONDS, poll=0.25):
    """Run `command` under the guard. Returns (returncode, kill_reason or None).

    stdout/stderr are file objects (or None to inherit). idle_timeout applies to the
    modification time of progress_path, which a test plugin updates as it goes.
    """
    mem_limit = mem_limit or default_mem_limit()
    probe = GpuProbe() if gpu_stall else None
    if throttle and os.path.exists("/usr/sbin/taskpolicy"):
        command = ["/usr/sbin/taskpolicy", "-c", "utility"] + list(command)
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                            start_new_session=True,
                            preexec_fn=_lower_priority if throttle else None)
    start = last_progress = time.time()
    reason = None
    peak = 0
    while proc.poll() is None:
        time.sleep(poll)
        now = time.time()
        used = group_footprint(proc.pid)
        peak = max(peak, used)
        if probe and probe.stalled_for() > gpu_stall:
            reason = f"GPU unresponsive for {probe.stalled_for():.0f}s (job monopolizing it)"
        elif used > mem_limit:
            reason = f"memory {used / 2**30:.1f} GB > limit {mem_limit / 2**30:.1f} GB"
        elif free_memory_percent() < min_free_percent and used > mem_limit // 4:
            reason = (f"system memory low ({free_memory_percent()}% free) while the job "
                      f"holds {used / 2**30:.1f} GB")
        elif timeout and now - start > timeout:
            reason = f"timeout after {timeout:.0f}s"
        elif idle_timeout and progress_path:
            try:
                last_progress = max(last_progress, os.path.getmtime(progress_path))
            except OSError:
                pass
            if now - last_progress > idle_timeout:
                reason = f"no progress for {idle_timeout:.0f}s (hung?)"
        if reason:
            kill_group(proc)
            break
    proc.wait()
    if probe:
        if reason and probe.stalled_for() > gpu_stall:  # report whether the GPU recovered
            deadline = time.time() + 30
            while probe.stalled_for() > 1 and time.time() < deadline:
                time.sleep(0.25)
            reason += ("; GPU recovered after the kill" if probe.stalled_for() <= 1
                       else "; GPU still unresponsive 30s after the kill")
        run.worst_gpu_ms = probe.worst_ms
        probe.close()
    run.peak_bytes = peak
    return (KILLED if reason else proc.returncode), reason


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mem-gb", type=float, help="memory cap (default: JAX_METAL_TEST_MEM_GB "
                        "or 30%% of RAM, at most 6 GB)")
    parser.add_argument("--timeout", type=float, default=3600, help="overall seconds (default 3600)")
    parser.add_argument("--idle-timeout", type=float, help="kill if --progress-file stops changing")
    parser.add_argument("--progress-file")
    parser.add_argument("--gpu-stall", type=float, default=GPU_STALL_SECONDS,
                        help="kill when the GPU does not answer for this many seconds (0: off)")
    parser.add_argument("--no-throttle", action="store_true",
                        help="keep normal priority (for benchmarks)")
    parser.add_argument("--no-wait", action="store_true", help="fail if another GPU job runs")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")
    with GpuLock(wait=not args.no_wait):
        code, reason = run(command, mem_limit=int(args.mem_gb * 2**30) if args.mem_gb else None,
                           timeout=args.timeout, idle_timeout=args.idle_timeout,
                           progress_path=args.progress_file, throttle=not args.no_throttle,
                           gpu_stall=args.gpu_stall)
    if reason:
        print(f"guard: killed {command[0]}: {reason}", file=sys.stderr, flush=True)
    print(f"guard: peak footprint {run.peak_bytes / 2**30:.2f} GB, worst GPU probe latency "
          f"{getattr(run, 'worst_gpu_ms', 0):.0f} ms", file=sys.stderr, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
