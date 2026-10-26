# jax-metal-shim

Runs current JAX (tested with jax/jaxlib 0.12.0.1) on Apple GPUs using Apple's unmodified
`jax-metal` 0.1.1 plugin, which is otherwise stuck on JAX 0.4.34. How it works and why:
[DESIGN.md](DESIGN.md).

## Install

Requires macOS on Apple silicon, Xcode command-line tools and a Python with jax/jaxlib 0.12.
Apple's `jax-metal` package must not be installed in the same environment (the shim carries
Apple's plugin library itself).

```sh
python3 -m venv .venv-shim
.venv-shim/bin/pip install jax==0.12.0.1 jaxlib==0.12.0.1
PYTHON=.venv-shim/bin/python shim/install.sh
```

`install.sh` builds `pjrt_plugin_metal_shim.dylib`, downloads the `jax-metal==0.1.1` wheel from
PyPI, checks the plugin library's sha256 and installs the `jax-metal-shim` package.

## Use

```python
import jax, jax.numpy as jnp
print(jax.devices())                     # [METAL(id=0)]
print(jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((4, 4))))
```

Environment variables:

| Variable | Effect |
|---|---|
| `JAX_METAL_SHIM_TRACE=1` | Log shim initialization and attribute queries to stderr |
| `JAX_METAL_SHIM_PLUGIN_PATH` | Load Apple's plugin from this path instead of the package directory |
| `JAX_USE_SHARDY_PARTITIONER` | If set, the shim leaves JAX's Shardy setting alone (the plugin needs it off) |

## Status

- `tests/test_jax_e2e.py`: 16 of 16 cases match the CPU, including grad, conv, sort,
  scatter, control flow and an MLP training step.
- `tests/probe_jax_ops.py`: 88 of 88 JAX ops pass, including linear algebra
  (`cholesky`, `solve`, `inv`), FFT, `reduce_precision`, bfloat16 and the `rbg` PRNG.
- `tests/accuracy_sweep.py`: every float32 elementwise op is within 18 ulps of the CPU (most
  within 1–4), with matching NaN/inf behavior.
- complex64 works throughout: inside jitted functions (arithmetic, transcendental functions,
  FFT) and as inputs/outputs of jitted functions and eager ops. A function returning complex
  values waits for completion before returning.
- float64 is not supported (no float64 in MPS or Apple GPU hardware). Double-float emulation
  measures ~1e-14 relative error on the M3 (`experimental/double_float.py`); see DESIGN.md.

## Tests

Run anything that uses the GPU through `tests/guard.py`. A runaway test once grew to 20 GB
on an 8 GB Mac and kernel-panicked it. The guard:

- kills the command's process group above a memory cap (30% of RAM by default; set
  `--mem-gb` or `JAX_METAL_TEST_MEM_GB`), Metal allocations included;
- kills the job when a trivial Metal command buffer gets no answer for 5 s (`tests/gpu_probe.m`);
- applies a timeout and lowers the command's priority;
- runs one GPU job at a time.

```sh
G=".venv-shim/bin/python shim/tests/guard.py --"
make -C shim test                                    # C++ load tests (plugin alone, then via shim)
$G .venv-shim/bin/python shim/tests/test_jax_e2e.py  # JAX end to end
$G .venv-shim/bin/python shim/tests/test_phase3.py   # linear algebra, precision, bf16, rbg
$G .venv-shim/bin/python shim/tests/test_complex.py  # complex numbers and FFT
$G .venv-shim/bin/python shim/tests/accuracy_sweep.py --strict
$G .venv-shim/bin/python shim/tests/probe_jax_ops.py re/findings/jax-op-probe.md
```

Killing a job does not free the GPU or the Metal compiler service immediately. A job that
monopolizes them can still stall WindowServer, whose watchdog then restarts it and logs the
user out. Prevention lives in the lowering rules: for example, Jacobi eigendecomposition
refuses sizes above 256. Try new algorithms on the CPU backend before running them on METAL.

Benchmarks use `guard.py --no-throttle --` so that timings run at normal priority.

`tests/jax_suite/run_jax_tests.py JAX_SRC OUT_DIR TEST...` runs JAX's own test files on METAL.
It applies the guard to each pytest process and also sets:

- a 120 s per-test timeout;
- an idle timeout for hangs;
- a soft memory limit that restarts pytest between tests.

A test that crashes, hangs or exceeds the cap is recorded as `crashed`, and the run continues.
Ctrl-C stops the run and kills its pytest process; finished results are kept.

Setup: JAX's tests at the installed version, plus pytest.

```sh
git clone --depth 1 --branch jax-v0.12.0.1 --filter=blob:none --sparse \
  https://github.com/jax-ml/jax ../jax-src && git -C ../jax-src sparse-checkout set tests
.venv-shim/bin/pip install pytest pytest-timeout absl-py
```

The files hold about 17,500 tests at the default `JAX_NUM_GENERATED_CASES=10`, so run them in
batches:

| Option | Effect |
|---|---|
| `-k EXPR` | pytest `-k` expression, applied to every TEST |
| `--shard I/N` | only the I-th of N contiguous slices (1-based) of what `-k` leaves; slices are stable between runs |
| `--resume` | skip tests that already have a result in OUT_DIR (crashed ones too) |
| `--budget-minutes M` | stop between tests after M minutes; continue later with `--resume` |

A TEST is a file under `tests/` or a node id prefix (`linalg_test.py::NumpyLinalgTest`).
Results go to `OUT_DIR/<file>.jsonl`, one record per test, merged across runs: re-running a
test replaces its record. `OUT_DIR/summary.md` covers every result in OUT_DIR and groups
failures by message; crash logs are in `OUT_DIR/<file>.jsonl.crash-N.txt`. Keep OUT_DIR under
`shim/build/` (git-ignored), for example:

```sh
R=".venv-shim/bin/python shim/tests/jax_suite/run_jax_tests.py ../jax-src shim/build/jax-suite"
$R fft_test.py random_test.py                     # 365 tests
$R linalg_test.py --shard 1/4                     # 250 of 1,000
$R lax_test.py --shard 3/10 --budget-minutes 30   # then the same command with --resume
```

`make test` needs test data from `tests/gen_artifact.py` in `shim/build/testdata`
(`.venv-latest/bin/python shim/tests/gen_artifact.py shim/build/testdata`).
