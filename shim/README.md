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
- complex64 works inside jitted functions (arithmetic, transcendental functions, FFT), but a
  jitted function cannot take or return complex arrays, so keep complex values internal.
- float64 is not supported (no float64 in MPS or Apple GPU hardware). Double-float emulation
  measures ~1e-14 relative error on the M3 (`experimental/double_float.py`); see DESIGN.md.

## Tests

```sh
make -C shim test                                    # C++ load tests (plugin alone, then via shim)
.venv-shim/bin/python shim/tests/test_jax_e2e.py     # JAX end to end
.venv-shim/bin/python shim/tests/test_phase3.py      # linear algebra, precision, bf16, rbg
.venv-shim/bin/python shim/tests/test_complex.py     # complex numbers and FFT
.venv-shim/bin/python shim/tests/accuracy_sweep.py --strict
.venv-shim/bin/python shim/tests/probe_jax_ops.py re/findings/jax-op-probe.md
```

`make test` needs test data from `tests/gen_artifact.py` in `shim/build/testdata`
(`.venv-latest/bin/python shim/tests/gen_artifact.py shim/build/testdata`).
