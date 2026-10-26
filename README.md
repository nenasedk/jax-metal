# jax-metal-shim: current JAX on Apple GPUs

Apple's [`jax-metal`](https://developer.apple.com/metal/jax/) plugin, which runs JAX on the
GPU of Apple-silicon Macs, stopped at JAX 0.4.34 (October 2024). This project runs current
JAX (0.12) on that GPU with Apple's plugin unmodified: a small PJRT shim between jaxlib and
the plugin, and JAX lowering rules for what the plugin cannot compile on its own.

```python
import jax, jax.numpy as jnp
print(jax.devices())                                     # [METAL(id=0)]
print(jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((4, 4))))
```

**Status: alpha.** Tested on one machine (M3, macOS 26.7.1) with jax 0.12.0.1.

- Works: jit, grad, control flow, convolutions, sorting, scatter/gather, random numbers,
  linear algebra (Cholesky, solve, inverse, QR, eigh, SVD), FFT, bfloat16/float16, complex64
  (inside programs and as inputs/outputs), and training loops.
- Accuracy: every float32 elementwise op is within 18 ulps of XLA on the CPU; several MPS
  accuracy bugs are corrected by the lowering rules.
- Not supported: float64 and complex128 (no float64 in MPS or Apple GPUs), multiple devices
  or processes, and JAX operations newer than StableHLO 0.19.4 that have no rule yet.
- Not yet measured: speed, against the CPU and against the old jax 0.4.34 + jax-metal setup.

See [PLAN.md](PLAN.md) for the full list of limitations and the roadmap.

## Install

Requires macOS 14.4 or later on Apple silicon, Python 3.12+, and an environment without
Apple's `jax-metal` package.

```sh
pip install jax-metal-shim                 # once released; until then: shim/install.sh
python -m jax_plugins.metal_shim fetch     # once: Apple's plugin library, from PyPI
```

The shim does not redistribute Apple's library. `fetch` downloads Apple's `jax-metal` 0.1.1
wheel from PyPI, verifies it, and keeps the library and Apple's license in a per-user cache.
Details, environment variables and tests: [shim/README.md](shim/README.md).

## How it works

- jaxlib 0.12 speaks PJRT C API 0.116 and serializes programs as recent StableHLO. Apple's
  plugin implements PJRT 0.47 and reads StableHLO up to 0.19.4. The shim presents the plugin's
  own function table to jaxlib, padded and with a few entries answered by the shim, and asks
  jaxlib to serialize programs for StableHLO 0.19.4.
- The Python package registers the `METAL` platform and replaces JAX's lowering of
  operations the plugin rejects, or computes wrongly, with equivalent programs built from
  operations it handles: linear algebra, FFT, complex numbers, accurate elementwise functions.

[shim/DESIGN.md](shim/DESIGN.md) has the details and the evidence behind each decision.

## Repository

| Path | Contents |
|---|---|
| `shim/` | The `jax-metal-shim` package: C++ shim (`src/`), Python package (`python/`), tests, benchmarks |
| `re/` | Reverse-engineering notes, scripts and evidence about Apple's plugin |
| `PLAN.md` | How the project got here, limitations, next steps |
| `CHANGELOG.md`, `RELEASING.md` | Release notes and the release checklist |

## License

The project is licensed under the Apache License 2.0 (`LICENSE`), as are the vendored PJRT C
API headers in `shim/third_party/` (OpenXLA). Apple's plugin library is Apple's, under the terms in its wheel, and is not part
of this repository or its packages.

This project is not affiliated with or endorsed by Apple.
