# Changelog

## 0.1.0 (unreleased)

First release: current JAX on Apple GPUs through Apple's unmodified `jax-metal` 0.1.1 plugin.

- PJRT shim (`pjrt_plugin_metal_shim.dylib`) between jaxlib 0.12 (PJRT C API 0.116) and the
  plugin (PJRT 0.47): padded function table, StableHLO 0.19.4 target, synthetic memory
  spaces, complex64 program inputs and outputs.
- METAL lowering rules for what the plugin rejects or computes wrongly: cumulative
  reductions, CHLO composites, linear algebra (Cholesky, triangular solve, QR, eigh, SVD),
  FFT, complex64 functions (including inverse trigonometric and hyperbolic functions),
  integer and bool matmul, `reduce_precision`, exact bfloat16 conversions, the `rbg` and
  philox PRNGs, multi-key sorts, and accuracy fixes for MPS's `expm1`, `log1p`, `tanh` and
  `sign`.
- NaN propagates through `max`, `min`, `clamp`, max/min reductions, `argmax`/`argmin` and
  max/min pooling, as with XLA (MPS drops it; `cond` of a singular matrix was 0, not inf).
- Programs that request a result accuracy or a dot algorithm, or use `lax.composite`, compile
  on METAL (the requests are dropped; composites run as their decomposition).
  `approx_max_k`/`approx_min_k` are computed exactly; complex `x.at[i].multiply`/`.power`
  work.
- `python -m jax_plugins.metal_shim fetch` downloads and verifies Apple's plugin library from
  PyPI into a per-user cache; `info` shows what is installed.
- Test tooling: a guard for GPU test runs (memory cap, GPU-stall detection, timeouts), a
  batch runner for JAX's own test suite on METAL, and a plugin that runs JAX's tests through
  the METAL rules on CPU. `make -C shim gpu-session` runs the whole GPU validation in one
  resumable command and labels each METAL failure by the same test's CPU result.
