# Changelog

For a full, maintained JAX backend on Apple GPUs, use
[jax-mps](https://github.com/tillahoffmann/jax-mps); see
[PLAN.md](PLAN.md#status-jax-mps-is-the-full-solution).

## 0.1.0 (unreleased)

First release: current JAX on Apple GPUs through Apple's unmodified `jax-metal` 0.1.1 plugin.

- PJRT shim (`pjrt_plugin_metal_shim.dylib`) between jaxlib 0.12 (PJRT C API 0.116) and the
  plugin (PJRT 0.47): padded function table, StableHLO 0.19.4 target, synthetic memory
  spaces, program inputs and outputs of types the plugin rejects (complex64, uint64, float8,
  int4).
- METAL lowering rules for what the plugin rejects or computes wrongly: cumulative
  reductions, CHLO composites, linear algebra (Cholesky, triangular solve, QR, eigh, SVD),
  FFT, complex64 functions (including inverse trigonometric and hyperbolic functions),
  integer and bool matmul, `reduce_precision`, exact bfloat16 conversions, the `rbg` and
  philox PRNGs, multi-key sorts, and accuracy fixes for MPS's `expm1`, `log1p`, `tanh` and
  `sign`.
- More of JAX's linear algebra: `eig`, `schur`, `hessenberg`, `geqp3` (QR with column
  pivoting), `tridiagonal` and a pivoting `tridiagonal_solve`, with LAPACK's conventions.
- Collectives on the plugin's one device (`psum`, `all_gather`, `ppermute`, `all_to_all`, ...),
  so `shard_map` and `pmap` programs run.
- uint64, float8 (e4m3fn, e5m2, e4m3b11fnuz, e4m3fnuz, e5m2fnuz) and int4/uint4 program inputs
  and outputs; float8, float4 and 2-/4-bit integer arithmetic, emulated with XLA's rounding.
  The newer narrow types (float8_e4m3, e3m4, e8m0fnu, float4_e2m1fn, int2, uint2) work
  inside jitted functions only.
- NaN propagates through `max`, `min`, `clamp`, max/min reductions, `argmax`/`argmin`,
  max/min pooling and `scatter_max`/`scatter_min`, as with XLA (MPS drops it; `cond` of a
  singular matrix was 0, not inf).
- Lowering rules work under `jax_numpy_dtype_promotion="strict"` and inside `shard_map`.
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
  `make -C shim unit-test` checks the shim against a fake plugin on any machine.
