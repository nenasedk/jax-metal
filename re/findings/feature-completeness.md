# Feature completeness of current JAX on METAL

What it would take for JAX 0.12 on an Apple GPU to offer what JAX offers on a single CUDA
GPU, the most complete GPU backend: what Apple's jax-metal lacked, what the shim adds, what
can still be added, and what the hardware rules out. Analysis of 2026-10-26; "works in
emulation" means the METAL rules on the CPU backend (`shim/tests/jax_suite/cpu_rules_plugin.py`)
with `metal_check.py` predicting no plugin issue, not yet confirmed on a GPU.

## Summary

- Most of JAX's single-device feature set is reachable with Apple's plugin: what is missing
  either decomposes into ops the plugin lowers (lowering rules) or can be fixed at the PJRT
  boundary (the shim).
- The hardware rules out native float64/complex128 arithmetic and more than one GPU, and its
  watchdog limits how long one program may hold the GPU.
- A second group is blocked by Apple's closed plugin, not the hardware: custom kernels
  (Pallas, FFI), host callbacks inside programs, a persistent compilation cache, native
  bfloat16 storage. These need an open replacement backend.

## What limits what

| Layer | Changeable | Decides |
|---|---|---|
| Apple GPU | no | no float64 units (Metal has no `double`), one GPU per Mac, GPU watchdog |
| Apple's plugin (jax-metal 0.1.1, frozen 2024-10) | no, closed | PJRT C API 0.47; StableHLO ≤ 0.19.4; 82 HLO→MPS patterns (`mps-op-coverage.md`); `custom_call` = LU only; no send/recv; no executable serialization; MPS semantics |
| The shim (`shim/src`) and its lowering rules (`shim/python`) | ours | anything expressible with the plugin's ops; relabeling buffers at program boundaries; intercepting PJRT calls |
| An open backend (PLAN.md Phase 4) | not built | lifts every plugin limit above |

## What Apple's jax-metal 0.1.1 lacked

- By Apple's own list: float64, complex64 and complex128, and "doesn't pass all tests"
  ([developer.apple.com/metal/jax](https://developer.apple.com/metal/jax/)). Pinned to
  jax 0.4.34.
- Missing from its patterns whatever the JAX version (`mps-op-coverage.md`, first probe in
  `jax-op-probe.md`): Cholesky, triangular solve (so `solve` and `inv`), FFT,
  real/imag/complex, `reduce_precision`, `cbrt`, `popcnt`, `clz`. QR, eigh and SVD lower to
  custom calls the plugin aborts on (it knows only LU).
- Crashes: `rng_bit_generator` (the `rbg` PRNG) aborts the process.
- Silent wrong results: naive `expm1`/`log1p`, `tanh(NaN) = 1`, `sign(NaN)`, unrounded
  bfloat16 conversions, max/min/clamp/reductions that drop NaN (`mps-nan-semantics.md`).

The shim fixes all of these, adds complex64 throughout (inside programs and at their
boundary), and runs current JAX (0.12). See `shim/DESIGN.md` for each rule.

## Remaining gaps and what each needs

| Feature | Status | Possible | How (rough effort) |
|---|---|---|---|
| `eig`, `eigvals`, `jnp.roots` | no lowering (JAX has LAPACK/cuSOLVER only) | yes, rules | Hessenberg reduction and shifted QR iteration in loops, complex results (1–2 weeks) |
| `schur`, `hessenberg`, `sqrtm`, `funm` | no lowering | yes, rules | same machinery as `eig` |
| QR with pivoting (`geqp3`), `tridiagonal` | no lowering | yes, rules | Householder loops like the existing QR (days) |
| `ormqr`, `tridiagonal_solve`, `cholesky_update`, `symmetric_product` | JAX's generic lowerings | works in emulation | — |
| `all_gather`, `ppermute`, `all_to_all`, `reduce_scatter`, `axis_index` on one device | no plugin pattern (`all_reduce` has one) | yes, rules | identities and reshapes on a mesh of size 1, so `shard_map`/`pmap` code runs (days) |
| uint64 program inputs/outputs | the plugin rejects them | yes, shim | relabel as int64, as complex64 is relabeled as float32 pairs (days) |
| int4/uint4, float8 variants | rejected; some cannot even be serialized for StableHLO 0.19.4 | emulation | 8-bit storage at the boundary, exact rounding inside programs, arithmetic in float16/float32; correct, not faster (1–2 weeks) |
| `jax.debug.print`, `pure_callback`, `io_callback` | JAX refuses them on METAL ("EmitPythonCallback not supported") | partly | split programs at callback sites, outside loops only; inside `scan`/`while` impossible with this plugin (2–4 weeks) |
| memory kinds, host offloading | not reported | yes, cosmetic | unified memory makes it moot: report a `pinned_host` space aliasing device memory, drop placement annotations (about a week, low value) |
| profiler traces | host side only | partly | a shim profiler extension with per-execution timings; per-kernel detail only in Instruments (about a week) |
| persistent compilation cache | the plugin cannot serialize executables | no, with this plugin | open backend |
| Pallas, FFI device kernels, Triton/Mosaic | `custom_call` = LU only | no, with this plugin | open backend that launches Metal kernels (months) |
| cuDNN attention, `scaled_matmul`, cuDNN fusions | CUDA-only features | not applicable | XLA attention works in emulation |
| ops in future StableHLO versions | each JAX release can add some | yes, ongoing | rules per release; the shim pins tested JAX versions |
| multiple processes, several Macs | the plugin is single-process | no, with this plugin | open backend with networking (MLX does this) |

Works in emulation (spot checks): `expm`, `eigh_tridiagonal`, `lu_factor`, `ragged_dot`,
`ormqr`, `tridiagonal_solve`, XLA `dot_product_attention`, `checkpoint`/`custom_vjp`,
sparse BCOO, `cumlogsumexp`, `psum` in `shard_map`/`pmap` on one device.

## Not possible on the hardware

- **float64 and complex128 arithmetic.** No float64 units; MPS has no float64 type, not even
  for constants. Emulation is possible as an explicit transform rather than transparent
  `jax_enable_x64` (`shim/DESIGN.md`, "float64"): double-float (`df64.jit`), about 48
  significand bits with float32's exponent range at 10–20 float32 operations per operation
  (measured about 1e-14 relative error on the M3), or exact IEEE binary64 from integer
  operations, an estimated 32–64 times slower than float32.
- **More than one GPU.** Apple-silicon Macs have one GPU (Ultra chips present one). Sharded
  code can run on a mesh of size 1, without speedup.
- **Long-running GPU work.** WindowServer's watchdog limits how long one program may hold the
  GPU; a 1008 × 1008 Jacobi eigendecomposition got WindowServer killed. Iterative algorithms
  must stay bounded (the Jacobi rule refuses n > 256) or be split.
- **8- and 4-bit floating-point arithmetic.** The plugin cannot express these types, so only
  emulation is possible; whatever newer Apple GPUs add, only an open backend could use it.

## Possible on the hardware, blocked by Apple's plugin

Custom kernels (Pallas, FFI), host callbacks inside loops, a persistent compilation cache,
native bfloat16 storage and speed (the plugin keeps bfloat16 in float32), packed int4 weights,
PJRT asynchronous transfers, StableHLO newer than 0.19.4 without rewriting. All of these need
an open replacement backend: StableHLO lowered to the public MPSGraph API (which has the
NaN-propagating ops, bfloat16 and 4-bit integer types), plus Metal kernels for custom calls.
Reaching today's coverage that way is plausibly 3–6 months; Pallas comes on top.

## Validation gap

JAX has 163 test files under `tests/`. 19 run in emulation as the GPU session's baseline
(`shim/build/jax-suite-cpu`, `jax-suite-cpu.md`); on METAL, only `fft_test` and part of
`linalg_test` have run. The next files are in "Emulation of further test files" below.

## Suggested order

1. Run more of JAX's test files in emulation (no GPU needed) to find gaps.
2. Run the GPU session (`make -C shim gpu-session`).
3. Single-device collectives and uint64 program inputs/outputs.
4. `geqp3`, `tridiagonal`, `hessenberg`, then `eig` and `schur`.
5. Narrow-dtype emulation (int4/uint4, float8).
6. Callbacks by program splitting.
7. `df64.jit` if users need float64.
8. Decide whether an open backend is worth building.
