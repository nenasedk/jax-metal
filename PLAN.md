# Reverse engineering jax-metal: plan of attack

Goal: get current JAX running on the Apple GPU of a modern Mac by reverse engineering
Apple's abandoned `jax-metal` PJRT plugin with REA (Reverse Engineer Anything).

## Environment

| Item | Value |
|---|---|
| Machine | Apple M3 (10-core GPU, Metal 4), 8 GB RAM |
| OS | macOS 26.7.1 (25G241) |
| REA | `rea-agents@6.7.0`, registered as MCP server `rea` in `~/.claude.json` |
| Analysis engine | Ghidra 12.1.4 at `~/tools/ghidra_12.1.4_PUBLIC` (arm64 natives built locally with `support/gradle/gradlew buildNatives`), OpenJDK 21 from Homebrew |
| Plugin under study | `jax-metal` 0.1.1 (2024-10-08, last release), `re/targets/pjrt_plugin_metal_14-0.1.1.dylib` (checksum in `re/targets/SHA256SUMS`; the binary itself is gitignored) |

## Baseline findings

Two virtualenvs live in the repo (gitignored):

| venv | Versions | Result |
|---|---|---|
| `.venv-baseline` | jax-metal 0.1.1, jax/jaxlib 0.4.34 | Works: `METAL(id=0)` device, arange, matmul, `jit`, `grad` |
| `.venv-latest` | jax-metal 0.1.1, jax/jaxlib 0.12.0.1 | Fails |

Failure modes with current JAX:

- Default: `Mismatched PJRT plugin PJRT API version (0.47) and framework PJRT API version 0.116`.
- `ENABLE_PJRT_COMPATIBILITY=1`: the device initializes, but every compile fails with
  `unknown attribute code: 22 ... in bytecode version 6 produced by: StableHLO_v1.16.3`.
  The plugin's embedded StableHLO deserializer is too old for the bytecode current jaxlib emits.

Conclusion: the plugin is not broken by macOS or the GPU. It is frozen at the October 2024
JAX interface.

Binary facts (`pjrt_plugin_metal_14.dylib`):

- 159 MB, thin arm64, not stripped (~470k local symbols, 55 exports, entry point `GetPjrtApi`).
- Mostly statically linked upstream code: LLVM 19.0.0git, MLIR, StableHLO/VHLO, MHLO, XLA, abseil `lts_20230802`, protobuf, Eigen.
- Apple-specific code: MLIR `mps` dialect (~11k symbols), `compileMlirHLOToMPS`, `registerHLOToMPSPass`,
  `registerMPSDialects`, `mps_client.cc`, `MetalStream`, `PjrtMetalstreamWrapper`. Executes through MPSGraph.

Current jaxlib picks the StableHLO version to serialize for a plugin from the plugin's
`stablehlo_current_version` attribute (see `jax/_src/xla_bridge.py`, `backend_stablehlo_version`).
jax-metal 0.1.1 predates that attribute, so jaxlib falls back to a version the plugin cannot read.

## Phases

### Phase 0: make Ghidra workable on 8 GB (½–1 day) — done, see `re/FINDINGS.md`

Lean analyzer profile, 3 GB heap, 60-minute REA startup timeout: 37 minutes to first answer,
1.8 GB peak RSS. `GetPjrtApi` decompiled; StableHLO version is 0.19.4.

A default full auto-analysis of the dylib through REA with a 5 GB heap ran for 40+ minutes and
pushed swap to 12.7/13.3 GB before it was stopped. Options, in order:

1. Lower the heap (`GHIDRA_HEADLESS_MAXMEM`) and disable the heaviest analyzers via
   `REA_GHIDRA_DISABLED_ANALYZERS`.
2. Keep one REA MCP session open so the binary is imported once, not per query.
3. If it still thrashes, analyze on a machine with 32 GB or more, or rely on targeted REA queries
   plus `nm`/`otool`/`lldb` for everything else.

### Phase 1: map the boundaries (2–4 days, REA-heavy) — done, see `re/FINDINGS.md`

PJRT 0.47 table recovered; StableHLO 0.19.4; 82 HLO ops with MPS patterns (79 of 88 probed JAX
ops work through the shim); 94.5% upstream (XLA `3c8260f6`, StableHLO `271e8634`), 4.7% Apple.


- Decompile `GetPjrtApi` and recover the PJRT 0.47 function table.
- Find the exact StableHLO/VHLO version the plugin can read.
- Recover the set of HLO ops the HLO→MPS lowering legalizes (the real op-coverage list).
- Tag regions of the binary as upstream (pinned to a commit) or Apple-proprietary, so effort goes
  to the Apple part.

### Phase 2: PJRT shim plugin (1–3 weeks, most likely path to a working result) — done, see `shim/`

Working shim with current JAX 0.12 on the M3 (16 of 16 end-to-end cases). The final design differs
from the sketch below: the shim keeps reporting PJRT 0.47 (jaxlib 0.12 accepts it in its default
compatibility mode), pads the table with nulls instead of `UNIMPLEMENTED` stubs, and adds synthetic
memory spaces. See `shim/DESIGN.md`.


A small C/C++ `pjrt_plugin_metal_shim.dylib` that:

- exposes PJRT C API 0.116 to current jaxlib and forwards to the 0.47 function table, filling new
  entries with `UNIMPLEMENTED` stubs;
- reports `stablehlo_current_version` set to the plugin's version, so jaxlib downgrades its output.

Main risk: StableHLO's forward-compatibility window is about two years, and the plugin is at that
edge. If current StableHLO cannot target the plugin's version, the shim must also rewrite the
module itself (+1–2 weeks).

### Phase 3: fill op gaps (ongoing) — op gaps closed

Run the JAX test suite against the shim, and use REA to explain each lowering failure. Fix gaps by
rewriting ops in the shim into forms the plugin already legalizes, without patching Apple's binary.
(In practice the rewrites live in the shim package's METAL lowering rules,
`shim/python/jax_plugins/metal_shim/_lowerings.py` and `_complex.py`; the C++ shim does not
parse programs.)

All 88 probed JAX ops pass. METAL lowering rules now cover cumulative reductions,
composite-wrapped CHLO ops, `cbrt`/`popcnt`/`clz`, `cholesky`/`triangular_solve` (so
`solve`/`inv`), `reduce_precision`, exact bfloat16 conversions, the `rbg` PRNG (MPS aborts on
it), MPS accuracy bugs (`expm1`, `log1p`, `tanh`, `sign(NaN)`), multi-key sorts, and complex64
inside programs including FFT, and complex64 values at program boundaries (dual pair/complex
programs with zero-copy buffer views). Remaining: float64 (no MPS support; double-float
emulation measured, see `shim/DESIGN.md`).

### Phase 4: open-source reimplementation (only if Phases 2–3 stall)

Use the reverse-engineered behavior as a spec for a new MPSGraph-based plugin. Multi-month effort
with ongoing maintenance.

## Limitations (as of 2026-10-24)

Correctness and coverage:

- **float64 / complex128**: unavailable; MPS has no float64 type. Double-float emulation is
  measured (~1e-14 relative error, float32 exponent range) but not wired into JAX.
- **Test coverage is our own**: 16 end-to-end cases, an 88-op probe, an elementwise float32
  accuracy sweep, linear algebra, complex/FFT and precision suites. JAX's own test suite and real
  models (Flax/Optax training, transformers) have not been run on METAL.
- **Accuracy beyond elementwise ops is unmeasured**: the sweep found silent MPS errors in
  `expm1`/`log1p`/`tanh`/`sign`; reductions, matmul precision, convolutions and float16/bfloat16
  arithmetic have not been swept the same way.
- **Newer-than-0.19.4 features**: JAX programs using ops newer than StableHLO 0.19.4, or CHLO ops
  XLA serializes as composites without a METAL rule (`mulhi`, `ragged_dot`, CHLO `scan`), fail.
- **bfloat16 is emulated in float32 by the plugin**: results are now exact, but there is no
  memory or speed benefit from bfloat16.

Performance:

- **Not benchmarked**, neither against the CPU nor against the old jax 0.4.34 + jax-metal setup.
- **Some rules are sequential loops**: `cholesky` and `triangular_solve` do n matrix-vector steps;
  FFT is radix-2 with a direct DFT for odd factors (O(n²) for large odd/prime lengths).
- **Programs returning complex values are synchronous** (the shim waits for the output before
  relabeling it), and are lowered/serialized twice at compile time.
- **No persistent compilation cache**: JAX's cache stays empty for METAL programs (the plugin
  does not serialize executables), so every new process recompiles.

Platform and maintenance:

- **Single device, single process**; no collectives or multi-process.
- **Pinned to jax/jaxlib 0.12**: the package hooks private JAX internals
  (`mlir.lower_jaxpr_to_module`, `compiler.compile_or_get_cached`, lowering-rule registries,
  `threefry_2x32`); `pyproject.toml` pins `<0.13` until a newer release is tested.
- **Depends on Apple's closed binary** from the `jax-metal==0.1.1` wheel on PyPI (fetched at
  install, checksum-verified). Tested on one machine (M3, macOS 26.7.1).
- **Shardy is turned off process-wide**, which affects multi-device sharding on other backends in
  the same process.
- **No CI**; the install builds locally and needs Xcode command-line tools.
- **REA analysis of the plugin costs ~37 minutes per session** on this 8 GB machine (REA
  re-imports the binary every session; its snapshot does not keep the Ghidra database).

## Recommended next steps

1. **Run JAX's own tests on METAL** (`lax_test`, `lax_numpy_test`, `linalg_test`, `fft_test`,
   `random_test`, `lax_autodiff_test`) to find remaining gaps systematically; triage failures
   into METAL lowering rules, accuracy fixes or documented limits.
2. **Real workloads and benchmarks**: train small Flax/Optax models (MLP, CNN, transformer),
   compare speed and loss curves with the CPU and with the jax 0.4.34 + jax-metal baseline.
3. **Extend the accuracy sweep** to reductions, matmul/conv precision and float16/bfloat16
   arithmetic, and add strict-mode checks to the test suite.
4. **Performance work where benchmarks point**: blocked Cholesky/triangular solve built on
   matmuls, a larger-radix FFT with Bluestein for prime lengths, asynchronous complex outputs.
5. **CI on a self-hosted Apple-silicon runner** running the C++ tests, the probe, the accuracy
   sweep and the JAX test subset, to catch JAX/macOS drift before users do.
6. **float64 via an explicit `df64.jit` transform** if users need it (estimated 2–4 weeks).
7. **Packaging**: a macOS arm64 wheel for the shim that fetches and verifies Apple's library on
   first use; a version check that warns on untested JAX releases.
8. **Report the MPS bugs** found here to Apple (Feedback Assistant): naive `expm1`/`log1p`,
   `tanh(NaN) = 1`, `sign(NaN)`, the `rng_bit_generator` abort, the missing complex `pad`
   kernel (`padOp_cf`), and the bfloat16 cast behavior.

## Complexity estimate

| Goal | Difficulty | Effort |
|---|---|---|
| JAX 0.4.34 on the GPU as-is | Trivial | Done (`.venv-baseline`) |
| Current JAX, core ops working, via the shim | Moderate–high | ~2–5 weeks |
| Broad JAX test-suite coverage on current JAX | High | +1–3 months, open-ended |
| Full open-source plugin rewrite | Very high | Months |

Overall about 7/10. Unstripped symbols and the large share of known upstream code make it much
easier. The StableHLO compatibility window and analyzing a 159 MB binary on 8 GB of RAM are what
make it hard.
