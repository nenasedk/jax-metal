# Findings: Phase 0 and first Phase 1 targets

Target: `re/targets/pjrt_plugin_metal_14-0.1.1.dylib`
(sha256 `e8e0656f24dc887507f5719926faf51f65546993809d3059add1a3c6c16961bf`).
REA evidence for the queries below: `re/evidence/phase0-phase1-bundle.json`.

## Phase 0: Ghidra on 8 GB

| Run | Heap | Analyzers | Result |
|---|---|---|---|
| REA default (first attempt) | 5 GB | Ghidra defaults | Stopped after 40+ min; swap 12.7/13.3 GB |
| Import only, `-noanalysis` | 3 GB | none | 107 s, 1.2 GB peak RSS |
| Lean profile, standalone headless | 3 GB | 13 disabled (below) | 2126 s (35.5 min), 1.8 GB peak RSS, swap flat |
| Lean profile through REA (`open_binary` + first query) | 3 GB | same | 37 min to first answer, then queries take seconds |

Disabled analyzers (`REA_GHIDRA_DISABLED_ANALYZERS`): Decompiler Switch Analysis, Stack,
Call Convention ID, Non-Returning Functions - Discovered, Function ID, Function Start Search,
Function Start Search After Code, Function Start Search After Data, Create Address Tables,
Embedded Media, Shared Return Calls, Apply Data Archives, Objective-C Message Analyzer.

Result of the lean analysis: 306,363 functions, 1,408,367 symbols. Time is dominated by
disassembly (Disassemble Entry Points ~21 min); Basic Constant Reference 6.4 min,
Mach-O Function Starts 2.4 min, Demangler GNU 2 min.

Settings that matter for REA:

- `REA_GHIDRA_STARTUP_TIMEOUT_MS` defaults to 330000 (5.5 min), which aborts before analysis of
  this binary finishes. Set to 3600000.
- REA imports with `-deleteProject`, so every new session re-runs the full analysis. The
  `close_binary` snapshot (74 KB) holds evidence records only, not the Ghidra database.
- `rea_session.py` (in `re/scripts/`) keeps one REA MCP session alive so the analysis runs once
  per working session. The Claude Code registration in `~/.claude.json` carries the same env.

## GetPjrtApi and the PJRT function table

REA decompilation (`re/findings/rea-GetPjrtApi.c`):

```c
undefined1 * _GetPjrtApi(void) { return &pjrt::metal_plugin::pjrt_api; }
```

`pjrt::metal_plugin::pjrt_api` is at `0x5acc6e0` (start of `__DATA_CONST,__const`):

- `struct_size` = 808, so 97 function pointers follow the 40-byte header.
- `pjrt_api_version` = **0.47**.
- All 97 entries match current upstream `PJRT_Api` field order exactly
  (`re/findings/pjrt_api_table.tsv`); upstream main has 138 entries (PJRT 0.117), so 41 are newer
  than the plugin.
- Apple-specific entries: index 10 `pjrt::metal_plugin::PJRT_Client_Create` and index 82
  `pjrt::metal_plugin::PJRT_MetalDeviceTopology_Create`. The other 95 are XLA's generic
  `pjrt::` wrapper implementations.
- `PJRT_Plugin_Attributes` (index 4, `0x25afaec`) returns zero attributes
  (`re/findings/rea-PJRT_Plugin_Attributes.c`). The plugin does not report
  `stablehlo_current_version`.

## StableHLO version

**The plugin's StableHLO/VHLO current version is 0.19.4; minimum is 0.9.0.**

Evidence:

- `mlir::vhlo::(anonymous namespace)::VhloToVersionPass::runOnOperation` (`0x3c7e848`) is the only
  reference to the string ` is greater than current version ` (`0x52f0c10`).
  Its checks (`re/findings/rea-VhloToVersionPass-version-checks.s`): reject minor < 9,
  minor > 0x13, or minor == 0x13 with patch >= 5. The current version printed in the error is
  16 bytes at `0x58f6130` (`major=0, minor=0x13`, read via REA `read_bytes`) plus patch `4`.
- Independent cross-check from the op inventory (`re/findings/vhlo-ops.txt`, 155 `vhlo.*_vN`
  op names): newest op present is `composite_v1` (introduced 0.19.0); `dynamic_conv_v2` (0.20.0)
  and every later op are absent. Attribute `quant_per_axis_v1` (0.18.0) is present; 1.2.0+ types
  are absent.

## Upstream pins (task 4)

No commit hashes are embedded, so the pins come from version history:

| Component | Pin | How |
|---|---|---|
| StableHLO | `271e8634` (0.19.4, 2024-03-27…29) | 0.19.4 existed only between `d96ce9be` and `d7205db5`; all 118 VHLO ops at `271e8634` are present in the plugin |
| XLA | `3c8260f6` (2024-03-29) … before `b0c6c262` (2024-04-05) | `3c8260f6` introduced PJRT 0.47 and pins StableHLO `271e8634`; `b0c6c262` moved to 0.19.5 |
| LLVM | ~`80aa52d8` ("19.0.0git") | LLVM pin of XLA `3c8260f6` |
| abseil | `lts_20230802` | inline namespace in mangled names |

So jax-metal 0.1.1 (released 2024-10-08) was built from late-March-2024 OpenXLA, which matches its
jaxlib ≥ 0.4.26 requirement. The PJRT header at `3c8260f6` is vendored in
`shim/third_party/xla_3c8260f6`.

## Upstream vs Apple code (task 4)

`re/scripts/tag_regions.py` tags every `__text` function by namespace, and tags file-local helpers
by their neighbours (the linker keeps each object file contiguous). Results:
`re/findings/region-summary.tsv`, `re/findings/region-ranges.tsv` (735 contiguous ranges, 367 of
them Apple), `re/findings/apple-functions.tsv` (21,584 functions).

| | Share of code (80 MB `__text`) |
|---|---|
| Upstream (LLVM 38%, MLIR 27%, XLA 12%, Eigen 5%, StableHLO 3%, mlir-hlo 3%, libc++, protobuf, abseil) | 94.5% |
| Apple-written | 3.0% |
| Upstream templates instantiated on Apple types (ODS/op traits for the `mps` dialect) | 1.7% |
| Unattributed | 0.8% |

Apple's code lives in: `mlir::mps` (MPS dialect + HLO→MPS lowering), `xla::mps`
(`MetalStreamExecutorClient`, `MetalExecutable`, `MetalXlaExecutable`, `MetalSimpleAllocator`,
`MetalTransferManager`, `StreamExecutorGpuDevice`), `metal_plugin` (StreamExecutor-style C device
API: allocate, memcpy, streams, events), `pjrt::metal_plugin` (`PJRT_Client_Create`,
`PJRT_MetalDeviceTopology_Create`, `pjrt_api`), `MetalStream`, `SP_Event_st`,
`PjrtMetalstreamWrapper` (Objective-C), `compileMlirHLOToMPS`, `runMPSGraphExecutable`.

## Op coverage (task 3)

`re/findings/mps-op-coverage.md` (generated by `re/scripts/op_coverage.py`):

- 52 TableGen patterns `mlir::mps::GeneratedConvertN`; each pattern's constructor (inlined in
  `RewritePattern::create<GeneratedConvertN>`) names its source op and target MPS op, e.g.
  `mhlo.add → mps.add`, `chlo.erf → mps.erf`.
- Hand-written converters `mlir::mps::(anonymous namespace)::*Converter` for 34 more ops (dot,
  dot_general, convolution, gather, scatter, sort, reduce, reduce_window, select_and_scatter,
  while/if/case, rng_bit_generator, top_k, …).
- 82 MHLO/CHLO ops have a pattern; 46 StableHLO 0.19.4 ops do not (`cholesky`,
  `triangular_solve`, `fft`, `reduce_precision`, `popcnt`, `count_leading_zeros`, `cbrt`, complex
  ops, collectives, dynamic-shape ops, quantization, …).
- Empirically (`re/findings/jax-op-probe.md`, through the shim with jax 0.12): 79 of 88 JAX ops
  pass after the METAL lowering rules in `shim/python/jax_plugins/metal_shim/_lowerings.py`.
- Compilation pipeline (REA decompilation of `compileMlirHLOToMPS`, `populatePreLegalization`,
  `populatePassManager`): StableHLO→MHLO, inliner, Apple's `DecomposeChloToHLOPass`, MHLO
  canonicalizations, Apple's `LegalizeControlFlowPass`, `ConvertHLOToMPSPass`,
  `RemoveZeroDimPass`. Nothing expands `composite` ops, which is why CHLO ops that current XLA
  serializes as composites fail.

## Direct-load test

`shim/tests/direct_load.cc` loads the plugin with no shim, compiles a jax 0.12 program serialized
as a StableHLO 0.19.4 portable artifact and runs it on the GPU: max abs error vs CPU 4.8e-7. The
same program serialized at StableHLO 1.16.3 fails with `unknown vhlo attribute code: 20`.

## Shim

Design and additional considerations: `shim/DESIGN.md`. Struct compatibility between the plugin's
PJRT 0.47 and jaxlib 0.12's PJRT 0.116: `re/findings/pjrt-0.47-vs-0.116.md` (all 97 shared entry
points take unchanged or append-only argument structs).
