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

## What this means for the shim (Phase 2)

- jaxlib picks the serialization target from the plugin's `stablehlo_current_version` attribute
  and falls back to `xla::GetDefaultStablehloVersion()` when it is missing
  (`xla/pjrt/c_api_client/pjrt_c_api_client.cc`). That fallback is why current JAX sends
  StableHLO the plugin cannot read.
- A shim that wraps `PJRT_Plugin_Attributes` to return `stablehlo_current_version = [0, 19, 4]`
  and `stablehlo_minimum_version = [0, 9, 0]` should make jaxlib downgrade its output.
- Upstream StableHLO still has `getMinimumVersion() == 0.9.0`, so targeting 0.19.4 is supported
  by the current serializer. Programs that use ops introduced after 0.19.4 (for example `tan`,
  `dot_general` with algorithms, variadic `all_reduce`) will fail to downgrade and need rewrites.
- The shim exposes PJRT 0.116+ to jaxlib, forwards the 97 existing entries, and returns
  `UNIMPLEMENTED` from the 41 newer ones. Args-struct growth between 0.47 and 0.116 is the next
  thing to check per function the shim forwards.
