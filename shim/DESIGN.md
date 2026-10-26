# jax-metal shim: design

Goal: run current JAX (jax/jaxlib 0.12, PJRT C API 0.116) on Apple GPUs using Apple's
unmodified `jax-metal` 0.1.1 plugin (`pjrt_plugin_metal_14.dylib`, PJRT 0.47, StableHLO 0.19.4).

## Evidence this works

`shim/tests/direct_load.cc` loads the plugin with no shim, compiles a program lowered by
jax 0.12 and serialized as a StableHLO 0.19.4 portable artifact, runs it on the M3 GPU and
matches the CPU result (max abs error 4.8e-7). The same program serialized at StableHLO 1.16.3
fails in `PJRT_Client_Compile` with `unknown vhlo attribute code: 20`. The StableHLO version is the
only thing blocking compilation.

## How jaxlib 0.12 treats an old plugin

From `xla/pjrt/pjrt_api.cc` and `xla/pjrt/c_api_client/pjrt_c_api_client.cc` at XLA `0bfeeb47`
(PJRT 0.116):

1. **Version gate.** With `ENABLE_PJRT_COMPATIBILITY` unset, compatibility mode is on: any plugin
   with major 0 and minor ≥ 29 is accepted. Setting the variable to `0` forces an exact match.
2. **StableHLO target.** jaxlib serializes programs at the plugin's `stablehlo_current_version`
   attribute and falls back to `xla::GetDefaultStablehloVersion()` (far newer than 0.19.4) when the
   attribute is missing. The plugin reports no attributes.
3. **Newer entry points.** Most calls to functions newer than the plugin are guarded by
   `minor_version` checks, so a 0.47 plugin gets the older code paths. Some are guarded only by a
   null check on the function pointer: `PJRT_Client_Load`, `PJRT_Event_Create`/`Set`,
   `PJRT_Buffer_Bitcast`, `PJRT_Device_ClearMemoryStats`,
   `PJRT_LoadedExecutable_AddressableDeviceLogicalIds` (read when every loaded executable is
   created) and `PJRT_TopologyDescription_GetMemorySpaceKindIds`. jaxlib reads those pointers past
   the end of the plugin's 808-byte table, so with the raw plugin it calls whatever bytes follow.
4. **Argument structs.** All 97 functions the plugin implements take argument structs that are
   unchanged or append-only between 0.47 and 0.116 (`re/findings/pjrt-0.47-vs-0.116.md`). The
   plugin accepts larger `struct_size` values and ignores trailing fields, and jaxlib
   zero-initializes the structs, so new output fields stay zero.

## Design

`pjrt_plugin_metal_shim.dylib` exports `GetPjrtApi` and presents the plugin's own table with two
changes.

1. **Zero-padded table.** The shim owns a `PJRT_Api` laid out per the 0.116 header, embedded in a
   larger zero-filled block (room for 512 entries) so frameworks newer than 0.116 also read null
   past the end. It copies the plugin's 808 bytes (header plus 97 entries) into it. Every slot
   the plugin lacks stays null, so null-guarded call sites take their fallbacks.
2. **Truthful version.** It reports PJRT 0.47 and `struct_size` 808, the plugin's real
   capabilities, so jaxlib's `minor_version` guards keep choosing the old paths.
3. **Attributes.** `PJRT_Plugin_Attributes` is replaced with a function that returns:
   - `stablehlo_current_version` = `[0, 19, 4]`
   - `stablehlo_minimum_version` = `[0, 9, 0]`
   - `jax_metal_shim_version` = shim version string
   jaxlib then downgrades every program to StableHLO 0.19.4 before handing it to the plugin.
4. **Loading the plugin.** On first `GetPjrtApi` call the shim `dlopen`s the plugin from
   `JAX_METAL_SHIM_PLUGIN_PATH` if set, else `pjrt_plugin_metal_14.dylib` next to the shim (the
   Python package sets the variable to the library it found, see below). It checks that the plugin reports PJRT 0.47 with `struct_size` 808 and refuses to load anything
   else, since the table copy is only valid for that layout.
5. **Synthetic memory spaces.** The plugin's device reports no memory spaces and
   `PJRT_Device_DefaultMemory` returns UNIMPLEMENTED. jaxlib 0.12 routes every host-to-device
   transfer (`device_put`, every jitted argument) through the device's default memory space, so
   without this every program with inputs fails with `default_memory_space is not supported`.
   The shim gives each device one memory of kind `device`:
   - It answers `PJRT_Client_AddressableMemories`, `PJRT_Device_AddressableMemories`,
     `PJRT_Device_DefaultMemory`, `PJRT_Memory_{Id,Kind,DebugString,ToString,AddressableByDevices}`
     and `PJRT_Buffer_Memory` itself.
   - It rewrites memory-addressed calls to the device form before forwarding:
     `PJRT_Client_BufferFromHostBuffer` (memory → device), `PJRT_Buffer_CopyToMemory` (→
     `PJRT_Buffer_CopyToDevice`) and `PJRT_Client_CreateViewOfDeviceBuffer`.
   - Shim memories start with a null `vtable`, since jaxlib reads `memory->vtable`.
   - Memory pointers the shim did not create go to the plugin unchanged, so the shim never
     manufactures `PJRT_Error`s.
   - `PJRT_Memory_Kind_Id` stays null; for a 0.47 plugin jaxlib derives the kind id from the kind
     string.
6. **Program inputs and outputs of types the plugin rejects** (complex64, uint64, float8,
   int4, uint4). See "Values at the program boundary" below. The shim wraps
   `PJRT_Client_Compile` (substitutes the boundary program),
   `PJRT_Executable_DeserializeAndLoad`, `PJRT_LoadedExecutable_Execute` (buffer views and
   relabeling) and `PJRT_LoadedExecutable_Destroy`, and exports `JaxMetalShim_SetBoundaryIo`
   for the package.
7. **Retyped buffers.** uint64, float8, float4 and 2-/4-bit integer buffers live in the plugin
   as a type holding the same bytes (int64; uint8; int8 for the signed narrow integers, one
   value per byte as frameworks transfer them). `PJRT_Client_BufferFromHostBuffer` creates
   them as that type, and `PJRT_Buffer_ElementType` reports the type the framework asked for
   (also after `PJRT_Buffer_CopyToDevice`; `PJRT_Buffer_Destroy` forgets the buffer).
8. **Tracing (opt-in).** `JAX_METAL_SHIM_TRACE=1` logs shim initialization and attribute queries
   to stderr.

Everything else is forwarded untouched: the remaining entries are the plugin's own function
pointers, and PJRT errors, events, buffers and executables stay plugin objects.

### Python package

`jax-metal-shim` installs `jax_plugins/metal_shim/__init__.py`, whose `initialize()` registers
platform `METAL` with the shim's library path, at the same priority Apple used (500).

JAX imports every module under `jax_plugins` and calls each `initialize()` in no fixed order, so
Apple's own `jax_plugins.metal_plugin` must not be installed alongside the shim, or `METAL` is
registered twice. The shim's `initialize()` detects that case and raises an error that says to run
`pip uninstall jax-metal`.

`initialize()` also:

- **Finds Apple's library.** The wheel never contains it. `python -m jax_plugins.metal_shim
  fetch` downloads the jax-metal 0.1.1 wheel from PyPI, verifies it (PyPI's sha256 for the
  wheel, a pinned sha256 for the library) and keeps the library and Apple's LICENSE in a
  per-user cache (`_plugin.py`). `initialize()` looks in `$JAX_METAL_SHIM_PLUGIN_PATH`, the
  package directory, then the cache, passes the result to the shim through
  `JAX_METAL_SHIM_PLUGIN_PATH`, and fails with instructions when there is none. It warns when
  jax or jaxlib is not a release this version was tested with.
- **Disables Shardy by default.** jax 0.12 lowers with the Shardy partitioner, whose `sdy` dialect
  appears on every jitted function with committed inputs (`sdy.mesh`, `sdy.sharding`); the plugin
  fails with `dialect 'sdy' is unknown`. Setting `jax_use_shardy_partitioner = False` makes JAX
  emit the classic `mhlo.sharding` attributes instead. An explicit `JAX_USE_SHARDY_PARTITIONER`
  in the environment wins. The setting is process-wide, which only matters for multi-device
  sharding on other backends in the same process.
- **Registers METAL lowering rules** (`_lowerings.py`) for JAX primitives the plugin rejects,
  each rewritten into ops the plugin lowers:
  - Cumulative reductions (`cumsum`, `cumprod`, `cummax`, `cummin`, `cumlogsumexp`): jax 0.12 emits
    a full-length `reduce_window` the plugin cannot legalize; a parallel prefix scan works.
  - CHLO ops that XLA's serializer always wraps in `stablehlo.composite`
    (`ChloPreserveHighLevelOpsPass`): `acos`, `acosh`, `asinh`, `atanh`, `sinh`, `cosh`, `erf`
    (Eigen/XLA float32 rational approximation), `erfc`, `top_k` (stable sort).
  - Ops without an MPS pattern: `cbrt`, `population_count`, `clz` (which also fixes `lgamma`).
  - Linear algebra: `cholesky` (column-by-column) and `triangular_solve` (forward substitution
    after transposing/reversing every flag combination into one canonical case), as `fori_loop`s
    of matrix-vector products. This also makes `linalg.solve`, `inv` and `cho_solve` work, since
    JAX's generic `lu` already lowered. Like JAX on CPU and GPU, `cholesky` of a matrix that is
    not positive definite is all NaN.
  - QR: `geqrf` and `householder_product` as Householder loops in LAPACK's packed form. JAX's
    generic lowering emits XLA's `Qr` custom call, which aborts the plugin (its
    `CustomCallConverter` only knows LU).
  - `eigh`: parallel cyclic Jacobi (round-robin pairs, one gather per round) for n ≤ 256 and for
    complex inputs; JAX's QDWH spectral divide-and-conquer, always batched, above. Jacobi refuses
    n > 256 at trace time: a 1008 × 1008 one stalled the GPU and WindowServer. The other rules'
    programs stay small, so no other cap is set: in StableHLO ops, QDWH `eigh` grows from
    2,300 (n = 257) to 7,000 (n = 2048), real SVD is 1,500 (64) to 4,900 (512), and Cholesky
    and QR are about 100–200 at any size (one loop each).
  - SVD: real inputs use JAX's generic QDWH SVD, always batched (unbatched, its `lax.cond`s inside
    a while loop crash MPSGraph at run time, `GPU::WhileOpHandler`). Complex inputs reduce to a
    min(m, n) square core with QR, then take the Jacobi eigendecomposition of [[0, A], [Aᴴ, 0]]
    (complex QDWH crashes MPSGraph). A requested SVD `algorithm` is a hint, and METAL uses its
    own (JAX's generic SVD would reject anything but POLAR).
  - Empty matrices: the linear algebra rules return early, since `fori_loop` traces its body even
    for zero trips and the body cannot index an empty axis.
  - `mulhi` (philox PRNG), which XLA serializes as a CHLO composite: 32-bit products from 16-bit
    limbs; 8- and 16-bit products in 32 bits.
  - `reduce_precision`: XLA's bit-level algorithm (round to nearest even, flush subnormals) on
    the float32 or native float16 bit pattern.
  - `bfloat16` conversions: the plugin emulates bfloat16 in float32, so its own float↔bfloat16
    casts skip rounding inside a program and flush subnormals at buffer boundaries. Conversions
    into bfloat16 round on the float32 bits and bitcast the top 16 bits; conversions out widen the
    bits. bfloat16 arithmetic itself was already exact.
  - `rng_bit_generator` (`jax.random` with `impl="rbg"`): MPS's random kernel aborts the process
    for this op whatever integer type it is asked for, so METAL generates the bits with
    Threefry-2x32 in counter mode. JAX defines these bits as backend-specific. Each word
    depends only on the key and its position, so a larger draw starts with the smaller one,
    as with XLA's generator (JAX's `vmap` of the `unsafe_rbg` PRNG relies on it).
  - Accuracy fixes for MPS elementwise functions, found by `tests/accuracy_sweep.py`: MPS computes
    `expm1`/`log1p` naively (1.5e7 ulps off near 0), returns 1 for `tanh(NaN)` and loses
    small-argument `tanh`, and returns a number for `sign(NaN)`. METAL uses Kahan/Goldberg
    `expm1`/`log1p` on MPS's accurate `exp`/`log`, `tanh` via `expm1`, XLA's float32 `erfc`,
    and a `log1p` form of `acosh`/`atanh`. Every float32 elementwise op is now within 18 ulps of
    XLA's CPU results (most within 1–4) with matching NaN/inf behavior. These rules compute
    float16 and bfloat16 inputs in float32 (their formulas assume its range: x² overflows
    float16 above 256), and `tanh` is exactly odd.
  - Multi-key sorts (`lexsort`, complex sorts): the plugin's sort takes one key, so n keys become
    n stable passes. `top_k` (any dtype, bool included) is a stable sort on an order-reversing
    key; `approx_max_k`/`approx_min_k`, whose generic lowering is an `ApproxTopK` custom call,
    are computed exactly, which is a valid result.
  - `dot_general` with an integer or bool result: MPS matmul takes floating point only, so these
    are exact multiply-and-sum (wrapping like XLA; bool is OR of ANDs), over slices of the
    contraction that keep the broadcast product within 4M elements. A contraction over a
    zero-length dimension returns zeros (it aborts MPSGraph).
  - Every rule traces with `jax_numpy_rank_promotion="allow"`: rules broadcast with jnp-style
    operators, and a user's `"raise"` (which JAX's own tests set) must not reject them.
  - Requests StableHLO 0.19.4 cannot express are dropped instead of failing the program: a
    requested result accuracy (`lax.exp(x, accuracy=...)`, StableHLO 1.9) and a dot algorithm
    (`precision=DotAlgorithmPreset...`, StableHLO 1.6). MPS has one float32 implementation of
    each function, and float32 products are at least as precise as any algorithm preset.
  - User `lax.composite` ops lower to their decomposition: nothing in the plugin's pipeline
    expands `stablehlo.composite`.
  - NaN in `max`, `min`, `clamp`, `reduce_max`/`reduce_min`, `argmax`/`argmin` and max/min
    pooling: XLA propagates it, the plugin does not. It builds `mps.reduction_max`/`min` with
    their `propagate_nans` flag off and maps the elementwise ops onto MPS's non-propagating
    `maximum`/`minimum`/`clamp` (`re/findings/mps-nan-semantics.md`, from the disassembly);
    `cond` of a singular matrix came out 0 instead of inf. For floating-point inputs the rules
    keep JAX's lowering and put NaN back where an input has one (a compare and a select, or
    one extra reduction).
  - Collectives on one device (`_collectives.py`): the plugin has one device and patterns for
    almost no collectives, so `psum`/`pmax`/`pmin`, `all_gather`, `all_to_all`,
    `reduce_scatter`, `ppermute`, `pbroadcast`, `axis_index` and `ragged_all_to_all` become
    what they compute on a one-device axis (identities, reshapes, zeros, 0). `shard_map` and
    `pmap` programs then run on METAL.
  - Decompositions JAX lowers only to LAPACK or cuSOLVER (`_decompositions.py`), with LAPACK's
    conventions and packed outputs: `geqp3` (QR with column pivoting), `hessenberg` (gehrd),
    `tridiagonal` (sytrd/hetrd), `schur` (Francis double-shift QR with LAPACK's deflation
    criteria and standardized 2 × 2 blocks; NaN after 30·n iterations without convergence),
    `eig` (from the real Schur form, eigenvectors normalized as `geev` does) and
    `tridiagonal_solve` (Gaussian elimination with partial pivoting, as `gtsv`, plus the
    `dlagts` perturbation JAX requests). Checked against LAPACK: same packed outputs, pivots
    and residuals.
  - NaN rules never apply inside reduction, window, scatter or all-reduce bodies: the plugin's
    converters recognize those by a single combining op. `scatter_max`/`scatter_min` propagate
    NaN with a scattered mask instead.
  - Every rule traces with standard dtype promotion and without varying manual axes, so rules
    work under `jax_numpy_dtype_promotion="strict"` and inside `shard_map`; JAX's generic `lu`,
    `lu_pivots_to_permutation` and `cholesky_update` go through the same wrapper.
- **Rewrites float8, float4 and 2-/4-bit integer programs** (`_narrow.py`), see "Narrow types"
  below.
- **Registers complex support** (`_complex.py`, `_boundary_io.py`), see "Complex numbers" below.

### Complex numbers and FFT

The plugin computes complex64 arithmetic (add, multiply, divide, `dot`, sums, `abs`) but has no
pattern for `mhlo.real`/`imag`/`complex`, has no complex kernels for `pad` (aborts the process),
`gather` or `scatter`, and no `fft` pattern. XLA's serializer also expands complex transcendental
functions into raw `real`/`imag` ops. `_complex.py`:

- builds `real` as a complex→real `convert` (MPS keeps the real part), `imag` as `real(z·(−i))`,
  and `complex(x, y)` as `convert(x) + convert(y)·i`;
- expresses complex `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `expm1`,
  `log1p`, `sign`, `rsqrt` and `pow` in real arithmetic, and the inverse functions `asin`,
  `acos`, `atan`, `asinh`, `acosh`, `atanh` with Kahan's branch-cut formulas (float32 results
  within 5e-6 relative of NumPy's complex128, on the cuts too). `sqrt` keeps the sign of a zero
  imaginary part, which picks the side of each branch cut;
- applies `pad`, `gather`, `scatter`, `scatter_add`/`sub` and `dynamic_update_slice` to real and
  imaginary parts separately, and sorts complex keys as (real, imag); `scatter_mul`
  (`x.at[i].multiply`, `.power`) multiplies magnitudes and adds angles, as real scatters;
- computes complex `reduce_max`/`reduce_min` (`jnp.max`, `nanmax`, `ptp`, ...), which order by
  real part and then imaginary part, with real reductions: JAX lowers them to a reduction
  whose body compares the parts (nine ops), and the plugin recognizes a reduction by its one
  combining op;
- implements `fft`/`ifft`/`rfft`/`irfft` (any rank) as a radix-2 transform with a direct DFT for
  odd lengths, with twiddle factors computed on the GPU from exact integer angles.

### Values at the program boundary

The plugin rejects any compiled function whose signature mentions complex, uint64, float8 or
4-bit integer types, including the small programs behind eager ops. `_boundary_io.py` and the
shim lift this without copies:

1. **Two programs.** For a METAL program with such inputs or outputs, the package hooks
   `mlir.lower_jaxpr_to_module` and lowers the original (its own signature) plus a boundary
   version in which each complex64 input/output is a float32[..., 2] (real, imag) pair, each
   uint64 one an int64 with the same bits, and each float8/int4/uint4 one its bytes (uint8, or
   int8 for int4), converted inside the program. The boundary version is serialized as a
   StableHLO 0.19.4 portable artifact with XLA's own serializer
   (`_jax.mlir.serialize_portable_artifact`) and attached to the original module.
2. **jaxlib compiles the original.** IFRT takes output dtypes, shapes and layouts from the MLIR
   module it is given (`ResultShapesOfModule`), not from the executable, so jaxlib must see the
   real signature.
3. **The plugin compiles the boundary program.** A `compiler.compile_or_get_cached` hook strips
   the attachment and calls `JaxMetalShim_SetBoundaryIo` (thread-local) with the boundary
   program and the positions and kinds of the changed values: 1 for complex pairs, `0x100 | T`
   for a retyped type `T` (a `PJRT_Buffer_Type`). The shim's `PJRT_Client_Compile` compiles the
   boundary program instead.
4. **Buffers are relabeled at execution.** The plugin checks argument shapes strictly, so
   complex64 inputs are passed as float32[..., 2] views of the same device memory
   (`PJRT_Client_CreateViewOfDeviceBuffer`; views are released when execution completes).
   Pair outputs are returned as complex64 views that own the pair buffers (deleting the view
   destroys the original). Retyped buffers already live as their bytes ("Retyped buffers"
   above); outputs of those kinds are relabeled with their type, and a buffer of such a type
   the plugin holds as that type (one the shim did not create) goes in as a view. Buffers
   then always match jaxlib's metadata, so every JAX path (host transfer, device_put to CPU,
   `addressable_data`, indexing, chaining) sees ordinary arrays of their own types.

Costs and limits:

- A program with complex outputs waits for completion before returning: a view counts as
  ready as soon as it exists, so the shim awaits the pair output first.
- Inputs of these types are never donated (donation is optional in JAX).
- Each such program is lowered and serialized twice at compile time.
- The newer narrow types (float8_e4m3, e3m4, e8m0fnu, float4_e2m1fn, int2, uint2) cannot be
  program inputs or outputs on METAL: StableHLO 0.19.4 has no such types, so jaxlib cannot
  serialize the original module. The package raises a `NotImplementedError` that says to
  convert inside the program. Their buffers do transfer (as uint8/int8).
- Rejected approaches, for the record: relabeling buffers alone (jaxlib's metadata then
  contradicts the buffers), pair-backed arrays with patched host fetches (leaks float32 pairs
  through `addressable_data`), and DLPack import (jaxlib refuses DLPack for METAL devices).

### Narrow types

MPS has no float8, float4, int4 or int2 types, and the plugin rejects programs that use them.
`_narrow.py` rewrites every METAL program with such values before lowering, interpreting its
jaxpr:

- **Carriers.** Each narrow value is held in a type the plugin has: float32 for the
  floating-point formats, int8/uint8 for the integers, always an exact value of the format.
- **Rounding per operation.** An operation with a narrow result runs on carriers and rounds
  once, the way XLA converts to the type: round to nearest even with subnormals; overflow to
  inf (e5m2, e4m3, e3m4), NaN (the `fn`/`fnuz` formats, e8m0fnu) or the largest value
  (float4_e2m1fn, whose NaN becomes -0); no -0 in `fnuz` formats; e8m0fnu as XLA (ties up,
  no 0); integers wrap, and from floating point truncate and saturate (NaN gives 0). The
  rounding works on bits, so backends that flush float32 subnormals (e8m0fnu's 2^-127 is one)
  do not change it. Conversions from float32, float16 and bfloat16 match XLA's CPU backend
  bit for bit for every format and value class.
- **No rounding where none is needed.** Data movement (reshape, slice, gather, select, sort,
  max/min, ...) keeps carriers as they are; reductions, windows and scatters combine in the
  carrier type and round their result once; cumulative sums and products round each partial
  result of a parallel scan.
- **Bits.** `bitcast_convert_type` encodes and decodes each format exactly (sign of NaN
  included); integer shifts, `population_count`, `clz` and `nextafter` work on the format's
  own bits.
- **Control flow.** `while`, `scan` and `cond` are rebuilt with carrier bodies; `jit`,
  `checkpoint`, `custom_jvp`/`custom_vjp` calls are inlined.

XLA's CPU and GPU backends compute float8 arithmetic in float16 and may keep excess precision
across operations, so results can differ from theirs in the last bit, and cumulative sums by
the order of rounding (XLA's CPU backend sums sequentially). Over JAX's narrow-type tests in
METAL emulation, see `re/findings/jax-suite-cpu.md`.

### float64

Not available, and not fixable with lowering rules:

- MPS has no float64 type. A float64 value cannot exist anywhere in a plugin program, not even
  as an internal constant (`'mps.constant' op result #0 must be tensor of mps native type`), and
  float64 program inputs/outputs are rejected. Apple GPUs have no float64 hardware.
- float64 buffers do transfer to the device (storage only).

**Double-float emulation works.** `experimental/double_float.py` represents a value as an
unevaluated sum of two float32s and carries rounding errors with error-free transformations
(Knuth TwoSum, Dekker TwoProduct). These need every float32 add and multiply rounded exactly per
IEEE 754, which holds on the M3 through MPS (no fast-math reassociation or FMA contraction:
TwoSum and TwoProduct are bit-exact over 65,536 random pairs). Measured against NumPy float64,
65,536 elements on the M3:

| Operation | max relative error | float32 alone | time |
|---|---|---|---|
| add | 6.4e-15 | 6.0e-8 | 0.62 ms |
| multiply | 1.4e-14 | 5.9e-8 | 0.45 ms |
| divide | 1.0e-14 | 5.9e-8 | 1.55 ms |
| sqrt | 1.1e-14 | 5.9e-8 | 0.68 ms |
| dot, 65,536 terms | 2.7e-16 | 1.8e-8 | 1.44 ms |

That is about 48 significand bits (true float64 has 53), but only float32's exponent range
(about 1e-38 to 3e38, no float64 subnormals), at roughly 10-20 float32 operations per operation.

**Possible delivery: an explicit transform, not transparent x64.** Transparent
`jax_enable_x64` would need float64 arrays passed straight to plugin programs, which the plugin
rejects; relabeling float64 buffers would also need an on-device bit conversion. An explicit
`df64.jit(f)` is feasible:

1. trace `f` with float64 avals (x64 enabled) to a jaxpr;
2. interpret the jaxpr, mapping each float64 variable to an (hi, lo) float32 pair and each
   primitive to its double-float algorithm (arithmetic, `dot_general` and reductions with
   compensated summation, comparisons, `select`, converts; shape ops apply to both parts);
3. split float64 inputs into pairs on the host before transfer and recombine outputs on the host.

Transcendental functions need double-float algorithms (range reduction plus series, as in the QD
library). Rough effort: 1-2 weeks for arithmetic, linear algebra and reductions; another 1-2 weeks
for transcendental functions and gradients.

**`philipturner/metal-float64` was evaluated and is not usable here** (checked 2026-10-24):

- It contains no float64 arithmetic. The repository is archived (last push 2023-05), its README
  says "still in the planning stage - this is not a finished library", `float64_t` is an empty
  class wrapping a `ulong`, `Vector.h` is vector/swizzle scaffolding, and its performance and
  precision tables are empty. The finished part is `MetalAtomic64` (64-bit atomics), which
  float64 math does not need.
- Even a finished version could not be called from JAX on METAL. It is Metal Shading Language
  code for hand-written compute kernels, and the plugin offers no way to run custom kernels inside
  its MPSGraph programs: its `CustomCallConverter` (`0x50e70`) recognizes a single target, which it
  maps to MPS's LU decomposition (`mps.MatrixDecompositionLUOp`).
- Its design notes are still useful. It plans two representations: `float32x2_t` (double-single,
  8-bit exponent, 48-bit significand), which is what `experimental/double_float.py` implements in
  JAX ops, and `float64_t` (IEEE binary64 emulated with integer arithmetic: full exponent range,
  subnormals, NaN/inf, estimated at 1/32-1/64 of float32 throughput). The second could be a
  later "exact IEEE" mode of `df64.jit`, built from uint32 operations, which the plugin supports.

Recommendation: implement `df64.jit` as a JAX-level transform on float32 pairs (no fork); revisit
an integer-based IEEE mode only if users need the float64 exponent range.

## Additional considerations

1. **Ops newer than StableHLO 0.19.4.** jaxlib fails to serialize a program that uses an op or
   attribute introduced after 0.19.4 (for example `stablehlo.tan` 1.4.0, `dot_general` with
   algorithm 1.6.0, `exponential` with `result_accuracy` 1.9.0, variadic `all_reduce` 1.5.0). The error
   is raised in jaxlib before the plugin sees the program. Fixing these needs a rewrite step
   (Phase 3), either in JAX lowering rules registered for platform `METAL` or inside the shim.
2. **What the plugin can lower.** Even valid 0.19.4 programs fail if Apple's HLO→MPS lowering has no
   pattern for an op. The op inventory is in `re/findings/mps-op-coverage.md`.
3. **Data types.** float64 is not available: MPS has no float64 type (not even for constants
   inside a program) and Apple GPUs have no float64 hardware; see "float64" below. complex64
   works inside programs and at their boundary (see above); complex128 needs float64. int64
   and uint64 work. float8, float4, int4 and int2 are emulated ("Narrow types"); the newer of
   them cannot be program inputs or outputs. float6 is out of scope (XLA's CPU backend
   crashes on it, so there is no reference).
4. **Shardy.** Handled by turning the Shardy partitioner off (see the Python package section).
   If a future JAX removes the GSPMD path, the shim would have to strip `sdy` attributes, which
   needs MLIR in the shim.
5. **Compile options.** jax 0.12 serializes a newer `CompileOptionsProto`. Unknown proto fields are
   skipped by the plugin's parser (the direct-load test passes the jax 0.12 options).
6. **Framework drift.** The design relies on the PJRT append-only rule and on jaxlib's version and
   null guards. A newer jaxlib that adds unguarded calls past 0.47 would need a shim update; the
   padded table makes such calls hit null rather than garbage.
7. **Single process, single device.** The plugin exposes one device. Collectives work on that
   device (they are identities on a one-device axis); multi-device and multi-process
   execution are out of scope.
8. **Licensing.** The shim and its tools are new code under Apache-2.0 (`LICENSE`), as are the
   vendored PJRT headers (see `third_party/*/SOURCE`). Apple's binary is not redistributed:
   each user fetches it from PyPI. Its wheel carries an MIT-style license file (Apple
   copyright) while its metadata says "All rights reserved", so bundling it would need that
   resolved first.

9. **Remaining gaps** (`re/findings/jax-op-probe.md`: 88 of 88 probed JAX ops pass):
   - float64 (see below);
   - float32→bfloat16 casts done by the plugin itself, outside these rules (none known), would
     still flush bfloat16 subnormals.

## Tests

- `tests/direct_load.cc`: plugin alone, StableHLO 0.19.4 artifact, GPU vs CPU result.
- `tests/shim_load.cc`: same flow through the shim; also checks the attributes and that slots past
  the plugin's table are null.
- `tests/shim_boundary_test.cc` (`make unit-test`, any machine): the shim against a fake PJRT
  0.47 plugin (`tests/fake_pjrt_plugin.cc`): retyped buffers, boundary program substitution,
  complex views and relabeled outputs.
- `tests/test_jax_e2e.py`: current JAX with the shim installed as the `METAL` plugin. Runs
  jit, grad, matmul, conv, sort, scatter, control flow, PRNG and an MLP training step on the GPU
  and compares with CPU (16 of 16 pass).
- `tests/probe_jax_ops.py`: 88 individual JAX ops, each in its own process, recorded in
  `re/findings/jax-op-probe.md` (88 of 88 pass).
- `tests/test_phase3.py`: linear algebra (every `triangular_solve` flag combination, batched),
  bit-exact `reduce_precision` and bfloat16 conversions, the `rbg` PRNG.
- `tests/test_complex.py`: complex arithmetic and transcendental functions, FFT family at lengths
  7–4096, multi-dimensional FFTs, FFT convolution, gradients through `rfft`; complex64 arrays
  in and out of programs (pytrees, scalars, loops, gradients, empty arrays, chaining, eager ops,
  transfer to CPU, donation, no buffer leak over 500 calls).
- `tests/test_narrow.py`: float8, float4, int4 and int2 conversions (each format's values,
  midpoints and neighbours, specials) bit for bit, arithmetic, reductions, matmul, bitcasts of
  every width, control flow and narrow program inputs/outputs, against XLA's CPU backend
  (rounding after every operation). `--emulate` runs the rewrite on the CPU instead.
- `tests/accuracy_sweep.py --strict`: every float32 elementwise op against the CPU across
  magnitudes 1e-12…1e4 and special values.
- `tests/accuracy_extended.py`: reductions, matmul, convolutions, float16/bfloat16 and binary
  ops against a float64 reference, flagged where METAL is clearly worse than XLA's CPU.
- `tests/jax_suite/run_jax_tests.py`: JAX's own test files on METAL, in batches, crash-tolerant.
  With `--cpu` it runs the same files in METAL emulation (below) and records each test's
  duration, memory and parameters; with `--compare-cpu`, its summary labels each METAL
  failure "METAL only" (MPS behavior) or "also on CPU" (the rules, or the test).
- `tests/gpu_session.py` (`make gpu-session`): all of the above on METAL in one resumable
  command, every GPU job under `tests/guard.py`, JAX's heaviest tests (by the CPU run) last.
- `tests/jax_suite/cpu_rules_plugin.py`: JAX's own tests on the CPU backend through these
  lowering rules, lowering the way METAL does (any machine, no GPU), with
  `tests/jax_suite/metal_check.py` predicting what the plugin could not compile. Over 56 of
  JAX's test files, 19,608 tests pass and 35 fail, every failure explained
  (`re/findings/jax-suite-cpu.md`); what remains for METAL is MPS behavior.
