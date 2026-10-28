# GPU suite failure investigation

Baseline: `shim/build/gpu-session/jax-suite/summary.md` (18,376 passed,
1,080 failed, 186 crashed). Work starts with its largest group: 183 numerical
`AssertionError` reports. The group contains several independent causes; the
message normalizer removes the numerical details after the first line.

## Boolean scatter consumers — resolved

On the M3, a boolean scatter returns the correct mask when it is a program
output, but consumers inside the same program read the original operand. For
example, `ones(4, bool).at[i].set(False)` with `i=[0]` returns
`[False, True, True, True]`; appending `.astype(int32)` returns `[1, 1, 1, 1]`.
The failure also occurs with nonconstant operands/updates, logical negation,
padding and reductions. Returning the mask alongside its reduction makes the
reduction correct. An optimization barrier does not fix it. Replacing the cast
with a select does not fix it either, isolating the failure to the boolean
scatter and its use inside the MPS graph, rather than numeric conversion.

`_indexing.py` performs boolean replacement/min/max scatters on int32 0/1
values and converts the result back to bool. Min/max preserve AND/OR semantics
with duplicate indices. Custom replacement update jaxprs retain their original
lowering rather than silently changing their update function.

Verification on Apple M3:

- All 48 JAX `testInsert*` and `testSegmentReduceBoolean*` tests pass: 21 were
  failures in the baseline, 27 were already passing.
- All 15 focused cases in `shim/tests/test_indexing.py` pass, including input
  operands/updates, duplicate reductions, and five kinds of downstream consumer.
- All 54 primitive comparisons in the development scatter probe pass (integer
  and floating scatters included).

## Interior padding aliases its fill tensor — resolved

The smallest reproducer returns both `lax.pad(x, 0, [(0, 0, 1)])` and an
independent zero tensor of the padded shape. For `x=[1, 2]`, both outputs become
`[1, 0, 2]`. The fill tensor is overwritten. This explains corruption in sparse
row pointers, counts, and set operations: the shim's parallel prefix scan uses
interior padding, and the overwritten zero tensor can be reused by later ops.

Disassembly of Apple's `PadConverter::matchAndRewrite` at `0x59324` confirms
that it builds an `mps.BroadcastToOp` followed by `mps.StridedSliceUpdateOp`
(`0x59838`, `0x59920`). Edge padding uses `mps.PadOp`. Replacing downstream
slice updates only hides some manifestations; the fix belongs at interior pad.

`_indexing.py` inserts a new dimension, concatenates padding after each input
element, reshapes, and trims the final gap. Ordinary edge padding handles the
remaining low/high padding. No interior pad reaches MPS.

Verification on Apple M3:

- All 49 development padding comparisons pass, including untouched fill tensors.
- All 29 focused indexing tests pass (the 15 boolean cases plus padding,
  preservation of shared buffers, multiple axes, negative edges and empty input).
- 81 of the 82 saved sparse assertions now pass, checked in separate batches
  under the memory guard. The remaining `test_bcoo_dynamic_slice3` uses
  out-of-range dynamic slice starts and is a separate indexing issue.

## Missing index clamping — resolved

MPS gather returns zeros for out-of-range indices where StableHLO requires the
nearest valid window. Dynamic slice and dynamic update slice likewise fail to
clamp the start to `dimension_size - window_size`; for a length-10 operand and
window size 4, a start of 9 must select/update elements 6 through 9. The native
slice lowering also rejects int8/uint8 starts.

`_indexing.py` clamps gather and dynamic slice/update indices explicitly. It
preserves gather's fill/drop mask, and clamps slice starts before converting to
a signed index carrier so that large unsigned values do not wrap negative.

Verification on Apple M3:

- All 41 focused indexing regressions pass, including unsigned maximum values,
  negative starts, multidimensional gather and explicit fill values.
- The bounds probe's 40 primitive comparisons pass.
- 19 more saved numerical assertions pass. This includes the last sparse
  assertion, nested out-of-bounds slices, negative indexing, clipped scatter
  derivatives, and take/take-along-axis cases.

## Ordering and gather payload precision — resolved

MPS sort ignores the comparator's canonicalization of zeros and NaNs: it puts
negative NaNs first and negative zero before positive zero. Standalone
TOTALORDER comparisons instead use ordinary floating comparisons, making all
NaN comparisons false. `_ordering.py` canonicalizes sort keys before sorting
and implements total-order comparisons using signed integer keys derived from
the floating bit patterns.

Both sorted integer values and ordinary gathered integer payloads lose bits
through a float32 conversion. For example, gathering 16,777,217 produces
16,777,216, and int32 maximum becomes int32 minimum. The permutation from sort
is correct even for neighboring large integer keys. Sort now gathers the
original operands using that permutation. Gather carries int32/uint32 bits as
float32, avoiding numeric conversion. It carries float16/bfloat16 bits as uint16
to preserve their NaN representations.

Verification on Apple M3:

- All 21 corresponding saved assertions pass (sort, top-k, unique, searchsorted,
  and scipy mode).
- All 64 upstream `testSort*`/`testTopK*` cases pass.
- All 20 focused ordering regressions pass: IEEE total-order comparisons,
  NaN/zero bits, adjacent large integer keys, axes, and multiple keys/payloads.
- All 27 ordering and 22 integer payload development comparisons pass.

Complex NaN construction/extraction remains a separate known limitation:
`_complex.py` uses multiplication by `1j`/`-1j`, contaminating one component
when the other is NaN. The ordering integration checks use finite complex data.

## Scatter windows, broadcast layout and unary updates — resolved

MPS writes the in-bounds portion of a scatter window whose start is out of range.
JAX's drop mode drops the entire window; its gather-based derivatives rely on
that contract. Explicitly masking such windows clears two scatter gradient
failures.

Three mixed-index update failures come from broadcast followed by transpose,
independent of scatter. Broadcasting `(3, 1, 2, 5)` to `(3, 2, 2, 5)` and
transposing axes `(1, 2, 0, 3)` reads the wrong repeated-value layout. Input
updates already materialized on the host work correctly. The shim materializes
non-scalar expanding broadcasts with concatenations (logarithmic in the repeat
count). Scalar fills and broadcasts without expansion keep their existing path.
This also clears the generalized eigensolver assertion.

Finally, MPS identifies a scatter combiner by its opcode and misinterprets
`square(x)` as `x * update`; scatter-apply supplies a dummy zero update. The shim
counts applications with scatter-add and evaluates the unary function outside
the scatter, repeatedly where duplicate indices require it.

Verification on Apple M3:

- All five targeted scatter/window assertions pass; the subsequent rerun also
  confirms `testGeneralizedEigh2` passes.
- All eight upstream `testIndexApply*` cases pass.
- All 66 permanent indexing regressions pass, including fill/clip windows,
  several broadcast layouts and types, and repeated unary applications.

## Narrow padding fills and integer bitwise reductions — resolved

Runtime padding fill values become zero for bool and 8-/16-bit integer operands,
although literal fills work. Widening integer pad does not suffice because the
compiler moves the cast through it. Concatenating fill tensors and cropped data
avoids this path, including empty inputs and negative edges.

MPS lowers integer AND/OR reductions to logical reduction kernels: OR of
`[0, 1, 2, 3]` returns 1 instead of 3. `_reductions.py` uses a balanced tree of
bitwise elementwise operations. Both named reductions and simple generic
bitwise reductions with runtime initial values use it.

Verification: all three saved assertions pass, and all 119 indexing/reduction
regressions pass on the M3 (48 reduction cases check both named and generic
forms, with empty dimensions, odd counts and several axis selections).

## Boolean slice updates in loops — resolved

Dynamic slice updates on bool work in isolation but lose writes inside a scan
loop. A scan that copies `x[i]` produces all false values; casting the scan's
output elements to int32 makes it correct, while casting only its reads does
not. Boolean dynamic updates now use int32 operands/updates, as boolean scatter
already does. All ten upstream binary-ufunc accumulation cases and all 75
indexing regressions pass, including direct scan copies and boolean prefix OR.

## Convolution layout heuristics — resolved

Equivalent IOHW and OIHW kernels give different answers: the MPS converter
treats IOHW as a transpose convolution and misinterprets strides. A `1×2`
kernel with vertical stride 2 reads the wrong rows. Filter derivatives with
swapped input batch/feature axes also enter a gradient kernel that aborts with
`Gradient and filter output channel mismatch`. Normalizing only the kernel is
insufficient. `_convolution.py` physically normalizes input, kernel and output
layouts; complex inputs first split into the existing real bilinear form.
Input dilation is explicit interior padding, and negative padding generated by
autodiff is explicit cropping/padding, avoiding unsupported native paths.

All four saved convolution gradient assertions pass. All 48 permanent
convolution cases pass, checking forward values, JVPs and VJPs in NCHW/NHWC,
four dtypes, multiple strides, input/kernel dilation and asymmetric padding.

## Cancellation in float32 factorizations — resolved

The Cholesky case has condition number about 25,467. Rounding a squared factor
before subtracting it from the next diagonal loses enough precision to fail the
solve comparison. Compensating that subtraction alone matches the CPU's stored
factor but still leaves error from rounded factors used in subsequent columns.
For float32 matrices up to 128, Cholesky now carries high/low float pairs through
the recurrence and rounds the final factors once. Larger matrices keep the
existing algorithm.

Small float32 LU uses compensated product subtraction in its trailing-matrix
updates. This also resolves the finite-difference second-derivative failure in
`test_custom_linear_solve_aux`. No tolerances were relaxed.

Both saved linear-solve assertions pass, including their gradient checks. All
12 factorization regressions pass: an ill-conditioned solve checked against
float64, rectangular/batched/empty LU, reconstruction for small and large
Cholesky, and non-positive-definite matrices.

## bfloat16 subnormal sign handling — resolved

Input/output bitcasts and nextafter preserve bfloat16 subnormals. MPS abs,
negation and select flush them to zero, breaking spacing's copysign fallback.
Sign changes now operate on uint16 bits and select chooses uint16 payloads
before converting back. All three upstream subnormal-spacing cases pass.
The permanent regression exhaustively checks all 65,536 bit patterns for abs,
negation and select, including NaN representations and signed zeros.

All 205 permanent regressions for the numerical fixes pass together on the M3
(peak footprint 0.58 GB under the guard).

The combined rerun passes **all 183 original numerical assertions**, in batches
of eight under the memory guard (peak footprint 1.55 GB). Per-test results and
logs are in `shim/build/numerical-final-verification/`.

The fixes account for 21 failures from
boolean scatter, 104 from interior padding, 19 from index clamping, and 21 from
ordering/payload precision, plus 18 from the causes above.
## Variadic optimization barriers — resolved

The next group contains 125 `null operand found` failures, mostly checkpoint
derivatives, plus direct barriers, mutable refs and control flow. All 125 still
failed after the numerical fixes. A one-input barrier works on the M3; two or
more inputs fail. Apple's `OptBarrierConverter::matchAndRewrite` (`0x58f50`)
explicitly reads operand zero (`0x58f84`), creates one `mps.IdentityOp`
(`0x58f9c`), then replaces the entire source operation with that one result
(`0x58fb4`). Other results become null operands in their consumers.

`_compiler_ir.py` packs the inputs into a single tensor, crosses one barrier,
and unpacks each result. Non-complex inputs use their byte representations;
complex inputs remain intact, with other inputs' bytes embedded as exact
complex values. This preserves NaN payloads, subnormals, signed zeros and wide
integer bits, as well as the shared dependency on all inputs. Splitting into
unrelated unary barriers would lose that dependency.

The rewrite works on a private compilation copy, catching barriers emitted
directly by remat as well as the ordinary primitive. It also prepares the
alternate boundary program before serialization. Public lowered IR and CPU
compilation keep their existing behavior.

This clears the null-operand error in all 125 cases. Four then exposed a second
compiler defect: trivial conditional results become `arith.select` during the
plugin's final canonicalization, but its MPS serializer cannot handle that op.
The pre-serialization MPS dump shows the unsupported selects; an equivalent
conditional with a multiply in one branch remains `scf.if` and works.
Unary barriers on branch returns prevent this fold, preserving lazy branch
execution. All six isolated conditional reproductions now pass.

All **125 original null-operand failures pass together** after both fixes.
Results are in `shim/build/null-operand-final-verification/`; peak footprint
was 0.44 GB and worst GPU probe latency was 3 ms. The 20 permanent barrier tests
include exhaustive 8-/16-bit patterns, wide integers, mixed types, complex and
narrow boundaries, empty inputs, checkpoint gradients and unchanged public IR.
All 51 new barrier/control-flow regressions pass, including both conditional
branches, multi-way switches, nested scans and repeated IR preparation. All
205 earlier numerical regressions also pass with the compiler repairs enabled.
All 350 previously passing upstream checkpoint/remat/barrier cases still pass
(`shim/build/barrier-passing-verification/`, peak footprint 0.51 GB).

## Decomposition crashes: loop arity and edge padding — resolved

Of the next 106-crash group, 91 cases exercise SVD, QDWH and spectral
divide-and-conquer. Two independent native defects caused their crashes:

- SCF canonicalization removes unused loop results and their body arguments,
  while retaining inputs needed by the condition. The resulting SCF is valid,
  but MPSGraph crashes on the unequal input/output arities. Returning all four
  QDWH outputs works; returning only its two matrices crashes. Native compiler
  dumps show the final convergence loop changing from three inputs/three
  outputs to three inputs/two outputs (`qdwh-mps-stage-{4,5}.log`).
- Complex QDWH's padded QR update crashes the GPU driver in
  `AGXG15GFamilyComputeContext::encodeEndWhile`. Replacing native edge padding
  with concatenation, keeping the same dynamic update and QR, removes the
  crash. `_indexing.py` now expresses all edge padding as concatenation/cropping,
  extending the earlier narrow-integer repair to floating types as well.

`_compiler_ir.py` joins changed loop results with a shared packed barrier.
Invariant captures are excluded. A result that is used in the initial module
may become dead after function inlining: this occurs inside the eigensolver
for 300×300 SVD. Detection also covers unused body arguments, so those results
remain protected when their current consumers disappear. Recognition of the
packed dependency makes preparation idempotent without adding attributes that
the plugin's old StableHLO reader cannot deserialize.

The first isolated rerun passed 90/91 decomposition cases
(`shim/build/decomposition-crash-verification/`). After extending protection
across inlining, the remaining upstream case,
`QdwhTest::testQdwhWithRandomMatrix1`, passes with all three subtests
(`shim/build/qdwh-large-final.log`). Its 300×300 SVD reconstruction error is
6.2e-6. All **282 permanent regression tests pass together**, including new
real/complex QDWH, unused convergence state, large SVD accuracy and floating
padding bit-pattern checks (`shim/build/compiler-fixes-regressions.log`;
peak footprint 0.79 GB, worst GPU probe latency 2 ms).

## Sparse empty arrays — resolved

Twelve cases in the crash group exercise sparse concatenation, reversal and
slicing. Layouts with no sparse index components have zero-sized index arrays.
Apple's `RemoveZeroDim` pass removes empty computations and outputs but leaves
empty input arguments in the MPS function. At execution, MPSGraph aborts in
`convertMPSGraphShapesToMLIRTypes`. A dense reproducer returning concatenated
nonempty data alongside concatenated empty indices has the same failure;
returning a constant empty output without an empty input succeeds.

`_boundary_io.py` now represents empty inputs as unused scalar uint8 arguments
and reconstructs their logical values from static metadata. The C++ shim
supplies one-byte placeholders, preserving argument positions and retaining
them until execution completes. Existing empty-output handling works unchanged,
including complex and narrow types. There is no host numerical computation.

Two sparse slices also hit native conversion of empty scatters: a zero-sized
update window becomes size one and exceeds the empty operand dimension, or
crashes conversion. Scatters with empty operands or updates now return their
operand before entering the native converter.

All **12 original sparse cases pass together**
(`shim/build/sparse-crash-final-verification/`, peak 0.34 GB, worst GPU probe
2 ms). All 42 new empty-array regressions and 20 existing barrier tests pass
together (`shim/build/empty-array-regressions.log`), covering seven dtypes,
donation, gradients, sparse layouts and all five scatter combiners. The C++
fake-plugin boundary test also verifies the scalar placeholder and unchanged
logical input metadata.

## Optional compiled-memory statistics — resolved

Two crashes, under `jit` and `pmap`, call `memory_analysis()`. The requested
`PJRT_Executable_GetCompiledMemoryStats` entry is at byte offset 808, exactly
beyond Apple's 808-byte API table. Current jaxlib calls the null slot without
checking it. The plugin binary contains an unregistered implementation that
only returns UNIMPLEMENTED; it provides no memory statistics.

The shim installs an entry returning a readable UNIMPLEMENTED error. JAX's
documented public behavior then returns `None`; no fabricated byte counts are
reported. Error code/message/destruction wrappers recognize this immutable
shim error and forward other errors to Apple. Both upstream cases and two
permanent regressions pass (`shim/build/memory-analysis-verification.log`),
including repeated calls, direct runtime errors and execution after querying.
The C++ fake-plugin test checks the error contract independently.

## Backend reset leaves a dangling native compute stream — resolved

The final case clears JAX's backends, creates a fresh one, then executes `pmap`.
Its native stack crashes in `metal_plugin::memcpy_htod + 140`, at the virtual
call through `MetalStream::gComputeStream` (binary offset `0x5d60f90`). The
pointer survives destruction of its owning client and refers to freed memory.
The transfer worker dereferences it while preparing the next host-to-GPU copy.

The shim keeps one native client across framework client lifetimes and releases
it at process exit, before the plugin's static teardown. JAX still recreates its
own client/device wrappers and clears its caches. This retains one native
allocator/stream across resets, rather than leaking a new client on each reset.
Apple's `PJRT_Client_Create` ignores create options and calls `GetMetalClient()`
using process-wide configuration, so sharing does not discard per-client options.

The upstream test passes (`shim/build/clear-backends-verification.log`). The
three runtime API regressions pass together, including five reset cycles with
mapped execution and complex/empty boundary values; peak footprint is 0.17 GB
and final teardown logs one MetalClient destruction
(`shim/build/runtime-api-regressions.log`). The C++ fake-plugin test verifies
that five create/destroy cycles allocate exactly one native client.

The combined rerun of all **106 original crash cases passes**
(`shim/build/crash-final-verification.log`; peak footprint 0.41 GB, worst GPU
probe latency 32 ms).

## Complex values in ops that emit real/imag directly — resolved (96 of 97)

The next group is 97 `failed to legalize operation 'mhlo.real'` failures. The
plugin has no `real`/`imag`/`complex` patterns, so `_complex.py` builds them
from converts and multiplications by ±i. Those rules replace the
`real_p`/`imag_p`/`complex_p` primitives, but some JAX lowerings emit
`stablehlo.real`/`imag` themselves, or emit complex ops that XLA's serializer
expands into them. Lowering representative cases for METAL (no compilation, no
GPU work) and reading the debug locations of the `real`/`imag` ops found five
sources:

- `max_p`/`min_p`: `mlir._minmax_hlo` orders complex values by real part, then
  imaginary part, with `hlo.real`/`imag`. This covers `jnp.maximum`/`minimum`,
  `logaddexp` (which uses `max`) and reductions with `initial` or `where` (the
  reduction is combined with the initial value by `max`). 45 cases.
- `exp2_p`: lowered to a complex exponential, which the serializer expands into
  `real`/`imag`.
- `log2_p`: `_log2_lower` scales the parts of the complex log with
  `hlo.real`/`imag`.
- `square_p`: `chlo.square`, whose complex decomposition uses `real`/`imag`.
  The second derivative of `tan` (`1 + tan²`) and the derivative of `sinc` also
  reach it.
- complex→bfloat16 `convert` (`nanmean`, `cumprod` with `dtype=bfloat16`): the
  shim's convert rule recognized real targets by `kind in "biuf"`, which
  excludes ml_dtypes' bfloat16 (kind `V`), so JAX's rule ran instead.

`_complex._chooser` implements complex `max`/`min` with the METAL `real`/`imag`,
comparisons and selects, following `_minmax_hlo` exactly. On real-part ties it
compares imaginary parts, and a NaN comparison is false, so it picks the second
operand. Real inputs keep the NaN-propagating rule. Complex `exp2` and `log2`
follow JAX's formulas. Complex `square` follows CHLO's decomposition, which XLA's
CPU backend matches bit for bit on about 60,000 checked inputs (finite ones,
signed zeros and infinities, with subnormal results flushed as on the CPU):

- real part: 0 where `|a| = |b|` and `a` is finite, else `(a − b)(a + b)`;
- imaginary part: `2(ab)`.

Plain `(a − b)(a + b)` disagrees on 12% of those inputs. The convert rule
splits any complex→non-complex conversion through the METAL `real`.

Verification:

- On the CPU with the METAL rules, all 97 cases pass.
- On the M3, 96 of 97 pass in guarded batches of eight
  (`shim/build/real-verification/`; peak footprint 0.32 GB, worst GPU probe
  latency 3 ms).
- `shim/tests/test_complex.py` gains exact METAL-vs-CPU checks:
  - complex `max`/`min` over all 1,296 pairs of a value grid, real-part ties,
    scalar operands, `initial`, `where` and loops;
  - `square` over a special-value grid and at `|re| = |im|`;
  - complex → bfloat16/float16/float32 converts;
  - tolerance checks for `exp2`, `log2`, `logaddexp`, `logaddexp2`, their
    gradients and the second derivative of `tan`.

  All pass, as do the other permanent regression files
  (`shim/build/regressions-real-fix.log`).

### Remaining: complex values with non-finite or negative-zero components

`testSincInfinities` now compiles, but returns NaN for `sinc(±inf + bj)`
instead of 0. `jnp.sinc` tests `|re(x)| = inf` and `isfinite(im(x))`. On METAL,
`imag(inf + 0j)` is NaN, because the shim computes it as `real(z·(−i))`, whose
real part is `a·0 + b`.

This is the known limitation recorded under ordering above, and it is broader
than recorded:

- Construction `convert(x) + convert(y)·i` computes `y·0`, so `complex(1, inf)`
  gets a NaN real part.
- Construction and extraction both drop the sign of a zero component.
- Program inputs and outputs use the same rules. An identity function on
  complex64 changes 77 of 144 grid values; all of them have a −0 or non-finite
  component.

Probes on the M3, with complex constants made inside the program and only real
outputs, show which operations MPS keeps exact:

- complex→real `convert` (the real part, NaN and infinities included);
- real→complex `convert`, negation, addition and `select` on complex values.

Every multiplication that isolates a component computes some `a·0`. StableHLO's
verifier rejects bitcasts between complex and real types ("cannot convert
between real and complex types"), so no exact data movement is available
either.

Exact construction appears feasible from the exact operations. For example,
`(x, −0) = −convert(−x)` plus `(−0, y) = −((−y, −0)·i)` for finite `y`, plus
constants `(−0, ±inf/NaN)` selected for non-finite `y`, since `−0` is the
additive identity. Extraction of the imaginary part cannot be exact when the
real part is infinite or NaN, unless the components are still available. That
is true at the program boundary, where inputs arrive as float32 pairs. This
remains open and is the natural next step for complex edge cases. Later
baseline failure groups (`reduce_window`, token arguments, multi-op reductions,
`mhlo.map`, ...) have not yet been taken up.
