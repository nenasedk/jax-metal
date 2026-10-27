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

Later baseline failure groups remain to be investigated.
