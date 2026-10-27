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

In total, 144 of the original 183 numerical assertions now pass (21 from
boolean scatter, 104 from interior padding, and 19 from index clamping).
The remaining baseline failure groups have not yet been resolved.
