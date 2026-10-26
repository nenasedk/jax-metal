# NaN in max, min and clamp on METAL

Question: do the plugin's max/min ops propagate NaN, as XLA's do? On METAL, `jnp.linalg.cond`
of a singular matrix returned 0 instead of inf (`NumpyLinalgTest.testCondSingular*`), which a
max that drops NaN would explain: `cond` computes `norm(A) * norm(inv(A))` and turns a NaN
result into inf, but a NaN-dropping max in `norm` never produces the NaN.

Answer, from the plugin binary alone: **no**. Reductions are built with NaN propagation
explicitly off, and elementwise max, min and clamp map one-to-one onto MPS's
non-propagating ops. The shim now restores XLA's behavior in its lowering rules
(`shim/python/jax_plugins/metal_shim/_lowerings.py`, "NaN propagation in max/min").

Target: `pjrt_plugin_metal_14.dylib` from jax-metal 0.1.1
(sha256 `e8e0656f24dc887507f5719926faf51f65546993809d3059add1a3c6c16961bf`), disassembled on
Linux with LLVM 18 (`llvm-nm`, `llvm-objdump`; the file is a plain arm64 Mach-O, `__text` at
file offset = address 0x6c00).

## The MPS dialect

The plugin lowers MHLO to Apple's `mps` MLIR dialect, serializes that module and runs it with
MPSGraph (`MPSGraphExecutable`), so what an `mps` op does with NaN is decided by MPSGraph. The
dialect has:

- `mps.maximum`, `mps.minimum`, `mps.clamp`: no NaN-related attribute (no `getPropagateNans`
  accessor; the dialect's operation list in `Dialect::addOperations<...>` has no separate
  NaN-propagating maximum or minimum).
- `mps.reduction_max`, `mps.reduction_min`: an optional unit attribute `propagate_nans`
  (`ReductionMaxOp::setPropagateNans(bool)`, `getPropagateNansAttr`, properties
  `keep_dims` at offset 0 and `propagate_nans` at offset 8). Absent means NaN is not
  propagated.
- `mps.reduction_argmax`, `mps.reduction_argmin`: no NaN attribute.

This mirrors MPSGraph's API, which pairs each op with a NaN-propagating variant
(`maximumWithNaNPropagationWithPrimaryTensor:`, `minimumWithNaNPropagation...`,
`reductionMaximumPropagateNaNWithTensor:`, `reductionMinimumPropagateNaN...`) and documents
the plain ones as not propagating NaN.

## Where the plugin builds them

A scan of `__text` for BL instructions to each builder (`re/scripts/bl_callers.py`) finds one
call site each, and none for `setPropagateNans` or the other `build` overloads:

| Builder | Caller |
|---|---|
| `OpBuilder::create<mps::ReductionMaxOp, Value, ConstantOp&, bool&, bool>` (0x5b37c) | 0x5a4ac in `mps::ReduceConverter::matchAndRewrite(mhlo::ReduceOp)` |
| `OpBuilder::create<mps::ReductionMinOp, ...>` (0x5b208) | 0x5a7a8, same function |
| `create<mps::ReductionArgMaxOp>` / `ArgMinOp` | 0x5a574 / 0x5a678, same function |
| `create<mps::MaximumOp>` (0x3ce10) | 0x3cbec in `GeneratedConvert39::matchAndRewrite` (reads `mhlo::MaxOp`) |
| `create<mps::MinimumOp>` (0x3d828) | 0x3d604 in `GeneratedConvert40` |
| `create<mps::ClampOp>` (0x414f8) | 0x412cc in `GeneratedConvert46` |

`GeneratedConvertNN` are tablegen (DRR) patterns: operands, result types and attributes of
the MHLO op are copied onto the MPS op, with nothing added for NaN.

The reduction call site passes `false` for `propagate_nans`. The template's arguments are
forwarding references, so each is passed by address; the last one, x5, points at a byte the
caller has just zeroed:

```
5a490:  strb  wzr, [sp, #0x8]      ; bool propagate_nans = false
5a494:  add   x2, sp, #0x18        ; Value input
5a498:  add   x3, sp, #0x30        ; ConstantOp axes
5a49c:  add   x4, sp, #0x10        ; bool keep_dims
5a4a0:  add   x5, sp, #0x8         ; bool propagate_nans
5a4ac:  bl    create<mps::ReductionMaxOp, ...>
```

(identical at 0x5a78c..0x5a7a8 for `ReductionMinOp`). `create<>` forwards the two bools as
`ldrb w4, [x21]` / `ldrb w5, [x20]` to `ReductionMaxOp::build(..., Value, Value, bool, bool)`
(0x21781c), which sets the property at offset 0 (`keep_dims`) from w4 and the one at offset 8
(`propagate_nans`) from w5, each only when the bool is true:

```
217880:  cbz  w22, 0x2178d8        ; keep_dims == false: skip
...
2178d4:  str  x22, [x8]            ; properties.keep_dims = UnitAttr
2178d8:  cbz  w21, 0x217930        ; propagate_nans == false: skip
...
21792c:  str  x21, [x8, #0x8]      ; properties.propagate_nans = UnitAttr
```

## Outside the binary

PyTorch's MPS backend ran into the same behavior and moved to the NaN-propagating MPSGraph
calls: `clamp` "doesn't propagate NaN through so simulate it as composition of
maximumWithNaNPropagationWithPrimaryTensor and minimumWithNaNPropagationWithPrimaryTensor"
([TensorCompare.mm, v2.3.0](https://github.com/pytorch/pytorch/blob/v2.3.0/aten/src/ATen/native/mps/operations/TensorCompare.mm)),
and its max/min reductions use `reductionMaximumPropagateNaNWithTensor` /
`reductionMinimumPropagateNaNWithTensor`
([ReduceOps.mm, v2.5.0](https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/mps/operations/ReduceOps.mm);
v2.1.0 still used the plain ones).

## What the shim does

For floating-point inputs, the METAL rules run JAX's lowering and then put NaN back:

- `max`, `min`: the first NaN operand wherever there is one; `clamp`: NaN where the operand is
  NaN;
- `reduce_max`, `reduce_min`: NaN where the reduced slice has a NaN;
- `argmax`, `argmin`: the index of the first NaN (XLA's and NumPy's result);
- `reduce_window_max`, `reduce_window_min` (max/min pooling): NaN for windows holding a NaN,
  found by max pooling the 0/1 NaN mask.

`cummax`/`cummin` already go through `max`/`min` (prefix scan rule). Integer inputs are left
alone. The cost is a compare and a select per input (elementwise), or one extra reduction.

Checked on the CPU backend through the rules (they are no-ops where max already propagates
NaN, and the fix-up functions repair NaN-dropping results); on METAL, the op probe cases
`nan_max_min`, `nan_pool_clip_cummax` and `singular_inv_cond` (`shim/tests/probe_jax_ops.py`)
check it.

Not answerable from the plugin: division by zero and other IEEE corner cases are decided by
the Metal kernels MPSGraph generates, not by this lowering. The probe case `divide_by_zero`
covers them.

## Reproducing

```sh
D=pjrt_plugin_metal_14.dylib   # python -m jax_plugins.metal_shim info prints its path
llvm-nm-18 "$D" | grep -E 'createINS_3mps14ReductionMaxOp|ReductionMaxOp16setPropagateNans'
python re/scripts/bl_callers.py "$D" 0x6c00 0x6c00 0x4cb35bc 5b37c 5b208 3ce10 3d828 414f8
llvm-objdump-18 -d --no-show-raw-insn --start-address=0x5a470 --stop-address=0x5a4b4 "$D"
llvm-objdump-18 -d --no-show-raw-insn --start-address=0x21781c --stop-address=0x217934 "$D"
```
