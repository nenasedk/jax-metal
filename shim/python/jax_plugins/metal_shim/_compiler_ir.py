"""Repair compiler IR before handing it to Apple's plugin.

OptBarrierConverter replaces an entire barrier with one mps.identity of operand
zero, leaving all remaining results null. Packing tensors into one buffer keeps
the all-inputs-to-all-outputs dependency; independent unary barriers would not.

The plugin also canonicalizes trivial scf.if branches into arith.select, which
its serializer cannot handle. Unary barriers on branch results keep these
branches in the supported control-flow dialect without evaluating both sides.

SCF canonicalization can also discard a loop result whose corresponding body
argument is unused (for example a convergence flag only read by the condition).
The resulting loop has different input/output arities and crashes MPSGraph.
Keeping these results linked to the consumed state prevents that rewrite.

Rewrite a compilation copy, including barriers emitted directly by remat's
lowering. The public lowered module retains JAX's original structure.
"""

from functools import wraps
from math import prod

from jax._src import compiler
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo


def _walk(operation):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                child = child.operation
                yield child
                yield from _walk(child)


def _barriers(operation):
    return (op for op in _walk(operation)
            if op.name == "stablehlo.optimization_barrier" and len(op.operands) > 1)


def _unprotected_returns(operation):
    for region in operation.regions:
        for block in region.blocks:
            terminator = list(block.operations)[-1].operation
            for index, value in enumerate(terminator.operands):
                if not isinstance(value.type, ir.RankedTensorType):
                    continue
                if (isinstance(value, ir.OpResult)
                        and value.owner.name == "stablehlo.optimization_barrier"
                        and value.owner.operation.block == block):
                    continue
                yield terminator, index, value


def _cases(operation):
    return (op for op in _walk(operation)
            if op.name == "stablehlo.case" and any(_unprotected_returns(op)))


def _loops(operation):
    for op in _walk(operation):
        if op.name != "stablehlo.while":
            continue
        body = op.regions[1].blocks[0]
        returned = list(body.operations)[-1].operands
        if (any(list(result.uses) for result in op.results)
                and any(arg != ret and not list(result.uses)
                        for arg, ret, result in zip(body.arguments, returned, op.results))):
            yield op


def _protect_loop_results(operation):
    body = operation.regions[1].blocks[0]
    returned = list(body.operations)[-1].operands
    # Invariant captures can be removed safely and need not be copied. Changed
    # state includes the otherwise-unused status and a result consumed outside.
    values = [r for r, arg, ret in zip(operation.results, body.arguments, returned)
              if arg != ret]
    if not any(list(value.uses) for value in values):
        values = list(operation.results)
    uses = [list(value.uses) for value in values]
    with ir.InsertionPoint(operation), operation.location:
        barrier = hlo.OptimizationBarrierOp(values)
    barrier.operation.move_after(operation)
    for old_uses, value in zip(uses, barrier.results):
        for use in old_uses:
            use.owner.operands[use.operand_number] = value


def _tensor(shape, element):
    return ir.RankedTensorType.get(shape, element)


def _encode(value, complex_carrier):
    original = ir.RankedTensorType(value.type)
    element = original.element_type
    if isinstance(element, ir.ComplexType):
        encoded = value
    else:
        # Bytes keep every float NaN payload, signed zero and wide integer bit.
        width = element.width
        byte_shape = (*original.shape, width // 8) if width > 8 else original.shape
        byte_type = _tensor(byte_shape, ir.IntegerType.get_unsigned(8))
        encoded = (hlo.convert(byte_type, value) if width == 1
                   else hlo.bitcast_convert(byte_type, value))
        if complex_carrier:
            # Complex cannot be bitcast to real in StableHLO. Keep complex inputs
            # intact, and embed other inputs' bytes as exact real values 0..255.
            encoded = hlo.convert(_tensor(byte_shape, complex_carrier), encoded)
    shape = ir.RankedTensorType(encoded.type).shape
    flat = hlo.reshape(_tensor((prod(shape),), ir.RankedTensorType(encoded.type).element_type),
                       encoded)
    return flat, encoded.type


def _decode(value, encoded_type, original_type, complex_carrier):
    value = hlo.reshape(encoded_type, value)
    element = ir.RankedTensorType(original_type).element_type
    if isinstance(element, ir.ComplexType):
        return value
    if complex_carrier:
        byte_type = _tensor(ir.RankedTensorType(encoded_type).shape,
                            ir.IntegerType.get_unsigned(8))
        value = hlo.convert(byte_type, value)
    return (hlo.convert(original_type, value) if element.width == 1
            else hlo.bitcast_convert(original_type, value))


def _rewrite_barrier(operation):
    operands = list(operation.operands)
    complex_types = [ir.RankedTensorType(v.type).element_type for v in operands
                     if isinstance(ir.RankedTensorType(v.type).element_type, ir.ComplexType)]
    complex_carrier = complex_types[0] if complex_types else None
    with ir.InsertionPoint(operation), operation.location:
        encoded = [_encode(v, complex_carrier) for v in operands]
        packed = hlo.concatenate([v for v, _ in encoded], 0)
        packed = hlo.OptimizationBarrierOp([packed]).results[0]
        offset = 0
        for result, (value, encoded_type) in zip(operation.results, encoded):
            size = ir.RankedTensorType(value.type).shape[0]
            part = hlo.slice(packed, [offset], [offset + size], [1])
            result.replace_all_uses_with(_decode(part, encoded_type, result.type, complex_carrier))
            offset += size
    operation.erase()


def prepare_module(module):
    """Return a private compilation module with supported IR, if needed."""
    if (not any(_barriers(module.operation)) and not any(_cases(module.operation))
            and not any(_loops(module.operation))):
        return module
    with module.context:
        result = ir.Module.parse(mlir.module_to_bytecode(module))
        for operation in list(_loops(result.operation)):
            _protect_loop_results(operation)
        for operation in list(_barriers(result.operation)):
            _rewrite_barrier(operation)
        for operation in list(_cases(result.operation)):
            for terminator, index, value in list(_unprotected_returns(operation)):
                with ir.InsertionPoint(terminator), terminator.location:
                    terminator.operands[index] = hlo.OptimizationBarrierOp([value]).results[0]
        result.operation.verify()
    return result


def register():
    original = compiler.compile_or_get_cached

    @wraps(original)
    def compile_(backend, computation, *args, **kwargs):
        if backend.platform == "METAL":
            computation = prepare_module(computation)
        return original(backend, computation, *args, **kwargs)

    compiler.compile_or_get_cached = compile_
