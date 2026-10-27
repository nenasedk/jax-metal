"""Reductions whose MPS implementations do not preserve StableHLO semantics."""

from functools import partial

import numpy as np
from jax import lax
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal

from jax_plugins.metal_shim import _complex


def _bitwise_reduce(x, *, axes, combine, identity):
    kept = tuple(d for d in range(x.ndim) if d not in axes)
    shape = tuple(x.shape[d] for d in kept)
    count = int(np.prod([x.shape[d] for d in axes]))
    if count == 0:
        value = np.iinfo(x.dtype).max if identity == -1 and x.dtype.kind == 'u' else identity
        return lax.full(shape, value, x.dtype)
    x = lax.reshape(lax.transpose(x, (*kept, *axes)), (*shape, count))
    while count > 1:
        half = count // 2
        left = lax.slice_in_dim(x, 0, half, axis=-1)
        right = lax.slice_in_dim(x, half, 2 * half, axis=-1)
        merged = combine(left, right)
        x = (lax.concatenate([merged, lax.slice_in_dim(x, 2 * half, count, axis=-1)],
                             x.ndim - 1) if count % 2 else merged)
        count = (count + 1) // 2
    return lax.reshape(x, shape)


def _bitwise_rule(primitive, combine, identity):
    default = _complex.current_rule(primitive)
    tree = mlir.lower_fun(partial(_bitwise_reduce, combine=combine, identity=identity),
                          multiple_results=False)

    def rule(ctx, operand, **params):
        if np.issubdtype(ctx.avals_in[0].dtype, np.integer):
            return tree(ctx, operand, **params)
        return default(ctx, operand, **params)

    return rule


def register(register_rule):
    # MPS's reduction_and/or kernels reduce truth values, even on integers.
    for primitive, combine, identity in (
            (lax_internal.reduce_and_p, lax.bitwise_and, -1),
            (lax_internal.reduce_or_p, lax.bitwise_or, 0),
            (lax_internal.reduce_xor_p, lax.bitwise_xor, 0)):
        register_rule(primitive, _bitwise_rule(primitive, combine, identity))
    default = _complex.current_rule(lax_internal.reduce_p)
    combiners = {lax_internal.and_p: (lax.bitwise_and, -1),
                 lax_internal.or_p: (lax.bitwise_or, 0),
                 lax_internal.xor_p: (lax.bitwise_xor, 0)}

    def general(ctx, *args, computation, jaxpr, dimensions):
        body = getattr(jaxpr, "jaxpr", jaxpr)
        if (len(args) == 2 and np.issubdtype(ctx.avals_in[0].dtype, np.integer)
                and len(body.eqns) == 1 and body.eqns[0].primitive in combiners
                and list(body.eqns[0].invars) == list(body.invars)
                and list(body.eqns[0].outvars) == list(body.outvars)):
            combine, identity = combiners[body.eqns[0].primitive]

            def reduce(x, init):
                result = _bitwise_reduce(x, axes=dimensions, combine=combine, identity=identity)
                return [combine(result, lax.broadcast_in_dim(init, result.shape, ()))]

            return mlir.lower_fun(reduce, multiple_results=True)(ctx, *args)
        return default(ctx, *args, computation=computation, jaxpr=jaxpr, dimensions=dimensions)

    register_rule(lax_internal.reduce_p, general)
