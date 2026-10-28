"""Reductions whose MPS implementations do not preserve StableHLO semantics.

Apple's ReduceConverter takes one operand and a body of one combining op. A variadic
reduce (jnp's min-and-max _reduce_minmax, reductions over pytrees) or a body of several ops
(argmin/argmax-style pairs, custom combiners) fails to legalize. A body that combines each
accumulator with its own element by one monoid op splits into one reduction per operand;
any other body is applied pairwise in a log-depth tree, which XLA's reduce semantics allow
(the body is taken to be associative and commutative).
"""

from functools import partial

import jax
import numpy as np
from jax import lax
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal

from jax_plugins.metal_shim import _complex


def _bitwise_reduce(x, *, axes, combine, identity):
    kept = tuple(d for d in range(x.ndim) if d not in axes)
    shape = tuple(x.shape[d] for d in kept)
    count = int(np.prod([x.shape[d] for d in axes]))
    if count == 0:
        if x.dtype == np.bool_:
            value = bool(identity)
        else:
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


def _bitwise_or_bool(dtype):
    """Integers (whose and/or MPS reduces as truth values) and bools: on a mask that is
    constant after folding (gather index checks, a constant `where=`), the plugin's MHLO
    canonicalizer leaves a native and/or reduction whose body only returns the init value,
    which its converter rejects."""
    return np.issubdtype(dtype, np.integer) or dtype == np.bool_


def _bitwise_rule(primitive, combine, identity):
    default = _complex.current_rule(primitive)
    tree = mlir.lower_fun(partial(_bitwise_reduce, combine=combine, identity=identity),
                          multiple_results=False)

    def rule(ctx, operand, **params):
        if _bitwise_or_bool(ctx.avals_in[0].dtype):
            return tree(ctx, operand, **params)
        return default(ctx, operand, **params)

    return rule


# Monoid ops a reduce body may apply per operand: the reduction by each (through the METAL
# rules of its primitive, NaN propagation and integer bitwise semantics included).
_SPLITTABLE = {lax_internal.add_p: (lax.reduce_sum, lax.add),
               lax_internal.mul_p: (lax.reduce_prod, lax.mul),
               lax_internal.max_p: (lax.reduce_max, lax.max),
               lax_internal.min_p: (lax.reduce_min, lax.min),
               lax_internal.and_p: (lax.reduce_and, lax.bitwise_and),
               lax_internal.or_p: (lax.reduce_or, lax.bitwise_or),
               lax_internal.xor_p: (lax.reduce_xor, lax.bitwise_xor)}


def _per_operand_ops(body):
    """The op combining accumulator i with element i, for each output i, if the body is
    exactly one such monoid op per output; else None."""
    n = len(body.outvars)
    if len(body.invars) != 2 * n or len(body.eqns) != n:
        return None
    ops = []
    for i, out in enumerate(body.outvars):
        eqn = next((e for e in body.eqns if e.outvars == [out]), None)
        if (eqn is None or eqn.primitive not in _SPLITTABLE or eqn.params
                or set(eqn.invars) != {body.invars[i], body.invars[n + i]}):
            return None
        ops.append(eqn.primitive)
    return ops


def _split_reduce(*args, ops, dimensions):
    n = len(args) // 2
    results = []
    for x, init, op in zip(args[:n], args[n:], ops):
        reduce, combine = _SPLITTABLE[op]
        reduced = reduce(x, tuple(dimensions))
        results.append(combine(lax.broadcast(init, reduced.shape), reduced))
    return results


def _tree_reduce(*args, jaxpr, consts, dimensions):
    """The body applied pairwise over the reduced elements, then to the init values."""
    n = len(args) // 2
    xs, inits = args[:n], args[n:]
    kept = tuple(d for d in range(xs[0].ndim) if d not in dimensions)
    shape = tuple(xs[0].shape[d] for d in kept)
    count = int(np.prod([xs[0].shape[d] for d in dimensions]))
    accs = [lax.broadcast(init, shape) for init in inits]
    body = getattr(jaxpr, "jaxpr", jaxpr)

    def mapped(depth):  # the scalar body, elementwise over `depth` dimensions
        fn = partial(core.eval_jaxpr, body, consts)
        for _ in range(depth):
            fn = jax.vmap(fn)
        return fn

    if count == 0:
        return accs
    xs = [lax.reshape(lax.transpose(x, (*kept, *dimensions)), (*shape, count)) for x in xs]
    pair = mapped(len(shape) + 1)
    while count > 1:
        half = count // 2
        left = [lax.slice_in_dim(x, 0, half, axis=-1) for x in xs]
        right = [lax.slice_in_dim(x, half, 2 * half, axis=-1) for x in xs]
        merged = pair(*left, *right)
        if count % 2:
            merged = [lax.concatenate([m, lax.slice_in_dim(x, 2 * half, count, axis=-1)],
                                      x.ndim - 1) for m, x in zip(merged, xs)]
        xs, count = merged, (count + 1) // 2
    return mapped(len(shape))(*accs, *[lax.reshape(x, shape) for x in xs])


def register(register_rule):
    # MPS's reduction_and/or kernels reduce truth values, even on integers; bools see
    # _bitwise_or_bool.
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
        if (len(args) == 2 and _bitwise_or_bool(ctx.avals_in[0].dtype)
                and len(body.eqns) == 1 and body.eqns[0].primitive in combiners
                and list(body.eqns[0].invars) == list(body.invars)
                and list(body.eqns[0].outvars) == list(body.outvars)):
            combine, identity = combiners[body.eqns[0].primitive]

            def reduce(x, init):
                result = _bitwise_reduce(x, axes=dimensions, combine=combine, identity=identity)
                return [combine(result, lax.broadcast_in_dim(init, result.shape, ()))]

            return mlir.lower_fun(reduce, multiple_results=True)(ctx, *args)
        if (len(args) == 2 and len(body.eqns) == 1  # one operand, one op: the plugin's form
                and not _complex.is_complex_aval(ctx.avals_in[0])):
            return default(ctx, *args, computation=computation, jaxpr=jaxpr, dimensions=dimensions)
        from jax_plugins.metal_shim._lowerings import outside_region  # pylint: disable=import-outside-toplevel
        consts = getattr(jaxpr, "consts", ())
        ops = None if consts else _per_operand_ops(body)
        with outside_region():  # these reductions have no region body
            if ops is not None:
                return mlir.lower_fun(partial(_split_reduce, ops=ops, dimensions=dimensions),
                                      multiple_results=True)(ctx, *args)
            return mlir.lower_fun(partial(_tree_reduce, jaxpr=jaxpr, consts=consts,
                                          dimensions=dimensions),
                                  multiple_results=True)(ctx, *args)

    register_rule(lax_internal.reduce_p, general)
