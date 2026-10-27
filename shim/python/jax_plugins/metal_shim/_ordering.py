"""Preserve JAX's ordering and payloads through MPS sort and comparisons."""

from functools import partial

import numpy as np
from jax import lax
from jax._src import dtypes
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal

from jax_plugins.metal_shim import _complex


def _float_order_key(x):
    """Signed integer key for IEEE totalOrder, including signed zeros and NaNs."""
    bits = np.dtype(x.dtype).itemsize * 8
    value = lax.bitcast_convert_type(x, np.dtype(f'int{bits}'))
    if bits < 32:
        value = lax.convert_element_type(value, np.int32)
    return lax.select(value < 0, lax.bitwise_xor(value, lax.full_like(value, (1 << (bits - 1)) - 1)),
                      value)


def _total_order_rule(primitive, compare):
    default = _complex.current_rule(primitive)
    ordered = mlir.lower_fun(lambda x, y: compare(_float_order_key(x), _float_order_key(y)),
                            multiple_results=False)

    def rule(ctx, *args, **params):
        if dtypes.issubdtype(ctx.avals_in[0].dtype, np.floating):
            return ordered(ctx, *args, **params)
        return default(ctx, *args, **params)

    return rule


def _sort_inputs(key, *, dimension):
    # Apple's comparator ignores JAX's zero/NaN canonicalization inside the
    # comparator region. Do it on the key tensor before entering the sort.
    if dtypes.issubdtype(key.dtype, np.floating):
        key = lax_internal._canonicalize_float_for_sort(key)
    return key, lax.broadcasted_iota(np.int32, key.shape, dimension)


def _permute(order, *operands, dimension):
    batch_dims = tuple(d for d in range(order.ndim) if d != dimension)
    dnums = lax.GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(dimension,), start_index_map=(dimension,),
        operand_batching_dims=batch_dims, start_indices_batching_dims=batch_dims)
    return [lax.gather(x, lax.expand_dims(order, (order.ndim,)), dnums,
                       (1,) * order.ndim, mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
            for x in operands]


def _sort_rule():
    default = _complex.current_rule(lax_internal.sort_p)

    def rule(ctx, *args, dimension, is_stable, num_keys):
        if num_keys > 1 or any(_complex.is_complex_aval(a) for a in ctx.avals_in):
            return default(ctx, *args, dimension=dimension, is_stable=is_stable,
                           num_keys=num_keys)
        dimension %= ctx.avals_in[0].ndim
        key_aval = ctx.avals_in[0]
        index_aval = key_aval.update(dtype=np.dtype(np.int32))
        sort_avals = [key_aval, index_aval]
        key, indices = mlir.lower_fun(partial(_sort_inputs, dimension=dimension),
                                     multiple_results=True)(
            ctx.replace(avals_in=[key_aval], avals_out=sort_avals, tokens_out=None), args[0])
        _, order = default(ctx.replace(avals_in=sort_avals, avals_out=sort_avals,
                                       tokens_out=None), key, indices,
                           dimension=dimension, is_stable=is_stable, num_keys=1)
        # MPS rounds returned integer keys through float32 (e.g. int32 max
        # becomes int32 min). Its permutation is correct; gather original data.
        return mlir.lower_fun(partial(_permute, dimension=dimension), multiple_results=True)(
            ctx.replace(avals_in=[index_aval, *ctx.avals_in], tokens_out=None), order, *args)

    return rule


def register(register_rule):
    for primitive, compare in ((lax_internal.eq_to_p, lax.eq),
                               (lax_internal.lt_to_p, lax.lt),
                               (lax_internal.le_to_p, lax.le)):
        register_rule(primitive, _total_order_rule(primitive, compare))
    register_rule(lax_internal.sort_p, _sort_rule())
