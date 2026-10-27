"""Indexing rules for differences between StableHLO and Apple's MPS lowering."""

from functools import partial

import numpy as np
import jax
from jax import lax
from jax._src import core, dtypes
from jax._src.interpreters import mlir
from jax._src.lax import slicing
from jax._src.lax import lax as lax_internal

from jax_plugins.metal_shim import _complex


def _clamp_indices(indices, bounds):
    # Clamp before changing the index type: a large unsigned index must go to the
    # upper bound, not wrap around and look negative in a signed carrier.
    bounds = np.minimum(bounds, np.iinfo(indices.dtype).max).astype(indices.dtype)
    bounds = lax.broadcast_in_dim(bounds, indices.shape,
                                  tuple(range(indices.ndim - bounds.ndim, indices.ndim)))
    return lax.min(lax.max(indices, lax.full_like(indices, 0)), bounds)


def _gather_rule():
    default = _complex.current_rule(slicing.gather_p)

    def rule(ctx, operand, indices, *, dimension_numbers, slice_sizes, mode, **params):
        dtype = ctx.avals_in[0].dtype
        carrier = (np.float32 if dtype in (np.dtype(np.int32), np.dtype(np.uint32)) else
                   np.uint16 if dtypes.issubdtype(dtype, np.floating) and dtype.itemsize == 2
                   else None)
        if carrier is not None and mode != lax.GatherScatterMode.FILL_OR_DROP:
            def exact(x, i):
                # MPS gather converts integer payloads through float32. Move their
                # bit patterns as floats instead, preserving every integer bit.
                # Conversely, 16-bit float NaNs survive a uint16 carrier exactly.
                value = lax.gather(lax.bitcast_convert_type(x, carrier), i,
                                   dimension_numbers, slice_sizes, mode=mode, **params)
                return lax.bitcast_convert_type(value, dtype)
            return mlir.lower_fun(exact, multiple_results=False)(ctx, operand, indices)
        if (mode != lax.GatherScatterMode.FILL_OR_DROP
                and core.is_constant_shape(ctx.avals_in[0].shape)
                and core.is_constant_shape(slice_sizes)):
            bounds = np.array([ctx.avals_in[0].shape[d] - slice_sizes[d]
                               for d in dimension_numbers.start_index_map], np.int64)
            aval = ctx.avals_in[1]
            clamp = mlir.lower_fun(partial(_clamp_indices, bounds=bounds), multiple_results=False)
            indices, = clamp(ctx.replace(avals_in=[aval], avals_out=[aval], tokens_out=None),
                             indices)
        # FILL_OR_DROP's generic lowering checks the original indices, then calls
        # a gather whose indices are clamped here. Its fill mask stays unchanged.
        return default(ctx, operand, indices, dimension_numbers=dimension_numbers,
                       slice_sizes=slice_sizes, mode=mode, **params)

    return rule


def _slice_rule(primitive, operand_count):
    default = _complex.current_rule(primitive)

    def rule(ctx, *args, **params):
        if operand_count == 2 and ctx.avals_in[0].dtype == np.dtype(np.bool_):
            def widened(x, update, *starts):
                result = primitive.bind(lax.convert_element_type(x, np.int32),
                                        lax.convert_element_type(update, np.int32),
                                        *starts, **params)
                return result != np.int32(0)
            return mlir.lower_fun(widened, multiple_results=False)(ctx, *args)
        shape = ctx.avals_in[0].shape
        sizes = params["slice_sizes"] if operand_count == 1 else ctx.avals_in[1].shape
        if not core.is_constant_shape(shape) or not core.is_constant_shape(sizes):
            return default(ctx, *args, **params)
        index_dtype = np.dtype(np.int64 if any(n > np.iinfo(np.int32).max for n in shape)
                               else np.int32)
        in_avals = ctx.avals_in[operand_count:]
        out_avals = [a.update(dtype=index_dtype) for a in in_avals]

        def bounded(*starts):
            return [lax.convert_element_type(_clamp_indices(i, np.array(n - s, np.int64)),
                                              index_dtype)
                    for i, n, s in zip(starts, shape, sizes)]

        starts = mlir.lower_fun(bounded, multiple_results=True)(
            ctx.replace(avals_in=in_avals, avals_out=out_avals, tokens_out=None),
            *args[operand_count:])
        return default(ctx.replace(avals_in=[*ctx.avals_in[:operand_count], *out_avals],
                                   tokens_out=None), *args[:operand_count], *starts, **params)

    return rule


def _interior_pad(operand, padding_value, *, padding_config):
    """Interleave data and padding with concatenation, then apply edge padding.

    Apple's interior pad can overwrite other tensors filled with the padding
    value. A separate dimension for each gap avoids that path and its aliasing.
    """
    for axis, (_, _, interior) in enumerate(padding_config):
        n = operand.shape[axis]
        if not interior or n <= 1:
            continue
        shape = operand.shape
        expanded = lax.expand_dims(operand, (axis + 1,))
        fill_shape = (*shape[:axis + 1], interior, *shape[axis + 1:])
        fill = lax.broadcast_in_dim(padding_value, fill_shape, ())
        joined = lax.concatenate([expanded, fill], axis + 1)
        joined = lax.reshape(joined, (*shape[:axis], n * (interior + 1), *shape[axis + 1:]))
        operand = lax.slice_in_dim(joined, 0, n + (n - 1) * interior, axis=axis)
    return lax.pad(operand, padding_value, [(low, high, 0) for low, high, _ in padding_config])


def _edge_pad(operand, padding_value, *, padding_config):
    for axis, (low, high, _) in enumerate(padding_config):
        n = operand.shape[axis]
        begin, end = max(0, -low), min(n, n + high)
        if end <= begin:
            shape = list(operand.shape)
            shape[axis] = n + low + high
            operand = lax.broadcast_in_dim(padding_value, shape, ())
            continue
        operand = lax.slice_in_dim(operand, begin, end, axis=axis)
        pieces = []
        for amount in (max(low, 0), max(high, 0)):
            shape = list(operand.shape)
            shape[axis] = amount
            pieces.append(lax.broadcast_in_dim(padding_value, shape, ()))
        operand = lax.concatenate([pieces[0], operand, pieces[1]], axis)
    return operand


def _pad_rule():
    default = _complex.current_rule(lax_internal.pad_p)
    interleaved = mlir.lower_fun(_interior_pad, multiple_results=False)

    def rule(ctx, operand, padding_value, *, padding_config):
        dtype = ctx.avals_in[0].dtype
        if dtype.kind in "biu" and dtype.itemsize < 4:
            fn = _interior_pad if any(i for _, _, i in padding_config) else _edge_pad
            return mlir.lower_fun(fn, multiple_results=False)(
                ctx, operand, padding_value, padding_config=padding_config)
        if any(interior for _, _, interior in padding_config):
            return interleaved(ctx, operand, padding_value, padding_config=padding_config)
        return default(ctx, operand, padding_value, padding_config=padding_config)

    return rule


def _materialized_broadcast(x, *, shape, broadcast_dimensions, **_):
    reshaped = [1] * len(shape)
    for axis, target in enumerate(broadcast_dimensions):
        reshaped[target] = x.shape[axis]
    x = lax.reshape(x, reshaped)
    for axis, (source, target) in enumerate(zip(reshaped, shape)):
        if source == target:
            continue
        if target == 0:
            x = lax.slice_in_dim(x, 0, 0, axis=axis)
            continue
        pieces, block, count = [], x, target
        while count:
            if count & 1:
                pieces.append(block)
            count >>= 1
            if count:
                block = lax.concatenate([block, block], axis)
        x = lax.concatenate(pieces, axis) if len(pieces) > 1 else pieces[0]
    return x


def _broadcast_rule():
    default = _complex.current_rule(lax_internal.broadcast_in_dim_p)
    materialized = mlir.lower_fun(_materialized_broadcast, multiple_results=False)

    def rule(ctx, operand, *, shape, broadcast_dimensions, **params):
        in_shape = ctx.avals_in[0].shape
        if (core.is_constant_shape(in_shape) and core.is_constant_shape(shape)
                and np.prod(in_shape) > 1 and np.prod(shape) > np.prod(in_shape)):
            return materialized(ctx, operand, shape=shape,
                                broadcast_dimensions=broadcast_dimensions, **params)
        return default(ctx, operand, shape=shape, broadcast_dimensions=broadcast_dimensions, **params)

    return rule


def _is_replacement(update):
    return (update is None or (not update.eqns
            and list(update.outvars) == [update.invars[1]]))


def _scatter_bounds_rule(primitive):
    default = _complex.current_rule(primitive)

    def rule(ctx, operand, indices, updates, *, dimension_numbers, mode, **params):
        dnums = dimension_numbers
        if mode == lax.GatherScatterMode.CLIP or not dnums.update_window_dims:
            return default(ctx, operand, indices, updates, dimension_numbers=dnums,
                           mode=mode, **params)
        # Drop a whole window if any part is out of bounds. MPS instead
        # writes the in-bounds portion, which disagrees with gather's VJP.
        shape = ctx.avals_in[0].shape
        sizes = [1] * len(shape)
        operand_window_dims = [d for d in range(len(shape))
                               if d not in dnums.inserted_window_dims
                               and d not in dnums.operand_batching_dims]
        for d, u in zip(operand_window_dims, dnums.update_window_dims):
            sizes[d] = ctx.avals_in[2].shape[u]
        bounds = np.array([shape[d] - sizes[d] for d in dnums.scatter_dims_to_operand_dims], np.int32)
        index_aval = ctx.avals_in[1]
        normalized_aval = index_aval.update(dtype=np.dtype(np.int32))

        def drop_partial(i):
            # Compare in the source type before conversion, for uint32 max.
            upper = np.minimum(bounds, np.iinfo(i.dtype).max).astype(i.dtype)
            valid = lax.full(i.shape[:-1], True, np.bool_)
            for component in range(i.shape[-1]):
                start = lax.index_in_dim(i, component, axis=i.ndim - 1, keepdims=False)
                valid = valid & (start >= 0) & (start <= upper[component])
            i = lax.convert_element_type(i, np.int32)
            return lax.select(lax.broadcast_in_dim(valid, i.shape, tuple(range(i.ndim - 1))),
                              i, lax.full_like(i, max(shape, default=0)))

        indices, = mlir.lower_fun(drop_partial, multiple_results=False)(
            ctx.replace(avals_in=[index_aval], avals_out=[normalized_aval], tokens_out=None), indices)
        return default(ctx.replace(avals_in=[ctx.avals_in[0], normalized_aval, ctx.avals_in[2]],
                                   tokens_out=None), operand, indices, updates,
                       dimension_numbers=dnums, mode=mode, **params)

    return rule


def _boolean_scatter(scatter, operand, indices, updates, *, dimension_numbers,
                     indices_are_sorted, unique_indices, mode, **_):
    # A boolean MPS scatter produces the right output when returned directly, but
    # its consumers see the original operand. Keep the scatter in an integer type
    # and convert back only after it. Boolean max is OR and min is AND.
    out = scatter(lax.convert_element_type(operand, np.int32), indices,
                  lax.convert_element_type(updates, np.int32), dimension_numbers,
                  indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
                  mode=mode)
    return out != np.int32(0)


def _boolean_scatter_rule(primitive, scatter):
    default = _complex.current_rule(primitive)
    widened = mlir.lower_fun(partial(_boolean_scatter, scatter), multiple_results=False)

    def rule(ctx, *args, **params):
        update = params.get("update_jaxpr")
        plain_set = _is_replacement(update)
        if (ctx.avals_in[0].dtype == np.dtype(np.bool_)
                and (primitive is not slicing.scatter_p or plain_set)):
            return widened(ctx, *args, **params)
        return default(ctx, *args, **params)

    return rule


def _scatter_apply_rule():
    default = _complex.current_rule(slicing.scatter_p)

    def rule(ctx, operand, indices, updates, *, update_jaxpr, update_consts, **params):
        update = update_jaxpr
        unary = (update is not None and update.invars[1] not in update.outvars
                 and all(update.invars[1] not in eqn.invars for eqn in update.eqns))
        if not unary:
            return default(ctx, operand, indices, updates, update_jaxpr=update,
                           update_consts=update_consts, **params)

        def apply(x, i, unused):
            if x.size == 0:
                return x
            # MPS recognizes a scatter by its body opcode, treating square(x)
            # as x * update (the dummy update is zero). Count applications and
            # evaluate the unary body outside the scatter, retaining duplicates.
            counts = lax.scatter_add(lax.full(x.shape, 0, np.int32), i,
                                     lax.full(unused.shape, 1, np.int32), **params)
            fn = jax.vmap(lambda value: core.eval_jaxpr(
                update, update_consts, value, lax.full((), 0, x.dtype))[0])

            def body(state):
                current, remaining = state
                transformed = lax.reshape(fn(lax.reshape(current, (x.size,))), x.shape)
                return lax.select(remaining > 0, transformed, current), remaining - 1

            return lax.while_loop(
                lambda state: lax.reduce_max(state[1], tuple(range(x.ndim))) > 0,
                body, (x, counts))[0]

        return mlir.lower_fun(apply, multiple_results=False)(ctx, operand, indices, updates)

    return rule


def register(register_rule):
    register_rule(lax_internal.pad_p, _pad_rule())
    register_rule(lax_internal.broadcast_in_dim_p, _broadcast_rule())
    register_rule(slicing.gather_p, _gather_rule())
    register_rule(slicing.dynamic_slice_p, _slice_rule(slicing.dynamic_slice_p, 1))
    register_rule(slicing.dynamic_update_slice_p, _slice_rule(slicing.dynamic_update_slice_p, 2))
    for primitive, scatter in (
            (slicing.scatter_p, lax.scatter),
            (slicing.scatter_min_p, lax.scatter_min),
            (slicing.scatter_max_p, lax.scatter_max)):
        register_rule(primitive, _boolean_scatter_rule(primitive, scatter))
    for primitive in (slicing.scatter_p, slicing.scatter_add_p, slicing.scatter_mul_p,
                      slicing.scatter_min_p, slicing.scatter_max_p, slicing.scatter_sub_p):
        register_rule(primitive, _scatter_bounds_rule(primitive))
    register_rule(slicing.scatter_p, _scatter_apply_rule())
