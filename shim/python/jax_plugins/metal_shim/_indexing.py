"""Indexing rules for differences between StableHLO and Apple's MPS lowering."""

from functools import partial

import numpy as np
from jax import lax
from jax._src import core
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
        if dtype in (np.dtype(np.int32), np.dtype(np.uint32)) and mode != lax.GatherScatterMode.FILL_OR_DROP:
            def exact(x, i):
                # MPS gather converts integer payloads through float32. Move their
                # bit patterns as floats instead, preserving every integer bit.
                value = lax.gather(lax.bitcast_convert_type(x, np.float32), i,
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


def _pad_rule():
    default = _complex.current_rule(lax_internal.pad_p)
    interleaved = mlir.lower_fun(_interior_pad, multiple_results=False)

    def rule(ctx, operand, padding_value, *, padding_config):
        if any(interior for _, _, interior in padding_config):
            return interleaved(ctx, operand, padding_value, padding_config=padding_config)
        return default(ctx, operand, padding_value, padding_config=padding_config)

    return rule


def _is_replacement(update):
    return (update is None or (not update.eqns
            and list(update.outvars) == [update.invars[1]]))


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


def register(register_rule):
    register_rule(lax_internal.pad_p, _pad_rule())
    register_rule(slicing.gather_p, _gather_rule())
    register_rule(slicing.dynamic_slice_p, _slice_rule(slicing.dynamic_slice_p, 1))
    register_rule(slicing.dynamic_update_slice_p, _slice_rule(slicing.dynamic_update_slice_p, 2))
    for primitive, scatter in (
            (slicing.scatter_p, lax.scatter),
            (slicing.scatter_min_p, lax.scatter_min),
            (slicing.scatter_max_p, lax.scatter_max)):
        register_rule(primitive, _boolean_scatter_rule(primitive, scatter))
