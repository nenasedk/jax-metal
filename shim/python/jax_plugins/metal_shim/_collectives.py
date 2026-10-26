"""Collectives for METAL, which has one device.

An Apple-silicon Mac has one GPU and the plugin exposes it as one device, so a METAL program
runs on a mesh of size 1 and every collective acts on a group of one device:

- psum, pmax, pmin and their invariant and unreduced forms return their operand (after
  reducing any positional axes);
- all_gather adds a dimension of size 1 (tiled: none), reduce_scatter removes one, and
  all_to_all moves one block onto itself: the data is unchanged, at most reshaped;
- ppermute keeps the operand when the permutation sends device 0 to itself and gives zeros
  otherwise; pbroadcast keeps it; ragged_all_to_all copies the chunk the device sends itself;
- axis_index is 0.

The plugin has no pattern for all_gather, all_to_all, collective_permute, reduce_scatter,
partition_id or replica_id (re/findings/mps-op-coverage.md). all_reduce has one, but the
identity is exact and free, and a NaN-propagating max would not be a reduction body the plugin
recognizes. A program lowered for a larger (abstract) mesh keeps JAX's lowering.
"""

import numpy as np
from jax import lax
from jax._src.interpreters import mlir
from jax._src.lax import parallel


def _one_device(ctx):
    axis_context = ctx.module_context.axis_context
    mesh = getattr(axis_context, "mesh", None)
    if mesh is not None:
        return mesh.size == 1
    return getattr(axis_context, "num_devices", 1) == 1


def _reshaped(ctx, x):
    """x as the primitive's output shape: the same elements in the same order."""
    aval_in, aval_out = ctx.avals_in[0], ctx.avals_out[0]
    if aval_in.shape == aval_out.shape:
        return x
    return mlir.reshape(ctx, x, aval_out)


def _reduction(positional_reduce):
    def rule(ctx, x, *, axes, **_):
        positional = tuple(axis for axis in axes if isinstance(axis, int))
        if not positional:
            return [x]
        reduce = mlir.lower_fun(lambda v: positional_reduce(v, positional),
                                multiple_results=False)
        return reduce(ctx, x)
    return rule


def _ppermute(ctx, x, *, perm, **_):
    if any(source == 0 and target == 0 for source, target in perm):
        return [x]
    return [mlir.full_like_aval(ctx, 0, ctx.avals_out[0])]


def _ragged_all_to_all_one(operand, output, input_offsets, send_sizes, output_offsets,
                           recv_sizes):
    """output with rows [output_offsets[0], +send_sizes[0]) taken from operand from row
    input_offsets[0]: the chunk this device sends to itself."""
    del recv_sizes  # equal to send_sizes for the device's message to itself
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    rows = lax.iota(np.int32, output.shape[0])
    start, size, source = (lax.convert_element_type(v[0], np.int32)
                           for v in (output_offsets, send_sizes, input_offsets))
    taken = jnp.take(operand, jnp.clip(rows - start + source, 0, operand.shape[0] - 1), axis=0)
    inside = (rows >= start) & (rows < start + size)
    return lax.select(lax.broadcast_in_dim(inside, output.shape, (0,)), taken, output)


_ONE_DEVICE_RULES = {
    parallel.psum_p: _reduction(lax.reduce_sum),
    parallel.pmax_p: _reduction(lax.reduce_max),
    parallel.pmin_p: _reduction(lax.reduce_min),
    parallel.psum_invariant_p: _reduction(lax.reduce_sum),
    parallel.unreduced_psum_p: _reduction(lax.reduce_sum),
    parallel.unreduced_pmax_p: _reduction(lax.reduce_max),
    parallel.unreduced_pmin_p: _reduction(lax.reduce_min),
    parallel.ppermute_p: _ppermute,
    parallel.pbroadcast_p: lambda ctx, x, **_: [x],
    parallel.pbroadcast_dynamic_p: lambda ctx, x, source, **_: [x],
    parallel.all_to_all_p: lambda ctx, x, **_: [_reshaped(ctx, x)],
    parallel.all_gather_p: lambda ctx, x, **_: [_reshaped(ctx, x)],
    parallel.all_gather_invariant_p: lambda ctx, x, **_: [_reshaped(ctx, x)],
    parallel.all_gather_reduced_p: lambda ctx, x, **_: [_reshaped(ctx, x)],
    parallel.reduce_scatter_p: lambda ctx, x, **_: [_reshaped(ctx, x)],
    parallel.unreduced_reduce_scatter_p: lambda ctx, x, **_: [_reshaped(ctx, x)],
    parallel.axis_index_p: lambda ctx, **_: [
        mlir.ir_constant(np.zeros((), ctx.avals_out[0].dtype))],
    parallel.ragged_all_to_all_p: lambda ctx, *args, **_: mlir.lower_fun(
        _ragged_all_to_all_one, multiple_results=False)(ctx, *args),
}


def _for_one_device(primitive, one_device_rule):
    entry = mlir._lowerings.get(primitive)  # pylint: disable=protected-access
    default = entry.rule if entry else None

    def rule(ctx, *args, **params):
        if _one_device(ctx):
            return one_device_rule(ctx, *args, **params)
        if default is None:
            raise NotImplementedError(f"METAL has one device; {primitive} over several")
        return default(ctx, *args, **params)

    return rule


def register(register_rule):
    for primitive, one_device_rule in _ONE_DEVICE_RULES.items():
        register_rule(primitive, _for_one_device(primitive, one_device_rule))
