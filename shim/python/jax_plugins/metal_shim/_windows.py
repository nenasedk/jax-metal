"""reduce_window for METAL.

Apple's ReduceWindowConverter maps mhlo.reduce_window to MPS's 4-D pooling kernels. On the M3
it legalizes only rank-4 operands whose body is one add or max; it refuses other ranks, base
dilation, min, multiply and multi-op or variadic bodies. Within that set, max pooling fills
padding with 0 instead of -inf, a sum ignores a non-zero init value and integer sums are
wrong. Window position, strides, SAME padding of sums and window dilation are handled.

XLA (CPU, GPU) leaves padding and base-dilation holes out of each window, which then holds
only operand elements: init_value is combined once. These rules keep the native pooling for
float32/float16 sums and maxima of operands up to rank 4 (reshaped to rank 4 with unit
dimensions; padding and base dilation applied first with lax.pad, holding the identity), and
compute every other window from strided slices:

- monoid windows (the sum/max/min primitives, and generic windows whose body is one add, max,
  min, mul, and or or): one dimension at a time, combining doubled blocks of taps, so a window
  of w taps costs O(log w) ops per dimension;
- other bodies (custom, variadic, select_and_gather_add's pairs): folded over the window
  positions in row-major order from the init value, acc = body(acc, element) where the
  element is an operand element (not padding or a hole), as XLA evaluates them.

select_and_scatter_add (the gradient of max/min pooling) lowers to MPS's 4-D
pooling_max_gradient, which refuses other ranks and min. Outside rank-4 max gradients without
padding it selects each window's first max (min) by a row-major fold and adds the source
values back at the selected positions with strided pads.
"""

import itertools
from functools import partial

import jax
import numpy as np
from jax import lax
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal
from jax._src.lax import windowed_reductions
from jax._src.lib.mlir.dialects import hlo

_NATIVE_DTYPES = (np.dtype(np.float32), np.dtype(np.float16))
# A custom body is traced once per window position; beyond this, refuse rather than build an
# enormous program.
_MAX_FOLDED_TAPS = 4096

# Native pooling on rank-4 operands without padding or base dilation: the subset of
# mhlo.reduce_window that MPS computes correctly.
_native_pool_p = core.Primitive("metal_native_pool")


def _native_pool_abstract(x, *, kind, window_dimensions, window_strides, window_dilation):
    del kind
    shape = windowed_reductions.reduce_window_shape_tuple(
        x.shape, window_dimensions, window_strides, [(0, 0)] * x.ndim, None, window_dilation)
    return core.ShapedArray(shape, x.dtype)


def _native_pool_lowering(ctx, x, *, kind, window_dimensions, window_strides, window_dilation):
    reduce_op, identity = {"sum": (hlo.add, lambda _: 0),
                           "max": (hlo.maximum, lax_internal._get_max_identity)}[kind]  # pylint: disable=protected-access
    return windowed_reductions._reduce_window_lower(  # pylint: disable=protected-access
        reduce_op, identity, ctx, x, window_dimensions=window_dimensions,
        window_strides=window_strides, padding=((0, 0),) * len(window_dimensions),
        base_dilation=(1,) * len(window_dimensions), window_dilation=window_dilation)


_native_pool_p.def_abstract_eval(_native_pool_abstract)
mlir.register_lowering(_native_pool_p, _native_pool_lowering)


def _pads(padding, base_dilation):
    return [(lo, hi, b - 1) for (lo, hi), b in zip(padding, base_dilation)]


def _native_pool(x, kind, window_dimensions, window_strides, window_dilation):
    """Native pooling of an operand of rank <= 4. MPS's max pooling skips NaN (a window
    holding one must be NaN) and returns 0 for a window of -inf only; one more pooling of
    per-element flags (2 for NaN, 1 above -inf) finds both."""
    lead = 4 - x.ndim
    x4 = lax.reshape(x, (1,) * lead + x.shape)
    window = dict(window_dimensions=(1,) * lead + tuple(window_dimensions),
                  window_strides=(1,) * lead + tuple(window_strides),
                  window_dilation=(1,) * lead + tuple(window_dilation))
    out = _native_pool_p.bind(x4, kind=kind, **window)
    if kind == "max":
        def flag(mask, value):
            return lax.convert_element_type(mask, x.dtype) * np.array(value, x.dtype)

        flags = flag(lax.ne(x4, x4), 2) + flag(lax.gt(x4, lax.full_like(x4, -np.inf)), 1)
        pooled = _native_pool_p.bind(flags, kind="max", **window)
        out = lax.select(lax.ge(pooled, lax.full_like(pooled, 1)), out,
                         lax.full_like(out, -np.inf))
        out = lax.select(lax.ge(pooled, lax.full_like(pooled, 2)), lax.full_like(out, np.nan), out)
    return lax.reshape(out, out.shape[lead:])


def _window_1d(x, axis, size, stride, dilation, combine):
    """op over taps i, i + d, ..., i + (size - 1) d along `axis`, at every stride-th i."""
    if size == 1:
        return lax.slice_in_dim(x, 0, None, stride, axis) if stride != 1 else x
    length = x.shape[axis] - (size - 1) * dilation  # outputs at stride 1
    if length <= 0:
        shape = list(x.shape)
        shape[axis] = 0
        return lax.full(shape, 0, x.dtype)
    # block[i] combines `width` taps from i; doubling it and taking the blocks for the set bits
    # of `size` covers the window in O(log size) ops.
    block, width, offset, acc, remaining = x, 1, 0, None, size
    while remaining:
        if remaining & 1:
            start = offset * dilation
            piece = lax.slice_in_dim(block, start, start + length, axis=axis)
            acc = piece if acc is None else combine(acc, piece)
            offset += width
        remaining >>= 1
        if remaining:
            n = block.shape[axis]
            shift = width * dilation
            block = combine(lax.slice_in_dim(block, 0, n - shift, axis=axis),
                            lax.slice_in_dim(block, shift, n, axis=axis))
            width *= 2
    return lax.slice_in_dim(acc, 0, None, stride, axis) if stride != 1 else acc


def _monoid_window(x, init, identity, *, combine, window_dimensions, window_strides, padding,
                   base_dilation, window_dilation):
    """A window reduction with an associative, commutative `combine` whose identity fills
    padding and base-dilation holes; `init` starts every window."""
    out_shape = windowed_reductions.reduce_window_shape_tuple(
        x.shape, window_dimensions, window_strides, padding, base_dilation, window_dilation)
    init = lax.convert_element_type(init, x.dtype)
    if 0 in out_shape or 0 in window_dimensions:
        return lax.broadcast(init, out_shape)
    x = lax.pad(x, lax.convert_element_type(identity, x.dtype), _pads(padding, base_dilation))
    for axis, (size, stride, dilation) in enumerate(zip(window_dimensions, window_strides,
                                                        window_dilation)):
        x = _window_1d(x, axis, size, stride, dilation, combine)
    return combine(x, lax.broadcast(init, x.shape))


def _native_eligible(x, window_dimensions):
    return x.dtype in _NATIVE_DTYPES and x.ndim <= 4 and 0 not in x.shape \
        and 0 not in window_dimensions


def _monoid(x, init, *, kind, combine, identity, window_dimensions, window_strides, padding,
            base_dilation, window_dilation):
    """Native pooling where MPS gets it right, else _monoid_window. `init` is None for the
    sum/max/min primitives, whose init is the identity."""
    identity = _identity(kind, x.dtype) if identity is None else identity
    if kind in ("sum", "max") and _native_eligible(x, window_dimensions):
        out_shape = windowed_reductions.reduce_window_shape_tuple(
            x.shape, window_dimensions, window_strides, padding, base_dilation, window_dilation)
        if 0 not in out_shape:
            fill = lax.convert_element_type(identity, x.dtype)
            padded = lax.pad(x, fill, _pads(padding, base_dilation))
            out = _native_pool(padded, kind, window_dimensions, window_strides, window_dilation)
            if init is None:
                return out
            return combine(out, lax.broadcast(lax.convert_element_type(init, x.dtype), out.shape))
    return _monoid_window(x, identity if init is None else init, identity, combine=combine,
                          window_dimensions=window_dimensions, window_strides=window_strides,
                          padding=padding, base_dilation=base_dilation,
                          window_dilation=window_dilation)


def _identity(kind, dtype):
    if kind == "max":
        return np.array(lax_internal._get_max_identity(dtype), dtype)  # pylint: disable=protected-access
    if kind == "min":
        return np.array(lax_internal._get_min_identity(dtype), dtype)  # pylint: disable=protected-access
    if kind == "and":  # all bits set
        return np.array(True if dtype == np.bool_ else ~np.zeros((), dtype), dtype)
    return np.array({"sum": 0, "mul": 1, "or": 0}[kind], dtype)


def _monoid_primitive(kind, combine, x, **window):
    return _monoid(x, None, kind=kind, combine=combine, identity=_identity(kind, x.dtype),
                   **window)


def _fold(*args, jaxpr, consts, window_dimensions, window_strides, padding, base_dilation,
          window_dilation):
    """Generic (custom or variadic) window: acc = body(acc, element) over the window positions
    in row-major order, accumulator first, from the init values, for operand elements only."""
    operands, inits = args[:len(args) // 2], args[len(args) // 2:]
    taps = int(np.prod(window_dimensions))
    if taps > _MAX_FOLDED_TAPS:
        raise NotImplementedError(
            f"METAL: reduce_window with a custom body over {taps} window positions "
            f"(at most {_MAX_FOLDED_TAPS})")
    out_shape = windowed_reductions.reduce_window_shape_tuple(
        operands[0].shape, window_dimensions, window_strides, padding, base_dilation,
        window_dilation)
    accs = [lax.broadcast(init, out_shape) for init in inits]
    if 0 in out_shape:
        return accs
    pads = _pads(padding, base_dilation)
    padded = [lax.pad(x, init, pads) for x, init in zip(operands, inits)]
    # Operand elements, as opposed to padding and holes, which windows leave out.
    real = lax.pad(lax.full(operands[0].shape, True, np.bool_), False, pads)
    body = getattr(jaxpr, "jaxpr", jaxpr)
    apply = partial(core.eval_jaxpr, body, consts)  # typed for scalars: map it elementwise
    for _ in out_shape:
        apply = jax.vmap(apply)
    for position in itertools.product(*(range(w) for w in window_dimensions)):
        starts = [p * d for p, d in zip(position, window_dilation)]
        limits = [s + (n - 1) * stride + 1
                  for s, n, stride in zip(starts, out_shape, window_strides)]
        elements = [lax.slice(x, starts, limits, window_strides) for x in padded]
        use = lax.slice(real, starts, limits, window_strides)
        accs = [lax.select(use, new, acc) for new, acc in zip(apply(*accs, *elements), accs)]
    return accs


# Single-op bodies that the monoid path computes: primitive -> (kind, combine).
_MONOIDS = {
    lax_internal.add_p: ("sum", lax.add), lax_internal.max_p: ("max", lax.max),
    lax_internal.min_p: ("min", lax.min), lax_internal.mul_p: ("mul", lax.mul),
    lax_internal.and_p: ("and", lax.bitwise_and), lax_internal.or_p: ("or", lax.bitwise_or),
}


def _single_op(jaxpr):
    """(kind, combine) if `jaxpr` is one commutative monoid op on its two inputs."""
    body = getattr(jaxpr, "jaxpr", jaxpr)
    if len(body.invars) != 2 or len(body.eqns) != 1:
        return None
    eqn = body.eqns[0]
    if (eqn.primitive in _MONOIDS and set(eqn.invars) == set(body.invars)
            and len(eqn.invars) == 2 and list(eqn.outvars) == list(body.outvars)):
        return _MONOIDS[eqn.primitive]
    return None


def _generic(*args, jaxpr, consts, **window):
    monoid = _single_op(jaxpr) if len(args) == 2 and not consts else None
    if monoid is None:
        return _fold(*args, jaxpr=jaxpr, consts=consts, **window)
    kind, combine = monoid
    x, init = args
    return [_monoid(x, init, kind=kind, combine=combine, identity=None, **window)]


def _select_and_scatter_add(source, operand, *, select_prim, window_dimensions,
                            window_strides, padding):
    """Add each source value at the position its window selects: the first element that
    select_prim keeps against every later one, in row-major order, as XLA selects."""
    taps = int(np.prod(window_dimensions))
    if taps > _MAX_FOLDED_TAPS:
        raise NotImplementedError(
            f"METAL: select_and_scatter_add over {taps} window positions (at most "
            f"{_MAX_FOLDED_TAPS})")
    zero = lax.full((), 0, source.dtype)
    if 0 in source.shape or 0 in operand.shape:
        return lax.broadcast(zero, operand.shape)
    identity = (lax_internal._get_max_identity if select_prim is lax.ge_p  # pylint: disable=protected-access
                else lax_internal._get_min_identity)(operand.dtype)  # pylint: disable=protected-access
    padded = lax.pad(operand, lax.full((), identity, operand.dtype),
                     [(lo, hi, 0) for lo, hi in padding])
    positions = list(itertools.product(*(range(w) for w in window_dimensions)))

    def window(position):
        limits = [p + (n - 1) * s + 1 for p, n, s in zip(position, source.shape, window_strides)]
        return lax.slice(padded, position, limits, window_strides)

    best, chosen = window(positions[0]), lax.full(source.shape, 0, np.int32)
    for k, position in enumerate(positions[1:], 1):
        element = window(position)
        keep = select_prim.bind(best, element)
        best = lax.select(keep, best, element)
        chosen = lax.select(keep, chosen, lax.full_like(chosen, k))
    combine = lax.bitwise_or if source.dtype == np.bool_ else lax.add
    result = lax.broadcast(zero, padded.shape)
    for k, position in enumerate(positions):
        hit = lax.select(lax.eq(chosen, lax.full_like(chosen, k)), source,
                         lax.broadcast(zero, source.shape))
        # Back to padded coordinates: position + i * stride for window i.
        pads = [(p, size - p - (n - 1) * s - 1, s - 1)
                for p, size, n, s in zip(position, padded.shape, source.shape, window_strides)]
        result = combine(result, lax.pad(hit, zero, pads))
    return lax.slice(result, [lo for lo, _ in padding],
                     [lo + n for (lo, _), n in zip(padding, operand.shape)])


def _select_and_scatter_add_rule(default):
    generic = mlir.lower_fun(_select_and_scatter_add, multiple_results=False)

    def rule(ctx, source, operand, *, select_prim, window_dimensions, window_strides, padding):
        params = dict(select_prim=select_prim, window_dimensions=window_dimensions,
                      window_strides=window_strides, padding=padding)
        aval = ctx.avals_in[1]
        if (aval.ndim == 4 and select_prim is lax.ge_p and aval.dtype in _NATIVE_DTYPES
                and all(lo == hi == 0 for lo, hi in padding)):
            return default(ctx, source, operand, **params)
        return generic(ctx, source, operand, **params)

    return rule


def register(register_rule):
    for primitive, kind, combine in (
            (windowed_reductions.reduce_window_sum_p, "sum", lax.add),
            (windowed_reductions.reduce_window_max_p, "max", lax.max),
            (windowed_reductions.reduce_window_min_p, "min", lax.min)):
        register_rule(primitive, mlir.lower_fun(partial(_monoid_primitive, kind, combine),
                                                multiple_results=False))
    register_rule(windowed_reductions.reduce_window_p,
                  mlir.lower_fun(_generic, multiple_results=True))
    from jax_plugins.metal_shim import _complex  # pylint: disable=import-outside-toplevel
    primitive = windowed_reductions.select_and_scatter_add_p
    register_rule(primitive, _select_and_scatter_add_rule(_complex.current_rule(primitive)))
