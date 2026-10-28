"""METAL-specific lowering rules for ops the plugin's HLO->MPS lowering rejects.

Each rule rewrites a JAX primitive into ops the plugin already handles. See
re/findings/mps-op-coverage.md for what the plugin can lower and
re/findings/jax-op-probe.md for what fails without these rules.
"""

import threading
from functools import partial

import numpy as np
import jax
from jax import lax
from jax._src import core
from jax._src import dtypes
from jax._src.interpreters import mlir
from jax._src.lax import ann
from jax._src.lax import lax as lax_internal
from jax._src.lax import linalg
from jax._src.lax import parallel
from jax._src.lax import slicing
from jax._src.lax import special
from jax._src.lax import windowed_reductions
from jax._src.lax.control_flow import loops
from jax._src.random.threefry2x32 import threefry_2x32

from jax_plugins.metal_shim import _collectives
from jax_plugins.metal_shim import _complex
from jax_plugins.metal_shim import _precision

PLATFORM = "METAL"


def _register_rule(primitive, rule):
    _complex.PLATFORM = PLATFORM
    _complex.register_rule(primitive, rule)


def _register(primitive, fn, multiple_results=False):
    _register_rule(primitive, mlir.lower_fun(fn, multiple_results=multiple_results))


def _default_rule(primitive):
    """JAX's platform-independent lowering for `primitive`."""
    return mlir._lowerings[primitive].rule  # pylint: disable=protected-access


# --- Cumulative reductions ---------------------------------------------------
# jax 0.12 lowers these to mhlo.reduce_window with a window as long as the axis,
# which the plugin fails to legalize. A parallel prefix scan uses only slices,
# pads and elementwise ops.

def _cumulative_scan(combine, x, *, axis, reverse):
    return loops.associative_scan(combine, x, reverse=reverse, axis=axis)


_CUMULATIVE = (
    (loops.cumsum_p, lax.add),
    (loops.cumprod_p, lax.mul),
    (loops.cummax_p, lax.max),
    (loops.cummin_p, lax.min),
    (loops.cumlogsumexp_p, loops.logaddexp),
)

# --- Accuracy fixes for MPS elementwise functions ---------------------------------
# MPS computes expm1 and log1p naively (exp(x) - 1, log(1 + x)), losing all precision
# near 0 (1.5e7 ulps at 1e-12), returns 1 for tanh(NaN) and loses small-argument tanh,
# and returns a number for sign(NaN). MPS exp and log are accurate (1-2 ulps), so the
# fixes build on them. Measured by shim/tests/accuracy_sweep.py.

def _log1p(x, **_):
    # Goldberg: log(u) * x / (u - 1) with u = 1 + x cancels u's rounding error.
    u = 1 + x
    accurate = lax.log(u) * x / (u - 1)
    result = lax.select(u == 1, x, accurate)
    return lax.select(x == np.inf, x, result)


def _expm1(x, **_):
    # Kahan: (u - 1) * x / log(u) with u = exp(x) cancels u's rounding error.
    u = lax.exp(x)
    accurate = (u - 1) * x / lax.log(u)
    result = lax.select(u == 1, x, lax.select(u - 1 == -1, lax.full_like(x, -1), accurate))
    return lax.select(u == np.inf, u, result)


def _tanh(x, **_):
    # tanh(x) = expm1(2x) / (expm1(2x) + 2) on |x|, so that tanh(-x) = -tanh(x) exactly;
    # |x| >= 10 is 1 in float32. NaN and +-0 pass through.
    clamped = lax.min(lax.abs(x), lax.full_like(x, 10.0))
    e = _expm1(2 * clamped)
    t = e / (e + 2)
    return lax.select(lax.ne(x, x) | (x == 0), x, lax.select(x < 0, -t, t))


def _with_nan_passthrough(primitive):
    """Run JAX's own lowering, then return NaN wherever the input was NaN."""
    default = _default_rule(primitive)
    fix = mlir.lower_fun(lambda x, y: lax.select(lax.ne(x, x), x, y), multiple_results=False)

    def rule(ctx, x, **params):
        y, = default(ctx, x, **params)
        return fix(ctx.replace(avals_in=[ctx.avals_in[0], ctx.avals_out[0]]), x, y)

    return rule


# --- NaN propagation in max/min ------------------------------------------------------
# XLA's max and min (elementwise, reductions, windows), clamp and argmax/argmin propagate NaN.
# The plugin's do not (re/findings/mps-nan-semantics.md): it builds mps.reduction_max/min with
# their propagate_nans flag off, and maps max, min and clamp one-to-one onto mps.maximum,
# minimum and clamp, MPSGraph's non-propagating forms (PyTorch's MPS backend moved to the
# ...NaNPropagation variants for this reason). On METAL, cond of a singular matrix came out 0
# instead of inf. These rules run JAX's lowering, then put NaN back where an input had it;
# integer inputs are left alone.

def _is_floating_aval(aval):
    # dtypes.issubdtype, unlike NumPy's, counts ml_dtypes' bfloat16 as floating.
    return isinstance(aval.dtype, np.dtype) and dtypes.issubdtype(aval.dtype, np.floating)


def _scatter_nan_propagating(primitive):
    """scatter_max/scatter_min, then NaN wherever the operand or an update scattered there is
    NaN: the update mask is scattered with scatter_max on 0/1 values (NaN-free, exact)."""
    default = _default_rule(primitive)
    scatter_max = _default_rule(slicing.scatter_max_p)
    nan_masks = mlir.lower_fun(
        lambda operand, updates: (lax.full_like(operand, 0),
                                  lax.convert_element_type(lax.ne(updates, updates), updates.dtype)),
        multiple_results=True)
    with_nan = mlir.lower_fun(
        lambda operand, hit, out: lax.select(
            lax.ne(operand, operand) | lax.gt(hit, lax.full_like(hit, 0)),
            lax.full_like(out, np.nan), out),
        multiple_results=False)

    def rule(ctx, operand, indices, updates, **params):
        out = default(ctx, operand, indices, updates, **params)
        if not _is_floating_aval(ctx.avals_in[0]):
            return out
        operand_aval, _, updates_aval = ctx.avals_in
        zeros, update_nan = nan_masks(
            ctx.replace(avals_in=[operand_aval, updates_aval],
                        avals_out=[operand_aval, updates_aval], tokens_out=None),
            operand, updates)
        # The mask always combines with max (scatter_min's own body would lose hits).
        max_jaxpr, max_consts = lax_internal._reduction_jaxpr(  # pylint: disable=protected-access
            lax.max, core.ShapedArray((), operand_aval.dtype))
        hit = scatter_max(ctx.replace(tokens_out=None), zeros, indices, update_nan,
                          **dict(params, update_jaxpr=max_jaxpr, update_consts=max_consts))
        return with_nan(ctx.replace(avals_in=[operand_aval, operand_aval, operand_aval],
                                    tokens_out=None), operand, *hit, *out)

    return rule


# Region bodies (the combiner of a reduce, scatter or all-reduce) must stay a
# single max or min: the plugin's converters recognize the reduction by that op and reject a
# compare-and-select body. Lowering such a primitive marks its body as inside a region, where
# the NaN rules keep JAX's plain lowering.
_region = threading.local()


def _in_region():
    return getattr(_region, "depth", 0) > 0


def _marking_region_bodies(rule):
    def wrapped(ctx, *args, **params):
        _region.depth = getattr(_region, "depth", 0) + 1
        try:
            return rule(ctx, *args, **params)
        finally:
            _region.depth -= 1
    return wrapped


_REGION_PRIMITIVES = (  # not reduce_window: _windows.py emits no region from its body
    lax_internal.reduce_p, windowed_reductions.select_and_scatter_p, slicing.scatter_p,
    slicing.scatter_add_p, slicing.scatter_sub_p, slicing.scatter_mul_p, slicing.scatter_min_p,
    slicing.scatter_max_p,
    parallel.psum_p, parallel.pmax_p, parallel.pmin_p,
)


def _nan_propagating(primitive, fix_fn):
    """JAX's lowering of `primitive`, then fix_fn(*inputs, output) for floating inputs."""
    default = _default_rule(primitive)

    def rule(ctx, *args, **params):
        out = default(ctx, *args, **params)
        if _in_region() or not any(_is_floating_aval(aval) for aval in ctx.avals_in):
            return out
        fix = mlir.lower_fun(partial(fix_fn, **params), multiple_results=False)
        # A fresh context: a lower_fun-based default (argmax) has set this one's tokens.
        fix_ctx = ctx.replace(avals_in=[*ctx.avals_in, ctx.avals_out[0]], tokens_out=None)
        return fix(fix_ctx, *args, *out)

    return rule


def _first_nan(*inputs_and_output, **_):
    """The first NaN input, broadcast to the output, wherever there is one; else the output."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    *inputs, out = inputs_and_output
    for x in reversed(inputs):
        # jnp.where broadcasts a scalar operand with the output's sharding (explicit meshes).
        out = jnp.where(lax.ne(x, x), x, out)
    return out


def _clamp_nan(low, x, high, out, **_):
    return _first_nan(x, out)


def _reduction_nan(x, out, *, axes, **_):
    has_nan = lax.reduce_or(lax.ne(x, x), axes)
    return lax.select(has_nan, lax.full_like(out, np.nan), out)


def _arg_reduction_nan(x, out, *, axes, index_dtype, **_):
    # The index of the first NaN, where there is one (argmax of the 0/1 NaN mask).
    is_nan = lax.ne(x, x)
    first = lax.argmax(lax.convert_element_type(is_nan, np.int32), axes[0], index_dtype)
    return lax.select(lax.reduce_or(is_nan, axes), first, out)


_NAN_PROPAGATING = (
    (lax_internal.max_p, _first_nan), (lax_internal.min_p, _first_nan),
    (lax_internal.clamp_p, _clamp_nan),
    (lax_internal.reduce_max_p, _reduction_nan), (lax_internal.reduce_min_p, _reduction_nan),
    (lax_internal.argmax_p, _arg_reduction_nan), (lax_internal.argmin_p, _arg_reduction_nan),
)


# --- CHLO ops serialized as composites -----------------------------------------
# XLA's serializer wraps these CHLO ops in stablehlo.composite
# (ChloPreserveHighLevelOpsPass), which the plugin cannot lower. Express them with
# ops the plugin supports instead.

_LARGE = 1e8  # beyond this, x*x overflows float32 in the formulas below


def _acos(x, **_):
    # atan2 form is exact at both ends: acos(1) = 0, acos(-1) = pi.
    return lax.atan2(lax.sqrt((1 - x) * (1 + x)), x)


def _acosh(x, **_):
    # log1p form is accurate near x = 1, where log(x + ...) cancels.
    t = x - 1
    small = _log1p(t + lax.sqrt(t * (x + 1)))
    return lax.select(x > _LARGE, lax.log(x) + np.log(2).astype(x.dtype), small)


def _asinh(x, **_):
    a = lax.abs(x)
    small = _log1p(a + a * a / (1 + lax.sqrt(1 + a * a)))
    return lax.sign(x) * lax.select(a > _LARGE, lax.log(a) + np.log(2).astype(x.dtype), small)


def _atanh(x, **_):
    # log1p form near 0; near +-1, 1 + x and 1 - x are exact, so the ratio form avoids
    # the cancellation in 1 + 2x / (1 - x).
    near_zero = 0.5 * _log1p(2 * x / (1 - x))
    near_one = 0.5 * lax.log((1 + x) / (1 - x))
    return lax.select(lax.abs(x) < 0.5, near_zero, near_one)


def _sinh(x, **_):
    return 0.5 * (_expm1(x) - _expm1(-x))


def _cosh(x, **_):
    return 0.5 * (lax.exp(x) + lax.exp(-x))


# Rational approximation of erf on [-4, 4] used by Eigen and XLA for float32
# (max abs error ~1e-7); erf is +-1 to float32 precision outside that range.
_ERF_ALPHA = (-2.72614225801306e-10, 2.77068142495902e-08, -2.10102402082508e-06,
              -5.69250639462346e-05, -7.34990630326855e-04, -2.95459980854025e-03,
              -1.60960333262415e-02)
_ERF_BETA = (-1.45660718464996e-05, -2.13374055278905e-04, -1.68282697438203e-03,
             -7.37332916720468e-03, -1.42647390514189e-02)


def _horner(x, coefficients):
    result = lax.full_like(x, coefficients[0])
    for coefficient in coefficients[1:]:
        result = result * x + coefficient
    return result


def _erf(x, **_):
    clamped = lax.clamp(lax.full_like(x, -4.0), x, lax.full_like(x, 4.0))
    x2 = clamped * clamped
    result = clamped * _horner(x2, _ERF_ALPHA) / _horner(x2, _ERF_BETA)
    return lax.select(lax.ne(x, x), x, result)  # clamp would turn NaN into +-4


# XLA's float32 erfc for |x| >= 1 (Cephes coefficients): exp(-x^2) / |x| * P(1/x^2),
# with separate polynomials below and above |x| = 2. Below 1, 1 - erf(x) is accurate.
_ERFC_P = (2.326819970068386e-2, -1.387039388740657e-1, 3.687424674597105e-1,
           -5.824733027278666e-1, 6.210004621745983e-1, -4.944515323274145e-1,
           3.404879937665872e-1, -2.741127028184656e-1, 5.638259427386472e-1)
_ERFC_R = (-1.047766399936249e1, 1.297719955372516e1, -7.495518717768503e0,
           2.921019019210786e0, -1.015265279202700e0, 4.218463358204948e-1,
           -2.820767439740514e-1, 5.641895067754075e-1)


def _erfc(x, **_):
    a = lax.abs(x)
    z = lax.exp(-x * x)
    q = 1 / a
    y = q * q
    p = lax.select(a < 2, _horner(y, _ERFC_P), _horner(y, _ERFC_R))
    tail = z * q * p
    tail = lax.select(x < 0, 2 - tail, tail)
    return lax.select(a < 1, 1 - _erf(x), tail)


def _cbrt(x, **_):
    # mhlo.cbrt has no MPS pattern. cbrt(0) = exp(-inf) = 0.
    return lax.sign(x) * lax.exp(lax.log(lax.abs(x)) / 3)


# --- mulhi ------------------------------------------------------------------------------
# chlo.mulhi (used by the philox PRNG) is serialized as a composite the plugin cannot lower.
# High 32 bits of a 32x32-bit product from 16-bit halves, exact in uint32 arithmetic.

def _mulhi(x, y, **_):
    dtype = np.dtype(x.dtype)
    if dtype.itemsize < 4:  # the full product fits in 32 bits
        wide = np.dtype(np.int32 if dtype.kind == "i" else np.uint32)
        product = lax.convert_element_type(x, wide) * lax.convert_element_type(y, wide)
        shift = lax.shift_right_arithmetic if dtype.kind == "i" else lax.shift_right_logical
        return lax.convert_element_type(shift(product, wide.type(8 * dtype.itemsize)), dtype)
    if dtype.itemsize != 4:
        raise NotImplementedError(f"METAL mulhi supports integers up to 32 bits, got {dtype}")
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    x, y = jnp.broadcast_arrays(x, y)  # lax broadcasts size-1 dimensions; select below does not
    u32 = np.uint32
    a, b = lax.bitcast_convert_type(x, u32), lax.bitcast_convert_type(y, u32)
    mask, shift = u32(0xFFFF), u32(16)
    al, ah = lax.bitwise_and(a, mask), lax.shift_right_logical(a, shift)
    bl, bh = lax.bitwise_and(b, mask), lax.shift_right_logical(b, shift)
    t = al * bl
    u = ah * bl + lax.shift_right_logical(t, shift)
    v = al * bh + lax.bitwise_and(u, mask)
    high = ah * bh + lax.shift_right_logical(u, shift) + lax.shift_right_logical(v, shift)
    if dtype.kind == "i":  # signed: subtract the other operand for each negative input
        zero = lax.full_like(high, 0)
        high = high - lax.select(x < 0, b, zero) - lax.select(y < 0, a, zero)
    return lax.bitcast_convert_type(high, dtype)


# --- Integer bit counting --------------------------------------------------------
# mhlo.popcnt and mhlo.count_leading_zeros have no MPS pattern; shifts and masks do.

def _as_unsigned(x):
    bits = np.dtype(x.dtype).itemsize * 8
    return lax.bitcast_convert_type(x, np.dtype(f"uint{bits}")), bits


def _population_count(x, **_):
    u, bits = _as_unsigned(x)
    one = lax.full_like(u, 1)
    count = lax.full_like(u, 0)
    for shift in range(bits):
        count = count + lax.bitwise_and(lax.shift_right_logical(u, lax.full_like(u, shift)), one)
    return lax.convert_element_type(count, x.dtype)


def _count_leading_zeros(x, **_):
    # Binary search: shift left while the top `step` bits are zero.
    u, bits = _as_unsigned(x)
    zeros = lax.full_like(u, 0)
    step = bits // 2
    while step:
        top_clear = lax.shift_right_logical(u, lax.full_like(u, bits - step)) == 0
        zeros = lax.select(top_clear, zeros + step, zeros)
        u = lax.select(top_clear, lax.shift_left(u, lax.full_like(u, step)), u)
        step //= 2
    zeros = lax.select(u == 0, zeros + 1, zeros)  # x == 0 has `bits` leading zeros
    return lax.convert_element_type(zeros, x.dtype)


def _order_reversing(x):
    """A monotonically decreasing map of x that never overflows."""
    if np.issubdtype(x.dtype, np.integer) or x.dtype == np.bool_:
        return lax.bitwise_not(x)
    return lax.neg(x)


def _top_k(operand, *, k, axis, **_):
    # Stable ascending sort on an order-reversing key gives descending values with
    # the lowest index first among ties, which is top_k's contract.
    axis = axis % operand.ndim
    indices = lax.broadcasted_iota(np.int32, operand.shape, axis)
    keys, order = lax.sort((_order_reversing(operand), indices), dimension=axis, num_keys=1,
                           is_stable=True)
    values = _order_reversing(keys)
    return [lax.slice_in_dim(values, 0, k, axis=axis), lax.slice_in_dim(order, 0, k, axis=axis)]


def _approx_top_k_rule():
    """approx_max_k/approx_min_k as an exact top-k, which is a valid result.

    Their generic lowering is an ApproxTopK custom call, which XLA's CPU and GPU compilers
    rewrite but the plugin does not know. The output has as many entries as JAX's shape for
    it: k when aggregated, more otherwise.
    """
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    def exact(operand, *, n, dimension, is_max_k):
        moved = jnp.moveaxis(operand, dimension, -1)
        values, indices = lax.top_k(moved if is_max_k else _order_reversing(moved), n)
        values = values if is_max_k else _order_reversing(values)
        return [jnp.moveaxis(values, -1, dimension), jnp.moveaxis(indices, -1, dimension)]

    def rule(ctx, operand, *, reduction_dimension, is_max_k, **_):
        n = ctx.avals_out[0].shape[reduction_dimension]
        return mlir.lower_fun(partial(exact, n=n, dimension=reduction_dimension, is_max_k=is_max_k),
                              multiple_results=True)(ctx, operand)

    return rule


# --- Linear algebra --------------------------------------------------------------
# mhlo.cholesky and mhlo.triangular_solve have no MPS pattern. Both become a loop over
# rows/columns of matrix-vector products, which the plugin lowers (while, dynamic
# slices, dot_general). O(n) sequential steps; fine for the small and medium matrices
# typical of ML use.

def _batch_matvec(matrix, vector):
    """matrix [..., n, k] contracted with vector [..., n] over n -> [..., k]."""
    batch = tuple(range(vector.ndim - 1))
    return lax.dot_general(vector, matrix, (((vector.ndim - 1,), (matrix.ndim - 2,)), (batch, batch)),
                           precision=lax.Precision.HIGHEST)


def _lu_compensated(a):
    if a.ndim > 2:
        return jax.vmap(_lu_compensated)(a)
    m, n = a.shape
    pivots = lax.full((min(m, n),), 0, np.int32)
    permutation = lax.iota(np.int32, m)
    if min(m, n) == 0:
        return a, pivots, permutation
    rows, cols = lax.iota(np.int32, m), lax.iota(np.int32, n)

    def step(k, state):
        matrix, pivots, permutation = state
        column = matrix[:, k]
        pivot = lax.argmax(lax.select(rows >= k, lax.abs(column),
                                      lax.full_like(column, -np.inf)), 0, np.int32)
        pivots = pivots.at[k].set(pivot)
        matrix = matrix.at[k].set(matrix[pivot]).at[pivot].set(matrix[k])
        permutation = permutation.at[k].set(permutation[pivot]).at[pivot].set(permutation[k])
        diagonal = matrix[k, k]
        column = lax.select((rows > k) & (diagonal != 0), matrix[:, k] / diagonal,
                            matrix[:, k])
        matrix = matrix.at[:, k].set(column)
        updated = _precision.subtract_product(matrix, column[:, None], matrix[k, None, :])
        matrix = lax.select((rows[:, None] > k) & (cols[None, :] > k), updated, matrix)
        return matrix, pivots, permutation

    return lax.fori_loop(0, min(m, n), step, (a, pivots, permutation))


def _lu_rule():
    default = _complex.current_rule(linalg.lu_p)
    compensated = mlir.lower_fun(_lu_compensated, multiple_results=True)

    def rule(ctx, a):
        aval = ctx.avals_in[0]
        if aval.dtype == np.float32 and min(aval.shape[-2:]) <= 128:
            return compensated(ctx, a)
        return default(ctx, a)

    return rule


def _cholesky_compensated(a):
    n = a.shape[-1]
    rows = lax.broadcasted_iota(np.int32, a.shape[:-1], a.ndim - 2)

    def column(j, state):
        high, low = state
        row = tuple(lax.dynamic_index_in_dim(part, j, axis=a.ndim - 2, keepdims=False)
                    for part in state)
        matrix = tuple(_swap_last(part) for part in state)
        product = _precision.multiply_pairs(matrix, (row[0][..., :, None], row[1][..., :, None]))
        total = _precision.sum_pairs(*product, a.ndim - 2)
        a_col = lax.dynamic_index_in_dim(a, j, axis=a.ndim - 1, keepdims=False)
        value = _precision.add_pairs((a_col, lax.full_like(a_col, 0)), (-total[0], -total[1]))
        pivot = tuple(lax.dynamic_index_in_dim(part, j, axis=part.ndim - 1, keepdims=True)
                      for part in value)
        diagonal = _precision.sqrt_pair(pivot)
        divided = _precision.divide_pairs(value, diagonal)
        output = []
        for matrix_part, quotient, diag in zip((high, low), divided, diagonal):
            col = lax.select(rows > j, quotient, lax.select(
                rows == j, lax.broadcast_in_dim(diag, quotient.shape, tuple(range(diag.ndim))),
                lax.full_like(quotient, 0)))
            output.append(lax.dynamic_update_index_in_dim(matrix_part, col[..., None], j,
                                                          axis=a.ndim - 1))
        return tuple(output)

    high, low = lax.fori_loop(0, n, column, (lax.full_like(a, 0), lax.full_like(a, 0)))
    return high + low


def _cholesky(a, **_):
    # Column-by-column Cholesky-Banachiewicz on the lower triangle (Hermitian for complex:
    # l[i, j] = (a[i, j] - sum_k l[i, k] conj(l[j, k])) / l[j, j], with a real diagonal).
    # Columns not yet computed are zero, so l @ conj(l[j]) only sums over k < j.
    n = a.shape[-1]
    if a.size == 0:  # fori_loop traces its body even for zero trips, which cannot index here
        return a
    row_axis, col_axis = a.ndim - 2, a.ndim - 1
    rows = lax.broadcasted_iota(np.int32, a.shape[:-1], a.ndim - 2)
    complex_input = np.issubdtype(a.dtype, np.complexfloating)

    def column(j, l):
        a_col = lax.dynamic_index_in_dim(a, j, axis=col_axis, keepdims=False)
        l_row = lax.dynamic_index_in_dim(l, j, axis=row_axis, keepdims=False)
        if complex_input:
            l_row = lax.conj(l_row)
        matrix = lax.transpose(l, (*range(a.ndim - 2), col_axis, row_axis))
        v = a_col - _batch_matvec(matrix, l_row)
        pivot = lax.dynamic_index_in_dim(v, j, axis=v.ndim - 1, keepdims=True)
        if complex_input:
            d = lax.complex(lax.sqrt(lax.real(pivot)), lax.full_like(lax.real(pivot), 0))
        else:
            d = lax.sqrt(pivot)
        col = lax.select(rows > j, v / d, lax.select(rows == j, lax.broadcast_in_dim(
            d, v.shape, tuple(range(v.ndim))), lax.full_like(v, 0)))
        return lax.dynamic_update_index_in_dim(l, col[..., None], j, axis=col_axis)

    l = (_cholesky_compensated(a) if a.dtype == np.float32 and n <= 128 else
         lax.fori_loop(0, n, column, lax.full_like(a, 0)))
    # Like JAX on CPU and GPU, a matrix that is not positive definite gives all NaN (here
    # NaN would only start at the failing column).
    diagonal = jnp_diagonal(l)
    ok = lax.reduce_and((lax.real(diagonal) if complex_input else diagonal) > 0, (l.ndim - 2,))
    return lax.select(lax.broadcast_in_dim(ok, l.shape, tuple(range(ok.ndim))), l,
                      lax.full_like(l, np.nan))


def _swap_last(x):
    return lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


def _forward_substitution(a, b, unit_diagonal):
    """Solve a x = b for lower-triangular a [..., m, m], b [..., m, k]."""
    m = a.shape[-1]
    cols = lax.broadcasted_iota(np.int32, a.shape[:-1], a.ndim - 2)

    def row(i, x):
        a_row = lax.dynamic_index_in_dim(a, i, axis=a.ndim - 2, keepdims=False)
        # Only k < i contributes; mask the rest so packed LU factors (U above the
        # diagonal) and any non-finite values there cannot leak in.
        strict = lax.select(cols < i, a_row, lax.full_like(a_row, 0))
        rhs = lax.dynamic_index_in_dim(b, i, axis=b.ndim - 2, keepdims=False)
        xi = rhs - _batch_matvec(x, strict)
        if not unit_diagonal:
            diagonal = lax.dynamic_index_in_dim(a_row, i, axis=a_row.ndim - 1, keepdims=True)
            xi = xi / diagonal
        return lax.dynamic_update_index_in_dim(x, xi[..., None, :], i, axis=x.ndim - 2)

    return lax.fori_loop(0, m, row, lax.full_like(b, 0))


def _triangular_solve(a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal, **_):
    if b.size == 0:  # the solution has b's shape; an empty a makes b empty too
        return b
    if conjugate_a and np.issubdtype(a.dtype, np.complexfloating):
        a = lax.conj(a)
    if transpose_a:
        a, lower = _swap_last(a), not lower
    if not left_side:  # x a = b  <=>  a^T x^T = b^T
        a, b, lower = _swap_last(a), _swap_last(b), not lower
    if not lower:  # reversing rows and columns turns upper-triangular into lower
        a = lax.rev(a, (a.ndim - 2, a.ndim - 1))
        b = lax.rev(b, (b.ndim - 2,))
    x = _forward_substitution(a, b, unit_diagonal)
    if not lower:
        x = lax.rev(x, (x.ndim - 2,))
    if not left_side:
        x = _swap_last(x)
    return x


def _is_complex(x):
    return np.issubdtype(x.dtype, np.complexfloating)


def _conj(x):
    return lax.conj(x) if _is_complex(x) else x


def _householder(col, j):
    """LAPACK larfg for column `col` [..., m] at row j: reflector v (v[j] = 1), tau, beta.

    H = I - tau v v^H maps col[j:] to (beta, 0, ..., 0); beta is real.
    """
    rows = lax.broadcasted_iota(np.int32, col.shape, col.ndim - 1)
    alpha = lax.dynamic_index_in_dim(col, j, axis=col.ndim - 1, keepdims=False)
    below = rows > j
    x = lax.select(below, col, lax.full_like(col, 0))
    xnorm2 = lax.reduce_sum(lax.real(x * _conj(x)) if _is_complex(x) else x * x, (x.ndim - 1,))
    alpha_re = lax.real(alpha) if _is_complex(alpha) else alpha
    alpha_im = lax.imag(alpha) if _is_complex(alpha) else lax.full_like(alpha, 0)
    norm = lax.sqrt(alpha_re * alpha_re + alpha_im * alpha_im + xnorm2)
    beta = lax.select(alpha_re >= 0, -norm, norm)
    trivial = (xnorm2 == 0) & (alpha_im == 0)  # already reduced: H = I
    safe_beta = lax.select(trivial, lax.full_like(beta, 1), beta)
    beta_c = lax.convert_element_type(safe_beta, col.dtype)
    tau = lax.select(trivial, lax.full_like(alpha, 0), (beta_c - alpha) / beta_c)
    denominator = lax.select(trivial, lax.full_like(alpha, 1), alpha - beta_c)
    v = lax.select(below, x / denominator[..., None],
                   lax.select(rows == j, lax.full_like(col, 1), lax.full_like(col, 0)))
    beta_out = lax.select(trivial, alpha, lax.convert_element_type(beta, col.dtype))
    return v, tau, beta_out


def _qr_column(a, taus, j):
    """One step of Householder QR in LAPACK's packed form: reduce column j of a [..., m, n]
    below the diagonal, apply the reflector to the later columns, store it below the
    diagonal and its tau in taus[..., j]."""
    rows = lax.broadcasted_iota(np.int32, a.shape[:-1], a.ndim - 2)
    cols = lax.broadcasted_iota(np.int32, (*a.shape[:-2], a.shape[-1]), a.ndim - 2)
    col = lax.dynamic_index_in_dim(a, j, axis=a.ndim - 1, keepdims=False)
    v, tau, beta = _householder(col, j)
    w = _batch_matvec(a, _conj(v))  # v^H a, one entry per column
    update = (_conj(tau)[..., None, None] * v[..., :, None]) * w[..., None, :]
    a = lax.select(lax.broadcast_in_dim(cols > j, a.shape, (*range(a.ndim - 2), a.ndim - 1)),
                   a - update, a)
    new_col = lax.select(rows < j, col, lax.select(rows == j, lax.broadcast_in_dim(
        beta, col.shape, tuple(range(beta.ndim))), v))
    a = lax.dynamic_update_index_in_dim(a, new_col[..., None], j, axis=a.ndim - 1)
    taus = lax.dynamic_update_index_in_dim(taus, tau[..., None], j, axis=taus.ndim - 1)
    return a, taus


def _geqrf(a, **_):
    # Householder QR in LAPACK's packed form: R on and above the diagonal, reflectors below,
    # taus separately. The generic lowering emits XLA's "Qr" custom call, which aborts the
    # plugin (its CustomCallConverter only knows LU).
    m, n = a.shape[-2:]
    taus = lax.full(a.shape[:-2] + (min(m, n),), 0, a.dtype)
    if a.size == 0:
        return [a, taus]
    return list(lax.fori_loop(0, min(m, n), lambda j, carry: _qr_column(*carry, j), (a, taus)))


def _householder_product(a, taus, **_):
    # Q = H_0 H_1 ... H_{k-1} applied to the first n columns of the identity.
    m, n = a.shape[-2:]
    k = taus.shape[-1]
    rows = lax.broadcasted_iota(np.int32, a.shape[:-1], a.ndim - 2)
    q = lax.broadcast_in_dim(lax_internal._eye(a.dtype, (m, n)), a.shape,  # pylint: disable=protected-access
                             (a.ndim - 2, a.ndim - 1))
    if a.size == 0 or k == 0:
        return q

    def step(i, q):
        j = k - 1 - i
        col = lax.dynamic_index_in_dim(a, j, axis=a.ndim - 1, keepdims=False)
        v = lax.select(rows > j, col, lax.select(rows == j, lax.full_like(col, 1), lax.full_like(col, 0)))
        tau = lax.dynamic_index_in_dim(taus, j, axis=taus.ndim - 1, keepdims=False)
        w = _batch_matvec(q, _conj(v))  # v^H q
        return q - (tau[..., None, None] * v[..., :, None]) * w[..., None, :]

    return lax.fori_loop(0, k, step, q)


def _round_robin(n):
    """Tournament schedule for n (even) indices: n - 1 rounds of n/2 disjoint pairs.

    Returns, per round, the permutation [p_0..p_{n/2-1}, q_0..q_{n/2-1}] and its inverse.
    """
    others = list(range(1, n))
    perms, inverses = [], []
    for _ in range(n - 1):
        order = [0] + others
        pairs = [tuple(sorted((order[i], order[n - 1 - i]))) for i in range(n // 2)]
        perm = [p for p, _ in pairs] + [q for _, q in pairs]
        perms.append(perm)
        inverses.append(list(np.argsort(perm)))
        others = others[-1:] + others[:-1]
    return np.array(perms, np.int32), np.array(inverses, np.int32)


# Largest matrix _jacobi_eigh accepts. A 1008 x 1008 Jacobi (complex pinv of 7 x 1000 via
# Jordan-Wielandt) tied up the GPU and the Metal compiler service, which WindowServer shares,
# until its watchdog killed WindowServer; killing the process did not free them in time.
# Refusing at trace time is harmless; 2 x 307 ran in seconds without stalling the GPU.
JACOBI_MAX_SIZE = 256


def _jacobi_eigh(a, sort_eigenvalues):
    """Parallel cyclic Jacobi for a Hermitian matrix [..., n, n]: (eigenvectors, eigenvalues).

    Each round applies n/2 disjoint 2x2 rotations at once (round-robin pairing), as row and
    column gathers, so a sweep is O(n^3). Indices with no coupling get identity rotations,
    so padded identity blocks keep their positions (the QDWH base case relies on this).
    """
    n = a.shape[-1]
    if n > JACOBI_MAX_SIZE:
        raise NotImplementedError(
            f"METAL: Jacobi eigendecomposition of size {n} exceeds {JACOBI_MAX_SIZE} (larger "
            "ones can stall the GPU and WindowServer); complex eigh/SVD is limited to this size")
    complex_input = _is_complex(a)
    real_dtype = np.finfo(a.dtype).dtype
    padded = n + (n % 2)
    if padded != n:  # an uncoupled extra index makes the size even
        widths = [(0, 0, 0)] * (a.ndim - 2) + [(0, 1, 0), (0, 1, 0)]
        a = lax.pad(a, lax_internal._zero(a), widths)  # pylint: disable=protected-access
    half = padded // 2
    perms, inverses = _round_robin(padded)
    perms, inverses = lax.convert_element_type(perms, np.int32), lax.convert_element_type(inverses, np.int32)
    batch = a.shape[:-2]
    v = lax.broadcast_in_dim(lax_internal._eye(a.dtype, (padded, padded)), a.shape,  # pylint: disable=protected-access
                             (a.ndim - 2, a.ndim - 1))
    eps = np.finfo(real_dtype).eps

    def magnitude2(x):
        return lax.real(x * lax.conj(x)) if complex_input else x * x

    def off_norm2(a):
        mask = 1 - lax_internal._eye(real_dtype, (padded, padded))  # pylint: disable=protected-access
        return lax.reduce_sum(magnitude2(a) * mask, (a.ndim - 2, a.ndim - 1))

    def rotate(r, carry):
        a, v = carry
        perm = lax.dynamic_index_in_dim(perms, r, keepdims=False)
        inverse = lax.dynamic_index_in_dim(inverses, r, keepdims=False)
        p, q = perm[:half], perm[half:]
        diag = jnp_diagonal(a)
        app, aqq = lax.real(diag[..., p]) if complex_input else diag[..., p], \
            lax.real(diag[..., q]) if complex_input else diag[..., q]
        apq = a[..., p, q]
        # |apq| without squaring (squares of tiny couplings underflow and break the unit
        # phase below); couplings negligible next to the diagonal are left alone.
        abs_pq = lax.abs(apq)
        coupled = abs_pq > eps * 1e-2 * (lax.abs(app) + lax.abs(aqq)) + np.finfo(real_dtype).tiny
        safe = lax.select(coupled, abs_pq, lax.full_like(abs_pq, 1))
        tau = (aqq - app) / (2 * safe)
        sign = lax.select(tau >= 0, lax.full_like(tau, 1), lax.full_like(tau, -1))
        t = sign / (lax.abs(tau) + lax.sqrt(1 + tau * tau))
        t = lax.select(coupled, t, lax.full_like(t, 0))
        c = lax.rsqrt(1 + t * t)
        s = t * c
        # J = diag(1, conj(phase)) R with R = [[c, s], [-s, c]] and phase = apq / |apq|
        # (a unit complex number, or the sign for real matrices).
        phase = apq / lax.convert_element_type(safe, a.dtype)
        phase = lax.select(coupled, phase, lax.full_like(phase, 1))
        cc, ss = lax.convert_element_type(c, a.dtype), lax.convert_element_type(s, a.dtype)
        jpp, jpq = cc, ss
        jqp, jqq = -ss * _conj(phase), cc * _conj(phase)

        def columns(x):  # x J on the paired columns
            xp, xq = x[..., :, p], x[..., :, q]
            new_p = xp * jpp[..., None, :] + xq * jqp[..., None, :]
            new_q = xp * jpq[..., None, :] + xq * jqq[..., None, :]
            return lax.concatenate([new_p, new_q], x.ndim - 1)[..., :, inverse]

        def rows(x):  # J^H x on the paired rows
            xp, xq = x[..., p, :], x[..., q, :]
            new_p = _conj(jpp)[..., :, None] * xp + _conj(jqp)[..., :, None] * xq
            new_q = _conj(jpq)[..., :, None] * xp + _conj(jqq)[..., :, None] * xq
            return lax.concatenate([new_p, new_q], x.ndim - 2)[..., inverse, :]

        return rows(columns(a)), columns(v)

    def sweep(carry):
        a, v, count = carry
        a, v = lax.fori_loop(0, padded - 1, rotate, (a, v))
        return a, v, count + 1

    def not_converged(carry):
        a, _, count = carry
        total = lax.reduce_sum(magnitude2(a), (a.ndim - 2, a.ndim - 1))
        return (count < 20) & lax.reduce_or(off_norm2(a) > (eps * eps) * total, tuple(range(len(batch))))

    a, v, _ = lax.while_loop(not_converged, sweep, (a, v, 0))
    w = jnp_diagonal(a)
    w = lax.real(w) if complex_input else w
    w, v = w[..., :n], v[..., :n, :n]
    if sort_eigenvalues:
        order = lax.broadcasted_iota(np.int32, w.shape, w.ndim - 1)
        w, order = lax.sort((w, order), dimension=w.ndim - 1, num_keys=1)
        v = jnp_take_columns(v, order)
    return v, w


def jnp_diagonal(a):
    """Diagonal of [..., n, n] as [..., n] (gather on an iota index)."""
    idx = np.arange(a.shape[-1])
    return a[..., idx, idx]


def jnp_take_columns(v, order):
    """v[..., :, order[...]] for a batch of column orders."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    return jnp.take_along_axis(v, order[..., None, :], axis=-1)


def _complex_svd(a, *, full_matrices, compute_uv, subset_by_index, **_):
    """Complex SVD: QR down to a k x k core (k = min(m, n)), then _square_complex_svd.

    Without the QR step Jacobi would run on m + n instead of 2k; for a 7 x 1000 matrix that
    kept the GPU busy long enough to stall WindowServer.
    JAX's QDWH SVD crashes MPSGraph at run time for complex inputs.
    """
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    m, n = a.shape[-2:]
    k = min(m, n)
    if m > n:  # a = q core
        q, core = jnp.linalg.qr(a)
    elif m < n:  # a^H = q r, so a = r^H q^H
        q, r = jnp.linalg.qr(_conj(jnp.swapaxes(a, -1, -2)))
        core = _conj(jnp.swapaxes(r, -1, -2))
    else:
        core = a
    s, u, v = _square_complex_svd(core, compute_uv)
    if subset_by_index is not None:
        lo, hi = subset_by_index
        s = s[..., lo:hi]
    if not compute_uv:
        return [s]
    if m > n:
        u = q @ u
    elif m < n:
        v = q @ v
    if full_matrices and m > k:  # complete U to a unitary m x m with QR
        u = lax.concatenate([u, jnp.linalg.qr(u, mode="complete")[0][..., :, k:]], u.ndim - 1)
    if full_matrices and n > k:
        v = lax.concatenate([v, jnp.linalg.qr(v, mode="complete")[0][..., :, k:]], v.ndim - 1)
    if subset_by_index is not None:
        u, v = u[..., lo:hi], v[..., lo:hi]
    return [s, u, _conj(jnp.swapaxes(v, -1, -2))]


def _square_complex_svd(a, compute_uv):
    """SVD of a square complex [..., k, k] as (s, u, v) from the Jacobi eigendecomposition
    of [[0, A], [A^H, 0]].

    Its eigenvalues are +-sigma_i with eigenvectors [u_i; v_i] / sqrt(2), so the largest k
    eigenpairs give the SVD without squaring the condition number (unlike A^H A).
    """
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    k = a.shape[-1]
    zeros = lax.full(a.shape, 0, a.dtype)
    ah = _conj(jnp.swapaxes(a, -1, -2))
    b = lax.concatenate([lax.concatenate([zeros, a], a.ndim - 1),
                         lax.concatenate([ah, zeros], a.ndim - 1)], a.ndim - 2)
    vectors, values = _jacobi_eigh(b, sort_eigenvalues=True)
    s = lax.max(lax.rev(values[..., k:], (values.ndim - 1,)), lax.full_like(values[..., :k], 0))
    if not compute_uv:
        return s, None, None
    top = lax.rev(vectors[..., k:], (vectors.ndim - 1,))
    u, v = top[..., :k, :], top[..., k:, :]
    # Columns for sigma ~ 0 mix the u and v halves; re-orthonormalize with Householder QR.
    # Columns come in descending sigma, so QR keeps the well-defined ones (up to the phase
    # restored from R's diagonal) and replaces the rest with an orthonormal completion.
    def orthonormalize(x):
        q, r = jnp.linalg.qr(x)
        d = jnp.diagonal(r, axis1=-2, axis2=-1)
        magnitude = lax.abs(d)
        phase = lax.select(magnitude > 0, d / lax.convert_element_type(magnitude + 1e-30, d.dtype),
                           lax.full_like(d, 1))
        return q * phase[..., None, :]

    return s, orthonormalize(u), orthonormalize(v)


def _svd_rule():
    """SVD for METAL.

    Real inputs: JAX's generic SVD (QDWH polar + eigh), always run with a batch dimension;
    unbatched, its QDWH while loop keeps lax.cond branches, which crash MPSGraph at run time
    (GPU::WhileOpHandler); with a batch dimension vmap turns them into selects.
    Complex inputs: _complex_svd (complex QDWH also crashes MPSGraph).
    A requested `algorithm` (QR, divide and conquer, ...) is a LAPACK-style hint; METAL has
    one algorithm per case and uses it, where JAX's generic SVD would reject the request.
    """
    default = _default_rule(linalg.svd_p)

    def batch_of_one(a, **params):
        outs = linalg.svd_p.bind(a[None], **params)
        return [out[0] for out in outs]

    batched = mlir.lower_fun(batch_of_one, multiple_results=True)
    complex_rule = mlir.lower_fun(_complex_svd, multiple_results=True)

    def rule(ctx, a, **params):
        params["algorithm"] = None
        shape = ctx.avals_in[0].shape
        if 0 in shape[-2:]:
            return default(ctx, a, **params)
        if _complex.is_complex_aval(ctx.avals_in[0]):
            return complex_rule(ctx, a, **params)
        if len(shape) > 2:
            return default(ctx, a, **params)
        return batched(ctx, a, **params)

    return rule


def _eigh(x, *, lower, sort_eigenvalues, subset_by_index, algorithm=None, **_):
    n = x.shape[-1]
    # Rebuild the Hermitian matrix from the referenced triangle.
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    tri = jnp.tril(x) if lower else jnp.triu(x)
    strict = jnp.tril(x, -1) if lower else jnp.triu(x, 1)
    h = tri + _conj(jnp.swapaxes(strict, -1, -2))
    full_range = subset_by_index is None or tuple(subset_by_index) == (0, n)
    if h.size == 0:  # Jacobi's loops would index empty axes
        v, w = h, lax.full(h.shape[:-1], 0, np.finfo(h.dtype).dtype)
    # Jacobi for small matrices and for complex ones (complex QDWH crashes MPSGraph).
    elif n <= JACOBI_MAX_SIZE or algorithm == linalg.EighImplementation.JACOBI or _is_complex(h):
        v, w = _jacobi_eigh(h, sort_eigenvalues or not full_range)
    else:
        # JAX's QDWH spectral divide-and-conquer (used on TPU) is pure JAX; it needs QR
        # (rules above) and calls eigh with JACOBI on blocks of at most 256.
        # Always batched: unbatched QDWH keeps lax.cond inside its while loop, which crashes
        # MPSGraph at run time (GPU::WhileOpHandler); vmap turns those conds into selects.
        from jax._src.tpu.linalg import eigh as tpu_eigh  # pylint: disable=import-outside-toplevel
        batched = h if h.ndim > 2 else h[None]
        v, w = tpu_eigh._eigh_qdwh_impl(  # pylint: disable=protected-access
            batched, lower=lower, sort_eigenvalues=sort_eigenvalues, subset_by_index=subset_by_index)
        return [v, w] if h.ndim > 2 else [v[0], w[0]]
    if not full_range:
        lo, hi = subset_by_index
        v, w = v[..., lo:hi], w[..., lo:hi]
    return [v, w]


# --- Floating-point precision ----------------------------------------------------

_F32_EXPONENT_BITS, _F32_MANTISSA_BITS = 8, 23

# Formats reduced on their own bit pattern: float dtype -> (unsigned dtype, exponent bits,
# mantissa bits). bfloat16 goes through float32, which is exact for it (same exponent
# layout, mantissa = top 7 float32 mantissa bits).
_NATIVE_FORMATS = {
    np.dtype(np.float32): (np.uint32, _F32_EXPONENT_BITS, _F32_MANTISSA_BITS),
    np.dtype(np.float16): (np.uint16, 5, 10),
}


def _reduce_precision_bits(x, exponent_bits, mantissa_bits):
    """XLA's reduce_precision on x's own bits (round to nearest even, flush subnormals)."""
    uint, source_exponent_bits, source_mantissa_bits = _NATIVE_FORMATS[np.dtype(x.dtype)]
    total_bits = 1 + source_exponent_bits + source_mantissa_bits
    mask = (1 << total_bits) - 1
    unsigned = lambda value: uint(value & mask)
    bits = lax.bitcast_convert_type(x, uint)
    if mantissa_bits < source_mantissa_bits:
        dropped = source_mantissa_bits - mantissa_bits
        last_kept = unsigned(1 << dropped)
        last_bit = lax.shift_right_logical(lax.bitwise_and(bits, last_kept), unsigned(dropped))
        bits = lax.bitwise_and(bits + last_bit + unsigned((1 << (dropped - 1)) - 1),
                               unsigned(~(int(last_kept) - 1)))
    if exponent_bits < source_exponent_bits:
        exponent_mask = unsigned(((1 << source_exponent_bits) - 1) << source_mantissa_bits)
        bias = (1 << (source_exponent_bits - 1)) - 1
        reduced_bias = (1 << (exponent_bits - 1)) - 1
        exponent = lax.bitwise_and(bits, exponent_mask)
        overflows = exponent > unsigned((bias + reduced_bias) << source_mantissa_bits)
        underflows = exponent <= unsigned((bias - reduced_bias) << source_mantissa_bits)
        signed_zero = lax.bitwise_and(bits, unsigned(1 << (total_bits - 1)))
        signed_inf = lax.bitwise_or(signed_zero, exponent_mask)
        bits = lax.select(overflows, signed_inf, lax.select(underflows, signed_zero, bits))
    result = lax.bitcast_convert_type(bits, x.dtype)
    return lax.select(lax.ne(x, x), x, result)  # keep NaNs


def _reduce_precision(x, *, exponent_bits, mantissa_bits, **_):
    if np.dtype(x.dtype) in _NATIVE_FORMATS:
        return _reduce_precision_bits(x, exponent_bits, mantissa_bits)
    wide = lax.convert_element_type(x, np.float32)
    return lax.convert_element_type(_reduce_precision_bits(wide, exponent_bits, mantissa_bits),
                                    x.dtype)


_BFLOAT16 = np.dtype("bfloat16")


def _to_bfloat16(x):
    """Any real x -> bfloat16 via float32 bits: round to nearest even, keep subnormals."""
    wide = lax.convert_element_type(x, np.float32)
    rounded = lax.bitcast_convert_type(_reduce_precision_bits(wide, _F32_EXPONENT_BITS, 7),
                                       np.uint32)
    high = lax.convert_element_type(lax.shift_right_logical(rounded, np.uint32(16)), np.uint16)
    # A NaN whose payload sat in the dropped low bits would read as inf; force it quiet.
    high = lax.select(lax.ne(wide, wide), lax.bitwise_or(high, np.uint16(0x0040)), high)
    return lax.bitcast_convert_type(high, _BFLOAT16)


def _from_bfloat16(x, dtype):
    """bfloat16 -> float32 (exact, by widening the bits), then to `dtype`."""
    bits = lax.convert_element_type(lax.bitcast_convert_type(x, np.uint16), np.uint32)
    wide = lax.bitcast_convert_type(lax.shift_left(bits, np.uint32(16)), np.float32)
    return wide if np.dtype(dtype) == np.float32 else lax.convert_element_type(wide, dtype)


def _bfloat16_sign_rule(primitive):
    default = _complex.current_rule(primitive)

    def change_sign(x):
        bits = lax.bitcast_convert_type(x, np.uint16)
        bits = (lax.bitwise_and(bits, np.uint16(0x7fff)) if primitive is lax_internal.abs_p
                else lax.bitwise_xor(bits, np.uint16(0x8000)))
        return lax.bitcast_convert_type(bits, _BFLOAT16)

    exact = mlir.lower_fun(change_sign, multiple_results=False)

    def rule(ctx, x):
        if ctx.avals_in[0].dtype == _BFLOAT16:
            return exact(ctx, x)
        return default(ctx, x)

    return rule


def _bfloat16_select_rule():
    default = _complex.current_rule(lax_internal.select_n_p)

    def select_bits(selector, *cases):
        chosen = lax.select_n(selector, *(lax.bitcast_convert_type(x, np.uint16) for x in cases))
        return lax.bitcast_convert_type(chosen, _BFLOAT16)

    exact = mlir.lower_fun(select_bits, multiple_results=False)

    def rule(ctx, *args):
        if ctx.avals_out[0].dtype == _BFLOAT16:
            return exact(ctx, *args)
        return default(ctx, *args)

    return rule


def _convert_element_type(ctx, operand, *, new_dtype, **params):
    # The plugin emulates bfloat16 in float32. Its own float<->bfloat16 casts do not round
    # when the result stays in the program, and flush subnormals at buffer boundaries.
    # bfloat16 arithmetic and bitcasts are exact, so convert through the bit pattern.
    if not isinstance(ctx.avals_in[0].dtype, np.dtype) or not isinstance(np.dtype(new_dtype), np.dtype):
        return _default_rule(lax_internal.convert_element_type_p)(ctx, operand, new_dtype=new_dtype,
                                                                  **params)
    source, target = np.dtype(ctx.avals_in[0].dtype), np.dtype(new_dtype)
    real = lambda dtype: dtype.kind in "biuf"
    if target == _BFLOAT16 and source != _BFLOAT16 and real(source):
        return mlir.lower_fun(_to_bfloat16, multiple_results=False)(ctx, operand)
    if source == _BFLOAT16 and target != _BFLOAT16 and real(target):
        return mlir.lower_fun(partial(_from_bfloat16, dtype=target), multiple_results=False)(
            ctx, operand)
    if source.kind == "c" and target.kind != "c":  # bfloat16 (kind "V") included
        # JAX's own rule emits mhlo.real, which the plugin cannot lower; use the METAL real_p.
        return mlir.lower_fun(lambda z: lax.convert_element_type(lax.real(z), target),
                              multiple_results=False)(ctx, operand)
    return _default_rule(lax_internal.convert_element_type_p)(ctx, operand, new_dtype=new_dtype,
                                                              **params)


# --- Random bits -----------------------------------------------------------------
# The plugin lowers rng_bit_generator to MPS's random kernel, which aborts the process
# ("MPSNDArrayRandom.mm: Invalid result MPSDataType") even when asked for signed bits.
# JAX defines rng_bit_generator output as backend-specific, so METAL generates it with
# Threefry-2x32 in counter mode instead: key words 0-1 (with word 3 mixed in) key the
# hash, word 2 is the counter offset, and the returned key advances the counter.

def _rng_bit_generator(key, *, shape, dtype, algorithm, **_):
    del algorithm  # always Threefry on METAL
    dtype = np.dtype(dtype)
    if key.shape != (4,) or key.dtype != np.uint32:
        raise NotImplementedError(f"METAL rng_bit_generator expects a uint32[4] key, got "
                                  f"{key.dtype}{list(key.shape)}")
    words_per_value = 2 if dtype.itemsize == 8 else 1
    count = int(np.prod(shape, dtype=np.int64)) * words_per_value
    padded = count + (count % 2)  # threefry hashes pairs of words
    counter = (lax.iota(np.uint32, padded) + key[2]).reshape(-1, 2)
    # threefry_2x32 pairs the first half of its input with the second; pairing consecutive
    # counters instead makes word i depend only on the key and i, so a larger draw starts
    # with the smaller one, as with XLA's generator (JAX's vmap of unsafe_rbg relies on it).
    hashed = threefry_2x32((key[0], lax.bitwise_xor(key[1], key[3])),
                           lax.concatenate([counter[:, 0], counter[:, 1]], 0))
    words = hashed.reshape(2, -1).T.reshape(-1)[:count]
    if words_per_value == 2:
        pairs = lax.convert_element_type(words.reshape(-1, 2), np.uint64)
        values = lax.bitwise_or(lax.shift_left(pairs[:, 0], np.uint64(32)), pairs[:, 1])
    elif dtype.itemsize < 4:
        values = lax.convert_element_type(
            lax.bitwise_and(words, np.uint32((1 << (8 * dtype.itemsize)) - 1)), dtype)
    else:
        values = words
    values = lax.bitcast_convert_type(values, dtype) if values.dtype != dtype else values
    new_key = key.at[2].add(np.uint32(padded & 0xFFFFFFFF))
    return [new_key, values.reshape(shape)]


def _in_float32(fn):
    """fn computed in float32 for float16/bfloat16 inputs: the formulas above assume
    float32's range (x * x overflows float16 above 256) and precision."""
    def wrapped(x, **params):
        if np.dtype(x.dtype) in (np.dtype(np.float16), _BFLOAT16):
            wide = fn(lax.convert_element_type(x, np.float32), **params)
            return lax.convert_element_type(wide, x.dtype)
        return fn(x, **params)
    return wrapped


_UNARY = (
    (lax_internal.acos_p, _acos),
    (lax_internal.acosh_p, _acosh),
    (lax_internal.asinh_p, _asinh),
    (lax_internal.atanh_p, _atanh),
    (lax_internal.sinh_p, _sinh),
    (lax_internal.cosh_p, _cosh),
    (lax_internal.cbrt_p, _cbrt),
    (lax_internal.log1p_p, _log1p),
    (lax_internal.expm1_p, _expm1),
    (lax_internal.tanh_p, _tanh),
    (special.erf_p, _erf),
    (special.erfc_p, _erfc),
    (lax_internal.population_count_p, _population_count),
    (lax_internal.clz_p, _count_leading_zeros),
)


# --- Requests StableHLO 0.19.4 cannot express --------------------------------------------
# jaxlib serializes programs for the plugin at StableHLO 0.19.4, and one feature it cannot
# express there fails the whole program. A requested result accuracy (lax.exp(x,
# accuracy=...), StableHLO 1.9) is dropped: MPS has one float32 implementation per function.
# (The METAL rules for expm1, log1p, tanh and cbrt ignore it already.)

_ACCURACY = (lax_internal.exp_p, lax_internal.exp2_p, lax_internal.log_p,
             lax_internal.logistic_p, lax_internal.sin_p, lax_internal.cos_p,
             lax_internal.tan_p, lax_internal.sqrt_p, lax_internal.rsqrt_p)


def _without_accuracy(rule):
    def wrapped(ctx, *args, **params):
        if params.get("accuracy") is not None:
            params["accuracy"] = None
        return rule(ctx, *args, **params)
    return wrapped


def _inline_composite(*args, call_jaxpr, **_):
    # The plugin does not expand stablehlo.composite (nothing in its pipeline does), so a
    # user's lax.composite runs as its decomposition, as on backends without the op.
    return core.jaxpr_as_fun(call_jaxpr)(*args)


def _real_or_complex(primitive, real_rule, complex_fn):
    """One METAL rule that picks the complex implementation when any input is complex."""
    complex_rule = mlir.lower_fun(lambda *args, **_: complex_fn(*args), multiple_results=False)

    def rule(ctx, *args, **params):
        if any(_complex.is_complex_aval(aval) for aval in ctx.avals_in):
            return complex_rule(ctx, *args, **params)
        return real_rule(ctx, *args, **params)

    return rule


def register():
    for primitive, combine in _CUMULATIVE:
        _register(primitive, partial(_cumulative_scan, combine))
    # Real-dtype rules for elementwise primitives; complex inputs dispatch to _complex.
    real_rules = {primitive: mlir.lower_fun(_in_float32(fn), multiple_results=False)
                  for primitive, fn in _UNARY}
    real_rules[lax_internal.sign_p] = _with_nan_passthrough(lax_internal.sign_p)
    for primitive in set(real_rules) | set(_complex.ELEMENTWISE) | set(_ACCURACY):
        real_rule = real_rules.get(primitive) or _without_accuracy(_default_rule(primitive))
        if primitive in _complex.ELEMENTWISE:
            real_rule = _real_or_complex(primitive, real_rule, _complex.ELEMENTWISE[primitive])
        _register_rule(primitive, real_rule)
    _register(lax_internal.top_k_p, _top_k, multiple_results=True)
    _register_rule(ann.approx_top_k_p, _approx_top_k_rule())
    _register(lax_internal.mulhi_p, _mulhi)
    _register(linalg.cholesky_p, _cholesky)
    _register_rule(linalg.lu_p, _lu_rule())
    _register(linalg.triangular_solve_p, _triangular_solve)
    _register(linalg.geqrf_p, _geqrf, multiple_results=True)
    _register(linalg.householder_product_p, _householder_product)
    _register(linalg.eigh_p, _eigh, multiple_results=True)
    _register_rule(linalg.svd_p, _svd_rule())
    _register(lax_internal.reduce_precision_p, _reduce_precision)
    _register_rule(lax_internal.convert_element_type_p, _convert_element_type)
    _register(lax_internal.rng_bit_generator_p, _rng_bit_generator, multiple_results=True)
    _register(lax_internal.composite_p, _inline_composite, multiple_results=True)
    for primitive, fix_fn in _NAN_PROPAGATING:
        _register_rule(primitive, _nan_propagating(primitive, fix_fn))
    from jax_plugins.metal_shim import _windows  # pylint: disable=import-outside-toplevel
    _windows.register(_register_rule)
    for primitive in (slicing.scatter_max_p, slicing.scatter_min_p):
        _register_rule(primitive, _scatter_nan_propagating(primitive))
    _collectives.register(_register_rule)
    from jax_plugins.metal_shim import _decompositions  # pylint: disable=import-outside-toplevel
    _decompositions.register(_register_rule)
    _complex.register()
    for primitive in (lax_internal.abs_p, lax_internal.neg_p):
        _register_rule(primitive, _bfloat16_sign_rule(primitive))
    _register_rule(lax_internal.select_n_p, _bfloat16_select_rule())
    from jax_plugins.metal_shim import _indexing  # pylint: disable=import-outside-toplevel
    _indexing.register(_register_rule)
    from jax_plugins.metal_shim import _ordering  # pylint: disable=import-outside-toplevel
    _ordering.register(_register_rule)
    from jax_plugins.metal_shim import _reductions  # pylint: disable=import-outside-toplevel
    _reductions.register(_register_rule)
    from jax_plugins.metal_shim import _convolution  # pylint: disable=import-outside-toplevel
    _convolution.register(_register_rule)
    # JAX's generic loop-based lowerings, traced the way this package's rules are (no
    # varying-axes checks inside shard_map; see _complex.allow_rank_promotion).
    for primitive in (linalg.lu_p, linalg.lu_pivots_to_permutation_p, linalg.cholesky_update_p):
        _register_rule(primitive, _complex.current_rule(primitive))
    # Last, so that it wraps the final rule of each primitive.
    for primitive in _REGION_PRIMITIVES:
        _register_rule(primitive, _marking_region_bodies(_complex.current_rule(primitive)))
