"""METAL-specific lowering rules for ops the plugin's HLO->MPS lowering rejects.

Each rule rewrites a JAX primitive into ops the plugin already handles. See
re/findings/mps-op-coverage.md for what the plugin can lower and
re/findings/jax-op-probe.md for what fails without these rules.
"""

from functools import partial

import numpy as np
from jax import lax
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal
from jax._src.lax import linalg
from jax._src.lax import special
from jax._src.lax.control_flow import loops
from jax._src.random.threefry2x32 import threefry_2x32

from jax_plugins.metal_shim import _complex

PLATFORM = "METAL"


def _register(primitive, fn, multiple_results=False):
    mlir.register_lowering(primitive, mlir.lower_fun(fn, multiple_results=multiple_results),
                           platform=PLATFORM)


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
    # tanh(x) = expm1(2x) / (expm1(2x) + 2); |x| >= 10 is +-1 in float32.
    clamped = lax.clamp(lax.full_like(x, -10.0), x, lax.full_like(x, 10.0))
    e = _expm1(2 * clamped)
    return lax.select(lax.ne(x, x), x, e / (e + 2))


def _with_nan_passthrough(primitive):
    """Run JAX's own lowering, then return NaN wherever the input was NaN."""
    default = _default_rule(primitive)
    fix = mlir.lower_fun(lambda x, y: lax.select(lax.ne(x, x), x, y), multiple_results=False)

    def rule(ctx, x, **params):
        y, = default(ctx, x, **params)
        return fix(ctx.replace(avals_in=[ctx.avals_in[0], ctx.avals_out[0]]), x, y)

    return rule


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
    if np.issubdtype(x.dtype, np.integer):
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


def _cholesky(a, **_):
    # Column-by-column Cholesky-Banachiewicz on the lower triangle. Columns not yet
    # computed are zero, so l @ l[j] only sums over k < j.
    n = a.shape[-1]
    row_axis, col_axis = a.ndim - 2, a.ndim - 1
    rows = lax.broadcasted_iota(np.int32, a.shape[:-1], a.ndim - 2)

    def column(j, l):
        a_col = lax.dynamic_index_in_dim(a, j, axis=col_axis, keepdims=False)
        l_row = lax.dynamic_index_in_dim(l, j, axis=row_axis, keepdims=False)
        v = a_col - _batch_matvec(lax.transpose(l, (*range(a.ndim - 2), col_axis, row_axis)), l_row)
        d = lax.sqrt(lax.dynamic_index_in_dim(v, j, axis=v.ndim - 1, keepdims=True))
        col = lax.select(rows > j, v / d, lax.select(rows == j, lax.broadcast_in_dim(
            d, v.shape, tuple(range(v.ndim))), lax.full_like(v, 0)))
        return lax.dynamic_update_index_in_dim(l, col[..., None], j, axis=col_axis)

    return lax.fori_loop(0, n, column, lax.full_like(a, 0))


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
            xi = xi / lax.dynamic_index_in_dim(a_row, i, axis=a_row.ndim - 1, keepdims=True)
        return lax.dynamic_update_index_in_dim(x, xi[..., None, :], i, axis=x.ndim - 2)

    return lax.fori_loop(0, m, row, lax.full_like(b, 0))


def _triangular_solve(a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal, **_):
    del conjugate_a  # real dtypes only; the plugin has no complex support
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


def _convert_element_type(ctx, operand, *, new_dtype, **params):
    # The plugin emulates bfloat16 in float32. Its own float<->bfloat16 casts do not round
    # when the result stays in the program, and flush subnormals at buffer boundaries.
    # bfloat16 arithmetic and bitcasts are exact, so convert through the bit pattern.
    source, target = np.dtype(ctx.avals_in[0].dtype), np.dtype(new_dtype)
    real = lambda dtype: dtype.kind in "biuf"
    if target == _BFLOAT16 and source != _BFLOAT16 and real(source):
        return mlir.lower_fun(_to_bfloat16, multiple_results=False)(ctx, operand)
    if source == _BFLOAT16 and target != _BFLOAT16 and real(target):
        return mlir.lower_fun(partial(_from_bfloat16, dtype=target), multiple_results=False)(
            ctx, operand)
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
    counter = lax.iota(np.uint32, padded) + key[2]
    words = threefry_2x32((key[0], lax.bitwise_xor(key[1], key[3])), counter)[:count]
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


def _real_or_complex(primitive, real_rule, complex_fn):
    """One METAL rule that picks the complex implementation when any input is complex."""
    complex_rule = mlir.lower_fun(lambda *args, **_: complex_fn(*args), multiple_results=False)

    def rule(ctx, *args, **params):
        if any(np.issubdtype(aval.dtype, np.complexfloating) for aval in ctx.avals_in):
            return complex_rule(ctx, *args, **params)
        return real_rule(ctx, *args, **params)

    return rule


def register():
    for primitive, combine in _CUMULATIVE:
        _register(primitive, partial(_cumulative_scan, combine))
    # Real-dtype rules for elementwise primitives; complex inputs dispatch to _complex.
    real_rules = {primitive: mlir.lower_fun(fn, multiple_results=False) for primitive, fn in _UNARY}
    real_rules[lax_internal.sign_p] = _with_nan_passthrough(lax_internal.sign_p)
    for primitive in set(real_rules) | set(_complex.ELEMENTWISE):
        real_rule = real_rules.get(primitive) or _default_rule(primitive)
        if primitive in _complex.ELEMENTWISE:
            real_rule = _real_or_complex(primitive, real_rule, _complex.ELEMENTWISE[primitive])
        mlir.register_lowering(primitive, real_rule, platform=PLATFORM)
    _register(lax_internal.top_k_p, _top_k, multiple_results=True)
    _register(linalg.cholesky_p, _cholesky)
    _register(linalg.triangular_solve_p, _triangular_solve)
    _register(lax_internal.reduce_precision_p, _reduce_precision)
    mlir.register_lowering(lax_internal.convert_element_type_p, _convert_element_type,
                           platform=PLATFORM)
    _register(lax_internal.rng_bit_generator_p, _rng_bit_generator, multiple_results=True)
    _complex.register()
