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
from jax._src.lax import special
from jax._src.lax.control_flow import loops

PLATFORM = "METAL"


def _register(primitive, fn, multiple_results=False):
    mlir.register_lowering(primitive, mlir.lower_fun(fn, multiple_results=multiple_results),
                           platform=PLATFORM)


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

# --- CHLO ops serialized as composites -----------------------------------------
# XLA's serializer wraps these CHLO ops in stablehlo.composite
# (ChloPreserveHighLevelOpsPass), which the plugin cannot lower. Express them with
# ops the plugin supports instead.

_LARGE = 1e8  # beyond this, x*x overflows float32 in the formulas below


def _acos(x, **_):
    # atan2 form is exact at both ends: acos(1) = 0, acos(-1) = pi.
    return lax.atan2(lax.sqrt((1 - x) * (1 + x)), x)


def _acosh(x, **_):
    small = lax.log(x + lax.sqrt((x - 1) * (x + 1)))
    return lax.select(x > _LARGE, lax.log(x) + np.log(2).astype(x.dtype), small)


def _asinh(x, **_):
    a = lax.abs(x)
    small = lax.log1p(a + a * a / (1 + lax.sqrt(1 + a * a)))
    return lax.sign(x) * lax.select(a > _LARGE, lax.log(a) + np.log(2).astype(x.dtype), small)


def _atanh(x, **_):
    return 0.5 * lax.log1p(2 * x / (1 - x))


def _sinh(x, **_):
    return 0.5 * (lax.expm1(x) - lax.expm1(-x))


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
    x = lax.clamp(lax.full_like(x, -4.0), x, lax.full_like(x, 4.0))
    x2 = x * x
    return x * _horner(x2, _ERF_ALPHA) / _horner(x2, _ERF_BETA)


def _erfc(x, **_):
    # Accurate in absolute terms; relative accuracy drops where erfc(x) < ~1e-7 (x > ~3.8).
    return 1 - _erf(x)


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


_UNARY = (
    (lax_internal.acos_p, _acos),
    (lax_internal.acosh_p, _acosh),
    (lax_internal.asinh_p, _asinh),
    (lax_internal.atanh_p, _atanh),
    (lax_internal.sinh_p, _sinh),
    (lax_internal.cosh_p, _cosh),
    (lax_internal.cbrt_p, _cbrt),
    (special.erf_p, _erf),
    (special.erfc_p, _erfc),
    (lax_internal.population_count_p, _population_count),
    (lax_internal.clz_p, _count_leading_zeros),
)


def register():
    for primitive, combine in _CUMULATIVE:
        _register(primitive, partial(_cumulative_scan, combine))
    for primitive, fn in _UNARY:
        _register(primitive, fn)
    _register(lax_internal.top_k_p, _top_k, multiple_results=True)
