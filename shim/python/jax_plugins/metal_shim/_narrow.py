"""float8, float4 and 2-/4-bit integer types on METAL, emulated.

MPS has none of these types and the plugin rejects any program that mentions one; jaxlib
cannot even serialize the newer ones (float8_e4m3, e3m4, e8m0fnu, float4_e2m1fn, int2,
uint2) for StableHLO 0.19.4. Programs that use them are rewritten before lowering: each
value of a narrow type is carried in a type the plugin has (float32 for the floating-point
formats, int8/uint8 for the integers) holding exactly a value of the narrow type, and every
operation that produces a narrow value rounds its carrier result the way XLA converts to
that type:

- floating point: round to nearest even, subnormals included; out of range is inf for the
  IEEE-like formats (e5m2, e4m3, e3m4), NaN for the finite-only ones (fn, fnuz, e8m0fnu),
  the largest value for float4_e2m1fn (which has no NaN either: NaN becomes -0); fnuz
  formats have no -0;
- integers: wrap from integers (two's complement); from floating point, truncate toward
  zero and saturate, NaN giving 0.

Arithmetic on narrow values therefore runs in float32 (or 8-bit integers) and is rounded
once per operation. (XLA's CPU and GPU backends compute float8 arithmetic in float16, so a
result can differ from theirs in the last bit where float16 rounding creates a tie.)
Operations that only move values (reshape, slice, gather, select, max, ...) need no
rounding; reductions, windows and scatters combine in the carrier type and round once.
bitcast_convert_type encodes and decodes the formats' bits exactly.

Narrow program inputs and outputs keep their types in the rewritten program, converted at
the edges. On METAL those edges never reach the plugin: _boundary_io.py has the plugin run
a version whose narrow inputs and outputs are their bytes (uint8, or int8 for the signed
integers), decoded and encoded inside, and the shim keeps narrow buffers as those bytes.
"""

import dataclasses

import jax
import ml_dtypes
import numpy as np
from jax import lax
from jax._src import core
from jax._src import source_info_util
from jax._src.interpreters import mlir

PLATFORM = "METAL"


@dataclasses.dataclass(frozen=True)
class Format:
    """A narrow type and how values convert to it."""
    dtype: np.dtype
    bits: int
    integer: bool
    signed: bool = True
    nmant: int = 0  # floating point from here on
    minexp: int = 0  # exponent of the smallest normal number
    max: float = 0.0
    subnormal: float = 0.0  # the smallest positive value with a 0 exponent field
    overflow: str = "inf"  # "inf", "nan" or "saturate"
    unsigned_zero: bool = False  # fnuz: no -0
    nan_to: float = np.nan  # what NaN converts to

    @property
    def carrier(self):
        """The type holding this type's values inside a rewritten program."""
        if not self.integer:
            return np.dtype(np.float32)
        return np.dtype(np.int8 if self.signed else np.uint8)

    @property
    def storage(self):
        """The type of this type's bytes (one value per byte, the code in the low bits)."""
        return np.dtype(np.int8 if self.integer and self.signed else np.uint8)


def _float_format(dtype, bits, overflow, unsigned_zero=False, nan_to=np.nan):
    dtype = np.dtype(dtype)
    info = ml_dtypes.finfo(dtype)
    signed = dtype != np.dtype(ml_dtypes.float8_e8m0fnu)
    return Format(dtype, bits, False, signed, int(info.nmant), int(info.minexp),
                  float(info.max), float(2.0 ** (int(info.minexp) - int(info.nmant))),
                  overflow, unsigned_zero, nan_to)


FORMATS = {f.dtype: f for f in (
    _float_format(ml_dtypes.float8_e4m3fn, 8, "nan"),
    _float_format(ml_dtypes.float8_e5m2, 8, "inf"),
    _float_format(ml_dtypes.float8_e4m3, 8, "inf"),
    _float_format(ml_dtypes.float8_e3m4, 8, "inf"),
    _float_format(ml_dtypes.float8_e4m3fnuz, 8, "nan", unsigned_zero=True),
    _float_format(ml_dtypes.float8_e5m2fnuz, 8, "nan", unsigned_zero=True),
    _float_format(ml_dtypes.float8_e4m3b11fnuz, 8, "nan", unsigned_zero=True),
    _float_format(ml_dtypes.float8_e8m0fnu, 8, "nan"),
    _float_format(ml_dtypes.float4_e2m1fn, 4, "saturate", nan_to=-0.0),
    Format(np.dtype(ml_dtypes.int4), 4, True, True),
    Format(np.dtype(ml_dtypes.uint4), 4, True, False),
    Format(np.dtype(ml_dtypes.int2), 2, True, True),
    Format(np.dtype(ml_dtypes.uint2), 2, True, False),
)}


def narrow_format(dtype):
    """The Format of a narrow dtype, else None (extended dtypes such as PRNG keys too)."""
    try:
        return FORMATS.get(np.dtype(dtype))
    except TypeError:
        return None


def _narrow_aval(aval):
    return narrow_format(getattr(aval, "dtype", None)) is not None


# --- Values of a format ------------------------------------------------------------------

def _round_float(x, fmt):
    """x (any real type) rounded to the nearest value of floating-point format fmt."""
    wide = np.dtype(np.float64 if np.dtype(x.dtype) == np.float64 else np.float32)
    uint, mantissa = (np.uint64, 52) if wide == np.float64 else (np.uint32, 23)
    x = lax.convert_element_type(x, wide)  # exact from narrower floats
    sign_bit = wide.itemsize * 8 - 1
    bits = lax.bitcast_convert_type(x, uint)
    # Magnitudes are compared as bits: a backend may flush float32 subnormals to zero.
    magnitude = lax.bitwise_and(bits, uint((1 << sign_bit) - 1))
    as_bits = lambda value: uint(np.array(value, wide).view(uint))
    shift = mantissa - fmt.nmant
    keep = uint(~((1 << shift) - 1) & np.iinfo(uint).max)
    half = uint(1 << (shift - 1))
    if fmt.nmant == 0:
        return _round_e8m0(magnitude, bits, x, uint, mantissa, sign_bit, as_bits)
    # Round the mantissa to nmant bits, ties to even.
    lsb = lax.bitwise_and(lax.shift_right_logical(magnitude, uint(shift)), uint(1))
    value = lax.bitcast_convert_type(lax.bitwise_and(magnitude + (half - uint(1)) + lsb, keep),
                                     wide)
    # Below the normal range: the nearest multiple of the smallest subnormal.
    scale = wide.type(fmt.subnormal)
    ax = lax.bitcast_convert_type(magnitude, wide)
    subnormal = lax.round(ax / scale, lax.RoundingMethod.TO_NEAREST_EVEN) * scale
    value = lax.select(magnitude < as_bits(2.0 ** fmt.minexp), subnormal, value)
    over = {"inf": np.inf, "nan": np.nan, "saturate": fmt.max}[fmt.overflow]
    value = lax.select(value > wide.type(fmt.max), lax.full_like(value, over), value)
    negative = lax.ne(lax.shift_right_logical(bits, uint(sign_bit)), uint(0))
    value = lax.select(negative, -value, value)
    if fmt.unsigned_zero:  # -0 becomes +0
        value = lax.select(lax.eq(value, wide.type(0)), lax.full_like(value, 0), value)
    value = lax.select(lax.ne(x, x), lax.full_like(value, fmt.nan_to), value)
    return lax.convert_element_type(value, np.float32)


def _round_e8m0(magnitude, bits, x, uint, mantissa, sign_bit, as_bits):
    """float8_e8m0fnu (powers of two from 2^-127, no 0, no negative values), as ml_dtypes
    and XLA convert float32: ties round up, everything up to 2^-127 is 2^-127 and what lies
    between 2^-127 and 2^-126 is 2^-126. Built from bits: 2^-127 is a float32 subnormal,
    which a backend may flush to zero in any arithmetic."""
    bias = 1023 if mantissa == 52 else 127
    exponent = lax.convert_element_type(
        lax.shift_right_logical(magnitude + uint(1 << (mantissa - 1)), uint(mantissa)), np.int32)
    code = exponent - np.int32(bias - 127)
    code = lax.select(magnitude < as_bits(2.0 ** -126), lax.full_like(code, 1), code)
    code = lax.select(magnitude <= as_bits(2.0 ** -127), lax.full_like(code, 0), code)
    invalid = (lax.ne(lax.shift_right_logical(bits, uint(sign_bit)), uint(0))
               | lax.eq(magnitude, uint(0)) | lax.ne(x, x) | (code > np.int32(254)))
    carrier = lax.select(lax.eq(code, np.int32(0)), lax.full_like(code, 0x00400000),
                         lax.shift_left(code, np.int32(23)))
    carrier = lax.select(invalid, lax.full_like(carrier, 0x7FC00000), carrier)
    return lax.bitcast_convert_type(carrier, np.float32)


def _wrap_int(x, fmt):
    """Integer x wrapped to fmt's bits (two's complement), in fmt's carrier."""
    x = lax.convert_element_type(x, np.int32)  # keeps the low 32 bits
    free = np.int32(32 - fmt.bits)
    x = lax.shift_left(x, free)
    x = (lax.shift_right_arithmetic if fmt.signed else lax.shift_right_logical)(x, free)
    return lax.convert_element_type(x, fmt.carrier)


def _saturate_int(x, fmt):
    """Floating-point x truncated toward zero and clamped to fmt's range; NaN gives 0."""
    low = -(2 ** (fmt.bits - 1)) if fmt.signed else 0
    high = 2 ** (fmt.bits - 1) - 1 if fmt.signed else 2 ** fmt.bits - 1
    dtype = np.dtype(x.dtype)
    x = lax.select(lax.lt(x, dtype.type(0)), lax.ceil(x), lax.floor(x))
    x = lax.clamp(dtype.type(low), x, dtype.type(high))
    x = lax.select(lax.ne(x, x), lax.full_like(x, 0), x)
    return lax.convert_element_type(x, fmt.carrier)


def to_format(x, fmt):
    """x (a carrier or any real value) converted to fmt, as its carrier."""
    if not fmt.integer:
        return _round_float(x, fmt)
    if np.issubdtype(np.dtype(x.dtype), np.floating):
        return _saturate_int(x, fmt)
    return _wrap_int(x, fmt)


# --- Bits of a format --------------------------------------------------------------------

def _table(fmt):
    """The float32 value of every code."""
    return np.arange(2 ** fmt.bits, dtype=np.uint8).view(fmt.dtype).astype(np.float32)


def _code_of(value, fmt):
    """The code ml_dtypes gives a float32 value (canonical NaN, +0, ...)."""
    return int(np.array(value, np.float32).astype(fmt.dtype).view(np.uint8)) & (2 ** fmt.bits - 1)


def decode(stored, fmt):
    """Carriers of the values whose codes are `stored` (one per byte, in the low bits)."""
    if fmt.integer:
        return _wrap_int(stored, fmt)
    index = lax.convert_element_type(
        lax.bitwise_and(lax.convert_element_type(stored, np.int32), np.int32(2 ** fmt.bits - 1)),
        np.int32)
    return jax.numpy.take(jax.numpy.asarray(_table(fmt)), index, mode="clip")


def encode(carrier, fmt):
    """The codes, in fmt's storage type, of carriers holding exact values of fmt."""
    if fmt.integer:
        return lax.convert_element_type(
            lax.bitwise_and(lax.convert_element_type(carrier, np.int32),
                            np.int32(2 ** fmt.bits - 1)), fmt.storage)
    x = lax.convert_element_type(carrier, np.float32)
    bits = lax.bitcast_convert_type(x, np.uint32)
    magnitude = lax.bitwise_and(bits, np.uint32(0x7FFFFFFF))
    exponent = lax.convert_element_type(lax.shift_right_logical(magnitude, np.uint32(23)),
                                        np.int32) - np.int32(127)
    if fmt.subnormal < 2.0 ** fmt.minexp:  # exponent field 0 holds the subnormals
        mantissa = lax.convert_element_type(
            lax.shift_right_logical(magnitude, np.uint32(23 - fmt.nmant)), np.int32)
        mantissa = lax.bitwise_and(mantissa, np.int32(2 ** fmt.nmant - 1))
        code = lax.shift_left(exponent - np.int32(fmt.minexp - 1), np.int32(fmt.nmant)) | mantissa
        ax = lax.bitcast_convert_type(magnitude, np.float32)
        subnormal = lax.convert_element_type(ax / np.float32(fmt.subnormal), np.int32)
        code = lax.select(ax < np.float32(2.0 ** fmt.minexp), subnormal, code)
    else:  # e8m0fnu: the code is the biased exponent
        code = exponent - np.int32(fmt.minexp)
    if fmt.overflow == "inf":
        code = lax.select(lax.eq(magnitude, np.uint32(0x7F800000)),
                          lax.full_like(code, _code_of(np.inf, fmt)), code)
    nan = lax.ne(x, x)
    if fmt.overflow != "saturate":  # float4_e2m1fn has no NaN
        code = lax.select(nan, lax.full_like(code, _code_of(np.nan, fmt)), code)
    if fmt.signed:  # NaN keeps its sign too, except in fnuz formats (whose NaN is -0's code)
        sign = lax.convert_element_type(lax.shift_right_logical(bits, np.uint32(31)), np.int32)
        if fmt.unsigned_zero:
            sign = lax.select(lax.eq(magnitude, np.uint32(0)) | nan, lax.full_like(sign, 0), sign)
        code = code | lax.shift_left(sign, np.int32(fmt.bits - 1))
    return lax.convert_element_type(code, fmt.storage)


# --- Running a jaxpr on carriers ---------------------------------------------------------

def uses_narrow(jaxpr):
    """Whether a jaxpr has a narrow-typed value anywhere, sub-jaxprs included."""
    if any(_narrow_aval(v.aval) for v in (*jaxpr.invars, *jaxpr.constvars, *jaxpr.outvars)):
        return True
    return any(_involves_narrow(eqn) for eqn in jaxpr.eqns)


def _involves_narrow(eqn):
    return (any(_narrow_aval(v.aval) for v in (*eqn.invars, *eqn.outvars))
            or any(uses_narrow(sub) for sub in core.jaxprs_in_params(eqn.params)))


def _carrier_value(value, aval):
    fmt = narrow_format(getattr(aval, "dtype", None))
    if fmt is None:
        return value
    return np.asarray(value).astype(np.int32 if fmt.integer else np.float32).astype(fmt.carrier)


def _carrier_aval(aval):
    fmt = narrow_format(getattr(aval, "dtype", None))
    return aval.update(dtype=fmt.carrier, weak_type=False) if fmt else aval


def _consts(jaxpr):
    return jaxpr.consts if jaxpr.consts is not None else ()


def _region(jaxpr):
    """A combining region (reduce, scatter, ...) on carriers, unrounded: the op combines in
    the carrier type and its result is rounded once."""
    avals = [_carrier_aval(v.aval) for v in jaxpr.invars]
    return jax.make_jaxpr(lambda *args: evaluate(jaxpr, _consts(jaxpr), args,
                                                 rounding=False))(*avals)


def _carrier_params(params):
    """params with narrow dtypes replaced by carriers, and combining regions on carriers."""
    params = dict(params)
    for name, value in list(params.items()):
        if isinstance(value, (np.dtype, type)) and narrow_format(value) is not None:
            params[name] = narrow_format(value).carrier
        elif isinstance(value, core.Jaxpr) and uses_narrow(value):
            params[name] = region = _region(value)
            if name.replace("jaxpr", "consts") in params:  # scatter's update_consts, ...
                params[name.replace("jaxpr", "consts")] = tuple(_consts(region))
    return params


# Sub-jaxprs run inline: lowering only uses these primitives' primal computation.
_INLINE = {"jit": "jaxpr", "pjit": "jaxpr", "closed_call": "call_jaxpr",
           "core_call": "call_jaxpr", "remat2": "jaxpr", "checkpoint": "jaxpr",
           "custom_jvp_call": "call_jaxpr", "custom_vjp_call": "call_jaxpr"}

# Primitives whose narrow results are narrow operands, moved or selected: no rounding.
_EXACT = {"reshape", "broadcast_in_dim", "transpose", "squeeze", "expand_dims", "rev", "slice",
          "dynamic_slice", "dynamic_update_slice", "gather", "concatenate", "pad", "select_n",
          "copy", "copy_p", "optimization_barrier", "split", "sort", "max", "min", "clamp",
          "reduce_max", "reduce_min", "cummax", "cummin", "reduce_window_max",
          "reduce_window_min", "scatter", "device_put", "sharding_constraint",
          "mesh_cast", "reshard"}


def _while(eqn, args):
    cond, body = eqn.params["cond_jaxpr"], eqn.params["body_jaxpr"]
    nc, nb = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
    cond_consts, body_consts, init = args[:nc], args[nc:nc + nb], args[nc + nb:]
    return list(lax.while_loop(
        lambda carry: evaluate(cond, _consts(cond), [*cond_consts, *carry])[0],
        lambda carry: tuple(evaluate(body, _consts(body), [*body_consts, *carry])),
        tuple(init)))


def _scan(eqn, args):
    p = eqn.params
    consts, init, xs = (list(part) for part in p["ft_in"].update(args).unpack())
    closed = p["jaxpr"]

    def step(carry, x):
        outs = evaluate(closed, _consts(closed), [*consts, *carry, *x])
        return tuple(outs[:len(init)]), tuple(outs[len(init):])

    carry, ys = lax.scan(step, tuple(init), tuple(xs), length=p["length"], reverse=p["reverse"],
                         unroll=p["unroll"])
    return [*carry, *ys]


def _cond(eqn, args):
    index, *operands = args
    branches = [lambda *ops, b=b: tuple(evaluate(b, _consts(b), ops))
                for b in eqn.params["branches"]]
    return list(lax.switch(index, branches, *operands))


def _convert(eqn, args):
    (x,), (out,) = args, eqn.outvars
    fmt = narrow_format(out.aval.dtype)
    if fmt is None:  # from a narrow type: the carrier holds the exact value
        return [lax.convert_element_type(x, out.aval.dtype)]
    return [to_format(x, fmt)]


def _edtype(eqn, args):
    """to_edtype/from_edtype of an extended dtype represented by a narrow one: through the
    narrow type itself. These programs keep a native narrow value (fine on CPU; METAL
    rejects them)."""
    (x,), (src,), (out,) = args, eqn.invars, eqn.outvars
    if narrow_format(src.aval.dtype) is not None:
        x = lax.convert_element_type(x, src.aval.dtype)
    out_value = eqn.primitive.bind(x, **eqn.params)
    fmt = narrow_format(out.aval.dtype)
    return [lax.convert_element_type(out_value, fmt.carrier) if fmt else out_value]


def _bits(dtype):
    fmt = narrow_format(dtype)
    return fmt.bits if fmt else np.dtype(dtype).itemsize * 8


def _codes(x, dtype, container):
    """The bits of x, of type dtype (a carrier if narrow), as unsigned `container` ints."""
    fmt = narrow_format(dtype)
    if fmt:
        return lax.convert_element_type(_unsigned_code(encode(x, fmt), fmt), container)
    return lax.convert_element_type(
        lax.bitcast_convert_type(x, np.dtype(f"uint{_bits(dtype)}")), container)


def _from_codes(codes, dtype):
    fmt = narrow_format(dtype)
    if fmt:
        return decode(codes, fmt)
    return lax.bitcast_convert_type(
        lax.convert_element_type(codes, np.dtype(f"uint{_bits(dtype)}")), dtype)


def _bitcast(eqn, args):
    """bitcast_convert_type through the formats' codes. Between widths, as XLA packs: the
    narrower type has a trailing axis, its first element in the lowest bits."""
    (x,), (src,), (out,) = args, eqn.invars, eqn.outvars
    src_bits, out_bits = _bits(src.aval.dtype), _bits(out.aval.dtype)
    container = np.dtype(np.uint64 if max(src_bits, out_bits) > 32 else np.uint32)
    codes = _codes(x, src.aval.dtype, container)
    if src_bits < out_bits:
        shifts = (np.arange(out_bits // src_bits) * src_bits).astype(container)
        shifts = jax.numpy.broadcast_to(shifts, codes.shape)
        codes = jax.numpy.sum(lax.shift_left(codes, shifts), axis=-1, dtype=container)
    elif src_bits > out_bits:
        shifts = (np.arange(src_bits // out_bits) * out_bits).astype(container)
        codes = jax.numpy.broadcast_to(codes[..., None], (*codes.shape, len(shifts)))
        codes = lax.bitwise_and(
            lax.shift_right_logical(codes, jax.numpy.broadcast_to(shifts, codes.shape)),
            container.type(2 ** out_bits - 1))
    return [_from_codes(codes, out.aval.dtype)]


def _unsigned_code(x, fmt):
    return lax.bitwise_and(lax.convert_element_type(x, np.int32), np.int32(2 ** fmt.bits - 1))


def _shift_right_logical(eqn, args):
    fmt = narrow_format(eqn.outvars[0].aval.dtype)
    x, amount = (_unsigned_code(a, fmt) for a in args)  # amounts count as unsigned
    shifted = lax.shift_right_logical(x, amount)
    return [_wrap_int(lax.select(amount >= fmt.bits, lax.full_like(x, 0), shifted), fmt)]


def _shift_left(eqn, args):
    fmt = narrow_format(eqn.outvars[0].aval.dtype)
    x, amount = args[0], _unsigned_code(args[1], fmt)
    shifted = lax.shift_left(lax.convert_element_type(x, np.int32), amount)
    return [_wrap_int(lax.select(amount >= fmt.bits, lax.full_like(shifted, 0), shifted), fmt)]


def _shift_right_arithmetic(eqn, args):
    fmt = narrow_format(eqn.outvars[0].aval.dtype)
    x, amount = (_unsigned_code(a, fmt) for a in args)
    free = np.int32(32 - fmt.bits)  # the top bit is the sign, unsigned types included
    x = lax.shift_right_arithmetic(lax.shift_left(x, free), free)
    return [_wrap_int(lax.shift_right_arithmetic(x, lax.min(amount, np.int32(fmt.bits - 1))),
                      fmt)]


def _bit_count(eqn, args):
    fmt = narrow_format(eqn.outvars[0].aval.dtype)
    x = _unsigned_code(args[0], fmt)
    if eqn.primitive.name == "population_count":
        return [_wrap_int(lax.population_count(x), fmt)]
    return [_wrap_int(lax.clz(x) - np.int32(32 - fmt.bits), fmt)]


def _next_after(eqn, args):
    """XLA's nextafter, on codes: one code up or down."""
    fmt = narrow_format(eqn.outvars[0].aval.dtype)
    x, y = args
    sign = np.int32(1 << (fmt.bits - 1))
    xc, yc = (lax.convert_element_type(encode(a, fmt), np.int32) & np.int32(2 ** fmt.bits - 1)
              for a in (x, y))
    x_abs, y_abs = xc & ~sign, yc & ~sign
    toward_zero = (x_abs > y_abs) | lax.ne(xc & sign, yc & sign)
    code = xc + lax.select(toward_zero, lax.full_like(xc, -1), lax.full_like(xc, 1))
    from_zero = lax.select(lax.eq(y_abs, np.int32(0)), yc, (yc & sign) | np.int32(1))
    code = lax.select(lax.eq(x_abs, np.int32(0)), from_zero, code)
    code = lax.select(lax.eq(x, y), yc, code)
    out = decode(code, fmt)
    if fmt.overflow == "saturate":  # no NaN
        return [out]
    return [lax.select(lax.ne(x, x) | lax.ne(y, y), lax.full_like(out, np.nan), out)]


_COMBINE = {"cumsum": lax.add, "cumprod": lax.mul, "cumlogsumexp": jax.numpy.logaddexp}


def _cumulative(eqn, args):
    """Cumulative sums and products round every partial result, as XLA's associative scan
    does on narrow types."""
    fmt, combine = narrow_format(eqn.outvars[0].aval.dtype), _COMBINE[eqn.primitive.name]
    return [lax.associative_scan(lambda a, b: to_format(combine(a, b), fmt), args[0],
                                 reverse=eqn.params["reverse"], axis=eqn.params["axis"])]


_SPECIAL = {"convert_element_type": _convert, "bitcast_convert_type": _bitcast,
            "while": _while, "scan": _scan, "cond": _cond,
            "shift_left": _shift_left, "shift_right_logical": _shift_right_logical,
            "shift_right_arithmetic": _shift_right_arithmetic,
            "population_count": _bit_count, "clz": _bit_count, "nextafter": _next_after,
            "to_edtype": _edtype, "from_edtype": _edtype,
            **{name: _cumulative for name in _COMBINE}}


def _apply(eqn, args, rounding):
    if not _involves_narrow(eqn):
        outs = eqn.primitive.bind(*args, **eqn.primitive.get_bind_params(eqn.params))
        return outs if eqn.primitive.multiple_results else [outs]
    name = eqn.primitive.name
    if name in _INLINE:
        inner = eqn.params[_INLINE[name]]
        return evaluate(inner, _consts(inner), args, rounding)
    if name in _SPECIAL:
        return _SPECIAL[name](eqn, args)
    if (name not in ("reduce", "reduce_window", "select_and_scatter")
            and not name.startswith("scatter") and any(core.jaxprs_in_params(eqn.params))):
        raise NotImplementedError(f"METAL: {name} with narrow types is not emulated")
    outs = eqn.primitive.bind(*args, **_carrier_params(eqn.params))
    outs = outs if eqn.primitive.multiple_results else [outs]
    if not rounding or name in _EXACT:
        return list(outs)
    return [to_format(o, narrow_format(v.aval.dtype)) if _narrow_aval(v.aval) else o
            for o, v in zip(outs, eqn.outvars)]


def evaluate(jaxpr, consts, args, rounding=True):
    """Run jaxpr on carriers: narrow-typed values are carriers in and out."""
    env = {}

    def read(v):
        if isinstance(v, core.Literal):
            return _carrier_value(v.val, v.aval)
        return env[v]

    for v, value in zip(jaxpr.constvars, consts):
        env[v] = _carrier_value(value, v.aval)
    for v, value in zip(jaxpr.invars, args):
        env[v] = value
    for eqn in jaxpr.eqns:
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        with (source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack),
              eqn.ctx.manager):
            outs = _apply(eqn, [read(v) for v in eqn.invars], rounding)
        for v, value in zip(eqn.outvars, outs):
            env[v] = value
    return [read(v) for v in jaxpr.outvars]


def carrier_jaxpr(jaxpr):
    """jaxpr run on carriers: no narrow type is left inside, and narrow inputs and outputs
    keep their types, converted (exactly) at the edges."""
    def body(*args):
        args = [lax.convert_element_type(a, narrow_format(aval.dtype).carrier)
                if _narrow_aval(aval) else a for a, aval in zip(args, jaxpr.in_avals)]
        outs = evaluate(jaxpr, _consts(jaxpr), args)
        return [lax.convert_element_type(o, aval.dtype) if _narrow_aval(aval) else o
                for o, aval in zip(outs, jaxpr.out_avals)]

    with jax.numpy_rank_promotion("allow"), jax.numpy_dtype_promotion("standard"):
        rewritten = jax.make_jaxpr(body)(*jaxpr.in_avals)
    return rewritten.replace(debug_info=jaxpr.debug_info)


# --- Lowering hook -----------------------------------------------------------------------

def _lower_jaxpr_to_module(original):
    def lower(module_name, jaxpr, **kwargs):
        if (tuple(kwargs.get("platforms") or ()) == (PLATFORM,)
                and kwargs.get("num_const_args", 0) == 0
                and len(jaxpr.in_avals) == len(kwargs["in_avals"])
                and not jaxpr.effects
                and uses_narrow(jaxpr)):
            jaxpr = carrier_jaxpr(jaxpr)
        return original(module_name, jaxpr, **kwargs)
    return lower


def register(platform=None):
    """Rewrite programs for `platform` (METAL; the CPU emulation passes "cpu")."""
    global PLATFORM  # pylint: disable=global-statement
    PLATFORM = platform or PLATFORM
    mlir.lower_jaxpr_to_module = _lower_jaxpr_to_module(mlir.lower_jaxpr_to_module)
