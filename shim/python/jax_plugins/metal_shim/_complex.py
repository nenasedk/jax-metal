"""complex64 support inside METAL programs, and FFT.

The plugin computes complex arithmetic (add, multiply, divide, dot, sum, abs) but has no
pattern for mhlo.real/imag/complex, the only ways in and out of the complex domain.
XLA's serializer also expands complex transcendental functions into real/imag ops
(StablehloComplexMathExpanderPass), and mhlo.fft has no pattern. These rules:

- build real/imag/complex from ops the plugin lowers (a complex -> real convert keeps the
  real part; multiplying by -i rotates the imaginary part into it);
- express complex transcendental functions in real arithmetic;
- implement FFT as a radix-2 transform with a direct DFT for odd lengths.

Complex values must stay inside the program: the plugin rejects any compiled function
whose inputs or outputs are complex. See shim/DESIGN.md.

A primitive has one rule per platform, so the dot_general rule here also covers its
non-complex cases (empty contractions, integer and bool results). allow_rank_promotion
wraps every rule this package registers.
"""

from functools import partial, wraps

import jax
import numpy as np
from jax import lax
from jax._src import config
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lax import convolution
from jax._src.lax import fft as lax_fft
from jax._src.lax import lax as lax_internal
from jax._src.lax import slicing
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo

PLATFORM = "METAL"


def is_complex(x):
    return np.issubdtype(x.dtype, np.complexfloating)


def is_complex_aval(aval):
    """Complex check that tolerates JAX's extended dtypes (PRNG keys are not NumPy dtypes)."""
    dtype = getattr(aval, "dtype", None)
    return isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.complexfloating)


def _invariant(aval):
    """aval without shard_map's varying-manual-axes type."""
    mat = getattr(aval, "manual_axis_type", None)
    if mat is not None and mat.varying:
        return aval.update(manual_axis_type=mat.update(varying=frozenset()))
    return aval


def allow_rank_promotion(rule):
    """`rule`, traced with jax_numpy_rank_promotion="allow", standard dtype promotion and
    avals without varying manual axes.

    Rules broadcast with jnp-style operators (a batched [b, n, n] times an [n, n] mask) and
    mix real and complex operands; a user's "raise"/"warn" rank promotion or "strict" dtype
    promotion, which JAX's own tests use, must not apply to them. Inside shard_map, a loop
    carry that starts from a constant (an identity, zeros, a counter) would not match the
    varying type of what the loop computes from the inputs. That type does not change the
    lowered program, so rules trace without it.
    """
    @wraps(rule)
    def wrapped(*args, **kwargs):
        if args and isinstance(args[0], mlir.LoweringRuleContext):
            ctx = args[0]
            ctx = ctx.replace(avals_in=[_invariant(a) for a in ctx.avals_in],
                              avals_out=[_invariant(a) for a in ctx.avals_out]
                              if ctx.avals_out is not None else None)
            args = (ctx, *args[1:])
        with jax.numpy_rank_promotion("allow"), jax.numpy_dtype_promotion("standard"), \
                config._check_vma(False):  # pylint: disable=protected-access
            return rule(*args, **kwargs)
    return wrapped


# Every METAL rule this package registers, by primitive (the latest one wins, as in JAX).
REGISTERED = {}


def register_rule(primitive, rule):
    REGISTERED[primitive] = allow_rank_promotion(rule)
    mlir.register_lowering(primitive, REGISTERED[primitive], platform=PLATFORM)


def current_rule(primitive):
    """The rule METAL lowers `primitive` with: this package's, else JAX's generic one."""
    if primitive in REGISTERED:
        return REGISTERED[primitive]
    return mlir._lowerings[primitive].rule  # pylint: disable=protected-access


_register_rule = register_rule


# --- real / imag / complex ---------------------------------------------------------

def _complex_constant(ctx, value, aval):
    ir_type = mlir.aval_to_ir_type(ctx.module_context, aval)
    return hlo.constant(ir.DenseElementsAttr.get(np.full(aval.shape, value, aval.dtype),
                                                 type=ir_type))


def _real_rule(ctx, z):
    return [hlo.convert(mlir.aval_to_ir_type(ctx.module_context, ctx.avals_out[0]), z)]


def _imag_rule(ctx, z):
    rotated = hlo.multiply(z, _complex_constant(ctx, -1j, ctx.avals_in[0]))
    return [hlo.convert(mlir.aval_to_ir_type(ctx.module_context, ctx.avals_out[0]), rotated)]


def _complex_rule(ctx, x, y):
    aval = ctx.avals_out[0]
    ir_type = mlir.aval_to_ir_type(ctx.module_context, aval)
    x, y = mlir.multi_broadcast_in_dim(ctx, (x, y), ctx.avals_in, aval.shape,  # scalar operands
                                       aval.sharding)
    return [hlo.add(hlo.convert(ir_type, x),
                    hlo.multiply(hlo.convert(ir_type, y), _complex_constant(ctx, 1j, aval)))]


# --- complex elementwise functions in real arithmetic --------------------------------

def _parts(z):
    return lax.real(z), lax.imag(z)


def exp(z):
    a, b = _parts(z)
    magnitude = lax.exp(a)
    return lax.complex(magnitude * lax.cos(b), magnitude * lax.sin(b))


def log(z):
    a, b = _parts(z)
    return lax.complex(lax.log(lax.abs(z)), lax.atan2(b, a))


def exp2(z):
    # JAX's formula, exp(z * ln 2) with a complex ln 2: its lowering is a complex
    # stablehlo.exponential, which XLA's serializer expands into real/imag ops.
    return exp(z * lax.full_like(z, np.log(2)))


def log2(z):
    # JAX's formula, log(z) / ln 2 per part: its lowering emits stablehlo.real/imag itself.
    a, b = _parts(z)
    scale = np.float32(1 / np.log(2))
    return lax.complex(lax.log(lax.abs(z)) * scale, lax.atan2(b, a) * scale)


def square(z):
    # CHLO's decomposition, which XLA's CPU backend matches bit for bit: the real part is
    # (a - b)(a + b), without cancellation, and exactly 0 when |a| == |b| (finite);
    # chlo.square on complex values expands into real/imag ops the plugin cannot lower.
    a, b = _parts(z)
    cancels = lax.is_finite(a) & lax.eq(lax.abs(a), lax.abs(b))
    re = lax.select(cancels, lax.full_like(a, 0), (a - b) * (a + b))
    return lax.complex(re, 2 * (a * b))


def _sign_bit(x):
    """x < 0, and also true for -0.0 (whose sign picks the side of a branch cut)."""
    return lax.bitcast_convert_type(x, np.int32) < 0


def sqrt(z):
    # Principal root without cancellation: t = sqrt((|a| + |z|) / 2). The imaginary part
    # takes b's sign, -0.0 included, as in C99's csqrt.
    a, b = _parts(z)
    t = lax.sqrt((lax.abs(a) + lax.abs(z)) / 2)
    safe_t = lax.select(t == 0, lax.full_like(t, 1), t)
    other = lax.abs(b) / (2 * safe_t)
    real = lax.select(a >= 0, t, other)
    imag = lax.select(a >= 0, b / (2 * safe_t), lax.select(_sign_bit(b), -t, t))
    zero = t == 0
    return lax.complex(lax.select(zero, t, real), lax.select(zero, b, imag))


def sin(z):
    a, b = _parts(z)
    return lax.complex(lax.sin(a) * lax.cosh(b), lax.cos(a) * lax.sinh(b))


def cos(z):
    a, b = _parts(z)
    return lax.complex(lax.cos(a) * lax.cosh(b), -lax.sin(a) * lax.sinh(b))


def sinh(z):
    a, b = _parts(z)
    return lax.complex(lax.sinh(a) * lax.cos(b), lax.cosh(a) * lax.sin(b))


def cosh(z):
    a, b = _parts(z)
    return lax.complex(lax.cosh(a) * lax.cos(b), lax.sinh(a) * lax.sin(b))


def tan(z):
    return sin(z) / cos(z)


def tanh(z):
    return sinh(z) / cosh(z)


def expm1(z):
    a, b = _parts(z)
    # Re: expm1(a) cos(b) - 2 sin^2(b/2) keeps precision near 0.
    half = lax.sin(b / 2)
    return lax.complex(lax.expm1(a) * lax.cos(b) - 2 * half * half, lax.exp(a) * lax.sin(b))


def log1p(z):
    a, b = _parts(z)
    # Re: 0.5 * log1p(2a + a^2 + b^2) keeps precision near 0.
    return lax.complex(0.5 * lax.log1p(2 * a + a * a + b * b), lax.atan2(b, 1 + a))


def sign(z):
    magnitude = lax.abs(z)
    safe = lax.select(magnitude == 0, lax.full_like(magnitude, 1), magnitude)
    unit = z / lax.complex(safe, lax.full_like(safe, 0))
    return lax.select(magnitude == 0, lax.full_like(z, 0), unit)


def rsqrt(z):
    return lax.full_like(z, 1) / sqrt(z)


# Inverse functions after Kahan ("Branch cuts for complex elementary functions"): no
# cancellation and no early overflow. The 1 -+ z, z -+ 1 terms keep the sign of a zero
# imaginary part, which picks the side of each branch cut.

def asin(z):
    a, b = _parts(z)
    s1m, s1p = sqrt(lax.complex(1 - a, -b)), sqrt(lax.complex(1 + a, b))
    return lax.complex(lax.atan2(a, lax.real(s1m * s1p)),
                       lax.asinh(lax.imag(lax.conj(s1m) * s1p)))


def acos(z):
    a, b = _parts(z)
    s1m, s1p = sqrt(lax.complex(1 - a, -b)), sqrt(lax.complex(1 + a, b))
    return lax.complex(2 * lax.atan2(lax.real(s1m), lax.real(s1p)),
                       lax.asinh(lax.imag(lax.conj(s1p) * s1m)))


def acosh(z):
    a, b = _parts(z)
    szm, szp = sqrt(lax.complex(a - 1, b)), sqrt(lax.complex(a + 1, b))
    return lax.complex(lax.asinh(lax.real(lax.conj(szm) * szp)),
                       2 * lax.atan2(lax.imag(szm), lax.real(szp)))


def atanh(z):
    a, b = _parts(z)
    return 0.5 * (log1p(z) - log1p(lax.complex(-a, -b)))


def _rotated(f, z):
    """-i f(i z): asinh from asin, atan from atanh."""
    a, b = _parts(z)
    w = f(lax.complex(-b, a))
    return lax.complex(lax.imag(w), -lax.real(w))


def asinh(z):
    return _rotated(asin, z)


def atan(z):
    return _rotated(atanh, z)


def pow(z, w):  # pylint: disable=redefined-builtin
    zero = lax.abs(z) == 0
    result = exp(w * log(lax.select(zero, lax.full_like(z, 1), z)))
    return lax.select(zero, lax.select(lax.abs(w) == 0, lax.full_like(z, 1), lax.full_like(z, 0)),
                      result)


# --- data movement on complex values ---------------------------------------------------
# MPS has no complex kernels for pad (it aborts the process: "padOp_cf was not found"),
# gather_nd or scatter_nd, and mhlo.sort has no complex lowering. These ops move values
# without combining them, so applying them to the real and imaginary parts separately is
# exact. Scatters are rebuilt through lax.scatter*/ on real parts because the primitive's
# update_jaxpr is typed for the complex dtype.

def _split(x):
    return (lax.real(x), lax.imag(x)) if is_complex(x) else (x, x)


def _pad(operand, padding_value, *, padding_config):
    (re, im), (pad_re, pad_im) = _split(operand), _split(padding_value)
    return lax.complex(lax.pad(re, pad_re, padding_config), lax.pad(im, pad_im, padding_config))


def _gather(operand, indices, *, fill_value, **params):
    re, im = _split(operand)
    fill_re, fill_im = (None, None) if fill_value is None else (np.real(fill_value), np.imag(fill_value))
    return lax.complex(slicing.gather_p.bind(re, indices, fill_value=fill_re, **params),
                       slicing.gather_p.bind(im, indices, fill_value=fill_im, **params))


def _dynamic_update_slice(operand, update, *starts):
    (re, im), (up_re, up_im) = _split(operand), _split(update)
    return lax.complex(lax.dynamic_update_slice(re, up_re, starts),
                       lax.dynamic_update_slice(im, up_im, starts))


def _scatter(apply, operand, indices, updates, *, dimension_numbers, indices_are_sorted,
             unique_indices, mode, **_):
    (re, im), (up_re, up_im) = _split(operand), _split(updates)
    kwargs = dict(indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode)
    return lax.complex(apply(re, indices, up_re, dimension_numbers, **kwargs),
                       apply(im, indices, up_im, dimension_numbers, **kwargs))


def _scatter_mul(operand, indices, updates, *, dimension_numbers, indices_are_sorted,
                 unique_indices, mode, **_):
    # A complex product does not split into real and imaginary parts, but into magnitude
    # and angle: magnitudes multiply (a real scatter_mul) and angles add (a real
    # scatter_add), repeated indices included. The operand is then scaled by the result.
    kwargs = dict(indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode)
    real_dtype = np.finfo(operand.dtype).dtype
    magnitude = lax.scatter_mul(lax.full(operand.shape, 1, real_dtype), indices, lax.abs(updates),
                                dimension_numbers, **kwargs)
    angle = lax.scatter_add(lax.full(operand.shape, 0, real_dtype), indices,
                            lax.atan2(lax.imag(updates), lax.real(updates)), dimension_numbers,
                            **kwargs)
    return operand * lax.complex(magnitude * lax.cos(angle), magnitude * lax.sin(angle))


def _reduce_chooser(reduce, init, operand, *, axes, **_):
    """reduce_max/min of complex values, ordered by real part, then imaginary part as XLA
    compares them, from real reductions: JAX's lowering is a reduction whose body compares
    the parts (nine ops), and the plugin recognizes a reduction by its one combining op."""
    kept = [d for d in range(operand.ndim) if d not in axes]
    if any(operand.shape[a] == 0 for a in axes):  # the identity, init + 0i
        return lax.full([operand.shape[d] for d in kept], init, operand.dtype)
    re, im = lax.real(operand), lax.imag(operand)
    best_re = reduce(re, axes)
    keep = lax.eq(re, lax.broadcast_in_dim(best_re, re.shape, kept))
    best_im = reduce(lax.select(keep, im, lax.full_like(im, init)), axes)
    return lax.complex(best_re, best_im)


def _chooser(compare, x, y):
    """max/min of complex values, ordered by real part, then imaginary part, as JAX lowers
    them (mlir._minmax_hlo), from the real/imag rules above: JAX's lowering emits
    stablehlo.real/imag itself, which the plugin cannot legalize. As in XLA, a NaN
    comparison is false, so the second operand is chosen."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    shape = np.broadcast_shapes(x.shape, y.shape)  # a scalar operand, as in _nary_lower_hlo
    x, y = jnp.broadcast_to(x, shape), jnp.broadcast_to(y, shape)
    rx, ry = lax.real(x), lax.real(y)
    pick_x = lax.select(lax.eq(rx, ry), compare(lax.imag(x), lax.imag(y)), compare(rx, ry))
    return lax.select(pick_x, x, y)


def _sort(*operands, dimension, is_stable, num_keys):
    # Complex keys sort lexicographically by (real, imag), like jnp.sort_complex. The
    # plugin's SortConverter takes a single key, so n keys become n stable single-key
    # passes from the last key to the first (also used for real multi-key sorts).
    del is_stable  # every pass is stable, which satisfies both settings
    expanded, keys, layout = [], 0, []
    for index, operand in enumerate(operands):
        parts = [lax.real(operand), lax.imag(operand)] if is_complex(operand) else [operand]
        expanded += parts
        keys += len(parts) if index < num_keys else 0
        layout.append(len(parts))
    for key in reversed(range(keys)):
        order = [key] + [i for i in range(len(expanded)) if i != key]
        moved = lax.sort([expanded[i] for i in order], dimension=dimension, is_stable=True,
                         num_keys=1)
        expanded = [None] * len(expanded)
        for position, source in enumerate(order):
            expanded[source] = moved[position]
    out, position = [], 0
    for width in layout:
        out.append(lax.complex(expanded[position], expanded[position + 1]) if width == 2
                   else expanded[position])
        position += width
    return out


def _sort_rule():
    """Complex operands or several keys: _sort; otherwise JAX's default lowering."""
    default = mlir._lowerings[lax_internal.sort_p].rule  # pylint: disable=protected-access
    multi_pass = mlir.lower_fun(_sort, multiple_results=True)

    def rule(ctx, *args, num_keys, **params):
        complex_input = any(is_complex_aval(a) for a in ctx.avals_in)
        if complex_input or num_keys > 1:
            return multi_pass(ctx, *args, num_keys=num_keys, **params)
        return default(ctx, *args, num_keys=num_keys, **params)

    return rule


def _bilinear(primitive):
    """A bilinear primitive (dot_general, conv) on complex operands as real products.

    The plugin's compiler segfaults on batched complex dot_general, so complex contractions
    never reach MPS: (a + ib)(c + id) = (ac - bd) + i(ad + bc).
    """

    def apply(lhs, rhs, **params):
        preferred = params.get("preferred_element_type")
        if preferred is not None and np.issubdtype(preferred, np.complexfloating):
            params["preferred_element_type"] = np.finfo(preferred).dtype
        product = lambda x, y: primitive.bind(x, y, **params)
        if is_complex(lhs) and is_complex(rhs):
            a, b, c, d = lax.real(lhs), lax.imag(lhs), lax.real(rhs), lax.imag(rhs)
            return lax.complex(product(a, c) - product(b, d), product(a, d) + product(b, c))
        if is_complex(lhs):
            return lax.complex(product(lax.real(lhs), rhs), product(lax.imag(lhs), rhs))
        return lax.complex(product(lhs, lax.real(rhs)), product(lhs, lax.imag(rhs)))

    return apply


def _while_rule():
    """while_loop with complex carries, carried as (real, imag) pairs.

    MPSGraph crashes at run time (GPU::WhileOpHandler) on some while loops that carry
    complex values (for example complex QDWH, used by complex SVD). Complex arithmetic inside
    the body is fine, so only the carried state is split.
    """
    from jax._src.lax.control_flow import loops  # pylint: disable=import-outside-toplevel
    default = mlir._lowerings[loops.while_p].rule  # pylint: disable=protected-access

    def split_loop(*args, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts):
        cond_consts = args[:cond_nconsts]
        body_consts = args[cond_nconsts:cond_nconsts + body_nconsts]
        init = args[cond_nconsts + body_nconsts:]
        flags = [is_complex(x) for x in init]

        def to_pairs(values):
            out = []
            for value, flag in zip(values, flags):
                out += [lax.real(value), lax.imag(value)] if flag else [value]
            return out

        def from_pairs(values):
            out, position = [], 0
            for flag in flags:
                if flag:
                    out.append(lax.complex(values[position], values[position + 1]))
                    position += 2
                else:
                    out.append(values[position])
                    position += 1
            return out

        def predicate(carry):
            return core.jaxpr_as_fun(cond_jaxpr)(*cond_consts, *from_pairs(carry))[0]

        def cond(carry):
            # After vmap the predicate can be batched: loop while any element is true.
            pred = predicate(carry)
            return lax.reduce_or(pred, tuple(range(pred.ndim))) if pred.ndim else pred

        def body(carry):
            new = to_pairs(core.jaxpr_as_fun(body_jaxpr)(*body_consts, *from_pairs(carry)))
            pred = predicate(carry)
            if not pred.ndim:
                return new
            # Batched predicate: only elements still running take the new value (as JAX does).
            return [lax.select(lax.broadcast_in_dim(pred, n.shape, tuple(range(pred.ndim))), n, o)
                    for n, o in zip(new, carry)]

        return from_pairs(lax.while_loop(cond, body, to_pairs(init)))

    split = mlir.lower_fun(split_loop, multiple_results=True)

    def rule(ctx, *args, cond_nconsts, body_nconsts, **params):
        carries = ctx.avals_in[cond_nconsts + body_nconsts:]
        if any(is_complex_aval(aval) for aval in carries):
            return split(ctx, *args, cond_nconsts=cond_nconsts, body_nconsts=body_nconsts,
                         cond_jaxpr=params["cond_jaxpr"], body_jaxpr=params["body_jaxpr"])
        return default(ctx, *args, cond_nconsts=cond_nconsts, body_nconsts=body_nconsts, **params)

    return rule


# Most elements _integer_dot_general's broadcast product holds at a time (16 MB of int32).
_INTEGER_DOT_CHUNK = 1 << 22


def _integer_dot_general(lhs, rhs, *, dimension_numbers, out_dtype):
    """dot_general with an integer or bool result, which MPS matmul rejects (it takes floating
    point only): exact multiply-and-sum, wrapping like XLA, over slices of the contraction so
    that the broadcast product stays within _INTEGER_DOT_CHUNK elements. bool is OR of ANDs.
    """
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_free = [d for d in range(lhs.ndim) if d not in (*lhs_contract, *lhs_batch)]
    rhs_free = [d for d in range(rhs.ndim) if d not in (*rhs_contract, *rhs_batch)]
    sizes = lambda x, dims: [x.shape[d] for d in dims]
    out_shape = (*sizes(lhs, lhs_batch), *sizes(lhs, lhs_free), *sizes(rhs, rhs_free))
    if 0 in out_shape:
        return lax.full(out_shape, 0, out_dtype)
    b, m, n, k = (int(np.prod(s)) for s in (sizes(lhs, lhs_batch), sizes(lhs, lhs_free),
                                             sizes(rhs, rhs_free), sizes(lhs, lhs_contract)))
    acc_dtype = np.dtype(np.int32) if out_dtype == np.bool_ else out_dtype
    l = lax.reshape(lax.convert_element_type(lhs, acc_dtype), (b, m, k),
                    dimensions=(*lhs_batch, *lhs_free, *lhs_contract))
    r = lax.reshape(lax.convert_element_type(rhs, acc_dtype), (b, k, n),
                    dimensions=(*rhs_batch, *rhs_contract, *rhs_free))
    chunk = max(1, min(k, _INTEGER_DOT_CHUNK // (b * m * n)))
    steps = -(-k // chunk)
    if steps * chunk != k:  # zero padding adds nothing to the sums
        zero = np.array(0, acc_dtype)
        l = lax.pad(l, zero, [(0, 0, 0), (0, 0, 0), (0, steps * chunk - k, 0)])
        r = lax.pad(r, zero, [(0, 0, 0), (0, steps * chunk - k, 0), (0, 0, 0)])

    def partial_sum(i):
        l_chunk = lax.dynamic_slice_in_dim(l, i * chunk, chunk, axis=2)
        r_chunk = lax.dynamic_slice_in_dim(r, i * chunk, chunk, axis=1)
        return lax.reduce_sum(lax.expand_dims(l_chunk, (3,)) * lax.expand_dims(r_chunk, (1,)),
                              (2,))

    out = partial_sum(0)
    if steps > 1:
        out = lax.fori_loop(1, steps, lambda i, acc: acc + partial_sum(i), out)
    out = out != 0 if out_dtype == np.bool_ else out
    return lax.reshape(out, out_shape)


def _dot_general_rule():
    """dot_general for METAL: empty contractions, complex operands, integer results and
    dot algorithms (dropped).

    A contraction over a zero-length dimension aborts MPSGraph ("MLIR pass manager failed");
    its result is all zeros. Complex operands become real products (see _bilinear). Integer
    and bool results use _integer_dot_general.
    """
    primitive = lax_internal.dot_general_p
    default = mlir._lowerings[primitive].rule  # pylint: disable=protected-access
    complex_rule = mlir.lower_fun(_bilinear(primitive), multiple_results=False)

    def rule(ctx, lhs, rhs, *, dimension_numbers, **params):
        (lhs_contract, _), _ = dimension_numbers
        out = ctx.avals_out[0]
        if isinstance(params.get("precision"), (lax.DotAlgorithm, lax.DotAlgorithmPreset)):
            # A dot algorithm (StableHLO 1.6) cannot be serialized for the plugin; METAL
            # computes in float32, at least as precise as any preset asks for.
            params["precision"] = None
        if any(ctx.avals_in[0].shape[d] == 0 for d in lhs_contract):
            empty = mlir.lower_fun(lambda lhs, rhs: lax.full(out.shape, 0, out.dtype),
                                   multiple_results=False)
            return empty(ctx, lhs, rhs)
        if any(is_complex_aval(aval) for aval in ctx.avals_in):
            return complex_rule(ctx, lhs, rhs, dimension_numbers=dimension_numbers, **params)
        if np.issubdtype(out.dtype, np.integer) or out.dtype == np.bool_:
            integer_rule = mlir.lower_fun(
                partial(_integer_dot_general, dimension_numbers=dimension_numbers,
                        out_dtype=np.dtype(out.dtype)),
                multiple_results=False)
            return integer_rule(ctx, lhs, rhs)
        return default(ctx, lhs, rhs, dimension_numbers=dimension_numbers, **params)

    return rule


def _when_complex(primitive, complex_fn, multiple_results=False, default=None):
    """Use complex_fn when any input is complex, `default` (JAX's default lowering) otherwise."""
    default = default or mlir._lowerings[primitive].rule  # pylint: disable=protected-access
    complex_rule = mlir.lower_fun(complex_fn, multiple_results=multiple_results)

    def rule(ctx, *args, **params):
        if any(is_complex_aval(aval) for aval in ctx.avals_in):
            return complex_rule(ctx, *args, **params)
        return default(ctx, *args, **params)

    return rule


# --- FFT ----------------------------------------------------------------------------

def _twiddles(n, count, dtype, inverse):
    """exp(-+2 pi i k / n) for k < count, computed on device from exact integer angles."""
    angle = (2 * np.pi / n) * lax.convert_element_type(lax.iota(np.int32, count), np.float32)
    direction = 1.0 if inverse else -1.0
    return lax.convert_element_type(lax.complex(lax.cos(angle), direction * lax.sin(angle)),
                                    dtype)


def _dft(x, inverse):
    """Direct DFT along the last axis: x @ W with W[j, k] = exp(-+2 pi i jk / n)."""
    n = x.shape[-1]
    rows = lax.broadcasted_iota(np.int32, (n, n), 0)
    cols = lax.broadcasted_iota(np.int32, (n, n), 1)
    # (j * k) mod n keeps every angle in [0, 2 pi) so cos/sin stay accurate.
    angle = (2 * np.pi / n) * lax.convert_element_type(lax.rem(rows * cols, np.int32(n)),
                                                       np.float32)
    direction = 1.0 if inverse else -1.0
    w = lax.convert_element_type(lax.complex(lax.cos(angle), direction * lax.sin(angle)),
                                 x.dtype)
    return lax.dot_general(x, w, (((x.ndim - 1,), (0,)), ((), ())),
                           precision=lax.Precision.HIGHEST)


def _fft_last_axis(x, inverse):
    """Unnormalized FFT along the last axis: radix-2 while the length is even, DFT below."""
    n = x.shape[-1]
    if n == 0:
        return x
    if n % 2 or n <= 16:
        return _dft(x, inverse)
    # Split even/odd samples into a new axis and transform both halves in one call.
    halves = lax.transpose(x.reshape(*x.shape[:-1], n // 2, 2),
                           (*range(x.ndim - 1), x.ndim, x.ndim - 1))
    transformed = _fft_last_axis(halves, inverse)
    even, odd = transformed[..., 0, :], transformed[..., 1, :]
    rotated = _twiddles(n, n // 2, x.dtype, inverse) * odd
    return lax.concatenate([even + rotated, even - rotated], x.ndim - 1)


def _along_axis(x, axis, fn):
    if axis == x.ndim - 1:
        return fn(x)
    order = [*range(x.ndim)]
    order[axis], order[-1] = order[-1], order[axis]
    return lax.transpose(fn(lax.transpose(x, order)), order)


def _fft(x, *, fft_type, fft_lengths, **_):
    fft_type = lax_fft.FftType(fft_type)
    dims = len(fft_lengths)
    axes = range(x.ndim - dims, x.ndim)
    if fft_type == lax_fft.FftType.RFFT:
        out = lax.convert_element_type(x, np.complex64)
        for axis in axes:
            out = _along_axis(out, axis, lambda v: _fft_last_axis(v, inverse=False))
        return lax.slice_in_dim(out, 0, fft_lengths[-1] // 2 + 1, axis=x.ndim - 1)
    if fft_type == lax_fft.FftType.IRFFT:
        n = fft_lengths[-1]
        # Inverse complex FFT over the leading axes first, then a 1-D inverse real FFT on
        # the last axis, whose Hermitian half spectrum is rebuilt as X[n - k] = conj(X[k]).
        out = x
        for axis in list(axes)[:-1]:
            out = _along_axis(out, axis, lambda v: _fft_last_axis(v, inverse=True))
        mirrored = lax.rev(lax.slice_in_dim(out, 1, n - out.shape[-1] + 1, axis=x.ndim - 1),
                           (x.ndim - 1,))
        full = lax.concatenate([out, lax.conj(mirrored)], x.ndim - 1)
        full = _fft_last_axis(full, inverse=True)
        return lax.real(full) / np.float32(np.prod(fft_lengths))
    inverse = fft_type == lax_fft.FftType.IFFT
    out = x
    for axis in axes:
        out = _along_axis(out, axis, lambda v: _fft_last_axis(v, inverse=inverse))
    if inverse:
        out = out / lax.convert_element_type(np.float32(np.prod(fft_lengths)), out.dtype)
    return out


def register():
    _register_rule(lax_internal.real_p, _real_rule)
    _register_rule(lax_internal.imag_p, _imag_rule)
    _register_rule(lax_internal.complex_p, _complex_rule)
    _register_rule(lax_fft.fft_p, mlir.lower_fun(_fft, multiple_results=False))
    splits = [
        (lax_internal.pad_p, _pad, False),
        (slicing.gather_p, _gather, False),
        (slicing.dynamic_update_slice_p, _dynamic_update_slice, False),
        (slicing.scatter_p, partial(_scatter, lax.scatter), False),
        (slicing.scatter_add_p, partial(_scatter, lax.scatter_add), False),
        (slicing.scatter_sub_p, partial(_scatter, lax.scatter_sub), False),
        (slicing.scatter_mul_p, _scatter_mul, False),
        (convolution.conv_general_dilated_p, _bilinear(convolution.conv_general_dilated_p), False),
    ]
    for primitive, fn, multiple_results in splits:
        _register_rule(primitive, _when_complex(primitive, fn, multiple_results))
    for primitive, reduce, init in ((lax_internal.reduce_max_p, lax.reduce_max, -np.inf),
                                    (lax_internal.reduce_min_p, lax.reduce_min, np.inf)):
        _register_rule(primitive, _when_complex(primitive, partial(_reduce_chooser, reduce, init),
                                                default=current_rule(primitive)))
    for primitive, compare in ((lax_internal.max_p, lax.gt), (lax_internal.min_p, lax.lt)):
        _register_rule(primitive, _when_complex(primitive, partial(_chooser, compare),
                                                default=current_rule(primitive)))
    _register_rule(lax_internal.sort_p, _sort_rule())
    _register_rule(lax_internal.dot_general_p, _dot_general_rule())
    from jax._src.lax.control_flow import loops  # pylint: disable=import-outside-toplevel
    _register_rule(loops.while_p, _while_rule())


# Complex implementations for primitives that _lowerings.py also handles for real dtypes.
ELEMENTWISE = {
    lax_internal.exp_p: exp, lax_internal.log_p: log, lax_internal.sqrt_p: sqrt,
    lax_internal.rsqrt_p: rsqrt, lax_internal.sin_p: sin, lax_internal.cos_p: cos,
    lax_internal.tan_p: tan, lax_internal.tanh_p: tanh, lax_internal.sinh_p: sinh,
    lax_internal.cosh_p: cosh, lax_internal.expm1_p: expm1, lax_internal.log1p_p: log1p,
    lax_internal.sign_p: sign, lax_internal.asin_p: asin, lax_internal.acos_p: acos,
    lax_internal.atan_p: atan, lax_internal.asinh_p: asinh, lax_internal.acosh_p: acosh,
    lax_internal.atanh_p: atanh, lax_internal.exp2_p: exp2, lax_internal.log2_p: log2,
    lax_internal.square_p: square,
}
