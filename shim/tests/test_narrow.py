"""float8, float4 and 2-/4-bit integers on METAL against XLA's CPU backend.

    .venv-shim/bin/python shim/tests/test_narrow.py            # METAL vs CPU
    python shim/tests/test_narrow.py --emulate                 # any machine: the rewrite on CPU

Covers conversions into every format from float32, float16, bfloat16 and integers (each
format's values, the midpoints between them and their neighbours, overflow, specials),
arithmetic, transcendental functions, comparisons, reductions, matmul, data movement,
scatter, bitcasts of every width, control flow, integer shifts and bit counts, and narrow
program inputs and outputs. Programs take and return float32/int32/bytes, so the newer types
(which cannot cross a METAL program boundary) are checked too. The CPU reference runs with
--xla_allow_excess_precision=false, so XLA rounds after every operation as the METAL rules
do; it computes float8 arithmetic in float16 rather than float32, though, so a check allows
a few differences in the last bit where it says so. Conversions and bit operations must
match exactly. Exits non-zero on any failure.

--emulate runs the shim's rewrite (_narrow.py) on the CPU backend instead of METAL.
"""

import argparse
import importlib.util
import os
import sys

# Before JAX starts its CPU client: round to narrow types after every operation.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_allow_excess_precision=false").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax._src import core  # noqa: E402

EMULATE = "--emulate" in sys.argv
if EMULATE:  # the repository's copy, without importing the package (which needs the dylib)
    _spec = importlib.util.spec_from_file_location("_narrow", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "python", "jax_plugins", "metal_shim",
        "_narrow.py"))
    _narrow = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_narrow)
else:
    from jax_plugins.metal_shim import _narrow

RNG = np.random.default_rng(0)
FLOATS = [ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2, ml_dtypes.float8_e4m3,
          ml_dtypes.float8_e3m4, ml_dtypes.float8_e4m3fnuz, ml_dtypes.float8_e5m2fnuz,
          ml_dtypes.float8_e4m3b11fnuz, ml_dtypes.float8_e8m0fnu, ml_dtypes.float4_e2m1fn]
INTS = [ml_dtypes.int4, ml_dtypes.uint4, ml_dtypes.int2, ml_dtypes.uint2]
# Formats StableHLO 0.19.4 has, which can be METAL program inputs and outputs.
BOUNDARY = [ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2, ml_dtypes.float8_e4m3fnuz,
            ml_dtypes.float8_e5m2fnuz, ml_dtypes.float8_e4m3b11fnuz, ml_dtypes.int4,
            ml_dtypes.uint4]
failures = []


def emulated(fn):
    """fn's jaxpr rewritten by _narrow and run on the CPU backend."""
    def run(*args):
        rewritten = _narrow.carrier_jaxpr(jax.make_jaxpr(fn)(*args))
        return jax.jit(core.jaxpr_as_fun(rewritten))(*args)
    return run


def on_target(fn, *args):
    if TARGET is None:
        return emulated(fn)(*args)
    with jax.default_device(TARGET):
        return jax.jit(fn)(*jax.device_put(args, TARGET))


def same(got, want):
    got, want = np.asarray(got), np.asarray(want)
    if got.dtype.kind == "f" or _narrow.narrow_format(got.dtype) is not None:
        g, w = got.astype(np.float64), want.astype(np.float64)
        return ((g == w) & (np.signbit(g) == np.signbit(w))) | (np.isnan(g) & np.isnan(w))
    return got.astype(np.int64) == want.astype(np.int64)


def check(name, fn, *args, tolerate=0, reference=None):
    """fn on the target against `reference` (default fn) on XLA's CPU backend."""
    try:
        with jax.default_device(jax.devices("cpu")[0]):
            want = jax.tree.leaves(jax.jit(reference or fn)(*args))
        got = jax.tree.leaves(on_target(fn, *args))
    except Exception as error:  # pylint: disable=broad-except
        failures.append(name)
        print(f"FAIL  {name}: {type(error).__name__}: {str(error).splitlines()[0][:200]}")
        return
    differ = 0
    for g, w in zip(got, want):
        if np.shape(g) != np.shape(w) or np.asarray(g).dtype != np.asarray(w).dtype:
            failures.append(name)
            print(f"FAIL  {name}: {np.asarray(g).dtype}{np.shape(g)} vs "
                  f"{np.asarray(w).dtype}{np.shape(w)}")
            return
        ok = same(g, w)
        differ += int(ok.size - ok.sum())
        if ok.size - ok.sum() > tolerate:
            bad = np.flatnonzero(~ok.ravel())[:4]
            failures.append(name)
            print(f"FAIL  {name}: {ok.size - ok.sum()} of {ok.size} differ, e.g. got "
                  f"{np.asarray(g).ravel()[bad]} want {np.asarray(w).ravel()[bad]}")
            return
    print(f"ok    {name}" + (f" ({differ} last-bit differences)" if differ else ""))


def values(dtype, shape):
    """Random values of a narrow type, finite ones for floats, as float32 or int32."""
    fmt = _narrow.narrow_format(dtype)
    table = np.arange(2 ** fmt.bits, dtype=np.uint8).view(dtype)
    pool = table.astype(np.int32) if fmt.integer else table.astype(np.float32)
    if not fmt.integer:
        pool = pool[np.isfinite(pool)]
    return RNG.choice(pool, shape)


def conversion_inputs(dtype, source):
    """Each value of the format, the midpoints between neighbours, values one source ulp
    away from both, overflow, specials, and random values over the source's range."""
    fmt = _narrow.narrow_format(dtype)
    table = np.arange(2 ** fmt.bits, dtype=np.uint8).view(dtype).astype(np.float64)
    vals = np.unique(table[~np.isnan(table)])
    mids = (vals[1:] + vals[:-1]) / 2
    finite = vals[np.isfinite(vals)]
    xs = [vals, mids, -vals, -mids,
          [0.0, -0.0, np.inf, -np.inf, np.nan, finite.max() * 1.0625, finite.max() * 2,
           finite[finite > 0].min() / 2, finite[finite > 0].min() / 4, 1e30, -1e30, 1e-30]]
    with np.errstate(over="ignore"):
        for v in (vals, mids):
            v = v.astype(source)
            xs += [np.nextafter(v, source(np.inf)), np.nextafter(v, source(-np.inf))]
        xs.append(RNG.standard_normal(4000) * np.exp2(RNG.integers(-30, 30, 4000)))
        return np.concatenate([np.asarray(x, np.float64).ravel() for x in xs]).astype(source)


def conversions():
    for source in (np.float32, np.float16, ml_dtypes.bfloat16):
        for dtype in FLOATS:
            check(f"convert {np.dtype(source).name} -> {np.dtype(dtype).name}",
                  lambda x, d=dtype: x.astype(d).astype(np.float32),
                  conversion_inputs(dtype, source))
    for dtype in INTS:
        ints = np.arange(-300, 300, dtype=np.int32)
        floats = np.concatenate([np.linspace(-20, 20, 801), [np.nan, np.inf, -np.inf, -0.0,
                                 0.5, -0.5, 7.5, -8.5, 15.99, 16.0, 1e10]]).astype(np.float32)
        check(f"convert int32 -> {np.dtype(dtype).name}",
              lambda x, d=dtype: x.astype(d).astype(np.int32), ints)
        check(f"convert float32 -> {np.dtype(dtype).name}",
              lambda x, d=dtype: x.astype(d).astype(np.int32), floats)


def float_ops(dtype):
    name = np.dtype(dtype).name
    n8 = lambda v: v.astype(dtype)
    f32 = lambda v: v.astype(np.float32)
    x, y = values(dtype, (64,)), values(dtype, (64,))
    check(f"{name} arithmetic", lambda a, b: [f32(op(n8(a), n8(b))) for op in (
        lax.add, lax.sub, lax.mul, lax.div, jnp.maximum, jnp.minimum)] + [
        f32(-n8(a)), f32(abs(n8(a)))], x, y, tolerate=4)
    check(f"{name} functions", lambda a: [f32(fn(n8(a))) for fn in (
        jnp.exp, lambda v: jnp.sqrt(abs(v)), jnp.sin, jnp.tanh, jnp.floor, jnp.round)],
        x, tolerate=4)
    check(f"{name} comparisons", lambda a, b: [n8(a) < n8(b), n8(a) == n8(b),
                                               jnp.isnan(n8(a)), jnp.isinf(n8(a)),
                                               f32(jnp.where(n8(a) > n8(b), n8(a), n8(b)))],
          x, y)
    check(f"{name} reductions", lambda a: [f32(jnp.sum(n8(a))), f32(jnp.max(n8(a))),
                                           f32(jnp.sum(n8(a).reshape(8, 8), axis=0)),
                                           jnp.argmax(n8(a)), f32(jnp.sort(n8(a)))],
          x, tolerate=8)
    xm, ym = values(dtype, (8, 8)), values(dtype, (8, 4))
    upcast = None
    if _narrow.narrow_format(dtype).bits < 8:  # XLA's CPU backend rejects float4 dots
        upcast = lambda a, b: [jnp.dot(a, b), f32(jnp.dot(a, b).astype(dtype))]
    check(f"{name} dot", lambda a, b: [jnp.dot(n8(a), n8(b), preferred_element_type=np.float32),
                                       f32(jnp.dot(n8(a), n8(b)))], xm, ym, tolerate=16,
          reference=upcast)
    idx = np.array([5, 1, 3])
    check(f"{name} movement", lambda a: [f32(v) for v in (
        n8(a).reshape(8, 8).T, n8(a)[::-1], jnp.concatenate([n8(a), n8(a)]), jnp.pad(n8(a), 2),
        n8(a)[idx], jnp.roll(n8(a), 3), n8(a).at[np.array([1, 2])].set(n8(a)[:2]))], x)
    check(f"{name} scatter-add", lambda a, b: f32(n8(a).at[np.array([0, 0, 5, 7])].add(
        n8(b)[:4])), x, y, tolerate=2)
    check(f"{name} control flow", lambda a: [f32(v) for v in (
        lax.fori_loop(0, 3, lambda i, c: c * c + n8(a), n8(a)),
        lax.scan(lambda c, e: (c + e, c * e), n8(a)[0], n8(a))[1],
        lax.cond(a[0] > a[1], lambda v: v + v, lambda v: -v, n8(a)),
        jax.jit(lambda v: jax.checkpoint(lambda w: w * w)(v) + v)(n8(a)))], x, tolerate=16)
    check(f"{name} nextafter", lambda a, b: f32(lax.nextafter(n8(a), n8(b))), x, y)
    if _narrow.narrow_format(dtype).bits == 8:
        codes = np.arange(256, dtype=np.uint8)
        check(f"{name} bitcast", lambda c: [
            lax.bitcast_convert_type(lax.bitcast_convert_type(c, dtype), np.uint8),
            f32(lax.bitcast_convert_type(c, dtype)),
            lax.bitcast_convert_type(lax.bitcast_convert_type(c.reshape(64, 4), np.uint32),
                                     dtype).astype(np.float32)], codes,
              # NaN codes come back as ml_dtypes' NaN of their sign: conversions drop payloads.
            reference=lambda c: [jnp.asarray(codes.view(dtype).astype(np.float32).astype(
                dtype).view(np.uint8)), f32(lax.bitcast_convert_type(c, dtype)),
                f32(lax.bitcast_convert_type(c, dtype)).reshape(64, 4)])


def int_ops(dtype):
    name = np.dtype(dtype).name
    fmt = _narrow.narrow_format(dtype)
    n = lambda v: v.astype(dtype)
    i32 = lambda v: v.astype(np.int32)
    x, y = values(dtype, (64,)), values(dtype, (64,))
    nonzero = np.where(y == 0, 1, y)
    amounts = np.arange(64) % 2 ** fmt.bits  # beyond the width once wrapped to the type
    check(f"{name} arithmetic", lambda a, b, c: [i32(v) for v in (
        n(a) + n(b), n(a) - n(b), n(a) * n(b), -n(a), jnp.maximum(n(a), n(b)), n(a) & n(b),
        n(a) | n(b), n(a) ^ n(b), ~n(a), lax.div(n(a), n(c)), lax.rem(n(a), n(c)))] + [
        n(a) == n(b), n(a) < n(b)], x, y, nonzero)
    check(f"{name} shifts", lambda a, s: [i32(v) for v in (
        n(a) << n(s), lax.shift_right_logical(n(a), n(s)),
        lax.shift_right_arithmetic(n(a), n(s)), lax.clz(n(a)))], x, amounts)
    unsigned = {ml_dtypes.int4: ml_dtypes.uint4, ml_dtypes.int2: ml_dtypes.uint2}.get(dtype)
    # XLA's CPU backend counts a signed narrow value's sign-extended byte; count its bits.
    check(f"{name} population_count", lambda a: i32(lax.population_count(n(a))), x,
          reference=(lambda a: i32(lax.bitcast_convert_type(lax.population_count(
              lax.bitcast_convert_type(n(a), unsigned)), dtype))) if unsigned else None)
    check(f"{name} convert", lambda a, f: [i32(n(a)), n(a).astype(np.float32), i32(n(f)),
                                           n(a).astype(ml_dtypes.float8_e4m3fn).astype(
                                               np.float32)],
          x, (RNG.standard_normal(64) * 10).astype(np.float32))
    sort = fmt.bits > 2  # XLA's CPU sort of int2 crashes
    check(f"{name} reductions, loops", lambda a: [i32(jnp.sum(n(a), dtype=dtype)),
                                                  i32(jnp.max(n(a))),
                                                  i32(lax.iota(dtype, 20)),
                                                  i32(lax.fori_loop(0, 5, lambda i, c: c + n(a),
                                                                    n(a)))]
          + ([i32(jnp.sort(n(a)))] if sort else []), x)


def bitcast_widths():
    bc = lax.bitcast_convert_type
    i4, u4, i2, u2 = ml_dtypes.int4, ml_dtypes.uint4, ml_dtypes.int2, ml_dtypes.uint2
    f4 = ml_dtypes.float4_e2m1fn
    i32 = lambda v: v.astype(np.int32)
    check("bitcast int4[..., 2] <-> int8", lambda a, b: [bc(a.astype(i4), np.int8),
                                                         i32(bc(b, i4))],
          values(i4, (5, 2)), RNG.integers(-128, 128, 5).astype(np.int8))
    check("bitcast uint4[..., 4] <-> float16", lambda a, b: [bc(a.astype(u4), np.float16),
                                                             i32(bc(b, u4))],
          values(u4, (5, 4)), RNG.standard_normal(5).astype(np.float16))
    check("bitcast float32 <-> int4[..., 8]", lambda a, b: [i32(bc(a, i4)),
                                                            bc(b.astype(i4), np.float32)],
          RNG.standard_normal(5).astype(np.float32), values(i4, (5, 8)))
    check("bitcast int2[..., 2] <-> int4", lambda a, b: [i32(bc(a.astype(i2), i4)),
                                                         i32(bc(b.astype(i4), i2))],
          values(i2, (5, 2)), values(i4, (5,)))
    check("bitcast uint2[..., 4] <-> float8_e5m2", lambda a, b: [
        bc(a.astype(u2), ml_dtypes.float8_e5m2).astype(np.float32),
        i32(bc(b.astype(ml_dtypes.float8_e4m3fn), u2))],
        values(u2, (5, 4)), values(ml_dtypes.float8_e4m3fn, (5,)))
    check("bitcast float4[..., 2] <-> uint8", lambda a, b: [bc(a.astype(f4), np.uint8),
                                                            bc(b, f4).astype(np.float32)],
          values(f4, (5, 2)), RNG.integers(0, 256, 5).astype(np.uint8))


def boundary():
    """Narrow arrays in and out of programs (METAL: _boundary_io and the shim)."""
    for dtype in BOUNDARY:
        name = np.dtype(dtype).name
        fmt = _narrow.narrow_format(dtype)
        x = values(dtype, (4, 8)).astype(dtype)
        y = values(dtype, (4, 8)).astype(dtype)
        fns = {"identity": lambda a, b: a,
               "arithmetic": (lambda a, b: a * b + a) if fmt.integer else
                             (lambda a, b: (a.astype(np.float32) * 2).astype(dtype) + b),
               "zeros": lambda a, b: jnp.zeros((3,), dtype),
               "to float32": lambda a, b: a.astype(np.float32)}
        for label, fn in fns.items():
            check(f"{name} in/out: {label}", fn, x, y, tolerate=2)
        if TARGET is not None:
            placed = jax.device_put(x, TARGET)
            ok = placed.dtype == x.dtype and bool(np.all(same(np.asarray(placed), x)))
            print(("ok    " if ok else "FAIL  ") + f"{name} device_put round trip")
            if not ok:
                failures.append(f"{name} device_put")
    if TARGET is not None:
        try:
            jax.jit(lambda a: a.astype(ml_dtypes.float8_e4m3))(
                jax.device_put(np.ones(3, np.float32), TARGET))
            print("FAIL  float8_e4m3 output: no error")
            failures.append("float8_e4m3 output")
        except NotImplementedError:
            print("ok    float8_e4m3 output: refused with NotImplementedError")


def main():
    global TARGET  # pylint: disable=global-statement
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emulate", action="store_true",
                        help="run the rewrite on the CPU backend instead of METAL")
    args = parser.parse_args()
    if args.emulate:
        jax.config.update("jax_platforms", "cpu")
    TARGET = None if args.emulate else jax.devices("METAL")[0]
    conversions()
    for dtype in FLOATS:
        float_ops(dtype)
    for dtype in INTS:
        int_ops(dtype)
    bitcast_widths()
    if args.emulate:
        print("(narrow program inputs and outputs: METAL only)")
    else:
        boundary()
    print(f"\n{len(failures)} failed" + (f": {', '.join(failures)}" if failures else ""))
    sys.exit(1 if failures else 0)


TARGET = None
if __name__ == "__main__":
    main()
