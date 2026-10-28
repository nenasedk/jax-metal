"""Verify complex64 support inside METAL programs, and FFT, against the CPU.

    .venv-shim/bin/python shim/tests/test_complex.py

The first group keeps complex values inside programs (real inputs, real outputs); the second
passes complex64 arrays in and out of programs. Exits non-zero on any failure.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

METAL, CPU = jax.devices("METAL")[0], jax.devices("cpu")[0]
RNG = np.random.default_rng(0)
failures = []


def on(device, fn, *args):
    placed = jax.device_put(args, device)
    with jax.default_device(device):
        return jax.tree.map(np.asarray, jax.jit(fn)(*placed))


def check(name, fn, *args, tol=1e-4):
    try:
        got, want = on(METAL, fn, *args), on(CPU, fn, *args)
        pairs = list(zip(jax.tree.leaves(got), jax.tree.leaves(want)))
        # Tolerance relative to the output's scale (FFT outputs grow with n).
        errors = [float(np.max(np.abs(g - w)) / max(1.0, float(np.max(np.abs(w))))) for g, w in pairs]
        ok = all(e <= tol for e in errors)
        detail = f"max scaled error {max(errors):.2e}"
    except Exception as error:  # report and continue
        ok, detail = False, (str(error).strip().splitlines() or [repr(error)])[0][:150]
    print(f"{'PASS' if ok else 'FAIL'} {name:40} {detail}")
    if not ok:
        failures.append(name)


def parts(z):
    return jnp.real(z), jnp.imag(z)


x, y = RNG.standard_normal(64).astype(np.float32), RNG.standard_normal(64).astype(np.float32)
z = lambda x, y: jax.lax.complex(x, y)

# Construction, arithmetic, reductions
check("real/imag of complex(x, y)", lambda x, y: parts(z(x, y)), x, y)
check("multiply, divide, conj", lambda x, y: parts(z(x, y) * z(y, x) / (z(x, y) + 2) + jnp.conj(z(y, x))), x, y)
check("abs, angle", lambda x, y: (jnp.abs(z(x, y)), jnp.angle(z(x, y))), x, y)
check("sum, mean, dot", lambda x, y: (*parts(jnp.sum(z(x, y))), *parts(jnp.mean(z(x, y) * z(y, x))),
                                      *parts(jnp.dot(z(x, y), z(y, x)))), x, y)
check("complex matmul", lambda a, b: parts(z(a, b) @ z(b, a).T), RNG.standard_normal((8, 16)).astype(np.float32),
      RNG.standard_normal((8, 16)).astype(np.float32))
check("exp(1j * x).real (rotation)", lambda x: jnp.exp(1j * x).real, x)

# Transcendental functions
small = lambda v: v * 0.5
for name, fn in [("exp", jnp.exp), ("log", jnp.log), ("sqrt", jnp.sqrt), ("sin", jnp.sin),
                 ("cos", jnp.cos), ("tan", jnp.tan), ("tanh", jnp.tanh), ("sinh", jnp.sinh),
                 ("cosh", jnp.cosh), ("expm1", jnp.expm1), ("log1p", jnp.log1p),
                 ("sign", jnp.sign), ("reciprocal sqrt", jax.lax.rsqrt),
                 ("exp2", jnp.exp2), ("log2", jnp.log2), ("square", jnp.square),
                 ("power", lambda v: jnp.power(v, 1.5 + 0.5j))]:
    check(f"complex {name}", lambda x, y, fn=fn: parts(fn(z(small(x), small(y)))), x, y, tol=1e-4)

# FFT family
for n in (8, 12, 15, 7, 64, 100, 1000, 1024, 4096):
    signal = RNG.standard_normal(n).astype(np.float32)
    check(f"fft n={n}", lambda s: parts(jnp.fft.fft(s)), signal)
    check(f"ifft(fft) n={n}", lambda s: parts(jnp.fft.ifft(jnp.fft.fft(s))), signal)
    check(f"rfft n={n}", lambda s: parts(jnp.fft.rfft(s)), signal)
    check(f"irfft(rfft) n={n}", lambda s: jnp.fft.irfft(jnp.fft.rfft(s), n=s.shape[-1]), signal)
image = RNG.standard_normal((4, 16, 12)).astype(np.float32)
check("fft2 batched", lambda v: parts(jnp.fft.fft2(v)), image)
check("rfftn / irfftn", lambda v: jnp.fft.irfftn(jnp.fft.rfftn(v), s=v.shape), image)
check("power spectrum", lambda s: jnp.abs(jnp.fft.rfft(s)) ** 2, RNG.standard_normal((3, 256)).astype(np.float32))
check("FFT convolution", lambda a, b: jnp.fft.irfft(jnp.fft.rfft(a, 128) * jnp.fft.rfft(b, 128), 128),
      RNG.standard_normal(64).astype(np.float32), RNG.standard_normal(64).astype(np.float32))
check("grad through rfft", jax.grad(lambda s: jnp.sum(jnp.abs(jnp.fft.rfft(s)) ** 2)),
      RNG.standard_normal(32).astype(np.float32))

# Complex values at the program boundary (inputs and outputs are complex64 arrays)
zc = (x + 1j * y).astype(np.complex64)
wc = (y - 2j * x).astype(np.complex64)


def check_complex_io(name, fn, *args, tol=1e-4):
    try:
        got, want = on(METAL, fn, *args), on(CPU, fn, *args)
        pairs = list(zip(jax.tree.leaves(got), jax.tree.leaves(want)))
        ok = all(g.dtype == w.dtype and g.shape == w.shape for g, w in pairs)
        errors = [float(np.max(np.abs(g - w), initial=0) / max(1.0, float(np.max(np.abs(w), initial=0))))
                  for g, w in pairs]
        ok = ok and all(e <= tol for e in errors)
        detail = f"max scaled error {max(errors, default=0):.2e}"
    except Exception as error:  # report and continue
        ok, detail = False, (str(error).strip().splitlines() or [repr(error)])[0][:150]
    print(f"{'PASS' if ok else 'FAIL'} {name:40} {detail}")
    if not ok:
        failures.append(name)


check_complex_io("complex in, complex out", lambda z: z * z + 1j, zc)
check_complex_io("complex in, real out", lambda z: jnp.abs(z) + jnp.angle(z), zc)
check_complex_io("real in, complex out", lambda a, b: jax.lax.complex(a, b) * (2 - 1j), x, y)
check_complex_io("several complex outputs", lambda z, w: (z + w, z * w, jnp.conj(z), jnp.sum(w)), zc, wc)
check_complex_io("mixed pytree outputs", lambda z, a: {"z": z * a, "n": jnp.linalg.norm(z), "a": a}, zc, x)
check_complex_io("complex scalar", lambda z: z[3] * z[5], zc)
check_complex_io("fft / ifft at the boundary", lambda z: jnp.fft.ifft(jnp.fft.fft(z) * 2), zc)
check_complex_io("complex matmul at the boundary",
                 lambda a, b: a @ b, zc[:48].reshape(6, 8), wc[:32].reshape(8, 4))
check_complex_io("fori_loop with complex carry",
                 lambda z: jax.lax.fori_loop(0, 5, lambda i, v: v * (0.9 + 0.1j) + 1, z), zc)
check_complex_io("grad wrt complex parameter",
                 jax.grad(lambda z: jnp.sum(jnp.abs(z * (1 + 2j)) ** 2)), zc)
check_complex_io("empty complex array", lambda z: z * 2, zc[:0])


def check_exact(name, fn, *args):
    """METAL and CPU agree exactly, NaN positions included."""
    try:
        got, want = on(METAL, fn, *args), on(CPU, fn, *args)
        for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
            np.testing.assert_array_equal(g, w)
        ok, detail = True, "exact"
    except Exception as error:  # report and continue
        ok, detail = False, (str(error).strip().splitlines() or [repr(error)])[0][:150]
    print(f"{'PASS' if ok else 'FAIL'} {name:40} {detail}")
    if not ok:
        failures.append(name)


def boundary_exact(*arrays):
    """Mask of elements that complex64 program inputs/outputs carry exactly: the boundary
    builds and splits complex values with multiplications by +-i, which turn a non-finite
    component into NaN in the other one and drop the sign of -0 (a known limitation, see
    re/findings/jax-suite-gpu-fixes.md)."""
    ok = np.ones(np.shape(arrays[0]), bool)
    for a in arrays:
        for part in (np.real(a), np.imag(a)):
            ok &= np.isfinite(part) & ~((part == 0) & np.signbit(part))
    return ok


# Complex max/min order by real part, then imaginary part; JAX lowers them with
# stablehlo.real/imag, which the plugin cannot legalize (_complex._chooser).
ties = (RNG.integers(-2, 3, 64) + 1j * RNG.integers(-2, 3, 64)).astype(np.complex64)
ties_b = (RNG.integers(-2, 3, 64) + 1j * RNG.integers(-2, 3, 64)).astype(np.complex64)
parts_grid = np.array([0.0, 1, -1, 3e38, -3e38, 1e-30], np.float32)
values = np.zeros(parts_grid.size ** 2, np.complex64)  # all (re, im) pairs: real-part ties
values.real, values.imag = np.repeat(parts_grid, parts_grid.size), np.tile(parts_grid, parts_grid.size)
special, special_b = np.repeat(values, values.size), np.tile(values, values.size)
check_exact("complex maximum/minimum, real-part ties", lambda a, b: (jnp.maximum(a, b), jnp.minimum(a, b)),
            ties, ties_b)
keep = boundary_exact(special, special_b)
check_exact("complex maximum/minimum, all pairs of a grid",
            lambda a, b: (jax.lax.max(a, b), jax.lax.min(a, b), jax.lax.max(b, a), jax.lax.min(b, a)),
            special[keep], special_b[keep])
check_exact("complex maximum with a scalar", lambda a: (jnp.maximum(a, 0), jnp.minimum(1j, a)), ties)
check_exact("complex max/min with initial", lambda a: (jnp.max(a, initial=1 + 0j), jnp.min(a, initial=0j)),
            ties)
check_exact("complex max with where and initial",
            lambda a: jnp.max(a.reshape(8, 8), axis=1, where=a.reshape(8, 8).real > 0, initial=-5 + 0j), ties)
check_exact("complex maximum inside a loop",
            lambda a, b: jax.lax.fori_loop(0, 3, lambda i, v: jnp.maximum(v * 1j, b), a), ties, ties_b)
# exp2, log2, square: JAX's lowerings emit stablehlo.real/imag (directly, or through a
# complex exp or chlo.square that XLA's serializer expands). square follows the CPU's CHLO
# formula exactly (checked where program inputs/outputs carry values exactly).
grid = np.array([0.0, 1, -1, 2, -2.5, 1e-30, 3e38, -3e38, 0.5, 1e19, -1e-20], np.float32)
square_in = np.zeros(grid.size ** 2, np.complex64)
square_in.real, square_in.imag = np.repeat(grid, grid.size), np.tile(grid, grid.size)
square_out = on(CPU, jnp.square, square_in)
check_exact("complex square, special values", jnp.square,
            square_in[boundary_exact(square_in, square_out)])
check_exact("complex square, |re| == |im|", jnp.square,
            (ties.real + 1j * ties.real * np.sign(ties.imag + 0.5)).astype(np.complex64))
check_complex_io("complex exp2/log2 at the boundary", lambda z: (jnp.exp2(z * 0.5), jnp.log2(z)), zc)
check_complex_io("second derivative of complex tan",
                 jax.grad(lambda z: jax.grad(lambda w: jnp.tan(w * 0.3).real)(z).real), zc[0])
check_exact("complex -> bfloat16/float16/float32", lambda z: (z.astype(jnp.bfloat16), z.astype(jnp.float16),
                                                              z.astype(jnp.float32)), zc)
check_complex_io("complex logaddexp", jnp.logaddexp, zc, wc)
check_complex_io("complex logaddexp2", jnp.logaddexp2, zc, wc)
check_complex_io("grad of complex logaddexp",
                 jax.grad(lambda a, b: jnp.sum(jnp.logaddexp(a, b)).real, holomorphic=False), zc, wc)


def chained_and_eager():
    with jax.default_device(METAL):
        a = jax.device_put(zc, METAL)
        b = jax.jit(lambda z: z * (1 - 1j))(a)          # complex output ...
        c = jax.jit(lambda z: jnp.exp(z) + z)(b)        # ... fed straight into another program
        d = (c[2:10] * 2).conj()                        # eager complex ops
        e = jnp.ones(3, jnp.complex64) * 1j             # complex creation on METAL
        cpu = jax.device_put(d, CPU)                    # transfer to another device
    expected = (np.exp(zc * (1 - 1j)) + zc * (1 - 1j))[2:10].conj() * 2
    return (np.allclose(np.asarray(d), expected, rtol=1e-4, atol=1e-4)
            and np.allclose(np.asarray(e), 1j) and np.allclose(np.asarray(cpu), expected, rtol=1e-4, atol=1e-4)
            and d.dtype == np.complex64 and list(cpu.devices())[0].platform == "cpu")


def donation():
    f = jax.jit(lambda z: z * 2, donate_argnums=0)
    a = jax.device_put(zc, METAL)
    return np.allclose(np.asarray(f(a)), zc * 2)


def no_buffer_leak():
    f = jax.jit(lambda z, w: (z * w, jnp.conj(z)))
    a, b = jax.device_put(zc, METAL), jax.device_put(wc, METAL)
    for _ in range(50):
        jax.block_until_ready(f(a, b))
    before = len(METAL.client.live_buffers())
    for _ in range(500):
        jax.block_until_ready(f(a, b))
    after = len(METAL.client.live_buffers())
    return after <= before + 4, f"live buffers {before} -> {after} over 500 calls"


for name, check_fn in [("chained jit, eager ops, device transfer", chained_and_eager),
                       ("donated complex input", donation), ("no buffer leak", no_buffer_leak)]:
    try:
        result = check_fn()
        ok, detail = result if isinstance(result, tuple) else (result, "")
    except Exception as error:  # report and continue
        ok, detail = False, (str(error).strip().splitlines() or [repr(error)])[0][:150]
    print(f"{'PASS' if ok else 'FAIL'} {name:40} {detail}")
    if not ok:
        failures.append(name)

print(f"\n{len(failures)} failure(s): {failures or 'none'}")
sys.exit(1 if failures else 0)
