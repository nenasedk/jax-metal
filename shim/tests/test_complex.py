"""Verify complex64 support inside METAL programs, and FFT, against the CPU.

    .venv-shim/bin/python shim/tests/test_complex.py

The plugin rejects complex values at a program's inputs and outputs, so every case takes
real inputs and returns real parts. Exits non-zero on any failure.
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

# Known limitation: complex values at the program boundary
try:
    on(METAL, lambda w: w * 2, (x + 1j * y).astype(np.complex64))
    print("NOTE complex program inputs now work (limitation lifted?)")
except Exception as error:  # expected
    print(f"ok   complex at program boundary still unsupported: {str(error).splitlines()[0][:90]}")

print(f"\n{len(failures)} failure(s): {failures or 'none'}")
sys.exit(1 if failures else 0)
