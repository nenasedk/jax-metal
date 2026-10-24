"""Verify the Phase 3 METAL lowering rules against the CPU backend.

    .venv-shim/bin/python shim/tests/test_phase3.py

Covers cholesky, triangular_solve (every flag combination, batched), linalg.solve/inv,
reduce_precision, bfloat16 rounding and the rbg PRNG. Exits non-zero on any failure.
"""

import itertools
import sys

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import lax

METAL, CPU = jax.devices("METAL")[0], jax.devices("cpu")[0]
RNG = np.random.default_rng(0)
failures = []


def on(device, fn, *args):
    placed = jax.device_put(args, device)
    with jax.default_device(device):
        return jax.tree.map(np.asarray, jax.jit(fn)(*placed))


def check(name, fn, *args, rtol=1e-4, atol=1e-4, exact=False):
    try:
        got, want = on(METAL, fn, *args), on(CPU, fn, *args)
        pairs = list(zip(jax.tree.leaves(got), jax.tree.leaves(want)))
        if exact:
            ok = all(np.array_equal(g.view(np.uint8), w.view(np.uint8)) for g, w in pairs)
        else:
            ok = all(np.allclose(g, w, rtol=rtol, atol=atol, equal_nan=True) for g, w in pairs)
        detail = "" if ok else f"max diff {max(float(np.nanmax(np.abs(g.astype(np.float64) - w))) for g, w in pairs):.3g}"
    except Exception as error:  # report and continue
        ok, detail = False, (str(error).strip().splitlines() or [repr(error)])[0][:150]
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def spd(*batch, n=6):
    a = RNG.standard_normal((*batch, n, n)).astype(np.float32)
    return (a @ np.swapaxes(a, -1, -2) + n * np.eye(n, dtype=np.float32)).astype(np.float32)


def well_conditioned_triangular(n=5, *batch):
    a = RNG.standard_normal((*batch, n, n)).astype(np.float32) * 0.3
    return a + 3 * np.eye(n, dtype=np.float32)


# Cholesky
check("cholesky 6x6", jnp.linalg.cholesky, spd())
check("cholesky batched 3x4x6x6", jnp.linalg.cholesky, spd(3, 4))
check("cholesky 32x32", jnp.linalg.cholesky, spd(n=32), rtol=1e-3, atol=1e-3)

# Triangular solve: every flag combination, batched
a, b_left, b_right = well_conditioned_triangular(5, 2), RNG.standard_normal((2, 5, 3)).astype(np.float32), RNG.standard_normal((2, 3, 5)).astype(np.float32)
for left, lower, trans, unit in itertools.product([True, False], repeat=4):
    b = b_left if left else b_right
    fn = lambda a, b, left=left, lower=lower, trans=trans, unit=unit: lax.linalg.triangular_solve(
        a, b, left_side=left, lower=lower, transpose_a=trans, unit_diagonal=unit)
    check(f"triangular_solve left={left} lower={lower} transpose={trans} unit={unit}", fn, a, b,
          rtol=1e-3, atol=1e-3)

# Things built on them
m = spd(n=8) + RNG.standard_normal((8, 8)).astype(np.float32)
check("linalg.solve", jnp.linalg.solve, m, RNG.standard_normal(8).astype(np.float32), rtol=1e-3, atol=1e-3)
check("linalg.inv", jnp.linalg.inv, m, rtol=1e-3, atol=1e-3)
check("cho_solve", lambda a, b: jsp.linalg.cho_solve(jsp.linalg.cho_factor(a), b), spd(n=8),
      RNG.standard_normal(8).astype(np.float32), rtol=1e-3, atol=1e-3)
check("grad through solve", jax.grad(lambda m, v: jnp.sum(jnp.linalg.solve(m, v) ** 2)), m,
      RNG.standard_normal(8).astype(np.float32), rtol=1e-3, atol=1e-3)

# reduce_precision: bit-exact against XLA's CPU implementation
special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40, -1e-40, 1e-8, 65504.0, 65520.0,
                    1e5, -1e5, 3.0e38, 1.0 + 2 ** -11, 1.0 + 3 * 2 ** -11, 1.0 + 2 ** -8], np.float32)
values = np.concatenate([special, RNG.standard_normal(256).astype(np.float32) * 10 ** RNG.uniform(-6, 6, 256).astype(np.float32)])
for exponent_bits, mantissa_bits in [(5, 10), (8, 7), (4, 3), (5, 2), (8, 23), (6, 0)]:
    check(f"reduce_precision e{exponent_bits} m{mantissa_bits}",
          lambda x, e=exponent_bits, mb=mantissa_bits: lax.reduce_precision(x, e, mb), values, exact=True)
check("reduce_precision on float16 input", lambda x: lax.reduce_precision(x, 5, 3),
      values.astype(np.float16), exact=True)

# bfloat16 rounding inside a program
ties = np.array([1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -(1.0 + 2 ** -8), 1.0 + 2 ** -9], np.float32)
check("f32 -> bf16 -> f32 round trip", lambda x: x.astype(jnp.bfloat16).astype(jnp.float32),
      np.concatenate([values, ties]), exact=True)
check("bf16 output buffer", lambda x: x.astype(jnp.bfloat16), values, exact=True)
check("f16 -> bf16 -> f32", lambda x: x.astype(jnp.bfloat16).astype(jnp.float32),
      values.astype(np.float16), exact=True)
check("bf16 -> f32 subnormals and specials", lambda x: x.astype(jnp.float32),
      np.concatenate([values, ties]).astype(jnp.bfloat16), exact=True)
check("bf16 -> f16 and int32 -> bf16", lambda x, i: (x.astype(jnp.float16), i.astype(jnp.bfloat16)),
      values.astype(jnp.bfloat16), np.array([0, 1, -7, 257, 65535, 2 ** 24 + 1, -2 ** 30], np.int32), exact=True)
bf16_a = (values[:64] / 1e3).astype(jnp.bfloat16)
bf16_b = RNG.standard_normal(64).astype(jnp.bfloat16)
check("bf16 arithmetic", lambda a, b: (a * b, a / b, a - b), bf16_a, bf16_b, exact=True)
check("bf16 transcendental (1 bf16 ulp)", lambda a: (jnp.exp(a), jnp.tanh(a)), bf16_a,
      rtol=2 ** -7, atol=1e-6)
check("bf16 matmul inputs", lambda a, b: (a.astype(jnp.bfloat16).astype(jnp.float32) @ b),
      RNG.standard_normal((4, 8)).astype(np.float32), RNG.standard_normal((8, 3)).astype(np.float32))

# rbg PRNG: bits are backend-defined, so check it runs, is deterministic and looks random
try:
    with jax.default_device(METAL):
        key = jax.random.key(42, impl="rbg")
        draw = jax.jit(lambda k: (jax.random.uniform(k, (4096,)), jax.random.bits(k, (1024,), jnp.uint8),
                                  jax.random.bits(k, (1024,), jnp.uint16)))
        (u, b8, b16), (u2, _, _) = draw(key), draw(key)
        u, b8, b16, u2 = map(np.asarray, (u, b8, b16, u2))
    ok = (np.array_equal(u, u2) and 0.45 < u.mean() < 0.55 and u.min() >= 0 and u.max() < 1
          and len(np.unique(b8)) > 200 and b16.dtype == np.uint16 and len(np.unique(b16)) > 900)
    detail = f"mean {u.mean():.3f}, unique u8 {len(np.unique(b8))}, unique u16 {len(np.unique(b16))}"
except Exception as error:  # the old failure mode aborts the process instead
    ok, detail = False, str(error).splitlines()[0][:150]
print(f"{'PASS' if ok else 'FAIL'} rbg PRNG {detail}")
if not ok:
    failures.append("rbg")

print(f"\n{len(failures)} failure(s): {failures or 'none'}")
sys.exit(1 if failures else 0)
