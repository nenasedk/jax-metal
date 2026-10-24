"""Proof of concept: float64-like arithmetic on METAL as unevaluated float32 pairs (hi + lo).

Apple GPUs and MPS have no float64. "Double-float" arithmetic represents a value as two
float32s and uses error-free transformations (Knuth TwoSum, Dekker TwoProduct) to carry the
rounding error, giving ~48 significand bits. It only works if the GPU rounds every float32
add and multiply exactly per IEEE 754 (no fast-math reassociation, no FMA contraction);
this script checks that first.

    .venv-shim/bin/python shim/experimental/double_float.py

Limits even when it works: the exponent range stays float32's (about 1e-38 to 3e38), and each
operation costs 10-20 float32 operations.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np


# --- error-free transformations -------------------------------------------------------

def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def quick_two_sum(a, b):  # requires |a| >= |b|
    s = a + b
    return s, b - (s - a)


def split(a):  # Dekker: a = hi + lo with 12-bit halves
    c = 4097.0 * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


# --- double-float operations on (hi, lo) pairs ----------------------------------------

def df_add(x, y):
    s, e = two_sum(x[0], y[0])
    t, f = two_sum(x[1], y[1])
    e = e + t
    s, e = quick_two_sum(s, e)
    e = e + f
    return quick_two_sum(s, e)


def df_mul(x, y):
    p, e = two_prod(x[0], y[0])
    e = e + (x[0] * y[1] + x[1] * y[0])
    return quick_two_sum(p, e)


def df_div(x, y):
    q1 = x[0] / y[0]
    r = df_add(x, df_neg(df_mul((q1, jnp.zeros_like(q1)), y)))
    q2 = r[0] / y[0]
    r = df_add(r, df_neg(df_mul((q2, jnp.zeros_like(q2)), y)))
    q3 = r[0] / y[0]
    q1, q2 = quick_two_sum(q1, q2)
    return df_add((q1, q2), (q3, jnp.zeros_like(q3)))


def df_neg(x):
    return -x[0], -x[1]


def df_sqrt(x):
    # One Newton step on the float32 root: sqrt(x) ~ a + (x - a^2) / (2a).
    a = jnp.sqrt(x[0])
    a2 = df_mul((a, jnp.zeros_like(a)), (a, jnp.zeros_like(a)))
    correction = df_add(x, df_neg(a2))[0] / (2 * a)
    return quick_two_sum(a, correction)


def df_sum(x):
    """Pairwise tree reduction along axis 0 (length must be a power of two)."""
    hi, lo = x
    while hi.shape[0] > 1:
        half = hi.shape[0] // 2
        hi, lo = df_add((hi[:half], lo[:half]), (hi[half:], lo[half:]))
    return hi[0], lo[0]


def df_dot(x, y):
    return df_sum(df_mul(x, y))


# --- host helpers ---------------------------------------------------------------------

def to_pair(values):
    values = np.asarray(values, np.float64)
    hi = values.astype(np.float32)
    return hi, (values - hi.astype(np.float64)).astype(np.float32)


def from_pair(pair):
    return np.asarray(pair[0], np.float64) + np.asarray(pair[1], np.float64)


def main():
    metal = jax.devices("METAL")[0]
    rng = np.random.default_rng(0)

    # 1. Is float32 arithmetic on the GPU exact enough for error-free transformations?
    a = rng.standard_normal(1 << 16).astype(np.float32)
    b = (rng.standard_normal(1 << 16) * 1e-3).astype(np.float32)
    s, e = jax.jit(two_sum)(*jax.device_put((a, b), metal))
    p, f = jax.jit(two_prod)(*jax.device_put((a, b), metal))
    exact_sum = np.array_equal(np.asarray(s, np.float64) + np.asarray(e, np.float64),
                               a.astype(np.float64) + b.astype(np.float64))
    exact_prod = np.array_equal(np.asarray(p, np.float64) + np.asarray(f, np.float64),
                                a.astype(np.float64) * b.astype(np.float64))
    print(f"TwoSum exact on METAL: {exact_sum}; TwoProduct exact on METAL: {exact_prod}")

    # 2. Accuracy of double-float operations against numpy float64.
    x64 = rng.uniform(0.5, 2.0, 1 << 16) * 10 ** rng.uniform(-5, 5, 1 << 16)
    y64 = rng.uniform(0.5, 2.0, 1 << 16) * 10 ** rng.uniform(-5, 5, 1 << 16)
    x, y = to_pair(x64), to_pair(y64)
    ops = {
        "add": (lambda x, y: df_add(x, y), x64 + y64),
        "mul": (lambda x, y: df_mul(x, y), x64 * y64),
        "div": (lambda x, y: df_div(x, y), x64 / y64),
        "sqrt": (lambda x, y: df_sqrt(x), np.sqrt(x64)),
        "dot (65536 terms)": (lambda x, y: df_dot(x, y), np.dot(x64, y64)),
    }
    for name, (fn, want) in ops.items():
        compiled = jax.jit(fn)
        args = jax.device_put((x, y), metal)
        got = from_pair(compiled(*args))
        start = time.perf_counter()
        for _ in range(10):
            jax.block_until_ready(compiled(*args))
        elapsed = (time.perf_counter() - start) / 10
        error = float(np.max(np.abs(got - want) / np.abs(want)))
        single = float(np.max(np.abs(np.asarray(want, np.float32).astype(np.float64) - want)
                              / np.abs(want)))
        print(f"{name:18} max rel error {error:.2e} (float32 alone {single:.2e}), {elapsed * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
