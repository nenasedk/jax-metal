"""GPU regressions for reduce_window and select_and_scatter_add (_windows.py), against the CPU.

Apple's converter computes only rank-4 float sum/max pooling correctly; these cases cover
the native subset (with padding, NaN and windows of -inf only) and the slice-based paths
(other ranks, min, integer and bool windows, base dilation, custom and variadic bodies,
pooling derivatives and gradients).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import ml_dtypes
import numpy as np
import pytest
from jax import lax

CPU = jax.devices("cpu")[0]
RNG = np.random.default_rng(0)


@pytest.fixture(scope="module", autouse=True)
def metal_device():
    with jax.default_device(jax.devices("METAL")[0]):
        yield


def compare(fn, *args, exact=True, rtol=1e-5):
    got = jax.tree.map(np.asarray, jax.jit(fn)(*args))
    want = jax.tree.map(np.asarray, jax.jit(fn)(*jax.device_put(args, CPU)))
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        assert g.shape == w.shape and g.dtype == w.dtype
        if exact:
            if g.dtype.kind == "V":  # ml_dtypes (bfloat16): NumPy only matches NaNs as float32
                g, w = g.astype(np.float32), w.astype(np.float32)
            np.testing.assert_array_equal(g, w)
        else:
            np.testing.assert_allclose(g.astype(np.float32), w.astype(np.float32), rtol=rtol,
                                       atol=rtol)


def data(shape, dtype, special=False):
    if np.dtype(dtype).kind in "iu":
        return RNG.integers(-50, 50, shape).astype(dtype)
    if dtype == np.bool_:
        return RNG.random(shape) < 0.3
    x = RNG.standard_normal(shape).astype(np.float32)
    if special and x.size >= 8:
        flat = x.reshape(-1)
        flat[RNG.choice(x.size, 4, replace=False)] = [np.nan, np.inf, -np.inf, -0.0]
    return x.astype(dtype)


# (shape, window, strides, padding, base_dilation, window_dilation)
WINDOWS = [
    ((10,), (3,), (1,), "VALID", (1,), (1,)),
    ((4, 6), (2, 2), (1, 1), "VALID", (1, 1), (1, 1)),
    ((4, 6), (2, 1), (2, 1), "SAME", (1, 1), (1, 1)),
    ((4, 6), (1, 2), (1, 2), ((0, 3), (1, 2)), (2, 3), (1, 2)),
    ((3, 5, 4), (2, 2, 1), (1, 2, 1), "SAME", (1, 1, 1), (1, 1, 1)),
    ((2, 5, 6, 3), (1, 2, 2, 1), (1, 1, 1, 1), "VALID", (1, 1, 1, 1), (1, 1, 1, 1)),
    ((2, 5, 6, 3), (1, 3, 3, 1), (1, 2, 2, 1), "SAME", (1, 1, 1, 1), (1, 1, 1, 1)),
    ((2, 5, 6, 3), (1, 2, 2, 1), (1, 1, 1, 1), ((0, 0), (2, 1), (1, 3), (0, 0)), (1, 2, 1, 1),
     (1, 1, 2, 1)),
    ((2, 3, 4, 5, 2), (1, 2, 2, 2, 1), (1, 1, 2, 1, 1), "SAME", (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
]


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"{w[0]}-{w[1]}")
@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16, np.int32])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_monoid_windows(window, dtype, op):
    shape, dims, strides, padding, base, dilation = window
    reduce, identity = {"sum": (lax.add, 0), "max": (lax.max, -np.inf), "min": (lax.min, np.inf)}[op]
    if np.dtype(dtype).kind in "iu" and op != "sum":
        identity = (np.iinfo(dtype).min if op == "max" else np.iinfo(dtype).max)
    init = np.array(identity, dtype)
    x = data(shape, dtype, special=(op != "sum"))
    fn = lambda x: lax.reduce_window(x, init, reduce, dims, strides, padding, base, dilation)
    floating_sum = op == "sum" and np.dtype(dtype).kind != "i"
    compare(fn, x, exact=not floating_sum, rtol=2e-2 if np.dtype(dtype).itemsize == 2 else 1e-5)


def test_max_pool_windows_of_negative_infinity():
    # MPS's max pooling returns 0 for a window holding only -inf.
    x = np.full((1, 4, 4, 2), -np.inf, np.float32)
    x[0, 0, 0, 0], x[0, 3, 3, 1] = 2.0, np.nan
    pads = ((0, 0), (1, 1), (0, 2), (0, 0))
    compare(lambda x: lax.reduce_window(x, -np.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), pads), x)


@pytest.mark.parametrize("reduce,init", [(lax.add, 1.5), (lax.max, 0.25), (lax.min, -0.5),
                                         (lax.mul, 2.0)])
def test_runtime_and_non_identity_init(reduce, init):
    x = data((2, 5, 6, 3), np.float32)
    fn = lambda x, init: lax.reduce_window(x, init, reduce, (1, 2, 3, 1), (1, 1, 2, 1),
                                           ((0, 0), (1, 2), (2, 0), (0, 0)), (1, 2, 1, 1))
    compare(fn, x, np.float32(init), exact=reduce in (lax.max, lax.min))


@pytest.mark.parametrize("dtype,reduce,init", [(np.bool_, lax.bitwise_or, False),
                                               (np.bool_, lax.bitwise_and, True),
                                               (np.int32, lax.bitwise_and, -1),
                                               (np.uint8, lax.bitwise_or, 0)])
def test_bitwise_windows(dtype, reduce, init):
    x = data((3, 7), dtype) if dtype == np.bool_ else data((3, 7), dtype) ^ np.array(5, dtype)
    compare(lambda x: lax.reduce_window(x, np.array(init, dtype), reduce, (2, 3), (1, 2), "SAME"), x)


@pytest.mark.parametrize("body", [lambda a, b: a - b, lambda a, b: a * 0.5 + b,
                                  lambda a, b: lax.max(a, b * b)], ids=["sub", "affine", "max-sq"])
def test_custom_bodies(body):
    # Folded in row-major order, accumulator first, from the init value, leaving out padding
    # and base-dilation holes, as XLA's CPU backend evaluates them.
    x = data((3, 6, 5), np.float32)
    compare(lambda x: lax.reduce_window(x, np.float32(0.5), body, (2, 3, 2), (1, 1, 1),
                                        ((1, 0), (0, 1), (1, 1)), (1, 1, 2), (1, 2, 1)), x)


def test_variadic_argmax_window():
    x = data((4, 9), np.float32)
    index = np.broadcast_to(np.arange(9, dtype=np.int32), (4, 9))

    def body(a, b):
        keep = (a[0] > b[0]) | ((a[0] == b[0]) & (a[1] < b[1]))
        return lax.select(keep, a[0], b[0]), lax.select(keep, a[1], b[1])

    compare(lambda x, i: lax.reduce_window((x, i), (np.float32(-np.inf), np.int32(9)), body,
                                           (1, 3), (1, 2), "SAME"), x, index)


@pytest.mark.parametrize("reduce,init", [(lax.max, -np.inf), (lax.min, np.inf)])
@pytest.mark.parametrize("shape,dims,strides,padding", [
    ((2, 6, 6, 3), (1, 2, 2, 1), (1, 2, 2, 1), "VALID"),
    ((2, 6, 6, 3), (1, 3, 3, 1), (1, 1, 1, 1), "SAME"),
    ((6, 7), (2, 3), (2, 1), ((1, 0), (0, 2))),
    ((2, 2, 5, 5, 3), (1, 1, 2, 2, 1), (1, 1, 2, 2, 1), "SAME"),
])
def test_pooling_derivatives(reduce, init, shape, dims, strides, padding):
    pool = lambda x: lax.reduce_window(x, init, reduce, dims, strides, padding)
    # select_and_gather_add (the JVP) on distinct values: on ties XLA's CPU backend does not
    # always keep the row-major first element, which these rules do (the order XLA defines).
    distinct = RNG.permutation(np.prod(shape)).reshape(shape).astype(np.float32)
    compare(lambda x, t: jax.jvp(pool, (x,), (t,)), distinct, data(shape, np.float32))
    # select_and_scatter_add (the gradient) with many ties: the first max (min) in row-major
    # order receives each window's gradient.
    x = RNG.integers(-3, 4, shape).astype(np.float32)
    compare(jax.grad(lambda x: (pool(x) * pool(x)).sum()), x)


def test_batched_pooling_gradient():
    x = RNG.integers(-3, 4, (3, 2, 6, 6, 2)).astype(np.float32)
    grad = jax.grad(lambda x: lax.reduce_window(x, -np.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1),
                                                "VALID").sum())
    compare(jax.vmap(grad), x)


def test_long_window():
    x = data((300,), np.float32, special=False)
    compare(lambda x: lax.reduce_window(x, -np.inf, lax.max, (100,), (3,), "VALID"), x)
    compare(lambda x: lax.reduce_window(x, 0.0, lax.add, (100,), (3,), "SAME"), x, exact=False,
            rtol=1e-4)


def test_empty_operand_and_output():
    compare(lambda x: lax.reduce_window(x, np.float32(0.5), lax.add, (2,), (1,), ((1, 2),)),
            np.zeros((0,), np.float32), exact=False)
    compare(lambda x: lax.reduce_window(x, -np.inf, lax.max, (5, 1), (1, 1), "VALID"),
            data((3, 4), np.float32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
