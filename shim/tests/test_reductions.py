"""GPU regressions for integer, boolean, variadic and multi-op reductions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


@pytest.fixture(scope="module", autouse=True)
def metal_device():
    with jax.default_device(jax.devices("METAL")[0]):
        yield


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int32, np.uint32])
@pytest.mark.parametrize("shape,axes", [((3, 4, 5), (1, 2)), ((2, 3, 5), (0, 2)),
                                        ((2, 0, 3), (1,)), ((2, 3), ())])
@pytest.mark.parametrize("operation,reference", [(lax.reduce_or, np.bitwise_or),
                                                (lax.reduce_and, np.bitwise_and),
                                                (lax.reduce_xor, np.bitwise_xor)])
def test_bitwise_reduction(dtype, shape, axes, operation, reference):
    x = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
    x = np.bitwise_xor(x, np.array(np.iinfo(dtype).max, dtype))
    expected = reference.reduce(x, axis=axes)
    actual = jax.jit(lambda x: operation(x, axes))(x)
    np.testing.assert_array_equal(actual, expected)
    combine = {lax.reduce_or: lax.bitwise_or, lax.reduce_and: lax.bitwise_and,
               lax.reduce_xor: lax.bitwise_xor}[operation]
    init = np.array(5, dtype)
    generic = jax.jit(lambda x, init: lax.reduce(x, init, combine, axes))
    np.testing.assert_array_equal(generic(x, init), reference.reduce(x, axis=axes, initial=init))



def _on_cpu(fn, *args):
    return jax.jit(fn)(*jax.device_put(args, jax.devices("cpu")[0]))


def _compare(fn, *args, exact=True):
    got, want = jax.jit(fn)(*args), _on_cpu(fn, *args)
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        if exact:
            np.testing.assert_array_equal(g, w)
        else:
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-6)


# Variadic reductions and multi-op bodies: the plugin's converter takes one operand and one
# combining op. Bodies of one monoid op per operand split; others reduce as a tree.
@pytest.mark.parametrize("axes", [(0,), (1,), (0, 2), (0, 1, 2), ()])
def test_variadic_minmax(axes):
    x = np.random.default_rng(0).standard_normal((3, 4, 5)).astype(np.float32)
    x[1, 2, 3] = np.nan
    i = (np.arange(60) % 7 - 3).astype(np.int8).reshape(3, 4, 5)

    def minmax(x, i):
        return lax.reduce((x, x, i, i), (np.float32(np.inf), np.float32(-np.inf), np.int8(127),
                                          np.int8(-128)),
                          lambda a, b: (lax.min(a[0], b[0]), lax.max(a[1], b[1]),
                                        lax.min(a[2], b[2]), lax.max(a[3], b[3])), axes)

    _compare(minmax, x, i)
    _compare(lambda x: (jnp.ptp(x, axis=axes or None), jnp.ptp(x[0], axis=-1)), x)


def test_variadic_mixed_monoids_and_init():
    x = np.random.default_rng(1).uniform(0.5, 1.5, (4, 6)).astype(np.float32)
    b = np.random.default_rng(2).random((4, 6)) < 0.5

    def mixed(x, b, init):
        return lax.reduce((x, x, b), (init, np.float32(2), np.bool_(False)),
                          lambda p, q: (p[0] + q[0], p[1] * q[1], p[2] | q[2]), (1,))

    _compare(mixed, x, b, np.float32(0.25), exact=False)

    def pair(x):  # (JAX cannot differentiate the bool operand's symbolic-zero tangent)
        return lax.reduce((x, x), (np.float32(0), np.float32(2)),
                          lambda p, q: (p[0] + q[0], p[1] * q[1]), (1,))

    _compare(jax.grad(lambda x: pair(x)[1].sum()), x, exact=False)


@pytest.mark.parametrize("shape,axes", [((7, 5), (0,)), ((7, 5), (1,)), ((3, 4, 5), (0, 2)),
                                        ((0, 5), (0,)), ((4,), (0,))])
def test_coupled_body_tree(shape, axes):
    # argmax-style pairs: the index follows the value, ties to the lower index.
    x = np.random.default_rng(3).integers(-4, 5, shape).astype(np.float32)
    idx = np.broadcast_to(np.arange(shape[axes[0]]).reshape(
        [-1 if d == axes[0] else 1 for d in range(len(shape))]), shape).astype(np.int32)

    def argmax(x, idx):
        def body(a, b):
            take_a = (a[0] > b[0]) | ((a[0] == b[0]) & (a[1] < b[1]))
            return lax.select(take_a, a[0], b[0]), lax.select(take_a, a[1], b[1])
        return lax.reduce((x, idx), (np.float32(-np.inf), np.int32(2**30)), body, axes)

    _compare(argmax, x, idx)


def test_pytree_and_complex_reductions():
    x = np.random.default_rng(4).standard_normal((3, 4)).astype(np.float32)
    z = (x + 1j * x[::-1]).astype(np.complex64)
    _compare(lambda x: lax.reduce({"a": x, "b": x * 2}, {"a": np.float32(0), "b": np.float32(1)},
                                  lambda p, q: {"a": p["a"] + q["a"], "b": p["b"] + q["b"]}, (0,)),
             x, exact=False)
    _compare(lambda z, init: lax.reduce(z, init, lax.max, (1,)), z, np.complex64(-5 + 0j))


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("reduction", [jnp.all, jnp.any])
def test_boolean_reductions_of_constants(constant, reduction):
    # On a mask that is constant after folding, the plugin's MHLO canonicalizer leaves a
    # native and/or reduction whose body only returns the init value.
    mask = np.full((8, 1), constant)
    x = np.arange(8, dtype=np.int32)
    _compare(lambda x: (reduction(jnp.asarray(mask), axis=1), reduction(mask[:0], axis=0),
                        jnp.repeat(x[:6], np.array([1, 3, 1, 1, 2, 0]), axis=0,
                                   total_repeat_length=8)), x)
    _compare(lambda x: jnp.sum(x.reshape(2, 4), axis=1, where=mask[:2] | True, initial=1), x)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
