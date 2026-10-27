"""Keep trivial conditional branches in the Metal serializer's supported IR."""

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


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float32, jnp.bfloat16, np.complex64])
@pytest.mark.parametrize("index", [-2, 0, 1, 2, 7])
def test_switch_constants_and_forwarded_values(dtype, index):
    x = np.arange(6).reshape(2, 3).astype(dtype)
    if dtype == np.complex64:
        x += np.complex64(2j)

    def fn(i, x):
        return lax.switch(i, (lambda x: (x, jnp.zeros_like(x)),
                              lambda x: (jnp.ones_like(x), x),
                              lambda x: (x, x)), x)

    got = jax.jit(fn)(np.int32(index), x)
    with jax.default_device(jax.devices("cpu")[0]):
        want = jax.jit(fn)(np.int32(index), x)
    for actual, expected in zip(got, want):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("custom_jvp", [False, True])
def test_checkpoint_conditional_gradient(positive, custom_jvp):
    def f(x):
        return x * jnp.arange(3, dtype=x.dtype)
    if custom_jvp:
        f = jax.custom_jvp(f)

        @f.defjvp
        def f_jvp(primals, tangents):
            (x,), (xdot,) = primals, tangents
            return f(x), xdot * jnp.arange(3, dtype=x.dtype)

    @jax.checkpoint
    def loss(x):
        return lax.cond(x.sum() > 0, f, lambda x: x, x).sum()

    x = np.array([1, 2, 3], np.float32) * (1 if positive else -1)
    expected = np.arange(3, dtype=np.float32) if positive else np.ones(3, np.float32)
    np.testing.assert_array_equal(jax.grad(loss)(x), expected)
    np.testing.assert_array_equal(jax.jit(jax.grad(loss))(x), expected)


def test_nested_conditional_scan():
    def body(carry, x):
        value = lax.cond(x > 0, lambda: lax.cond(x > 2, lambda: x, lambda: carry),
                         lambda: jnp.float32(0))
        return value, value

    x = np.array([1, 3, -1, 2, 4], np.float32)
    actual = jax.jit(lambda x: lax.scan(body, jnp.float32(5), x))(x)
    np.testing.assert_array_equal(actual[0], 4)
    np.testing.assert_array_equal(actual[1], [5, 3, 0, 0, 4])


def test_preparation_is_idempotent_and_keeps_lazy_branches():
    from jax_plugins.metal_shim import _compiler_ir

    def fn(p, x):
        return lax.cond(p, lambda: x, lambda: jnp.zeros_like(x))

    lowered = jax.jit(fn).lower(True, np.ones(3, np.float32))
    module = lowered.compiler_ir()
    before = str(module)
    prepared = _compiler_ir.prepare_module(module)
    assert _compiler_ir.prepare_module(prepared) is prepared
    assert str(module) == before
    assert str(prepared).count('"stablehlo.case"') == 1
    # Leaves the case in place: branch computations are still evaluated lazily.
    assert not list(_compiler_ir._cases(prepared.operation))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
