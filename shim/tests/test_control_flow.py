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


@pytest.mark.parametrize("limit", [0, 1, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_unused_convergence_flag(limit, dtype):
    def fn(x, limit):
        def cond(state):
            i, _, running = state
            return (i < limit) & running

        def body(state):
            i, x, _ = state
            x = lax.fori_loop(0, 2, lambda _, x: x * 0.5, x)
            return i + 1, x, jnp.max(jnp.abs(x)) > 0.2

        return lax.while_loop(cond, body, (np.int32(0), x, True))[1]

    x = np.arange(1, 7).reshape(2, 3).astype(dtype)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(jax.jit(fn)(x, np.int32(limit)))
    np.testing.assert_array_equal(jax.jit(fn)(x, np.int32(limit)), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("shape", [(8, 4), (6, 6)])
@pytest.mark.parametrize("keep_status", [False, True])
def test_nested_qdwh_loops(dtype, shape, keep_status):
    from jax._src.tpu.linalg import qdwh

    rng = np.random.default_rng(41)
    x = rng.normal(size=shape).astype(dtype)
    if dtype == np.complex64:
        x += 1j * rng.normal(size=shape).astype(np.float32)
    x += 2 * np.eye(*shape, dtype=dtype)

    def fn(x):
        result = qdwh.qdwh(x)
        return result if keep_status else result[:2]

    compiled = jax.jit(fn).lower(x).compile()
    result = compiled(jax.device_put(x, jax.devices("METAL")[0]))
    u, h = map(np.asarray, result[:2])
    # Unique polar factors for this full-rank input, evaluated in double precision.
    wide = x.astype(np.complex128 if dtype == np.complex64 else np.float64)
    left, singular, right = np.linalg.svd(wide, full_matrices=False)
    np.testing.assert_allclose(u, left @ right, rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(h, (right.T.conj() * singular) @ right,
                               rtol=1e-5, atol=2e-6)
    if keep_status:
        assert 0 < int(result[2]) <= 10
        assert bool(result[3])


@pytest.mark.parametrize("keep_status", [False, True])
def test_loop_preparation_is_idempotent(keep_status):
    from jax_plugins.metal_shim import _compiler_ir

    def fn(x):
        def body(state):
            x, _ = state
            return x + 1, x < 3
        result = lax.while_loop(lambda state: state[1], body, (x, True))
        return result if keep_status else result[0]

    module = jax.jit(fn).lower(np.float32(0)).compiler_ir()
    before = str(module)
    prepared = _compiler_ir.prepare_module(module)
    assert prepared is not module
    assert _compiler_ir.prepare_module(prepared) is prepared
    assert str(module) == before
    assert prepared.operation.verify()
    assert not list(_compiler_ir._loops(prepared.operation))


def test_svd_with_inlined_convergence_loop():
    # Above 256, SVD's eigensolver uses spectral divide-and-conquer. Its QDWH
    # helper returns a status that becomes unused only after native inlining;
    # dropping that loop result used to crash MPSGraph at compilation.
    x = np.random.default_rng(2).uniform(size=(300, 300)).astype(np.float32)
    u, s, vh = map(np.asarray, jax.jit(
        lambda x: jnp.linalg.svd(x, full_matrices=False))(x))
    np.testing.assert_allclose(s, np.linalg.svd(x.astype(np.float64), compute_uv=False),
                               rtol=2e-4, atol=2e-5)
    assert np.linalg.norm((u * s) @ vh - x) / np.linalg.norm(x) < 2e-5
    np.testing.assert_allclose(u.T @ u, np.eye(300), atol=2e-5)
    np.testing.assert_allclose(vh @ vh.T, np.eye(300), atol=2e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
