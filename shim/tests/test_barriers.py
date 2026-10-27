"""Variadic barriers preserve values, dependencies and checkpoint derivatives."""

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
    with jax.default_device(jax.devices("METAL")[0]), jax.enable_x64():
        yield


def assert_bits(actual, expected):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        got, want = np.asarray(got), np.asarray(want)
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got.tobytes(), want.tobytes())


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16,
                                  np.int32, np.uint32, np.int64, np.uint64,
                                  np.float16, jnp.bfloat16, np.float32])
def test_mixed_barrier_bits(dtype):
    # Exhaust every 8-/16-bit pattern, and include large integers, floating NaN
    # payloads, infinities and signed zeros in the wider types.
    width = np.dtype(dtype).itemsize * 8
    if width <= 16:
        raw = np.arange(2**width, dtype=f"uint{width}")
    else:
        raw = np.array([0, 1, 2**width - 1, 2**(width - 1), 2**24 + 1,
                        0x7fc01234, 0x7f800000, 0x80000001], f"uint{width}")
    args = (raw.view(dtype), np.array([[True, False], [False, True]]),
            np.array(-7, np.int16))
    assert_bits(jax.jit(lax.optimization_barrier)(args), args)


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3), ()])
def test_empty_and_scalar(shape):
    args = (np.zeros(shape, np.float32), np.zeros(shape, np.int32))
    assert_bits(jax.jit(lax.optimization_barrier)(args), args)
    assert jax.jit(lambda: lax.optimization_barrier(()))() == ()


def test_mixed_complex_boundary():
    args = (np.array([1 + 2j, -3 + 4j], np.complex64),
            np.array([16777217, 2147483647], np.int32),
            np.array([0x8000, 1, 0x7fc1], np.uint16).view(jnp.bfloat16))
    assert_bits(jax.jit(lax.optimization_barrier)(args), args)


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.int4, jnp.uint4])
def test_narrow_boundary(dtype):
    args = (np.array([1, 2, 3], dtype), np.array([5, -2], np.int32))
    assert_bits(jax.jit(lax.optimization_barrier)(args), args)


def test_checkpoint_scan_derivatives():
    def loss(x, weights):
        @jax.checkpoint
        def body(carry, w):
            a, b = lax.optimization_barrier((carry, w))
            value = jnp.sin(a * b)
            return value, value * value
        carry, values = lax.scan(body, x, weights)
        return carry.sum() + values.sum()

    x = np.array([0.2, 0.5], np.float32)
    weights = np.arange(8, dtype=np.float32).reshape(4, 2) / 10
    fn = jax.jit(jax.value_and_grad(loss, (0, 1)))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = jax.tree.map(np.asarray, fn(x, weights))
    actual = fn(x, weights)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-7)


def test_compilation_preserves_public_lowering():
    from jax_plugins.metal_shim import _compiler_ir

    args = (np.arange(3, dtype=np.float32), np.arange(2, dtype=np.int32))
    lowered = jax.jit(lambda *xs: lax.optimization_barrier(xs)).lower(*args)
    module = lowered.compiler_ir()
    before = str(module)
    prepared = _compiler_ir.prepare_module(module)
    assert prepared is not module
    assert str(module) == before
    assert prepared.operation.verify()
    assert not list(_compiler_ir._barriers(prepared.operation))
    # There is one shared barrier, not separate per-input barriers or a no-op.
    assert str(prepared).count("stablehlo.optimization_barrier") == 1
    assert_bits(lowered.compile()(*args), args)
    assert str(module) == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
