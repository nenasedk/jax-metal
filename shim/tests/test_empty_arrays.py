"""Empty values retain their shapes without reaching native MPS input shapes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import sparse


@pytest.fixture(scope="module", autouse=True)
def metal_device():
    with jax.default_device(jax.devices("METAL")[0]), jax.enable_x64():
        yield


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.float32, np.complex64,
                                  np.uint64, jnp.bfloat16, jnp.int4])
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
@pytest.mark.parametrize("donate", [False, True])
def test_mixed_empty_boundary(dtype, shape, donate):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    empty = np.empty(shape, dtype)

    def fn(x, empty):
        return x + 1, lax.concatenate([empty, empty], 0)

    compiled = jax.jit(fn, donate_argnums=(1,) if donate else ()).lower(x, empty).compile()
    got, got_empty = compiled(jax.device_put(x), jax.device_put(empty))
    np.testing.assert_array_equal(got, x + 1)
    expected = np.concatenate([empty, empty], 0)
    assert got_empty.shape == expected.shape
    assert got_empty.dtype == expected.dtype
    np.testing.assert_array_equal(got_empty, expected)


@pytest.mark.parametrize("scatter", [lax.scatter, lax.scatter_add, lax.scatter_mul,
                                    lax.scatter_min, lax.scatter_max])
@pytest.mark.parametrize("empty_operand", [False, True])
def test_empty_scatter(scatter, empty_operand):
    if empty_operand:
        x = np.empty((0, 3), np.float32)
        indices = np.zeros((2, 1), np.int32)
        updates = np.empty((2, 0, 3), np.float32)
        dims = lax.ScatterDimensionNumbers((1, 2), (), (0,))
    else:
        x = np.arange(9, dtype=np.float32).reshape(3, 3)
        indices = np.empty((0, 1), np.int32)
        updates = np.empty((0, 3), np.float32)
        dims = lax.ScatterDimensionNumbers((1,), (0,), (0,))
    result = jax.jit(lambda x, i, u: scatter(x, i, u, dims))(x, indices, updates)
    np.testing.assert_array_equal(result, x)


@pytest.mark.parametrize("operation", ["concatenate", "reverse", "slice"])
def test_sparse_without_index_components(operation):
    x = np.arange(60, dtype=np.float32).reshape(3, 5, 4)
    matrix = sparse.BCOO.fromdense(x, n_batch=2, n_dense=1)
    assert matrix.indices.shape[-1] == 0
    if operation == "concatenate":
        fn = lambda a: sparse.bcoo_concatenate([a, a], dimension=0)
        expected = np.concatenate([x, x], 0)
    elif operation == "reverse":
        fn = lambda a: sparse.bcoo_rev(a, dimensions=(0, 1))
        expected = x[::-1, ::-1]
    else:
        fn = lambda a: sparse.bcoo_dynamic_slice(a, (1, 1, 0), (2, 0, 4))
        expected = x[1:3, 1:1, :]
    result = jax.jit(fn)(matrix)
    np.testing.assert_array_equal(result.todense(), expected)


def test_empty_input_gradient():
    def loss(x, empty):
        return (x * x).sum() + (empty * empty).sum()
    x, empty = np.arange(3, dtype=np.float32), np.empty((2, 0), np.float32)
    dx, de = jax.jit(jax.grad(loss, (0, 1)))(x, empty)
    np.testing.assert_array_equal(dx, 2 * x)
    assert de.shape == empty.shape
    assert de.dtype == empty.dtype
