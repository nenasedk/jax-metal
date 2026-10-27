"""GPU regressions for indexing failures found in JAX's test suite.

Run through tests/guard.py. This file imports the shim from the checkout.
"""

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


@pytest.mark.parametrize("operation", ["set", "min", "max"])
@pytest.mark.parametrize("consumer", ["cast", "sum", "not", "nonzero", "pad"])
def test_boolean_scatter_consumers(operation, consumer):
    operand = np.array([True, False, True, False, True])
    indices = np.array([0, 1, 1, 3], np.int32)
    updates = np.array([False, False, True, True])
    # Avoid duplicate replacement writes, whose winner is not specified.
    if operation == "set":
        indices, updates = indices[[0, 2, 3]], updates[[0, 2, 3]]
    expected = operand.copy()
    combine = {"min": np.logical_and, "max": np.logical_or}
    if operation == "set":
        expected[indices] = updates
    else:
        combine[operation].at(expected, indices, updates)
    consume = {
        "cast": lambda x: x.astype(np.int32),
        "sum": lambda x: jnp.sum(x),
        "not": lambda x: ~x,
        "nonzero": lambda x: jnp.nonzero(x, size=5, fill_value=-1)[0],
        "pad": lambda x: jnp.pad(x, (1, 2)),
    }[consumer]

    def compute(x, i, u):
        return consume(getattr(x.at[i], operation)(u))

    # Compute the reference with ordinary CPU lowerings, including nonzero padding.
    with jax.default_device(jax.devices("cpu")[0]):
        reference = np.asarray(consume(jnp.asarray(expected)))
    result = jax.jit(compute)(operand, indices, updates)
    np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("method", ["scatter", "dynamic_update_slice"])
def test_slice_update_preserves_shared_zero(method):
    matrix = np.array([[1, 2, 3, 0, 0], [0, 4, 5, 6, 0], [0, 7, 8, 0, 0],
                       [9, 0, 0, 0, 0], [0, 10, 0, 11, 12], [13, 0, 0, 14, 0],
                       [0, 15, 0, 0, 0], [16, 17, 18, 0, 19]], np.float32)

    def compute(x):
        rows = jnp.nonzero(x, size=19)[0]
        counts = jnp.cumsum(jnp.bincount(rows, length=8))
        zero = jnp.zeros(9, np.int32)
        updated = (zero.at[1:].set(counts) if method == "scatter" else
                   lax.dynamic_update_slice(zero, counts, (1,)))
        return updated, zero

    updated, original = jax.jit(compute)(matrix)
    np.testing.assert_array_equal(updated, [0, 3, 6, 8, 9, 12, 14, 15, 19])
    np.testing.assert_array_equal(original, np.zeros(9, np.int32))


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.bool_, np.complex64])
@pytest.mark.parametrize("fill", [0, 1])
def test_interior_pad_preserves_shared_fill(dtype, fill):
    x = np.array([2, 0, 3], dtype)

    def compute(x, value):
        return lax.pad(x, value, [(0, 0, 1)]), jnp.full((5,), value, dtype)

    padded, untouched = jax.jit(compute)(x, np.array(fill, dtype))
    expected = np.full(5, fill, dtype)
    expected[::2] = x
    np.testing.assert_array_equal(padded, expected)
    np.testing.assert_array_equal(untouched, np.full(5, fill, dtype))


@pytest.mark.parametrize("shape,config", [
    ((2, 3), ((1, 2, 1), (0, 1, 2))),
    ((2, 3), ((-1, 0, 1), (1, -2, 2))),
    ((0, 3), ((1, 2, 1), (0, 1, 2))),
    ((1, 3), ((0, 0, 3), (1, 1, 1))),
])
def test_interior_pad_axes_and_edges(shape, config):
    x = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    fn = jax.jit(lambda x, value: lax.pad(x, value, config))
    with jax.default_device(jax.devices("cpu")[0]):
        reference = np.asarray(fn(x, np.int32(7)))
    np.testing.assert_array_equal(fn(x, np.int32(7)), reference)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int32, np.uint32])
@pytest.mark.parametrize("update", [False, True])
def test_dynamic_slice_clamps_before_index_conversion(dtype, update):
    x = np.arange(10, dtype=np.int32)
    fn = jax.jit(lambda x, i: (
        lax.dynamic_update_slice(x, jnp.array([91, 92, 93, 94]), (i,),
                                 allow_negative_indices=False) if update else
        lax.dynamic_slice(x, (i,), (4,), allow_negative_indices=False)))
    starts = [0, 6, 9, np.iinfo(dtype).max]
    if np.issubdtype(dtype, np.signedinteger):
        starts.append(np.iinfo(dtype).min)
    for start in starts:
        clipped = min(max(start, 0), 6)
        expected = x.copy() if update else x[clipped:clipped + 4]
        if update:
            expected[clipped:clipped + 4] = [91, 92, 93, 94]
        np.testing.assert_array_equal(fn(x, dtype(start)), expected)


@pytest.mark.parametrize("mode", ["clip", "fill"])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_gather_bounds_and_fill(mode, dtype):
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    indices = np.array([[[0, 0], [3, 4]], [[4, 0], [1, np.iinfo(dtype).max]]], dtype)
    if np.issubdtype(dtype, np.signedinteger):
        indices[0, 0] = [-1, -2]
    dnums = lax.GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0, 1),
                                       start_index_map=(0, 1))
    fn = jax.jit(lambda x, i: lax.gather(x, i, dnums, (1, 1), mode=mode,
                                        fill_value=-123))
    with jax.default_device(jax.devices("cpu")[0]):
        reference = np.asarray(fn(x, indices))
    np.testing.assert_array_equal(fn(x, indices), reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
