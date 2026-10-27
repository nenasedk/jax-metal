"""GPU regressions for exact integer reductions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
