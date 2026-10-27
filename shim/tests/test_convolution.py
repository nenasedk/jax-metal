"""GPU regressions for convolution layouts used by differentiation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16, np.complex64])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("stride,dilation", [(1, 2), (2, 1), (2, 2)])
def test_convolution_and_transpose_derivatives(dtype, layout, stride, dilation):
    x = (np.arange(48).reshape(2, 2, 3, 4) % 5 - 2).astype(dtype)
    w = (np.arange(12).reshape(3, 2, 1, 2) % 3 - 1).astype(dtype)
    if dtype == np.complex64:
        x = x + np.complex64(1j) * x[::-1]
        w = w + np.complex64(1j)
    if layout == "NHWC":
        x = x.transpose(0, 2, 3, 1)
        w = w.transpose(2, 3, 1, 0)
    kernel = "OIHW" if layout == "NCHW" else "HWIO"

    def conv(x, w):
        return lax.conv_general_dilated(
            x, w, (stride, 1), ((0, 0), (1, 1)), lhs_dilation=(dilation, 1),
            dimension_numbers=(layout, kernel, layout))

    @jax.jit
    def compute(x, w):
        value, pullback = jax.vjp(conv, x, w)
        return value, pullback(jnp.ones_like(value)), jax.jvp(
            conv, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))[1]

    with jax.default_device(jax.devices("cpu")[0]):
        expected = jax.tree.map(np.asarray, compute(x, w))
    with jax.default_device(jax.devices("METAL")[0]):
        actual = jax.tree.map(np.asarray, compute(x, w))
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
