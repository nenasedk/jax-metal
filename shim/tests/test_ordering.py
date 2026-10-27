"""Apple GPU regressions for sorting, total order, and integer payloads."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.lax import lax as lax_internal


@pytest.fixture(scope="module", autouse=True)
def metal_device():
    with jax.default_device(jax.devices("METAL")[0]):
        yield


def compare_cpu(fn, *args):
    fn = jax.jit(fn)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = jax.tree.map(np.asarray, fn(*args))
    actual = jax.tree.map(np.asarray, fn(*args))
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        # Compare bit patterns too: payloads must retain NaN payloads and zero signs.
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32])
def test_stable_sort_special_values(dtype):
    x = np.array([-np.inf, 0., -0., np.inf, np.nan, -np.nan], dtype)
    compare_cpu(lambda x: lax.sort_key_val(x, lax.iota(np.int32, x.size), is_stable=True), x)


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32])
@pytest.mark.parametrize("primitive", [lax_internal.eq_to_p, lax_internal.lt_to_p,
                                       lax_internal.le_to_p])
def test_total_order_bits(dtype, primitive):
    bits = np.dtype(dtype).itemsize * 8
    mantissa = {np.dtype(np.float16): 10, np.dtype(jnp.bfloat16): 7,
                np.dtype(np.float32): 23}[np.dtype(dtype)]
    sign = 1 << (bits - 1)
    inf = sign - (1 << mantissa)
    raw = np.array([0, 1, inf, inf + 1, inf + (1 << (mantissa - 1)),
                    sign, sign + 1, sign + inf, sign + inf + 1], f'uint{bits}')
    x = raw.view(dtype)
    compare_cpu(lambda x: primitive.bind(x[:, None], x[None, :]), x)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_gather_integer_bits(dtype):
    # Includes float NaN, infinity and subnormal bit patterns after reinterpretation.
    x = np.array([0, 1, 0x7f800000, 0x7f800001, 0x7fc00001, 0x80000000,
                  0xff800001, 0xffffffff, 0x40000001], np.uint32).view(dtype)
    indices = np.arange(x.size - 1, -2, -1, dtype=np.int32)
    compare_cpu(lambda x, i: x.at[i].get(mode='fill', fill_value=123456789), x, indices)
    compare_cpu(lambda x, i: x.at[i].get(mode='clip'), x, indices)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("axis", [0, 1])
def test_sort_and_topk_integer_precision(dtype, axis):
    x = np.random.default_rng(27).integers(0, np.iinfo(dtype).max, (5, 7), dtype=dtype)
    x[0, :5] = [2**30 + 3, 2**30 + 1, 2**30 + 2, 2**30 + 4, np.iinfo(dtype).max]
    compare_cpu(lambda x: (lax.sort(x, dimension=axis), lax.top_k(x, 3, axis=axis)), x)


@pytest.mark.parametrize("complex_key", [False, True])
def test_sort_multiple_keys_and_payload(complex_key):
    primary = np.array([1., np.nan, 0., -0., -np.nan, 1.], np.float32)
    secondary = np.array([3, 1, 5, 2, 0, 1], np.int32)
    if complex_key:
        primary = primary.astype(np.complex64) + secondary * np.complex64(1j)
    payload = np.array([2**30 + i for i in range(6)], np.int32)
    compare_cpu(lambda a, b, c: lax.sort((a, b, c), num_keys=2), primary, secondary, payload)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
