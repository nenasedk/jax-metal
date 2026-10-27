"""Preserve every bfloat16 bit pattern when changing its sign."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def test_all_bfloat16_sign_patterns():
    bits = np.arange(65536, dtype=np.uint16)
    x = bits.view(jnp.bfloat16)
    with jax.default_device(jax.devices("METAL")[0]):
        absolute, negative, selected = jax.jit(lambda x: (
            lax.bitcast_convert_type(lax.abs(x), np.uint16),
            lax.bitcast_convert_type(lax.neg(x), np.uint16),
            lax.bitcast_convert_type(lax.select(
                (lax.bitcast_convert_type(x, np.uint16) & np.uint16(1)) != 0,
                x, lax.neg(x)), np.uint16)))(x)
    np.testing.assert_array_equal(absolute, bits & np.uint16(0x7fff))
    np.testing.assert_array_equal(negative, bits ^ np.uint16(0x8000))
    np.testing.assert_array_equal(selected, np.where(bits & 1, bits, bits ^ np.uint16(0x8000)))


def test_bfloat16_spacing_subnormals():
    bits = np.array([0x8005, 0x8001, 0, 1, 5], np.uint16)
    with jax.default_device(jax.devices("METAL")[0]):
        result = jax.jit(lambda x: lax.bitcast_convert_type(jnp.spacing(x), np.uint16))(
            bits.view(jnp.bfloat16))
    np.testing.assert_array_equal(result, [0x8001, 0x8001, 1, 1, 1])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
