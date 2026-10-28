"""PJRT optional APIs and client lifecycle remain safe on the older plugin."""

import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import numpy as np
import pytest


@pytest.mark.parametrize("mapped", [False, True])
def test_memory_analysis_unavailable(mapped):
    with jax.default_device(jax.devices("METAL")[0]):
        x = np.arange(3, dtype=np.float32).reshape(1, 3)
        fn = (jax.pmap if mapped else jax.jit)(lambda x: x + 1)
        compiled = fn.lower(x).compile()
        for _ in range(3):
            assert compiled.memory_analysis() is None
        np.testing.assert_array_equal(compiled(x), x + 1)
        with pytest.raises(jax.errors.JaxRuntimeError, match="UNIMPLEMENTED.*memory statistics"):
            compiled.runtime_executable().get_compiled_memory_stats()


def test_repeated_backend_resets():
    from jax._src import api

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    z = (x + 2j).astype(np.complex64)
    empty = np.empty((2, 0), np.int32)
    for _ in range(5):
        with jax.default_device(jax.devices("METAL")[0]):
            fn = jax.jit(lambda z, empty: (z * 2, empty))
            result, result_empty = fn(z, empty)
            np.testing.assert_array_equal(result, z * 2)
            assert result_empty.shape == empty.shape
            mapped = jax.pmap(lambda x: x + 1)(x[None])
            np.testing.assert_array_equal(mapped, x[None] + 1)
            del result, result_empty, mapped, fn
        api.clear_backends()
        gc.collect()


def test_tokens_cross_program_boundaries():
    # The plugin rejects !mhlo.token in a signature; tokens are unread uint8 placeholders
    # that the shim reports as PJRT token buffers.
    from jax._src import core

    with jax.default_device(jax.devices("METAL")[0]):
        token = jax.lax.create_token()
        assert isinstance(token, core.Token)
        with pytest.raises(RuntimeError, match="Cannot convert a token-shape buffer"):
            token._buf._value
        x = np.arange(4, dtype=np.float32)
        doubled, out = jax.jit(lambda x, t: (x * 2, t))(x, token)  # token in and out
        np.testing.assert_array_equal(doubled, x * 2)
        assert isinstance(out, core.Token) and out is not token
        chained = jax.jit(lambda t, x: (t, x + 1))(out, x)  # an output token as an input
        np.testing.assert_array_equal(chained[1], x + 1)
        assert isinstance(jax.jit(jax.lax.create_token)(1.0), core.Token)
        _, captured = jax.jit(lambda x: (x + 2, token))(7)
        assert isinstance(captured, core.Token)
        captured.block_until_ready()
        mixed = jax.jit(lambda z, t: (z * 1j, t))((x + 1j).astype(np.complex64), token)
        np.testing.assert_array_equal(mixed[0], (x + 1j).astype(np.complex64) * 1j)
