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
