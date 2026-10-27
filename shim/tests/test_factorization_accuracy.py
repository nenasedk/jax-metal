"""GPU regression tests for cancellation-sensitive float32 factorizations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest
from jax import lax


@pytest.fixture(scope="module", autouse=True)
def metal_device():
    with jax.default_device(jax.devices("METAL")[0]):
        yield


def test_ill_conditioned_cholesky():
    a = np.array([[1.6183797, -0.67463326], [-0.67463326, 0.2813133]], np.float32)
    b = np.array([0.14424941, -0.3589493], np.float32)
    factor = jax.jit(jnp.linalg.cholesky)(a)
    expected = np.linalg.cholesky(a.astype(np.float64))
    np.testing.assert_allclose(factor, expected, rtol=2e-7, atol=1e-9)
    solution = jsp.linalg.cho_solve((factor, True), b)
    np.testing.assert_allclose(solution, np.linalg.solve(a.astype(np.float64), b), rtol=2e-6)


@pytest.mark.parametrize("shape", [(2, 2), (3, 5), (5, 3), (2, 4, 4), (0, 3), (3, 0)])
def test_lu_reconstruction(shape):
    a = np.random.default_rng(53).normal(size=shape).astype(np.float32)
    packed, pivots, permutation = jax.tree.map(np.asarray, jax.jit(lax.linalg.lu)(a))
    m, n = shape[-2:]
    rank = min(m, n)
    lower = np.tril(packed[..., :, :rank], -1) + np.eye(m, rank, dtype=np.float32)
    upper = np.triu(packed[..., :rank, :])
    permuted = np.take_along_axis(a, permutation[..., :, None], axis=-2)
    np.testing.assert_allclose(lower @ upper, permuted, rtol=2e-6, atol=2e-7)
    assert pivots.shape == (*shape[:-2], rank)


@pytest.mark.parametrize("shape", [(0, 0), (3, 3), (2, 5, 5), (129, 129)])
def test_cholesky_reconstruction(shape):
    base = np.random.default_rng(23).normal(size=shape).astype(np.float32)
    a = base @ np.swapaxes(base, -1, -2) + np.eye(shape[-1], dtype=np.float32)
    factor = np.asarray(jax.jit(jnp.linalg.cholesky)(a))
    np.testing.assert_allclose(factor @ np.swapaxes(factor, -1, -2), a, rtol=2e-5, atol=3e-5)


def test_non_positive_definite_cholesky():
    for a in [np.zeros((3, 3), np.float32), np.diag(np.array([1., -1., 1.], np.float32))]:
        actual = np.asarray(jax.jit(jnp.linalg.cholesky)(a))
        assert np.isnan(actual[np.tril_indices(3)]).all()
        np.testing.assert_array_equal(actual[np.triu_indices(3, 1)], 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
