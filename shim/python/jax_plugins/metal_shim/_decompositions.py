"""METAL rules for the decompositions JAX lowers only to LAPACK or cuSOLVER.

- geqp3: QR with column pivoting (jax.scipy.linalg.qr(pivoting=True));
- hessenberg: Hessenberg reduction, LAPACK gehrd (jax.scipy.linalg.hessenberg);
- tridiagonal: Hermitian tridiagonal reduction, LAPACK sytrd/hetrd;
- schur: Schur decomposition, LAPACK gees (jax.scipy.linalg.schur, sqrtm, funm);
- eig: nonsymmetric eigendecomposition, LAPACK geev (jnp.linalg.eig, eigvals, roots);
- tridiagonal_solve: with partial pivoting (LAPACK gtsv) and, for inverse iteration
  (jax.scipy.linalg.eigh_tridiagonal), perturbed singular pivots. JAX's generic lowering
  neither pivots nor perturbs.

Each follows LAPACK's conventions and output formats (packed Householder reflectors with
LAPACK's sign choice, 1-based pivots, standardized 2 x 2 blocks of the real Schur form,
eigenvectors of unit norm with their largest component real), so the JAX functions built on
these primitives work unchanged.

Schur is the Francis double-shift QR iteration on the Hessenberg form, updating whole rows
and columns as LAPACK does when the full Schur form is wanted (dlahqr): one while loop step
either deflates a 1 x 1 block, deflates and standardizes a 2 x 2 block (LAPACK's dlanv2 for
real matrices; complex 2 x 2 blocks are triangularized), or performs one double-shift sweep
of 3 x 3 Householder reflectors. eig converts the Schur form to complex triangular form and
back-substitutes for the eigenvectors (LAPACK's trevc).

None of the loops contains lax.cond, which crashes MPSGraph inside a while loop; every
branch is a select. Batches go through vmap, which also keeps per-matrix loop state.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.interpreters import mlir
from jax._src.lax import linalg

from jax_plugins.metal_shim import _lowerings

HIGHEST = lax.Precision.HIGHEST


def _matmul(a, b):
    return jnp.matmul(a, b, precision=HIGHEST)


def _is_complex(x):
    return jnp.issubdtype(x.dtype, jnp.complexfloating)


def _conj(x):
    return jnp.conj(x) if _is_complex(x) else x


def _real_dtype(dtype):
    return jnp.finfo(dtype).dtype


def _over_batch(fn, args, core_ndims):
    """fn on one problem, vmapped over the flattened leading batch dimensions of args.

    Always batched, even for one matrix: vmap keeps per-matrix loop state in the while
    loops, and a batch of one costs nothing.
    """
    batch = args[0].shape[:args[0].ndim - core_ndims[0]]
    size = int(np.prod(batch)) if batch else 1
    flat = [x.reshape((size,) + x.shape[x.ndim - nd:]) for x, nd in zip(args, core_ndims)]
    outs = jax.vmap(fn)(*flat)
    return [o.reshape(batch + o.shape[1:]) for o in outs]


# --- Hessenberg reduction (LAPACK gehrd) ----------------------------------------------

def _hessenberg_one(a):
    """(packed, taus): H on and above the first subdiagonal, reflector j below it in column
    j (v[j + 1] = 1 implicit), taus of length n - 1. A = Q H Q^H, Q = H_0 ... H_{n-2}."""
    n = a.shape[0]
    taus = jnp.zeros((max(n - 1, 0),), a.dtype)
    if n <= 1:
        return a, taus
    index = jnp.arange(n)

    def step(j, carry):
        a, taus = carry
        col = lax.dynamic_index_in_dim(a, j, axis=1, keepdims=False)
        v, tau, beta = _lowerings._householder(col, j + 1)  # pylint: disable=protected-access
        a = a - tau * jnp.outer(_matmul(a, v), _conj(v))  # A H, columns > j
        update = _conj(tau) * jnp.outer(v, _matmul(_conj(v), a))  # H^H A
        a = jnp.where(index[None, :] > j, a - update, a)
        col = jnp.where(index <= j, col, jnp.where(index == j + 1, beta, v))
        a = lax.dynamic_update_index_in_dim(a, col[:, None], j, axis=1)
        taus = lax.dynamic_update_index_in_dim(taus, tau[None], j, axis=0)
        return a, taus

    return lax.fori_loop(0, n - 1, step, (a, taus))


def _hessenberg_q(packed, taus):
    """Q = H_0 ... H_{n-2} of _hessenberg_one, as jax.scipy.linalg.hessenberg builds it."""
    n = packed.shape[0]
    q = jnp.eye(n, dtype=packed.dtype)
    if n <= 1:
        return q
    inner = _lowerings._householder_product(packed[1:, :-1], taus)  # pylint: disable=protected-access
    return q.at[1:, 1:].set(inner)


def _hessenberg_rule(a, **_):
    return _over_batch(_hessenberg_one, [a], [2])


# --- QR with column pivoting (LAPACK geqp3) -------------------------------------------

def _geqp3_one(a, jpvt):
    """(packed, jpvt, taus) as LAPACK geqp3: at step j, the remaining column of largest norm
    (the first, on ties) becomes column j; columns with jpvt != 0 come first, unpivoted.
    jpvt out is 1-based: column jpvt[k] - 1 of A is column k of A P."""
    m, n = a.shape
    k = min(m, n)
    fixed = jpvt != 0
    order = jnp.argsort(jnp.logical_not(fixed), stable=True).astype(np.int32)
    a = a[:, order]
    taus = jnp.zeros((k,), a.dtype)
    rows, cols = jnp.arange(m), jnp.arange(n, dtype=np.int32)
    leading = jnp.sum(fixed)

    def step(j, carry):
        a, perm, taus = carry
        magnitude = jnp.abs(a) ** 2
        norms = jnp.sum(jnp.where(rows[:, None] >= j, magnitude, 0), axis=0)
        p = jnp.argmax(jnp.where(cols >= j, norms, -1)).astype(np.int32)
        p = jnp.where(j < leading, j, p)
        swap = cols.at[j].set(p).at[p].set(j)
        a, perm = a[:, swap], perm[swap]
        a, taus = _lowerings._qr_column(a, taus, j)  # pylint: disable=protected-access
        return a, perm, taus

    a, perm, taus = lax.fori_loop(0, k, step, (a, order, taus))
    return a, (perm + 1).astype(jpvt.dtype), taus


def _geqp3_rule(a, jpvt, **_):
    return _over_batch(_geqp3_one, [a, jpvt], [2, 1])


# --- Tridiagonal reduction (LAPACK sytrd/hetrd) ---------------------------------------

def _tridiagonal_lower_one(a):
    """(packed, d, e, taus) for the Hermitian matrix in a's lower triangle: d and e on the
    diagonal and first subdiagonal, reflector j below it in column j, the strict upper
    triangle of a unchanged. Q^H A Q = T with Q = H_0 ... H_{n-2}."""
    n = a.shape[0]
    index = jnp.arange(n)
    below = index[:, None] > index[None, :]
    h = jnp.where(below, a, 0) + _conj(jnp.where(below, a, 0)).T
    h = h + jnp.diag(jnp.real(jnp.diagonal(a)).astype(a.dtype))
    vectors = jnp.zeros_like(a)
    taus = jnp.zeros((n - 1,), a.dtype)

    def step(j, carry):
        h, vectors, taus = carry
        col = lax.dynamic_index_in_dim(h, j, axis=1, keepdims=False)
        v, tau, beta = _lowerings._householder(col, j + 1)  # pylint: disable=protected-access
        h = h - tau * jnp.outer(_matmul(h, v), _conj(v))  # H^H A H
        h = h - _conj(tau) * jnp.outer(v, _matmul(_conj(v), h))
        reduced = jnp.where(index == j + 1, beta, jnp.where(index > j + 1, 0, col))
        h = lax.dynamic_update_index_in_dim(h, reduced[:, None], j, axis=1)
        h = lax.dynamic_update_index_in_dim(h, _conj(reduced)[None, :], j, axis=0)
        vectors = lax.dynamic_update_index_in_dim(vectors, v[:, None], j, axis=1)
        taus = lax.dynamic_update_index_in_dim(taus, tau[None], j, axis=0)
        return h, vectors, taus

    if n > 1:  # fori_loop traces its body even for no trips
        h, vectors, taus = lax.fori_loop(0, n - 1, step, (h, vectors, taus))
    d = jnp.real(jnp.diagonal(h))
    e = jnp.real(jnp.diagonal(h, -1))
    rows, cols = index[:, None], index[None, :]
    packed = jnp.where(rows > cols + 1, vectors,
                       jnp.where(rows == cols + 1, jnp.diag(e.astype(a.dtype), -1),
                                 jnp.where(rows == cols, jnp.diag(d.astype(a.dtype)), a)))
    return packed, d, e, taus


def _tridiagonal_upper_one(a):
    # LAPACK's upper reduction runs from the last column back; on the index-reversed matrix
    # it is the lower reduction, with the same reflectors (reversed) and taus (reversed).
    packed, d, e, taus = _tridiagonal_lower_one(a[::-1, ::-1])
    return packed[::-1, ::-1], d[::-1], e[::-1], taus[::-1]


def _tridiagonal_rule(a, *, lower, **_):
    one = _tridiagonal_lower_one if lower else _tridiagonal_upper_one
    return _over_batch(one, [a], [2])


# --- Schur decomposition (LAPACK gees / hseqr) ----------------------------------------

def _sign(x):
    return jnp.where(x >= 0, jnp.ones_like(x), -jnp.ones_like(x))  # Fortran SIGN(1, x)


def _dlanv2(a, b, c, d):
    """LAPACK dlanv2: the standardized form (a, b, c, d) of a real 2 x 2 block and the
    rotation (cs, sn) giving it: upper triangular if its eigenvalues are real, else
    a == d and b * c < 0."""
    eps = jnp.finfo(a.dtype).eps
    one, zero = jnp.ones_like(a), jnp.zeros_like(a)
    temp = a - d
    p = 0.5 * temp
    bcmax = jnp.maximum(jnp.abs(b), jnp.abs(c))
    bcmis = jnp.minimum(jnp.abs(b), jnp.abs(c)) * _sign(b) * _sign(c)
    scale = jnp.maximum(jnp.abs(p), bcmax)
    safe_scale = jnp.where(scale == 0, one, scale)
    z = (p / safe_scale) * p + (bcmax / safe_scale) * bcmis
    # Real eigenvalues: triangularize.
    zr = p + _sign(p) * jnp.sqrt(scale) * jnp.sqrt(jnp.maximum(z, 0))
    safe_zr = jnp.where(zr == 0, one, zr)
    tau_r = jnp.hypot(c, zr)
    safe_tau_r = jnp.where(tau_r == 0, one, tau_r)
    real_case = (d + zr, b - c, zero, d - (bcmax / safe_zr) * bcmis, zr / safe_tau_r,
                 c / safe_tau_r)
    # Complex or nearly equal real eigenvalues: make the diagonal equal.
    sigma = b + c
    tau_c = jnp.hypot(sigma, temp)
    safe_tau_c = jnp.where(tau_c == 0, one, tau_c)
    cs = jnp.sqrt(0.5 * (1 + jnp.abs(sigma) / safe_tau_c))
    sn = -(p / (safe_tau_c * cs)) * _sign(sigma)
    aa, bb = a * cs + b * sn, -a * sn + b * cs
    cc, dd = c * cs + d * sn, -c * sn + d * cs
    b2, c2 = bb * cs + dd * sn, -aa * sn + cc * cs
    mid = 0.5 * ((aa * cs + cc * sn) + (-bb * sn + dd * cs))
    sab, sac = jnp.sqrt(jnp.abs(b2)), jnp.sqrt(jnp.abs(c2))
    p3 = _sign(c2) * sab * sac
    tau3 = 1 / jnp.sqrt(jnp.where(b2 + c2 == 0, one, jnp.abs(b2 + c2)))
    cs1, sn1 = sab * tau3, sac * tau3
    complex_real = (mid + p3, b2 - c2, zero, mid - p3, cs * cs1 - sn * sn1, cs * sn1 + sn * cs1)
    complex_swap = (mid, -c2, zero, mid, -sn, cs)
    complex_case = (mid, b2, c2, mid, cs, sn)
    pick = lambda flag, x, y: tuple(jnp.where(flag, u, w) for u, w in zip(x, y))
    complex_case = pick((c2 != 0) & (b2 != 0) & (_sign(b2) == _sign(c2)), complex_real,
                        pick((c2 != 0) & (b2 == 0), complex_swap, complex_case))
    general = pick(z >= 4 * eps, real_case, complex_case)
    unchanged = (a, b, c, d, one, zero)
    result = pick(c == 0, unchanged,
                  pick(b == 0, (d, -c, zero, a, zero, one),
                       pick((temp == 0) & (_sign(b) != _sign(c)), unchanged, general)))
    return result


def _apply_rotation(h, z, i, g):
    """H := G^H H G on rows/columns i, i + 1 (whole rows and columns), Z := Z G."""
    gh = _conj(g).T
    rows = lax.dynamic_slice_in_dim(h, i, 2, axis=0)
    h = lax.dynamic_update_slice_in_dim(h, _matmul(gh, rows), i, axis=0)
    cols = lax.dynamic_slice_in_dim(h, i, 2, axis=1)
    h = lax.dynamic_update_slice_in_dim(h, _matmul(cols, g), i, axis=1)
    zcols = lax.dynamic_slice_in_dim(z, i, 2, axis=1)
    z = lax.dynamic_update_slice_in_dim(z, _matmul(zcols, g), i, axis=1)
    return h, z


def _set(h, i, j, value):
    return lax.dynamic_update_slice(h, jnp.reshape(value, (1, 1)).astype(h.dtype), (i, j))


def _get(h, i, j):
    return lax.dynamic_slice(h, (i, j), (1, 1))[0, 0]


def _triangularize_2x2(h, z, i, lam):
    """Make the 2 x 2 block at (i, i) upper triangular with lam (one of its eigenvalues)
    first, by the unitary whose first column is the eigenvector (lam - d, c)."""
    d, c = _get(h, i + 1, i + 1), _get(h, i + 1, i)
    x1, x2 = lam - d, c
    norm = jnp.sqrt(jnp.abs(x1) ** 2 + jnp.abs(x2) ** 2)
    trivial = (c == 0) | (norm == 0)
    safe = jnp.where(trivial, 1, norm)
    x1 = jnp.where(trivial, 1, x1 / safe)
    x2 = jnp.where(trivial, 0, x2 / safe)
    g = jnp.stack([jnp.stack([x1, -jnp.conj(x2)]), jnp.stack([x2, jnp.conj(x1)])])
    h, z = _apply_rotation(h, z, i, g)
    return _set(h, i + 1, i, 0), z


def _eigenvalues_2x2(a, b, c, d):
    """Both eigenvalues of [[a, b], [c, d]] (complex), the one farther from d first."""
    tr = (a + d) / 2
    disc = jnp.sqrt(((a - d) / 2) ** 2 + b * c)
    l1, l2 = tr + disc, tr - disc
    swap = jnp.abs(l2 - d) > jnp.abs(l1 - d)
    return jnp.where(swap, l2, l1), jnp.where(swap, l1, l2)


def _standardize_2x2(h, z, i):
    a, b, c, d = (_get(h, i, i), _get(h, i, i + 1), _get(h, i + 1, i), _get(h, i + 1, i + 1))
    if _is_complex(h):
        lam, _ = _eigenvalues_2x2(a, b, c, d)
        return _triangularize_2x2(h, z, i, lam)
    aa, bb, cc, dd, cs, sn = _dlanv2(a, b, c, d)
    g = jnp.stack([jnp.stack([cs, -sn]), jnp.stack([sn, cs])])
    h, z = _apply_rotation(h, z, i, g)
    block = jnp.stack([jnp.stack([aa, bb]), jnp.stack([cc, dd])])
    return lax.dynamic_update_slice(h, block, (i, i)), z


def _double_shift_sweep(h, z, lo, hi, its):
    """One Francis double-shift QR sweep over the active block [lo, hi] (hi - lo >= 2) of the
    padded Hessenberg matrix h, with exceptional shifts after 10 and 20 sweeps (dlahqr)."""
    n = z.shape[0]
    hii, hi1i1 = _get(h, hi, hi), _get(h, hi - 1, hi - 1)
    trace = hi1i1 + hii
    det = hi1i1 * hii - _get(h, hi - 1, hi) * _get(h, hi, hi - 1)
    s_lo = jnp.abs(_get(h, lo + 1, lo)) + jnp.abs(_get(h, lo + 2, lo + 1))
    s_hi = jnp.abs(_get(h, hi, hi - 1)) + jnp.abs(_get(h, hi - 1, hi - 2))
    for count, s, base in ((10, s_lo, _get(h, lo, lo)), (20, s_hi, hii)):
        h11 = 0.75 * s + base
        trace = jnp.where(its == count, 2 * h11, trace)
        det = jnp.where(its == count, h11 * h11 + 0.4375 * s * s, det)
    h00, h01 = _get(h, lo, lo), _get(h, lo, lo + 1)
    h10, h11, h21 = _get(h, lo + 1, lo), _get(h, lo + 1, lo + 1), _get(h, lo + 2, lo + 1)
    first = jnp.stack([h00 * h00 + h01 * h10 - trace * h00 + det,
                       h10 * (h00 + h11 - trace), h10 * h21])
    scale = jnp.sum(jnp.abs(first))
    first = first / jnp.where(scale == 0, 1, scale).astype(first.dtype)

    def chase(k, carry):
        h, z = carry
        active = (k >= lo) & (k <= hi - 1)
        at_start = k == lo
        column = lax.dynamic_slice(h, (k, jnp.maximum(k - 1, 0)), (3, 1))[:, 0]
        x = jnp.where(at_start, first, column)
        x = x.at[2].set(jnp.where(k == hi - 1, 0, x[2]))  # the last reflector has two rows
        v, tau, beta = _lowerings._householder(x, 0)  # pylint: disable=protected-access
        rows = lax.dynamic_slice_in_dim(h, k, 3, axis=0)
        rows = rows - _conj(tau) * jnp.outer(v, _matmul(_conj(v), rows))
        new = lax.dynamic_update_slice_in_dim(h, rows, k, axis=0)
        cols = lax.dynamic_slice_in_dim(new, k, 3, axis=1)
        cols = cols - tau * jnp.outer(_matmul(cols, v), _conj(v))
        new = lax.dynamic_update_slice_in_dim(new, cols, k, axis=1)
        bulge = jnp.stack([beta, jnp.zeros_like(beta), jnp.zeros_like(beta)])
        fixed = lax.dynamic_update_slice(new, bulge[:, None], (k, jnp.maximum(k - 1, 0)))
        new = jnp.where(at_start, new, fixed)
        zcols = lax.dynamic_slice_in_dim(z, k, 3, axis=1)
        zcols = zcols - tau * jnp.outer(_matmul(zcols, v), _conj(v))
        new_z = lax.dynamic_update_slice_in_dim(z, zcols, k, axis=1)
        return jnp.where(active, new, h), jnp.where(active, new_z, z)

    return lax.fori_loop(0, max(n - 1, 0), chase, (h, z))


def _abs1(x):
    """|re| + |im|, LAPACK's cabs1 (|x| for real x)."""
    return jnp.abs(jnp.real(x)) + jnp.abs(jnp.imag(x)) if _is_complex(x) else jnp.abs(x)


def _negligible(h, index, small_number, ulp):
    """Which subdiagonal entries h[k, k - 1] (k in index) can be set to zero: LAPACK's test in
    dlahqr/zlahqr, a cheap bound refined by Ahues and Tisseur's, which keeps small
    eigenvalues accurate."""
    n = h.shape[0] - 2
    sub, sup = h[index, index - 1], h[index - 1, index]
    d0, d1 = h[index - 1, index - 1], h[index, index]
    test = _abs1(d0) + _abs1(d1)
    before = jnp.where(index >= 2, _abs1(h[index - 1, jnp.maximum(index - 2, 0)]), 0)
    after = jnp.where(index + 1 <= n - 1, _abs1(h[jnp.minimum(index + 1, n - 1), index]), 0)
    test = jnp.where(test == 0, before + after, test)
    ab = jnp.maximum(_abs1(sub), _abs1(sup))
    ba = jnp.minimum(_abs1(sub), _abs1(sup))
    aa = jnp.maximum(_abs1(d1), _abs1(d0 - d1))
    bb = jnp.minimum(_abs1(d1), _abs1(d0 - d1))
    s = aa + ab
    safe_s = jnp.where(s == 0, 1, s)
    refined = ba * (ab / safe_s) <= jnp.maximum(small_number, ulp * (bb * (aa / safe_s)))
    return (_abs1(sub) <= small_number) | ((_abs1(sub) <= ulp * test) & refined)


def _schur_one(a):
    """(T, Z) with A = Z T Z^H: T upper triangular (complex a) or in real Schur form (real a,
    standardized 2 x 2 blocks for complex conjugate eigenvalues); NaN if the iteration does
    not converge within LAPACK's limit (30 max(10, n) sweeps)."""
    n = a.shape[0]
    if n <= 1:
        return a, jnp.eye(n, dtype=a.dtype)
    real_dtype = _real_dtype(a.dtype)
    ulp = jnp.finfo(real_dtype).eps
    small_number = jnp.finfo(real_dtype).tiny * (n / ulp)
    packed, taus = _hessenberg_one(a)
    h = jnp.pad(jnp.triu(packed, -1), ((0, 2), (0, 2)))  # room for 3-row reflectors
    z = jnp.pad(_hessenberg_q(packed, taus), ((0, 0), (0, 2)))
    index = jnp.arange(1, n)
    limit = 30 * max(10, n)

    def iterate(state):
        h, z, hi, its, total = state
        sub = h[index, index - 1]
        small = _negligible(h, index, small_number, ulp)
        sub = jnp.where(small & (index <= hi), 0, sub)
        h = h.at[index, index - 1].set(sub)
        lo = jnp.max(jnp.where((index <= hi) & (sub == 0), index, 0))
        size = hi - lo + 1
        h2, z2 = _standardize_2x2(h, z, jnp.maximum(hi - 1, 0))
        h3, z3 = _double_shift_sweep(h, z, lo, jnp.maximum(hi, lo + 2), its)
        sweep = size >= 3
        h = jnp.where(size == 2, h2, jnp.where(sweep, h3, h))
        z = jnp.where(size == 2, z2, jnp.where(sweep, z3, z))
        hi = jnp.where(size == 1, hi - 1, jnp.where(size == 2, hi - 2, hi))
        return h, z, hi, jnp.where(sweep, its + 1, 0), total + 1

    h, z, hi, _, _ = lax.while_loop(lambda state: (state[2] > 0) & (state[4] < limit), iterate,
                                    (h, z, jnp.int32(n - 1), jnp.int32(0), jnp.int32(0)))
    t, z = h[:n, :n], z[:, :n]
    rows, cols = jnp.arange(n)[:, None], jnp.arange(n)[None, :]
    t = jnp.where(rows > cols + (0 if _is_complex(a) else 1), 0, t)
    failed = (hi > 0) | ~jnp.all(jnp.isfinite(a))
    nan = jnp.full_like(t, jnp.nan)
    return jnp.where(failed, nan, t), jnp.where(failed, nan, z)


def _schur_rule(a, *, compute_schur_vectors, sort_eig_vals, **_):
    if sort_eig_vals:
        raise NotImplementedError("METAL: schur with sorted eigenvalues is not implemented")
    if a.shape[-1] == 0:
        return [a, a] if compute_schur_vectors else [a]
    t, z = _over_batch(_schur_one, [a], [2])
    return [t, z] if compute_schur_vectors else [t]


# --- Nonsymmetric eigendecomposition (LAPACK geev) ------------------------------------

def _real_schur_eigenvalues(t):
    """Eigenvalues of a real Schur form: real for 1 x 1 blocks, a +- i sqrt(-bc) for the
    standardized 2 x 2 blocks [[a, b], [c, a]] (positive imaginary part first), as geev."""
    n = t.shape[0]
    diag = jnp.diagonal(t)
    sub = jnp.concatenate([jnp.diagonal(t, -1), jnp.zeros((1,), t.dtype)])
    sup = jnp.concatenate([jnp.diagonal(t, 1), jnp.zeros((1,), t.dtype)])
    imag = jnp.sqrt(jnp.abs(sub)) * jnp.sqrt(jnp.abs(sup))
    starts = sub != 0
    seconds = jnp.concatenate([jnp.zeros((1,), bool), starts[:-1]])
    previous = jnp.concatenate([jnp.zeros((1,), t.dtype), imag[:-1]])
    imag = jnp.where(starts, imag, jnp.where(seconds, -previous, 0))
    del n
    return lax.complex(diag, imag), starts, seconds


def _normalized(v):
    """Columns of unit 2-norm with their largest component real and positive (geev)."""
    v = v / jnp.linalg.norm(v, axis=0, keepdims=True).astype(v.dtype)
    largest = jnp.argmax(jnp.abs(v), axis=0)
    pivot = jnp.take_along_axis(v, largest[None, :], axis=0)
    return v * (jnp.conj(pivot) / jnp.abs(pivot).astype(v.dtype))


def _eig_one(a, *, left, right):
    n = a.shape[0]
    complex_dtype = jnp.result_type(a.dtype, np.complex64)
    real_dtype = _real_dtype(a.dtype)
    ulp = jnp.finfo(real_dtype).eps
    small_number = jnp.finfo(real_dtype).tiny * (n / ulp)
    t, z = _schur_one(a)
    if _is_complex(a):
        w = jnp.diagonal(t)
        tc, zc = t, z
        real_pairs = None
    else:
        w, starts, seconds = _real_schur_eigenvalues(t)
        tc, zc = t.astype(complex_dtype), z.astype(complex_dtype)

        def to_triangular(i, carry):  # the 2 x 2 blocks, lam = a + i sqrt(-bc) first
            tc, zc = carry
            m = jnp.int32(n - 2) - i
            lam = jnp.take(w, m)
            new = _triangularize_2x2(tc, zc, m, lam)
            keep = ~jnp.take(starts, m)
            return jnp.where(keep, tc, new[0]), jnp.where(keep, zc, new[1])

        if n > 1:  # fori_loop traces its body even for no trips
            tc, zc = lax.fori_loop(0, n - 1, to_triangular, (tc, zc))
        real_pairs = (starts, seconds)
    index = jnp.arange(n)
    floor = jnp.maximum(ulp * (jnp.abs(jnp.real(w)) + jnp.abs(jnp.imag(w))), small_number)

    def safe(denominator):
        return jnp.where(jnp.abs(denominator) < floor, floor.astype(denominator.dtype),
                         denominator)

    outputs = [w]
    if left:
        def left_step(j, y):  # T^H y = conj(w) y, y[k] = 1, y[i < k] = 0
            column = jnp.where(index < j, jnp.conj(tc[:, j]), 0)
            value = -_matmul(column, y) / safe(jnp.conj(tc[j, j] - w))
            return y.at[j, :].set(jnp.where(index < j, value, y[j, :]))
        y = lax.fori_loop(1, n, left_step, jnp.eye(n, dtype=complex_dtype))
        outputs.append(_normalized(_matmul(zc, y)))
    if right:
        def right_step(i, x):  # T x = w x, x[k] = 1, x[i > k] = 0
            j = n - 2 - i
            row = jnp.where(index > j, tc[j, :], 0)
            value = -_matmul(row, x) / safe(tc[j, j] - w)
            return x.at[j, :].set(jnp.where(index > j, value, x[j, :]))
        x = lax.fori_loop(0, n - 1, right_step, jnp.eye(n, dtype=complex_dtype))
        outputs.append(_normalized(_matmul(zc, x)))
    if real_pairs is not None:  # real vectors for real eigenvalues, exact conjugate pairs
        starts, seconds = real_pairs
        real = jnp.imag(w) == 0
        for i in range(1, len(outputs)):
            v = jnp.where(real[None, :], jnp.real(outputs[i]).astype(complex_dtype), outputs[i])
            conjugate_of_previous = jnp.conj(jnp.roll(v, 1, axis=1))
            outputs[i] = jnp.where(seconds[None, :], conjugate_of_previous, v)
    failed = ~jnp.all(jnp.isfinite(t))
    return [jnp.where(failed, jnp.full_like(o, jnp.nan), o) for o in outputs]


def _eig_rule(a, *, compute_left_eigenvectors, compute_right_eigenvectors, **_):
    n = a.shape[-1]
    complex_dtype = jnp.result_type(a.dtype, np.complex64)
    if n == 0:
        w = jnp.zeros(a.shape[:-1], complex_dtype)
        vectors = [jnp.zeros(a.shape, complex_dtype)] * (
            compute_left_eigenvectors + compute_right_eigenvectors)
        return [w, *vectors]

    def one(x):
        return _eig_one(x, left=compute_left_eigenvectors, right=compute_right_eigenvectors)

    return _over_batch(one, [a], [2])


# --- Tridiagonal solve (LAPACK gtsv) -----------------------------------------------------

def _tridiagonal_solve_one(dl, d, du, b, *, perturb_singular):
    """x with T x = b for T = diag(dl[1:], -1) + diag(d) + diag(du[:-1], 1), by Gaussian
    elimination with partial pivoting (LAPACK gtsv). With perturb_singular, pivots smaller
    than eps max|T| become +-eps max|T|, so singular systems (inverse iteration) give a
    finite answer, as LAPACK's dlagts does."""
    n = d.shape[0]
    if n == 1:
        return b / d[0]
    lower = dl[1:]  # lower[i] = T[i + 1, i]
    second = jnp.zeros_like(d)  # second[i] = U[i, i + 2], filled by row interchanges
    tol = jnp.finfo(_real_dtype(d.dtype)).eps * jnp.max(
        jnp.concatenate([jnp.abs(d), jnp.abs(dl), jnp.abs(du)]))
    tol = jnp.where(tol == 0, jnp.finfo(_real_dtype(d.dtype)).tiny, tol)

    def pivot(x):
        if not perturb_singular:
            return x
        sign = jnp.where(jnp.real(x) < 0, -1, 1).astype(x.dtype)
        return jnp.where(jnp.abs(x) < tol, sign * tol, x)

    def eliminate(i, carry):
        d, du, second, lower, b = carry
        di, li, di1, dui = d[i], lower[i], d[i + 1], du[i]
        dui1 = jnp.where(i + 1 < n - 1, du[jnp.minimum(i + 1, n - 1)], 0)
        bi, bi1 = b[i], b[i + 1]
        swap = jnp.abs(di) < jnp.abs(li)
        # No interchange: row i + 1 -= (l / d) row i.
        fact = li / jnp.where(di == 0, 1, pivot(di))
        keep = (pivot(di), dui, jnp.zeros_like(di), di1 - fact * dui, dui1, bi, bi1 - fact * bi)
        # Interchange rows i and i + 1, then eliminate.
        fact_s = di / jnp.where(li == 0, 1, li)
        swapped = (li, di1, dui1, dui - fact_s * di1, -fact_s * dui1, bi1, bi - fact_s * bi1)
        new = [jnp.where(swap, s_, k_) for s_, k_ in zip(swapped, keep)]
        d = d.at[i].set(new[0]).at[i + 1].set(new[3])
        du = du.at[i].set(new[1])
        du = du.at[jnp.minimum(i + 1, n - 1)].set(jnp.where(i + 1 < n - 1, new[4],
                                                             du[jnp.minimum(i + 1, n - 1)]))
        second = second.at[i].set(new[2])
        b = b.at[i].set(new[5]).at[i + 1].set(new[6])
        return d, du, second, lower, b

    d, du, second, lower, b = lax.fori_loop(0, n - 1, eliminate, (d, du, second, lower, b))
    d = d.at[n - 1].set(pivot(d[n - 1]))
    x = b.at[n - 1].set(b[n - 1] / d[n - 1])
    x = x.at[n - 2].set((x[n - 2] - du[n - 2] * x[n - 1]) / d[n - 2])

    def substitute(k, x):
        i = n - 3 - k
        return x.at[i].set((x[i] - du[i] * x[i + 1] - second[i] * x[i + 2]) / d[i])

    if n > 2:
        x = lax.fori_loop(0, n - 2, substitute, x)
    return x


def _tridiagonal_solve_rule(dl, d, du, b, *, perturb_singular, **_):
    if d.shape[-1] == 0 or b.shape[-1] == 0:
        return [b]
    one = partial(_tridiagonal_solve_one, perturb_singular=perturb_singular)
    return _over_batch(lambda *args: [one(*args)], [dl, d, du, b], [1, 1, 1, 2])


def register(register_rule):
    for primitive, fn in ((linalg.tridiagonal_solve_p, _tridiagonal_solve_rule),
                          (linalg.hessenberg_p, _hessenberg_rule),
                          (linalg.geqp3_p, _geqp3_rule),
                          (linalg.tridiagonal_p, _tridiagonal_rule),
                          (linalg.schur_p, _schur_rule),
                          (linalg.eig_p, _eig_rule)):
        register_rule(primitive, mlir.lower_fun(fn, multiple_results=True))
