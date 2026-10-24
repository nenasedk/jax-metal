"""Sweep float32 elementwise ops on METAL against the CPU across magnitudes and specials.

    .venv-shim/bin/python shim/tests/accuracy_sweep.py [--strict]

Reports, per op, the worst relative error (in float32 ulps, ignoring results below 1e-30)
and any NaN/inf disagreement. With --strict, exits non-zero if an op exceeds MAX_ULPS.
"""

import sys

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

MAX_ULPS = 64  # transcendental implementations differ; flag only real accuracy bugs

magnitudes = np.logspace(-12, 4, 161, dtype=np.float64)
base = np.concatenate([magnitudes, -magnitudes, [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0]])
X = base.astype(np.float32)
UNIT = np.clip(X, -0.999999, 0.999999)
POSITIVE = np.abs(X) + np.float32(0)

OPS = {
    "exp": (jnp.exp, X), "expm1": (jnp.expm1, X), "log": (jnp.log, POSITIVE),
    "log1p": (jnp.log1p, X), "sqrt": (jnp.sqrt, POSITIVE), "rsqrt": (jax.lax.rsqrt, POSITIVE),
    "cbrt": (jnp.cbrt, X), "sin": (jnp.sin, X), "cos": (jnp.cos, X), "tan": (jnp.tan, X),
    "tanh": (jnp.tanh, X), "sinh": (jnp.sinh, X), "cosh": (jnp.cosh, X),
    "arcsinh": (jnp.arcsinh, X), "arccosh": (jnp.arccosh, POSITIVE + 1),
    "arctanh": (jnp.arctanh, UNIT), "arcsin": (jnp.arcsin, UNIT), "arccos": (jnp.arccos, UNIT),
    "arctan": (jnp.arctan, X), "erf": (jsp.special.erf, X), "erfc": (jsp.special.erfc, X),
    "erfinv": (jsp.special.erfinv, UNIT), "sigmoid": (jax.nn.sigmoid, X),
    "softplus": (jax.nn.softplus, X), "logaddexp": (lambda x: jnp.logaddexp(x, 0.5 * x), X),
    "pow_2.5": (lambda x: jnp.power(x, 2.5), POSITIVE), "square": (jnp.square, X),
    "reciprocal": (jnp.reciprocal, X), "sign": (jnp.sign, X), "abs": (jnp.abs, X),
}


def ulps(got, want):
    spacing = np.spacing(np.maximum(np.abs(want), np.float32(1e-30)).astype(np.float32))
    return np.abs(got.astype(np.float64) - want) / spacing


def main():
    metal, cpu = jax.devices("METAL")[0], jax.devices("cpu")[0]
    bad = []
    for name, (fn, x) in OPS.items():
        got, want = (np.asarray(jax.jit(fn)(jax.device_put(x, d))) for d in (metal, cpu))
        nan_mismatch = np.isnan(got) != np.isnan(want)
        finite = np.isfinite(got) & np.isfinite(want)
        inf_mismatch = ~nan_mismatch & ~finite & (got != want) & ~(np.isnan(got) & np.isnan(want))
        worst = float(ulps(got[finite], want[finite]).max()) if finite.any() else 0.0
        where = x[finite][ulps(got[finite], want[finite]).argmax()] if finite.any() else None
        issues = []
        if worst > MAX_ULPS:
            issues.append(f"{worst:.3g} ulps at x={where:.3g}")
        if nan_mismatch.any():
            issues.append(f"NaN mismatch at x={x[nan_mismatch][:3]}")
        if inf_mismatch.any():
            issues.append(f"inf mismatch at x={x[inf_mismatch][:3]}")
        print(f"{'BAD ' if issues else 'ok  '} {name:10} worst {worst:9.3g} ulps  {'; '.join(issues)}")
        if issues:
            bad.append(name)
    print(f"\nops outside {MAX_ULPS} ulps or with NaN/inf mismatches: {bad or 'none'}")
    if "--strict" in sys.argv and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
