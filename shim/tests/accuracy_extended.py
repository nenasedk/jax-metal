"""Accuracy of METAL beyond elementwise float32: reductions, matmul, conv, f16/bf16, binary ops.

    .venv-shim/bin/python shim/tests/accuracy_extended.py [OUT.md] [--strict] [--device NAME]

Every check measures METAL's error against a float64 (NumPy) reference and compares it with
the CPU backend's error on the same float32/float16/bfloat16 inputs. A check is flagged when
METAL is clearly worse than XLA's CPU (more than RATIO times its error and above an absolute
floor), or disagrees on NaN/inf. With --strict, exits non-zero if anything is flagged.
"""

import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax

# --device picks the device under test (default METAL; cpu to dry-run the script anywhere).
DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "METAL"
METAL, CPU = jax.devices(DEVICE)[0], jax.devices("cpu")[0]
RNG = np.random.default_rng(0)
RATIO = 8.0
rows = []


def on(device, fn, *args):
    with jax.default_device(device):
        return np.asarray(jax.jit(fn)(*jax.device_put(args, device)))


def rel_error(got, truth):
    got, truth = np.asarray(got, np.float64), np.asarray(truth, np.float64)
    finite = np.isfinite(truth)
    scale = max(float(np.max(np.abs(truth[finite]), initial=0)), 1e-30)
    return float(np.max(np.abs(got[finite] - truth[finite]), initial=0)) / scale


def check(group, name, fn, args, truth, floor):
    try:
        metal, cpu = on(METAL, fn, *args), on(CPU, fn, *args)
        special = (np.isnan(metal) != np.isnan(np.asarray(truth))) | (
            np.isinf(metal) != np.isinf(np.asarray(truth)))
        e_metal, e_cpu = rel_error(metal, truth), rel_error(cpu, truth)
        flagged = bool(special.any()) or (e_metal > floor and e_metal > RATIO * max(e_cpu, floor / RATIO))
        note = "NaN/inf mismatch" if special.any() else ""
    except Exception as error:  # report and continue
        e_metal = e_cpu = float("nan")
        flagged, note = True, (str(error).strip().splitlines() or [repr(error)])[0][:120]
    rows.append((group, name, e_metal, e_cpu, flagged, note))
    print(f"{'BAD ' if flagged else 'ok  '} {group:10} {name:34} metal {e_metal:9.2e}  cpu {e_cpu:9.2e}  {note}",
          flush=True)


def f32(*shape, scale=1.0):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


# --- reductions ----------------------------------------------------------------------------
for n in (1_000, 100_000, 1_000_000):
    x = f32(n) + 3.0  # positive mean exposes naive summation error
    x64 = x.astype(np.float64)
    check("reduce", f"sum n={n}", jnp.sum, (x,), x64.sum(), 1e-6)
    check("reduce", f"mean n={n}", jnp.mean, (x,), x64.mean(), 1e-6)
    check("reduce", f"var n={n}", jnp.var, (x,), x64.var(), 1e-5)
    check("reduce", f"std n={n}", jnp.std, (x,), x64.std(), 1e-5)
    check("reduce", f"norm n={n}", jnp.linalg.norm, (x,), np.linalg.norm(x64), 1e-6)
    check("reduce", f"logsumexp n={n}", jax.scipy.special.logsumexp, (x,),
          np.log(np.sum(np.exp(x64 - x64.max()))) + x64.max(), 1e-6)
    check("reduce", f"max/min n={n}", lambda v: jnp.stack([v.max(), v.min()]), (x,),
          np.array([x64.max(), x64.min()]), 0)
x = (1 + f32(4096, scale=1e-3)).astype(np.float32)
check("reduce", "prod n=4096", jnp.prod, (x,), np.prod(x.astype(np.float64)), 1e-5)
x = f32(256, 4096) + 1.0
check("reduce", "sum axis=1 (256x4096)", lambda v: v.sum(1), (x,), x.astype(np.float64).sum(1), 1e-6)
x = f32(100_000) + 1.0
check("reduce", "cumsum n=100000", jnp.cumsum, (x,), np.cumsum(x.astype(np.float64)), 1e-6)

# --- matmul ---------------------------------------------------------------------------------
for n in (64, 512, 2048):
    a, b = f32(n, n), f32(n, n)
    truth = a.astype(np.float64) @ b.astype(np.float64)
    check("matmul", f"{n}x{n} default", lambda a, b: a @ b, (a, b), truth, 1e-6)
    check("matmul", f"{n}x{n} HIGHEST", lambda a, b: jnp.matmul(a, b, precision=lax.Precision.HIGHEST),
          (a, b), truth, 1e-6)
a, b = f32(8, 64, 256), f32(8, 256, 32)
check("matmul", "batched 8x64x256 @ 8x256x32", lambda a, b: a @ b, (a, b),
      a.astype(np.float64) @ b.astype(np.float64), 1e-6)

# --- convolution ----------------------------------------------------------------------------
def conv_reference(x, k, padding):
    """Direct float64 NCHW/OIHW convolution with stride 1."""
    x, k = x.astype(np.float64), k.astype(np.float64)
    if padding == "SAME":
        ph, pw = k.shape[2] // 2, k.shape[3] // 2
        x = np.pad(x, ((0, 0), (0, 0), (ph, k.shape[2] - 1 - ph), (pw, k.shape[3] - 1 - pw)))
    n, c, h, w = x.shape
    o, _, kh, kw = k.shape
    out = np.zeros((n, o, h - kh + 1, w - kw + 1))
    for i in range(kh):
        for j in range(kw):
            out += np.einsum("nchw,oc->nohw", x[:, :, i:i + out.shape[2], j:j + out.shape[3]], k[:, :, i, j])
    return out


for (n, c, h, o, kh, padding) in [(2, 3, 32, 16, 3, "SAME"), (4, 64, 16, 64, 3, "SAME"),
                                  (2, 16, 28, 8, 5, "VALID"), (1, 128, 8, 128, 1, "VALID")]:
    x, k = f32(n, c, h, h), f32(o, c, kh, kh)
    check("conv", f"{n}x{c}x{h}x{h} k{o}x{kh}x{kh} {padding}",
          lambda x, k, p=padding: lax.conv_general_dilated(x, k, (1, 1), p), (x, k),
          conv_reference(x, k, padding), 1e-6)

# --- float16 / bfloat16 ---------------------------------------------------------------------
for dtype, name, floor in ((np.float16, "f16", 1e-3), (ml_dtypes.bfloat16, "bf16", 8e-3)):
    a = (RNG.uniform(-4, 4, 4096)).astype(dtype)
    b = (RNG.uniform(0.5, 4, 4096) * RNG.choice([-1, 1], 4096)).astype(dtype)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    # Correctly rounded arithmetic: the truth is the float64 result rounded to the dtype.
    for op, fn, truth in [("add", lambda a, b: a + b, a64 + b64), ("mul", lambda a, b: a * b, a64 * b64),
                          ("div", lambda a, b: a / b, a64 / b64)]:
        check(name, op, fn, (a, b), truth.astype(dtype).astype(np.float64), floor)
    pos = np.abs(b)
    check(name, "sqrt", jnp.sqrt, (pos,), np.sqrt(pos.astype(np.float64)).astype(dtype), floor)
    for op, fn, ref in [("exp", jnp.exp, np.exp), ("log", jnp.log, np.log), ("tanh", jnp.tanh, np.tanh),
                        ("sin", jnp.sin, np.sin), ("sigmoid", jax.nn.sigmoid, lambda v: 1 / (1 + np.exp(-v)))]:
        args = (pos,) if op == "log" else (a,)
        check(name, op, fn, args, ref(args[0].astype(np.float64)), floor)
    m, k = (RNG.standard_normal((128, 256))).astype(dtype), (RNG.standard_normal((256, 64))).astype(dtype)
    check(name, "matmul 128x256x64", lambda m, k: m @ k, (m, k),
          m.astype(np.float64) @ k.astype(np.float64), floor)
    v = (RNG.standard_normal(10_000) + 1).astype(dtype)
    check(name, "sum n=10000", jnp.sum, (v,), v.astype(np.float64).sum(), floor * 10)

# --- float32 binary ops ---------------------------------------------------------------------
a, b = f32(4096, scale=3), f32(4096, scale=3)
pa = np.abs(a) + 0.1
a64, b64, pa64 = a.astype(np.float64), b.astype(np.float64), pa.astype(np.float64)
check("binary", "pow (x>0)", jnp.power, (pa, b), pa64 ** b64, 1e-6)
check("binary", "atan2", jnp.arctan2, (a, b), np.arctan2(a64, b64), 1e-6)
check("binary", "remainder", jnp.remainder, (a, pa), np.remainder(a64, pa64), 1e-6)
check("binary", "fmod", jnp.fmod, (a, pa), np.fmod(a64, pa64), 1e-6)
check("binary", "hypot", jnp.hypot, (a, b), np.hypot(a64, b64), 1e-6)
check("binary", "logaddexp", jnp.logaddexp, (a, b), np.logaddexp(a64, b64), 1e-6)
nan_mix = a.copy()
nan_mix[::7] = np.nan
check("binary", "maximum/minimum with NaN", lambda x, y: jnp.stack([jnp.maximum(x, y), jnp.minimum(x, y)]),
      (nan_mix, b), np.stack([np.maximum(nan_mix.astype(np.float64), b64),
                              np.minimum(nan_mix.astype(np.float64), b64)]), 1e-7)
check("binary", "integer power x**3", lambda x: x ** 3, (a,), a64 ** 3, 1e-6)

flagged = [r for r in rows if r[4]]
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        out.write("# Extended accuracy: METAL vs CPU against a float64 reference\n\n"
                  "Generated by `shim/tests/accuracy_extended.py`. Errors are max |got - truth| / "
                  f"max |truth|; a row is flagged when METAL is more than {RATIO:g}x worse than "
                  "XLA's CPU on the same inputs (above a per-check floor) or disagrees on NaN/inf.\n\n"
                  "| Group | Check | METAL error | CPU error | Flagged | Note |\n|---|---|---|---|---|---|\n")
        for group, name, em, ec, bad, note in rows:
            out.write(f"| {group} | {name} | {em:.2e} | {ec:.2e} | {'**yes**' if bad else ''} | {note} |\n")
print(f"\n{len(flagged)} flagged: {[f'{r[0]}:{r[1]}' for r in flagged] or 'none'}")
if "--strict" in sys.argv and flagged:
    sys.exit(1)
