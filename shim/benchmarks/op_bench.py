"""Time individual ops on METAL vs the CPU backend (median of repeated jitted calls).

    .venv-shim/bin/python shim/benchmarks/op_bench.py OUT.json

Covers ops the shim implements with METAL lowering rules (cholesky, triangular_solve, FFT,
cumsum, sort) next to native ones (matmul, conv) for reference.
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

RNG = np.random.default_rng(0)


def f32(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def spd(n):
    a = f32(n, n)
    return (a @ a.T / n + np.eye(n, dtype=np.float32)).astype(np.float32)


def timed(fn, args, device, repeats):
    compiled = jax.jit(fn)
    placed = jax.device_put(args, device)
    with jax.default_device(device):
        jax.block_until_ready(compiled(*placed))  # compile + warm up
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(compiled(*placed))
            samples.append(time.perf_counter() - start)
    return 1e3 * float(np.median(samples))


def cases():
    out = {}
    for n in (1024, 2048):
        out[f"matmul {n}"] = (lambda a, b: a @ b, (f32(n, n), f32(n, n)))
    out["conv 32x64x56x56 k3"] = (lambda x, k: lax.conv_general_dilated(x, k, (1, 1), "SAME"),
                                  (f32(32, 64, 56, 56), f32(64, 64, 3, 3)))
    for n in (64, 256, 1024):
        out[f"cholesky {n}"] = (jnp.linalg.cholesky, (spd(n),))
        out[f"solve {n}"] = (jnp.linalg.solve, (spd(n), f32(n, 8)))
        out[f"triangular_solve {n}"] = (
            lambda a, b: lax.linalg.triangular_solve(a, b, left_side=True, lower=True),
            (np.tril(spd(n)), f32(n, 8)))
    for n in (1024, 65536, 1 << 20):
        out[f"fft {n}"] = (lambda x: jnp.abs(jnp.fft.fft(x)), (f32(n),))
        out[f"rfft {n}"] = (lambda x: jnp.abs(jnp.fft.rfft(x)), (f32(n),))
    for n in (1009, 4099):  # primes: direct DFT path
        out[f"fft prime {n}"] = (lambda x: jnp.abs(jnp.fft.fft(x)), (f32(n),))
    out["fft2 batch 64x256x256"] = (lambda x: jnp.abs(jnp.fft.fft2(x)), (f32(64, 256, 256),))
    out["cumsum 1M"] = (jnp.cumsum, (f32(1 << 20),))
    out["sort 1M"] = (jnp.sort, (f32(1 << 20),))
    return out


def main(out_path):
    metal, cpu = jax.devices("METAL")[0], jax.devices("cpu")[0]
    results = {}
    for name, (fn, args) in cases().items():
        row = {}
        for label, device in (("metal_ms", metal), ("cpu_ms", cpu)):
            try:
                row[label] = timed(fn, args, device, repeats=10)
            except Exception as error:  # record and continue
                row[label] = None
                row[label + "_error"] = str(error).splitlines()[0][:200]
        results[name] = row
        ratio = (row["cpu_ms"] / row["metal_ms"]) if row.get("metal_ms") and row.get("cpu_ms") else None
        print(f"{name:26} metal {row.get('metal_ms') or float('nan'):9.2f} ms   cpu {row.get('cpu_ms') or float('nan'):9.2f} ms"
              f"   speedup {ratio if ratio else float('nan'):6.2f}x", flush=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
