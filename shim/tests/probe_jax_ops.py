"""Probe which JAX operations run on the METAL device through the shim.

    .venv-shim/bin/python shim/tests/probe_jax_ops.py OUT.md

Each case runs in its own subprocess (a plugin crash only loses that case), on METAL and
on CPU, and is reported as PASS, MISMATCH, ERROR or CRASH.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np


def _cases():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax import lax

    rng = np.random.default_rng(0)
    f = lambda *s: rng.standard_normal(s, dtype=np.float32)
    pos = lambda *s: np.abs(f(*s)) + 0.5
    unit = lambda *s: np.tanh(f(*s)) * 0.9
    i32 = lambda *s: rng.integers(-50, 50, s, dtype=np.int32)
    spd = (lambda a: a @ a.T + 4 * np.eye(4, dtype=np.float32))(f(4, 4))
    image = f(2, 3, 8, 8)

    def max_pool(x):
        return lax.reduce_window(x, -np.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")

    def avg_pool(x):
        return lax.reduce_window(x, 0.0, lax.add, (1, 1, 2, 2), (1, 1, 2, 2), "VALID") / 4

    return {
        # Elementwise unary
        "abs": (jnp.abs, f(16)), "ceil": (jnp.ceil, f(16)), "floor": (jnp.floor, f(16)),
        "round": (jnp.round, f(16)), "sign": (jnp.sign, f(16)), "negative": (jnp.negative, f(16)),
        "exp": (jnp.exp, f(16)), "expm1": (jnp.expm1, f(16)), "log": (jnp.log, pos(16)),
        "log1p": (jnp.log1p, pos(16)), "sqrt": (jnp.sqrt, pos(16)), "rsqrt": (lax.rsqrt, pos(16)),
        "cbrt": (jnp.cbrt, f(16)), "sin": (jnp.sin, f(16)), "cos": (jnp.cos, f(16)),
        "tan": (jnp.tan, unit(16)), "tanh": (jnp.tanh, f(16)), "arcsin": (jnp.arcsin, unit(16)),
        "arccos": (jnp.arccos, unit(16)), "arctan": (jnp.arctan, f(16)),
        "sinh": (jnp.sinh, f(16)), "cosh": (jnp.cosh, f(16)), "arcsinh": (jnp.arcsinh, f(16)),
        "arccosh": (jnp.arccosh, pos(16) + 1), "arctanh": (jnp.arctanh, unit(16)),
        "erf": (jsp.special.erf, f(16)), "erfinv": (jsp.special.erfinv, unit(16)),
        "erfc": (jsp.special.erfc, f(16)), "lgamma": (jsp.special.gammaln, pos(16)),
        "digamma": (jsp.special.digamma, pos(16)), "sigmoid": (jax.nn.sigmoid, f(16)),
        "isfinite": (jnp.isfinite, f(16)), "isnan": (jnp.isnan, f(16)),
        "nextafter": (lambda x: jnp.nextafter(x, x + 1), f(16)),
        "reduce_precision": (lambda x: lax.reduce_precision(x, 5, 10), f(16)),
        # Elementwise binary / integer / bitwise
        "add_sub_mul_div": (lambda x, y: (x + y, x - y, x * y, x / y), f(16), pos(16)),
        "pow": (jnp.power, pos(16), f(16)), "remainder": (jnp.remainder, f(16), pos(16)),
        "max_min": (lambda x, y: (jnp.maximum(x, y), jnp.minimum(x, y)), f(16), f(16)),
        "arctan2": (jnp.arctan2, f(16), f(16)), "compare": (lambda x, y: (x < y, x == y), f(16), f(16)),
        "int32_arith": (lambda x, y: (x + y, x * y, x // (jnp.abs(y) + 1)), i32(16), i32(16)),
        "bitwise": (lambda x, y: (x & y, x | y, x ^ y, ~x), i32(16), i32(16)),
        "shifts": (lambda x: (x << 2, x >> 1, lax.shift_right_logical(x, 1)), i32(16)),
        "popcount": (lax.population_count, i32(16)), "clz": (lax.clz, i32(16)),
        "where_clip": (lambda x: (jnp.where(x > 0, x, 0.0), jnp.clip(x, -0.5, 0.5)), f(16)),
        # Dtypes
        "float16": (lambda x: (x.astype(jnp.float16) * 2).astype(jnp.float32), f(16)),
        "bfloat16": (lambda x: (x.astype(jnp.bfloat16) * 2).astype(jnp.float32), f(16)),
        "uint8_convert": (lambda x: (jnp.abs(x) * 10).astype(jnp.uint8), f(16)),
        "bitcast": (lambda x: lax.bitcast_convert_type(x, jnp.int32), f(16)),
        "complex64": (lambda x: jnp.abs(x * (1 + 1j)), f(16)),
        # Shapes and indexing
        "transpose_reshape": (lambda x: x.T.reshape(-1), f(4, 6)),
        "slice_reverse_pad": (lambda x: (x[1:5:2], x[::-1], jnp.pad(x, 2)), f(8)),
        "concat_broadcast": (lambda x: (jnp.concatenate([x, x]), jnp.broadcast_to(x, (3, 8))), f(8)),
        "iota": (lambda: jnp.arange(12).reshape(3, 4), ),
        "dynamic_slice": (lambda x: lax.dynamic_slice(x, (2,), (3,)), f(8)),
        "dynamic_update_slice": (lambda x: lax.dynamic_update_slice(x, jnp.ones(3), (2,)), f(8)),
        "gather": (lambda x: x[jnp.array([3, 0, 2])], f(6, 4)),
        "scatter_add": (lambda x: x.at[jnp.array([0, 2, 2])].add(1.0), f(6)),
        "sort_argsort": (lambda x: (jnp.sort(x), jnp.argsort(x)), f(32)),
        "top_k": (lambda x: lax.top_k(x, 3), f(32)),
        # Reductions and windows
        "reduce_sum_max_prod": (lambda x: (x.sum(1), x.max(0), x.prod()), f(4, 5)),
        "reduce_and_or": (lambda x: (jnp.all(x > 0), jnp.any(x > 0)), f(16)),
        "argmax_argmin": (lambda x: (jnp.argmax(x, 1), jnp.argmin(x, 0)), f(4, 5)),
        "logsumexp": (lambda x: jsp.special.logsumexp(x, axis=1), f(4, 5)),
        "max_pool": (max_pool, image), "avg_pool": (avg_pool, image),
        "max_pool_grad": (jax.grad(lambda x: max_pool(x).sum()), image),
        "cumsum": (lambda x: jnp.cumsum(x), f(16)),
        # Linear algebra and convolution
        "matmul_batched": (lambda a, b: a @ b, f(3, 4, 5), f(3, 5, 6)),
        "einsum": (lambda a, b: jnp.einsum("ij,jk->ik", a, b), f(4, 5), f(5, 6)),
        "conv2d": (lambda x, k: lax.conv_general_dilated(x, k, (1, 1), "SAME"), image, f(4, 3, 3, 3)),
        "conv2d_grad": (jax.grad(lambda k, x: lax.conv_general_dilated(x, k, (1, 1), "SAME").sum()),
                        f(4, 3, 3, 3), image),
        "conv_transpose": (lambda x, k: lax.conv_transpose(x, k, (2, 2), "SAME",
                                                           dimension_numbers=("NCHW", "OIHW", "NCHW")),
                           image, f(4, 3, 3, 3)),
        "cholesky": (jnp.linalg.cholesky, spd), "solve": (jnp.linalg.solve, spd, f(4)),
        "inv": (jnp.linalg.inv, spd), "triangular_solve": (
            lambda a, b: jsp.linalg.solve_triangular(a, b, lower=True), np.tril(spd), f(4)),
        "fft": (lambda x: jnp.abs(jnp.fft.fft(x)), f(16)),
        # Control flow and randomness
        "while_fori": (lambda x: lax.fori_loop(0, 5, lambda i, v: v * 1.1, x), f(4)),
        "cond": (lambda x: lax.cond(x.sum() > 0, jnp.sin, jnp.cos, x), f(4)),
        "switch": (lambda x: lax.switch(1, [jnp.sin, jnp.cos, jnp.tanh], x), f(4)),
        "scan": (lambda x: lax.scan(lambda c, v: (c + v, c * v), 0.0, x), f(8)),
        "random_threefry": (lambda: jax.random.uniform(jax.random.key(1), (8,)), ),
        "random_rbg": (lambda: jax.random.uniform(jax.random.key(1, impl="rbg"), (8,)), ),
        # Networks
        "softmax_layernorm": (lambda x: (jax.nn.softmax(x), (x - x.mean(-1, keepdims=True))
                                         / jnp.sqrt(x.var(-1, keepdims=True) + 1e-5)), f(4, 16)),
        "float64_x64_off": (lambda x: x.astype(jnp.float64) * 2, f(4)),
    }


def run_case(name):
    import jax
    case = _cases()[name]
    fn, args = case[0], case[1:]
    results = {}
    for platform in ("METAL", "cpu"):
        device = jax.devices(platform)[0]
        placed = jax.device_put(args, device)
        with jax.default_device(device):
            out = jax.jit(fn)(*placed)
        results[platform] = [np.asarray(leaf) for leaf in jax.tree.leaves(out)]
    ok = all(g.shape == w.shape and np.allclose(g.astype(np.float64), w.astype(np.float64),
                                                rtol=1e-3, atol=1e-3, equal_nan=True)
             for g, w in zip(results["METAL"], results["cpu"]))
    return "PASS" if ok else "MISMATCH", ""


def main(out_path):
    names = list(_cases())
    rows = []
    for name in names:
        proc = subprocess.run([sys.executable, __file__, "--case", name], capture_output=True,
                              text=True, timeout=600, env=dict(os.environ))
        # The plugin also prints to stdout, so find the result line.
        results = [line for line in proc.stdout.splitlines() if line.startswith('{"status"')]
        if proc.returncode == 0 and results:
            result = json.loads(results[-1])
            status, detail = result["status"], result["detail"]
        else:
            status, detail = "CRASH", f"exit {proc.returncode}: " + (proc.stderr.strip().splitlines() or [""])[-1][:120]
        rows.append((name, status, detail))
        print(f"{status:8} {name:22} {detail[:110]}", flush=True)
    counts = {s: sum(1 for _, st, _ in rows if st == s) for s in ("PASS", "MISMATCH", "ERROR", "CRASH")}
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# JAX op probe: jax 0.12 on METAL through the shim\n\n")
        out.write("Generated by `shim/tests/probe_jax_ops.py`; each case runs under `jax.jit` on "
                  "METAL and CPU and compares results (rtol/atol 1e-3).\n\n")
        out.write(", ".join(f"{k}: {v}" for k, v in counts.items()) + "\n\n")
        out.write("| Case | Status | Detail |\n|---|---|---|\n")
        for name, status, detail in rows:
            out.write(f"| `{name}` | {status} | {detail.replace('|', '/')} |\n")
    print(counts)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--case"]:
        try:
            status, detail = run_case(sys.argv[2])
        except Exception as error:  # report the failing op, else the first line
            message = str(error)
            ops = re.findall(r"failed to legalize operation '([\w.]+)'", message)
            first = (message.strip().splitlines() or [repr(error)])[0]
            status, detail = "ERROR", (f"no lowering for {', '.join(dict.fromkeys(ops))}" if ops
                                       else first[:200])
        print(json.dumps({"status": status, "detail": detail}))
    else:
        main(sys.argv[1])
