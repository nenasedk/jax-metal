"""Run JAX programs on the METAL device through the shim and compare with the CPU.

    .venv-shim/bin/python shim/tests/test_jax_e2e.py

Each case runs independently and reports PASS, MISMATCH or ERROR, so one unsupported
op does not hide the others. Exits non-zero if any case in CORE fails.
"""

import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

CORE = {"arange_add", "sin_matmul_add", "grad_cubic", "elementwise_math", "reductions",
        "softmax", "matmul_512", "fori_loop", "mlp_train_step"}


def _x(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


def mlp_train_step():
    params = {"w1": _x((16, 32), 1), "b1": _x((32,), 2), "w2": _x((32, 4), 3)}
    x, y = _x((8, 16), 4), _x((8, 4), 5)

    def loss(p, x, y):
        h = jnp.tanh(x @ p["w1"] + p["b1"])
        return jnp.mean((h @ p["w2"] - y) ** 2)

    def step(p, x, y):
        grads = jax.grad(loss)(p, x, y)
        return jax.tree.map(lambda a, g: a - 0.1 * g, p, grads)

    return step, (params, x, y)


CASES = {
    "arange_add": (lambda: jnp.arange(10.0) + 1.0, ()),
    "sin_matmul_add": (lambda x, y: jnp.sin(x) @ y + 1.0, (_x((4, 8)), _x((8, 3), 1))),
    "grad_cubic": (jax.grad(lambda x: jnp.sum(x ** 3)), (_x((5,)),)),
    "elementwise_math": (lambda x: jnp.exp(x) + jnp.log1p(jnp.abs(x)) + jnp.tanh(x) * jnp.sqrt(jnp.abs(x)),
                         (_x((64,)),)),
    "reductions": (lambda x: (x.sum(0), x.mean(1), x.max(), jnp.argmax(x, 1)), (_x((6, 7)),)),
    "softmax": (lambda x: jax.nn.softmax(x, axis=-1), (_x((4, 10)),)),
    "matmul_512": (lambda a, b: a @ b, (_x((512, 512)), _x((512, 512), 1))),
    "fori_loop": (lambda x: lax.fori_loop(0, 10, lambda i, v: v * 0.9 + i, x), (_x((8,)),)),
    "cumsum": (lambda x: jnp.cumsum(x, axis=0), (_x((16, 3)),)),
    "cumprod_cummax": (lambda x: (jnp.cumprod(x * 0.1 + 1.0, axis=1), lax.cummax(x, axis=0, reverse=True)),
                       (_x((5, 9)),)),
    "sort": (lambda x: jnp.sort(x), (_x((32,)),)),
    "gather_scatter": (lambda x: x.at[jnp.array([0, 2, 4])].add(10.0)[jnp.array([4, 0])], (_x((6,)),)),
    "conv2d": (lambda x, k: lax.conv_general_dilated(x, k, (1, 1), "SAME"),
               (_x((1, 3, 8, 8)), _x((4, 3, 3, 3), 1))),
    "random_normal": (lambda: jax.random.normal(jax.random.key(0), (4,)), ()),
    "tan": (lambda x: jnp.tan(x), (_x((8,)) * 0.5,)),
    "mlp_train_step": mlp_train_step(),
}


def run(fn, args, device):
    args = jax.device_put(args, device)
    with jax.default_device(device):
        out = jax.jit(fn)(*args)
    return jax.tree.map(np.asarray, out)


def main():
    metal, cpu = jax.devices("METAL")[0], jax.devices("cpu")[0]
    print(f"jax {jax.__version__}; METAL device: {metal} ({metal.device_kind})\n")
    failed_core = []
    for name, (fn, args) in CASES.items():
        try:
            got, want = run(fn, args, metal), run(fn, args, cpu)
            ok = all(np.allclose(g, w, rtol=1e-4, atol=1e-4)
                     for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)))
            status = "PASS" if ok else "MISMATCH"
            detail = ""
        except Exception as error:  # report every case, keep going
            status = "ERROR"
            detail = str(error).strip().splitlines()[0][:150] if str(error).strip() else repr(error)
            if "-v" in sys.argv:
                traceback.print_exc()
        print(f"{status:8} {name:18} {detail}")
        if status != "PASS" and name in CORE:
            failed_core.append(name)
    print(f"\ncore cases failing: {failed_core or 'none'}")
    sys.exit(1 if failed_core else 0)


if __name__ == "__main__":
    main()
