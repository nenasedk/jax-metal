"""Training benchmark in plain JAX, runnable on any JAX version and backend.

    python shim/benchmarks/train_bench.py DEVICE OUT.json [--steps N]

DEVICE is a platform name such as METAL or cpu. Uses only APIs present in both jax 0.4.34
(the jax-metal 0.1.1 baseline) and jax 0.12 (the shim), so the same models run on the old
setup, the shim and the CPU. Each model trains with Adam on synthetic data labelled by a fixed
random "teacher", so the loss genuinely decreases. Records compile time (first step), median
steady-state step time and the loss curve.
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

tree_map = jax.tree_util.tree_map


def dense_init(key, fan_in, fan_out):
    return {"w": jax.random.normal(key, (fan_in, fan_out)) * np.sqrt(2.0 / fan_in),
            "b": jnp.zeros((fan_out,))}


def dense(p, x):
    return x @ p["w"] + p["b"]


# --- models -------------------------------------------------------------------------------

def mlp_init(key):
    k = jax.random.split(key, 3)
    return [dense_init(k[0], 784, 512), dense_init(k[1], 512, 512), dense_init(k[2], 512, 10)]


def mlp_apply(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(dense(layer, x))
    return dense(params[-1], x)


def conv_init(key, cin, cout):
    return {"w": jax.random.normal(key, (cout, cin, 3, 3)) * np.sqrt(2.0 / (cin * 9)),
            "b": jnp.zeros((cout, 1, 1))}


def conv(p, x):
    return lax.conv_general_dilated(x, p["w"], (1, 1), "SAME") + p["b"]


def pool(x):
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")


def cnn_init(key):
    k = jax.random.split(key, 4)
    return {"c1": conv_init(k[0], 3, 32), "c2": conv_init(k[1], 32, 64),
            "c3": conv_init(k[2], 64, 64), "out": dense_init(k[3], 64 * 4 * 4, 10)}


def cnn_apply(params, x):
    x = pool(jax.nn.relu(conv(params["c1"], x)))
    x = pool(jax.nn.relu(conv(params["c2"], x)))
    x = pool(jax.nn.relu(conv(params["c3"], x)))
    return dense(params["out"], x.reshape(x.shape[0], -1))


D_MODEL, HEADS, LAYERS, SEQ, VOCAB = 256, 4, 4, 128, 512


def layer_norm(x, scale):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) * lax.rsqrt(var + 1e-5) * scale


def transformer_init(key):
    keys = iter(jax.random.split(key, 4 + 6 * LAYERS))
    params = {"embed": jax.random.normal(next(keys), (VOCAB, D_MODEL)) * 0.02,
              "pos": jax.random.normal(next(keys), (SEQ, D_MODEL)) * 0.02, "layers": []}
    for _ in range(LAYERS):
        params["layers"].append({
            "qkv": dense_init(next(keys), D_MODEL, 3 * D_MODEL),
            "proj": dense_init(next(keys), D_MODEL, D_MODEL),
            "ff1": dense_init(next(keys), D_MODEL, 4 * D_MODEL),
            "ff2": dense_init(next(keys), 4 * D_MODEL, D_MODEL),
            "ln1": jnp.ones((D_MODEL,)), "ln2": jnp.ones((D_MODEL,))})
        next(keys), next(keys)
    params["head"] = dense_init(next(keys), D_MODEL, VOCAB)
    return params


def transformer_apply(params, tokens):
    batch = tokens.shape[0]
    x = params["embed"][tokens] + params["pos"]
    mask = jnp.tril(jnp.ones((SEQ, SEQ), bool))
    head_dim = D_MODEL // HEADS
    for layer in params["layers"]:
        h = layer_norm(x, layer["ln1"])
        q, k, v = jnp.split(dense(layer["qkv"], h), 3, axis=-1)
        split = lambda t: t.reshape(batch, SEQ, HEADS, head_dim).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(head_dim)
        scores = jnp.where(mask, scores, -1e9)
        attn = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, -1), v)
        x = x + dense(layer["proj"], attn.transpose(0, 2, 1, 3).reshape(batch, SEQ, D_MODEL))
        h = layer_norm(x, layer["ln2"])
        x = x + dense(layer["ff2"], jax.nn.gelu(dense(layer["ff1"], h)))
    return dense(params["head"], x)


# --- data, loss, optimizer ------------------------------------------------------------------

def make_data(name, rng):
    if name == "mlp":
        x = rng.standard_normal((256, 784)).astype(np.float32)
        teacher = rng.standard_normal((784, 10)).astype(np.float32)
        return x, (x @ teacher).argmax(-1).astype(np.int32)
    if name == "cnn":
        x = rng.standard_normal((64, 3, 32, 32)).astype(np.float32)
        teacher = rng.standard_normal((3 * 32 * 32, 10)).astype(np.float32)
        return x, (x.reshape(64, -1) @ teacher).argmax(-1).astype(np.int32)
    tokens = rng.integers(0, VOCAB, (16, SEQ + 1)).astype(np.int32)
    tokens[:, 1:] = (tokens[:, :-1] * 7 + 3) % VOCAB  # learnable next-token rule
    return tokens[:, :-1], tokens[:, 1:]


def cross_entropy(logits, labels):
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, labels[..., None], axis=-1))


def adam_step(params, grads, state, step, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    mhat = tree_map(lambda m: m / (1 - b1 ** step), m)
    vhat = tree_map(lambda v: v / (1 - b2 ** step), v)
    params = tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v}


MODELS = {"mlp": (mlp_init, mlp_apply), "cnn": (cnn_init, cnn_apply),
          "transformer": (transformer_init, transformer_apply)}


def bench(name, device, steps):
    init, apply = MODELS[name]
    x, y = make_data(name, np.random.default_rng(0))

    def loss_fn(params, x, y):
        return cross_entropy(apply(params, x), y)

    @jax.jit
    def train_step(params, state, step, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params, state = adam_step(params, grads, state, step)
        return params, state, loss

    with jax.default_device(device):
        params = init(jax.random.PRNGKey(0))
        state = {"m": tree_map(jnp.zeros_like, params), "v": tree_map(jnp.zeros_like, params)}
        x, y = jax.device_put(x, device), jax.device_put(y, device)
        start = time.perf_counter()
        params, state, loss = train_step(params, state, 1.0, x, y)
        loss.block_until_ready()
        compile_time = time.perf_counter() - start
        losses, times = [float(loss)], []
        for step in range(2, steps + 1):
            start = time.perf_counter()
            params, state, loss = train_step(params, state, float(step), x, y)
            loss.block_until_ready()
            times.append(time.perf_counter() - start)
            losses.append(float(loss))
    return {"compile_s": compile_time, "step_ms_median": 1e3 * float(np.median(times)),
            "step_ms_p90": 1e3 * float(np.percentile(times, 90)),
            "loss_first": losses[0], "loss_last": losses[-1], "losses": losses}


def main(device_name, out_path, steps):
    device = jax.devices(device_name)[0]
    result = {"jax": jax.__version__, "device": str(device), "steps": steps, "models": {}}
    for name in MODELS:
        try:
            result["models"][name] = bench(name, device, steps)
            r = result["models"][name]
            print(f"{name:12} compile {r['compile_s']:6.2f}s  step {r['step_ms_median']:8.2f} ms  "
                  f"loss {r['loss_first']:.3f} -> {r['loss_last']:.3f}", flush=True)
        except Exception as error:  # record and continue
            result["models"][name] = {"error": str(error).splitlines()[0][:300]}
            print(f"{name:12} ERROR {result['models'][name]['error']}", flush=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 100
    main(sys.argv[1], sys.argv[2], steps)
