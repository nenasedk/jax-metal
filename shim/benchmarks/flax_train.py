"""Train Flax (Linen) models with Optax on METAL and on the CPU and compare.

    .venv-flax/bin/python shim/benchmarks/flax_train.py OUT.json [--steps N]

Both devices start from the same parameters and data, so their loss curves should track each
other closely (they are not bit-identical: float32 reductions are ordered differently).
Reports the largest relative gap between the curves, final losses and step times.
"""

import json
import sys
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        for features in (32, 64):
            x = nn.Conv(features, (3, 3))(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), (2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dropout(0.1, deterministic=True)(nn.relu(nn.Dense(128)(x)))
        return nn.Dense(10)(x)


class Block(nn.Module):
    heads: int
    dim: int

    @nn.compact
    def __call__(self, x):
        mask = nn.make_causal_mask(jnp.ones(x.shape[:2]))
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads)(nn.LayerNorm()(x), mask=mask)
        h = nn.Dense(4 * self.dim)(nn.LayerNorm()(x))
        return x + nn.Dense(self.dim)(nn.gelu(h))


class TinyGPT(nn.Module):
    vocab: int = 256
    dim: int = 128
    layers: int = 2

    @nn.compact
    def __call__(self, tokens):
        x = nn.Embed(self.vocab, self.dim)(tokens)
        x = x + self.param("pos", nn.initializers.normal(0.02), (tokens.shape[1], self.dim))
        for _ in range(self.layers):
            x = Block(heads=4, dim=self.dim)(x)
        return nn.Dense(self.vocab)(nn.LayerNorm()(x))


def data(name, rng):
    if name == "cnn":
        x = rng.standard_normal((64, 28, 28, 3)).astype(np.float32)
        teacher = rng.standard_normal((28 * 28 * 3, 10)).astype(np.float32)
        return x, (x.reshape(64, -1) @ teacher).argmax(-1).astype(np.int32)
    tokens = rng.integers(0, 256, (16, 65)).astype(np.int32)
    tokens[:, 1:] = (tokens[:, :-1] * 5 + 1) % 256
    return tokens[:, :-1], tokens[:, 1:]


def train(model, x, y, device, steps):
    optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    with jax.default_device(jax.devices("cpu")[0]):
        params = model.init(jax.random.PRNGKey(0), x[:1])  # identical start on every device

    def loss_fn(params, x, y):
        logits = model.apply(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    with jax.default_device(device):
        params = jax.device_put(params, device)
        opt_state = optimizer.init(params)
        x, y = jax.device_put((x, y), device)
        start = time.perf_counter()
        params, opt_state, loss = step(params, opt_state, x, y)
        losses = [float(loss)]
        compile_s = time.perf_counter() - start
        times = []
        for _ in range(steps - 1):
            start = time.perf_counter()
            params, opt_state, loss = step(params, opt_state, x, y)
            loss.block_until_ready()
            times.append(time.perf_counter() - start)
            losses.append(float(loss))
    return {"compile_s": compile_s, "step_ms_median": 1e3 * float(np.median(times)), "losses": losses}


def main(out_path, steps):
    results = {"jax": jax.__version__, "steps": steps}
    for name, model in (("cnn", CNN()), ("tiny_gpt", TinyGPT())):
        x, y = data(name, np.random.default_rng(0))
        runs = {d: train(model, x, y, jax.devices(d)[0], steps) for d in ("METAL", "cpu")}
        metal, cpu = np.array(runs["METAL"]["losses"]), np.array(runs["cpu"]["losses"])
        gap = float(np.max(np.abs(metal - cpu) / np.maximum(np.abs(cpu), 1e-3)))
        results[name] = {"runs": runs, "max_relative_loss_gap": gap}
        print(f"{name:9} loss {cpu[0]:.3f} -> metal {metal[-1]:.4f} / cpu {cpu[-1]:.4f}  max gap {gap:.2e}  "
              f"step metal {runs['METAL']['step_ms_median']:.2f} ms / cpu {runs['cpu']['step_ms_median']:.2f} ms",
              flush=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 100
    main(sys.argv[1], steps)
