"""Lower a JAX program with current JAX and write it as a StableHLO 0.19.4 portable artifact.

Run with the .venv-latest interpreter (jax/jaxlib 0.12):

    .venv-latest/bin/python shim/tests/gen_artifact.py OUT_DIR [TARGET_VERSION]

Writes program.mlirbc, compile_options.pb, input/expected .bin files and manifest.txt,
which shim/tests/direct_load.c reads.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # lowering only; never touch the Metal plugin

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import compiler
from jaxlib.mlir.dialects import stablehlo


def program(x, y):
    return jnp.sin(x) @ y + 1.0


def main(out_dir, target_version):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8), dtype=np.float32)
    y = rng.standard_normal((8, 3), dtype=np.float32)
    expected = np.asarray(program(x, y), dtype=np.float32)

    module_text = jax.jit(program).lower(x, y).as_text()
    with open(os.path.join(out_dir, "program.mlir"), "w") as handle:
        handle.write(module_text)
    artifact = stablehlo.serialize_portable_artifact_str(module_text, target_version)
    with open(os.path.join(out_dir, "program.mlirbc"), "wb") as handle:
        handle.write(artifact)

    options = compiler.get_compile_options(num_replicas=1, num_partitions=1)
    with open(os.path.join(out_dir, "compile_options.pb"), "wb") as handle:
        handle.write(options.SerializeAsString())

    manifest = []
    for role, name, array in (("in", "x", x), ("in", "y", y), ("out", "expected", expected)):
        path = os.path.abspath(os.path.join(out_dir, name + ".bin"))
        array.tofile(path)
        dims = " ".join(str(d) for d in array.shape)
        manifest.append(f"{role} f32 {array.ndim} {dims} {path}")
    with open(os.path.join(out_dir, "manifest.txt"), "w") as handle:
        handle.write("\n".join(manifest) + "\n")

    print(f"jax {jax.__version__}, StableHLO current {stablehlo.get_current_version()}, "
          f"target {target_version}, artifact {len(artifact)} bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0.19.4")
