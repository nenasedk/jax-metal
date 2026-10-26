#!/bin/zsh
# Install jax-metal-shim from this checkout into the current (or given) Python: pip compiles
# the shim library (needs Xcode command-line tools), then Apple's plugin library is fetched
# from PyPI (jax-metal 0.1.1, checksum-verified) into the per-user cache.
#
#   shim/install.sh                 # uses `python3` on PATH
#   PYTHON=.venv/bin/python shim/install.sh

set -euo pipefail

SHIM_DIR=${0:A:h}
PYTHON=${PYTHON:-python3}

if "$PYTHON" -c '
import importlib.util, sys
try:
    found = importlib.util.find_spec("jax_plugins.metal_plugin") is not None
except ModuleNotFoundError:  # no jax_plugins namespace at all
    found = False
sys.exit(0 if found else 1)'; then
  echo "Apple's jax-metal package is installed in $PYTHON; uninstall it first: $PYTHON -m pip uninstall jax-metal" >&2
  exit 1
fi

"$PYTHON" -m pip install --quiet "$SHIM_DIR"
"$PYTHON" -m jax_plugins.metal_shim fetch
"$PYTHON" -m jax_plugins.metal_shim info
