#!/bin/zsh
# Build the shim, fetch Apple's plugin library from the jax-metal 0.1.1 wheel on PyPI,
# and install the jax-metal-shim package into the current (or given) Python.
#
#   shim/install.sh                 # uses `python3` on PATH
#   PYTHON=.venv/bin/python shim/install.sh

set -euo pipefail

SHIM_DIR=${0:A:h}
PYTHON=${PYTHON:-python3}
PACKAGE_DIR=$SHIM_DIR/python/jax_plugins/metal_shim
PLUGIN_SHA256=e8e0656f24dc887507f5719926faf51f65546993809d3059add1a3c6c16961bf

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

make -C "$SHIM_DIR"
cp "$SHIM_DIR/build/pjrt_plugin_metal_shim.dylib" "$PACKAGE_DIR/"

plugin=$PACKAGE_DIR/pjrt_plugin_metal_14.dylib
if [[ ! -f $plugin ]] || [[ $(shasum -a 256 "$plugin" | cut -d' ' -f1) != "$PLUGIN_SHA256" ]]; then
  download=$(mktemp -d)
  trap 'rm -rf "$download"' EXIT
  "$PYTHON" -m pip download --quiet --no-deps --only-binary :all: \
    --platform macosx_13_0_arm64 jax-metal==0.1.1 -d "$download"
  unzip -q -j "$download"/jax_metal-0.1.1-*.whl \
    'jax_plugins/metal_plugin/pjrt_plugin_metal_14.dylib' -d "$download/lib"
  actual=$(shasum -a 256 "$download/lib/pjrt_plugin_metal_14.dylib" | cut -d' ' -f1)
  if [[ $actual != "$PLUGIN_SHA256" ]]; then
    echo "unexpected pjrt_plugin_metal_14.dylib sha256 $actual (expected $PLUGIN_SHA256)" >&2
    exit 1
  fi
  cp "$download/lib/pjrt_plugin_metal_14.dylib" "$plugin"
fi

"$PYTHON" -m pip install --quiet "$SHIM_DIR/python"
echo "installed jax-metal-shim into $("$PYTHON" -c 'import sys; print(sys.prefix)')"
