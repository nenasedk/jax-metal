"""Register Apple's jax-metal 0.1.1 plugin with current JAX through the PJRT shim.

JAX imports every module under the ``jax_plugins`` namespace and calls its
``initialize()``. This one registers platform ``METAL`` with
``pjrt_plugin_metal_shim.dylib``, which loads Apple's ``pjrt_plugin_metal_14.dylib``
(see shim/DESIGN.md). Apple's library is fetched once with
``python -m jax_plugins.metal_shim fetch``.
"""

import importlib.util
import os
import warnings
from pathlib import Path

import jax
import jax._src.xla_bridge as xb
import jaxlib

from jax_plugins.metal_shim import _boundary_io
from jax_plugins.metal_shim import _lowerings
from jax_plugins.metal_shim import _plugin

_HERE = Path(__file__).resolve().parent
SHIM_LIBRARY = _HERE / "pjrt_plugin_metal_shim.dylib"
__version__ = "0.1.0"
# jax/jaxlib releases this version was tested with; the package hooks JAX internals.
TESTED_JAX = ("0.12.0", "0.12.0.1")


def initialize():
    """Register the METAL platform; called by JAX's plugin discovery."""
    # Apple's package registers METAL itself; two registrations conflict.
    if importlib.util.find_spec("jax_plugins.metal_plugin") is not None:
        raise RuntimeError(
            "jax-metal-shim and Apple's jax-metal package are both installed and both "
            "register the METAL platform. Run `pip uninstall jax-metal`; the shim loads "
            "Apple's plugin library itself.")
    if not SHIM_LIBRARY.exists():
        raise RuntimeError(f"{SHIM_LIBRARY} is missing; reinstall jax-metal-shim")
    plugin = _plugin.find_plugin()
    if plugin is None:
        raise RuntimeError(
            "Apple's pjrt_plugin_metal_14.dylib was not found"
            + (f" at $JAX_METAL_SHIM_PLUGIN_PATH ({os.environ['JAX_METAL_SHIM_PLUGIN_PATH']})"
               if os.environ.get("JAX_METAL_SHIM_PLUGIN_PATH") else "")
            + ". Fetch it once with `python -m jax_plugins.metal_shim fetch`, which downloads "
            "jax-metal 0.1.1 from PyPI and keeps the library in a per-user cache.")
    os.environ["JAX_METAL_SHIM_PLUGIN_PATH"] = str(plugin)  # read by the shim when loaded
    for name, version in (("jax", jax.__version__), ("jaxlib", jaxlib.__version__)):
        if version not in TESTED_JAX:
            warnings.warn(f"jax-metal-shim {__version__} is tested with {name} "
                          f"{' and '.join(TESTED_JAX)}, not {version}; programs may fail "
                          "to lower on METAL", stacklevel=2)
    # The plugin predates Shardy and cannot parse the `sdy` dialect; lower with GSPMD
    # shardings instead unless the user chose explicitly.
    if "JAX_USE_SHARDY_PARTITIONER" not in os.environ:
        jax.config.update("jax_use_shardy_partitioner", False)
    xb.register_plugin("METAL", priority=500, library_path=str(SHIM_LIBRARY))
    _lowerings.register()
    _boundary_io.register(SHIM_LIBRARY)


def version():
    """Version of the shim package."""
    return __version__
