"""Register Apple's jax-metal 0.1.1 plugin with current JAX through the PJRT shim.

JAX imports every module under the ``jax_plugins`` namespace and calls its
``initialize()``. This one registers platform ``METAL`` with
``pjrt_plugin_metal_shim.dylib``, which loads Apple's ``pjrt_plugin_metal_14.dylib``
from this directory (see shim/DESIGN.md).
"""

import importlib.util
import os
from pathlib import Path

import jax
import jax._src.xla_bridge as xb

from jax_plugins.metal_shim import _lowerings

_HERE = Path(__file__).resolve().parent
SHIM_LIBRARY = _HERE / "pjrt_plugin_metal_shim.dylib"
PLUGIN_LIBRARY = _HERE / "pjrt_plugin_metal_14.dylib"
__version__ = "0.1.0"


def initialize():
    """Register the METAL platform; called by JAX's plugin discovery."""
    # Apple's package registers METAL itself; two registrations conflict.
    if importlib.util.find_spec("jax_plugins.metal_plugin") is not None:
        raise RuntimeError(
            "jax-metal-shim and Apple's jax-metal package are both installed and both "
            "register the METAL platform. Run `pip uninstall jax-metal`; the shim already "
            "carries Apple's plugin library.")
    for library in (SHIM_LIBRARY, PLUGIN_LIBRARY):
        if not library.exists():
            raise RuntimeError(f"{library} is missing; install with shim/install.sh")
    # The plugin predates Shardy and cannot parse the `sdy` dialect; lower with GSPMD
    # shardings instead unless the user chose explicitly.
    if "JAX_USE_SHARDY_PARTITIONER" not in os.environ:
        jax.config.update("jax_use_shardy_partitioner", False)
    xb.register_plugin("METAL", priority=500, library_path=str(SHIM_LIBRARY))
    _lowerings.register()


def version():
    """Version of the shim package."""
    return __version__
