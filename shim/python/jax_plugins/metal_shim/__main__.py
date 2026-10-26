"""Command line for jax-metal-shim.

    python -m jax_plugins.metal_shim fetch [--force]   # download Apple's plugin (once)
    python -m jax_plugins.metal_shim info              # versions and library paths
"""

import argparse
import os
import sys

from jax_plugins.metal_shim import _plugin


def info():
    import jax  # pylint: disable=import-outside-toplevel
    import jaxlib  # pylint: disable=import-outside-toplevel

    from jax_plugins import metal_shim  # pylint: disable=import-outside-toplevel
    plugin = _plugin.find_plugin()
    verified = plugin is not None and _plugin.sha256(plugin) == _plugin.PLUGIN_SHA256
    rows = [
        ("jax-metal-shim", metal_shim.__version__),
        ("jax / jaxlib", f"{jax.__version__} / {jaxlib.__version__} "
                         f"(tested: {', '.join(metal_shim.TESTED_JAX)})"),
        ("shim library", f"{metal_shim.SHIM_LIBRARY} "
                         f"({'present' if metal_shim.SHIM_LIBRARY.exists() else 'MISSING'})"),
        ("Apple's plugin", f"{plugin} ({'verified' if verified else 'sha256 MISMATCH'})"
                           if plugin else "not found; run `python -m jax_plugins.metal_shim fetch`"),
        ("plugin cache", str(_plugin.cache_dir())),
        ("$JAX_METAL_SHIM_PLUGIN_PATH", os.environ.get("JAX_METAL_SHIM_PLUGIN_PATH", "(unset)")),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0 if plugin and verified and metal_shim.SHIM_LIBRARY.exists() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m jax_plugins.metal_shim",
                                     description="jax-metal-shim utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download Apple's pjrt_plugin_metal_14.dylib "
                                "(jax-metal 0.1.1 from PyPI) into the per-user cache")
    fetch.add_argument("--force", action="store_true", help="download again")
    commands.add_parser("info", help="show versions and where the libraries are")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        _plugin.fetch(force=args.force)
        return 0
    return info()


if __name__ == "__main__":
    sys.exit(main())
