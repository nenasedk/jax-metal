"""Locate Apple's pjrt_plugin_metal_14.dylib, and fetch it from PyPI on request.

The jax-metal-shim wheel does not contain Apple's library. `python -m jax_plugins.metal_shim
fetch` downloads the jax-metal 0.1.1 wheel from PyPI, checks the wheel against PyPI's digest
and the library against a pinned sha256, and stores the library (with Apple's LICENSE) in a
per-user cache. Lookup order: $JAX_METAL_SHIM_PLUGIN_PATH, this package's directory (source
installs), the cache.
"""

import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PLUGIN_NAME = "pjrt_plugin_metal_14.dylib"
PLUGIN_SHA256 = "e8e0656f24dc887507f5719926faf51f65546993809d3059add1a3c6c16961bf"
JAX_METAL_VERSION = "0.1.1"
WHEEL_NAME = f"jax_metal-{JAX_METAL_VERSION}-py3-none-macosx_13_0_arm64.whl"
WHEEL_PLUGIN = "jax_plugins/metal_plugin/" + PLUGIN_NAME
WHEEL_LICENSE = f"jax_metal-{JAX_METAL_VERSION}.dist-info/LICENSE"
PYPI_JSON = f"https://pypi.org/pypi/jax-metal/{JAX_METAL_VERSION}/json"
PACKAGE_DIR = Path(__file__).resolve().parent


def cache_dir():
    """$JAX_METAL_SHIM_CACHE, else ~/Library/Caches (macOS) or $XDG_CACHE_HOME, per version."""
    if os.environ.get("JAX_METAL_SHIM_CACHE"):
        return Path(os.environ["JAX_METAL_SHIM_CACHE"])
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "jax-metal-shim" / f"jax-metal-{JAX_METAL_VERSION}"


def find_plugin():
    """Path of Apple's library, or None. An explicit $JAX_METAL_SHIM_PLUGIN_PATH is final."""
    if os.environ.get("JAX_METAL_SHIM_PLUGIN_PATH"):
        path = Path(os.environ["JAX_METAL_SHIM_PLUGIN_PATH"])
        return path if path.is_file() else None
    for path in (PACKAGE_DIR / PLUGIN_NAME, cache_dir() / PLUGIN_NAME):
        if path.is_file():
            return path
    return None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wheel_url():
    """URL and PyPI's sha256 of the jax-metal wheel."""
    with urllib.request.urlopen(PYPI_JSON, timeout=60) as response:
        release = json.load(response)
    for entry in release["urls"]:
        if entry["filename"] == WHEEL_NAME:
            return entry["url"], entry["digests"]["sha256"]
    raise RuntimeError(f"{WHEEL_NAME} is not listed on PyPI ({PYPI_JSON})")


def fetch(force=False, log=print):
    """Download Apple's library into cache_dir() unless a verified copy is there."""
    target = cache_dir() / PLUGIN_NAME
    if target.is_file() and not force and sha256(target) == PLUGIN_SHA256:
        log(f"{target} is present and verified")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url, wheel_sha256 = _wheel_url()
    log(f"downloading {url}")
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        wheel = Path(scratch) / WHEEL_NAME
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=600) as response, open(wheel, "wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != wheel_sha256:
            raise RuntimeError(f"{WHEEL_NAME}: sha256 {digest.hexdigest()} does not match "
                               f"PyPI's {wheel_sha256}")
        with zipfile.ZipFile(wheel) as archive:
            archive.extract(WHEEL_PLUGIN, scratch)
            archive.extract(WHEEL_LICENSE, scratch)
        plugin = Path(scratch) / WHEEL_PLUGIN
        if sha256(plugin) != PLUGIN_SHA256:
            raise RuntimeError(f"{PLUGIN_NAME} from {WHEEL_NAME} has sha256 {sha256(plugin)}, "
                               f"expected {PLUGIN_SHA256}")
        os.replace(Path(scratch) / WHEEL_LICENSE, target.parent / "LICENSE-jax-metal")
        os.replace(plugin, target)
    log(f"installed {target} (Apple's license: {target.parent / 'LICENSE-jax-metal'})")
    return target
