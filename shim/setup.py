"""Build hook for jax-metal-shim: compile pjrt_plugin_metal_shim.dylib into the package.

Metadata lives in pyproject.toml. The library is a plain C++ shared library (no Python
extension), so the wheel is tagged py3-none-<platform>: one wheel per platform serves every
Python 3. Apple's pjrt_plugin_metal_14.dylib is not part of the build; users fetch it from
PyPI (python -m jax_plugins.metal_shim fetch).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "python" / "jax_plugins" / "metal_shim"
LIBRARY = "pjrt_plugin_metal_shim.dylib"
# Apple's plugin needs macOS Sonoma 14.4+ on Apple silicon.
MACOS_PLATFORM = "macosx_14_0_arm64"


def package_version():
    text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.MULTILINE).group(1)


def compile_shim(out_dir):
    command = [os.environ.get("CXX", "clang++"), "-std=c++20", "-O2", "-Wall", "-Wextra",
               "-Wno-missing-field-initializers", "-fPIC", "-fvisibility=hidden", "-shared",
               f"-I{HERE / 'third_party' / 'xla_0bfeeb47'}",
               f'-DJAX_METAL_SHIM_VERSION="{package_version()}"',
               str(HERE / "src" / "pjrt_plugin_metal_shim.cc"), "-o", str(out_dir / LIBRARY)]
    if sys.platform == "darwin":
        command += ["-arch", "arm64", "-mmacosx-version-min=14.0",
                    "-install_name", f"@rpath/{LIBRARY}"]
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


class BuildPy(build_py):
    """Python sources as usual, plus the compiled shim library next to them."""

    def run(self):
        super().run()
        compile_shim(PACKAGE if self.editable_mode else
                     Path(self.build_lib) / "jax_plugins" / "metal_shim")


class BinaryDistribution(Distribution):
    """Platform-specific, though without Python extensions."""

    def has_ext_modules(self):
        return True


class BdistWheel(bdist_wheel):
    """Platform wheel for any Python 3: the package holds a native library, no extension."""

    def finalize_options(self):
        if sys.platform == "darwin" and not self.plat_name:
            self.plat_name = MACOS_PLATFORM  # the library is arm64-only
        super().finalize_options()

    def get_tag(self):
        return ("py3", "none", super().get_tag()[2])


setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
