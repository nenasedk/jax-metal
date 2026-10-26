# Releasing jax-metal-shim

## Before the first public release

- Choose the project license: add `LICENSE` at the repository root and in `shim/` (the
  package root, so the sdist and wheel carry it), and set `license` in `shim/pyproject.toml`.
- Decide what of `re/` stays public (it holds decompiled excerpts of Apple's plugin).
- One-time PyPI setup: on pypi.org, add a trusted publisher for project `jax-metal-shim`
  (owner `nenasedk`, repository `jax-metal`, workflow `release.yml`, environment `pypi`). In
  the GitHub repository, create the environment `pypi` with yourself as required reviewer, so
  that publishing waits for an approval.

## Each release

1. **Version.** Set `__version__` in `shim/python/jax_plugins/metal_shim/__init__.py`. It is
   the only copy: `pyproject.toml` reads it, and `setup.py` and the `Makefile` pass it to the
   C++ shim. If the release supports a new JAX, update `TESTED_JAX` there and the `jax`/`jaxlib`
   pins in `shim/pyproject.toml`.
2. **Changelog.** Move the notes in `CHANGELOG.md` under the new version.
3. **CPU checks** (any machine): JAX's tests through the METAL rules on CPU
   (`shim/tests/jax_suite/cpu_rules_plugin.py`, see `shim/README.md`). Only known failures may
   remain.
4. **GPU validation** on an Apple-silicon Mac, every command through `shim/tests/guard.py`:
   `make -C shim test`, `test_jax_e2e.py`, `test_phase3.py`, `test_complex.py`,
   `accuracy_sweep.py --strict`, `probe_jax_ops.py` (all PASS), and JAX's test files with
   `run_jax_tests.py`. Commit the probe results and `shim/build/jax-suite/summary.md`.
5. **Build and smoke-test locally** on the Mac:

   ```sh
   python -m build --outdir dist shim
   python -m twine check dist/*
   python -m venv /tmp/release-check && /tmp/release-check/bin/pip install dist/*.whl
   /tmp/release-check/bin/python -m jax_plugins.metal_shim fetch
   /tmp/release-check/bin/python -c "import jax; print(jax.devices()); print(jax.numpy.ones(3) * 2)"
   ```

   The wheel is `jax_metal_shim-<version>-py3-none-macosx_14_0_arm64.whl` and contains
   `pjrt_plugin_metal_shim.dylib` (arm64) but never Apple's `pjrt_plugin_metal_14.dylib`.
6. **Tag and push** `v<version>` on `main`. `.github/workflows/release.yml` builds the sdist
   and wheel on an Apple-silicon runner, checks them, drafts a GitHub release with both
   files, and waits for approval of the `pypi` environment before publishing to PyPI.
7. Review and publish the draft GitHub release.
