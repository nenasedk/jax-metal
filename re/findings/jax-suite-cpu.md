# JAX's tests on CPU in METAL emulation

Generated from a run of `shim/tests/jax_suite/cpu_rules_plugin.py` (jax 0.12.0.1,
`JAX_NUM_GENERATED_CASES=10`): the CPU backend lowers programs as METAL does (the shim's
rules, JAX's generic lowerings, METAL test selection with one device), and `metal_check.py`
predicts what Apple's plugin could not compile. Predictions are confirmed or refuted by a
METAL run.

Run on 2026-10-26 on Linux with the runner, three pytest-xdist workers, over the 56 files in
`DEFAULT_FILES` (19 core, 37 extended); the narrow-type and complex-reduction tests were run
again after those rules landed. The results, the predictions and each test's cost (seconds,
memory, parameters) are in `shim/build/jax-suite-cpu/`, which the GPU session compares
METAL's results with:

```sh
python shim/tests/jax_suite/run_jax_tests.py re/work/jax-src shim/build/jax-suite-cpu \
  --cpu --workers 3        # JAX_SUITE_TIMEOUT=300 for slow machines
```

## What remains, and why

All 35 failures are understood:

- By design (13). `lax_test`: 11 `CompositeTest`s assert `stablehlo.composite` in the lowered
  program, which METAL replaces by the decomposition (the plugin cannot expand composites);
  `testDotAlgorithmConfig` asserts the dot algorithm attribute and
  `testDotAlgorithmInvalidFloat8Type` expects JAX's algorithm validation, both of which METAL
  skips by dropping the algorithm (it cannot be serialized for StableHLO 0.19.4).
- Emulation artifacts (10). `lax_control_flow_test::test_platform_dependent*` (6):
  `lax.platform_dependent` picks the branch for the platform the program is lowered for,
  `cpu` here, while the tests expect the METAL branch. `api_test` (4): three tests ask for
  the METAL backend by name, and `test_print_environment_info` reads the backend's version.
- Callbacks and effects (5): `checkify`'s `check` (2), `debug_callback`
  (`custom_root_test`), `debug_print` (`shard_map_test`) and ordered effects in
  `custom_linear_solve`. METAL has no host callbacks (the plugin has no send/recv).
- `eigh` accuracy and size (7): `eigh_test` `testEigh`, `testEighRankDeficient`,
  `testEighTinyNorm` and `testEighBatching` (one of them a complex 300 × 300 matrix, above the
  Jacobi limit of 256) and two `lax_scipy_spectral_dac_test` cases: the METAL `eigh` (Jacobi
  for n ≤ 256, QDWH above) misses JAX's tightest tolerances on rank-deficient and tiny-norm
  inputs.

Predicted METAL issues (79 tests, all documented limits):

- float64 and complex128 (43): MPS has no float64.
- Newer narrow types as program inputs or outputs (28: float8_e4m3, e3m4, e8m0fnu,
  float4_e2m1fn, float6): StableHLO 0.19.4 has no such types, so jaxlib cannot serialize the
  program for the plugin. Inside a program they are emulated (`_narrow.py`); float6 is out of
  scope (XLA's CPU backend crashes on it).
- `stablehlo.create_token` (5, `pjit_test` token inputs), `annotate_device_placement`
  (`compute_on`), `dce_sink` (a testing aid), and one extended dtype represented as float8
  (`test_custom_tangent_dtype`), which keeps a native float8 value.

The narrow-type rewrite took the tests with predicted METAL issues among JAX's narrow-type
tests from 88 to 20 (the rest: the newer types above), and the complex `max`/`min` reduction
rule removed 24 more (a nine-op comparator body the plugin cannot match). Not visible on
CPU: MPS crashes, MPS accuracy and the plugin's per-pattern limits. Those need the METAL run
(`shim/tests/jax_suite/run_jax_tests.py`).

| File | passed | failed | skipped |
|---|---|---|---|
| `fft_test` | 109 | 0 | 2 |
| `random_test` | 238 | 0 | 16 |
| `lax_test` | 2525 | 13 | 76 |
| `lax_numpy_operators_test` | 1626 | 0 | 2 |
| `lax_numpy_reducers_test` | 1043 | 0 | 4 |
| `lax_numpy_indexing_test` | 390 | 0 | 0 |
| `lax_autodiff_test` | 744 | 0 | 1 |
| `lax_control_flow_test` | 539 | 6 | 483 |
| `nn_test` | 387 | 0 | 56 |
| `lax_vmap_test` | 318 | 0 | 10 |
| `lax_vmap_op_test` | 405 | 0 | 295 |
| `lax_numpy_einsum_test` | 176 | 0 | 0 |
| `random_lax_test` | 649 | 0 | 60 |
| `lax_scipy_special_functions_test` | 467 | 0 | 0 |
| `lax_scipy_test` | 103 | 0 | 18 |
| `scipy_signal_test` | 76 | 0 | 24 |
| `ann_test` | 32 | 0 | 14 |
| `linalg_test` | 787 | 0 | 193 |
| `lax_numpy_test` | 3311 | 0 | 37 |
| `lax_numpy_ufuncs_test` | 207 | 0 | 8 |
| `lax_numpy_setops_test` | 155 | 0 | 0 |
| `lax_numpy_vectorize_test` | 35 | 0 | 0 |
| `batching_test` | 232 | 0 | 0 |
| `checkify_test` | 112 | 2 | 4 |
| `custom_linear_solve_test` | 13 | 1 | 2 |
| `custom_root_test` | 22 | 1 | 0 |
| `x64_context_test` | 23 | 0 | 2 |
| `dtypes_test` | 834 | 0 | 31 |
| `random_impl_test` | 60 | 0 | 0 |
| `mutable_array_test` | 184 | 0 | 34 |
| `image_test` | 57 | 0 | 30 |
| `ode_test` | 14 | 0 | 0 |
| `stax_test` | 71 | 0 | 0 |
| `optimizers_test` | 24 | 0 | 0 |
| `experimental_rnn_test` | 10 | 0 | 12 |
| `scipy_stats_test` | 1001 | 0 | 0 |
| `scipy_fft_test` | 46 | 0 | 1 |
| `scipy_ndimage_test` | 16 | 0 | 0 |
| `scipy_interpolate_test` | 4 | 0 | 0 |
| `scipy_optimize_test` | 13 | 0 | 1 |
| `scipy_spatial_test` | 71 | 0 | 0 |
| `lax_scipy_sparse_test` | 55 | 0 | 36 |
| `sparse_test` | 410 | 0 | 54 |
| `sparse_bcoo_bcsr_test` | 420 | 0 | 25 |
| `polynomial_test` | 20 | 0 | 0 |
| `eigh_test` | 79 | 5 | 11 |
| `svd_test` | 65 | 0 | 27 |
| `qdwh_test` | 46 | 0 | 0 |
| `lobpcg_test` | 55 | 0 | 0 |
| `lax_scipy_spectral_dac_test` | 3 | 2 | 5 |
| `cholesky_update_test` | 2 | 0 | 2 |
| `shard_map_test` | 67 | 1 | 420 |
| `pmap_test` | 213 | 0 | 91 |
| `pjit_test` | 138 | 0 | 750 |
| `array_test` | 70 | 0 | 168 |
| `api_test` | 836 | 4 | 175 |
| **total** | 19608 | 35 | 3180 |

## Failures on CPU (by test)

| File | Test | Count |
|---|---|---|
| `api_test` | `test_device_to_device_copy_between_backends` | 1 |
| `api_test` | `test_explicit_backend` | 1 |
| `api_test` | `test_mismatched_nested_backends` | 1 |
| `api_test` | `test_print_environment_info` | 1 |
| `checkify_test` | `test_assert_primitive_lowering` | 1 |
| `checkify_test` | `test_debug_check_noop` | 1 |
| `custom_linear_solve_test` | `test_custom_linear_solve_ordered_effects` | 1 |
| `custom_root_test` | `test_custom_root_solve_runs_once` | 1 |
| `eigh_test` | `testEigh` | 1 |
| `eigh_test` | `testEighBatching` | 2 |
| `eigh_test` | `testEighRankDeficient` | 1 |
| `eigh_test` | `testEighTinyNorm` | 1 |
| `lax_control_flow_test` | `test_platform_dependent` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_batched` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_grad` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_lowering` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_multiple_identical_branches` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_no_args` | 1 |
| `lax_scipy_spectral_dac_test` | `test_spectral_dac_eigh` | 2 |
| `lax_test` | `testDotAlgorithmConfig` | 1 |
| `lax_test` | `testDotAlgorithmInvalidFloat8Type` | 1 |
| `lax_test` | `test_composite` | 1 |
| `lax_test` | `test_composite_attribute_dtypes` | 1 |
| `lax_test` | `test_composite_decorator` | 1 |
| `lax_test` | `test_composite_with_array_consts` | 1 |
| `lax_test` | `test_composite_with_attributes` | 1 |
| `lax_test` | `test_composite_with_attributes_unstable` | 1 |
| `lax_test` | `test_composite_with_jit_function` | 1 |
| `lax_test` | `test_composite_with_no_args` | 1 |
| `lax_test` | `test_composite_with_non_default_version` | 1 |
| `lax_test` | `test_composite_with_symbolic_shapes` | 1 |
| `lax_test` | `test_composite_with_variadic_input_output` | 1 |
| `shard_map_test` | `test_debug_print_eager` | 1 |

## Predicted METAL issues (tests that passed or failed on CPU)

| Tests | Passed on CPU | Issue | Example |
|---|---|---|---|
| 42 | 42 | type f64 | `tests/dtypes_test.py::TestPromotionTables::testFloatArrayCreation3` |
| 28 | 28 | serialize to StableHLO 0.19.4: INVALID_ARGUMENT: Failed to serialize StableHLO to plugin version 0.19.4; | `tests/dtypes_test.py::TestPromotionTables::testArrayRepr48` |
| 5 | 5 | op stablehlo.create_token | `tests/pjit_test.py::ArrayPjitTest::test_jit_token_input` |
| 1 | 1 | custom_call annotate_device_placement | `tests/api_test.py::APITest::test_cache_miss_explanations_compute_on` |
| 1 | 1 | custom_call dce_sink | `tests/lax_test.py::LaxTest::test_dce_sink_prevents_xla_dce` |
| 1 | 1 | type complex<f64> | `tests/lax_numpy_test.py::LaxBackedNumpyTests::testArrayExplicitDtypes` |
| 1 | 1 | type f8E5M2 | `tests/dtypes_test.py::ExtendedDTypeTest::test_custom_tangent_dtype` |
