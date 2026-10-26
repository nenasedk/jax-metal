# JAX's tests on CPU in METAL emulation

Generated from a run of `shim/tests/jax_suite/cpu_rules_plugin.py` (jax 0.12.0.1,
`JAX_NUM_GENERATED_CASES=10`): the CPU backend lowers programs as METAL does (the shim's
rules, JAX's generic lowerings, METAL test selection), and `metal_check.py` predicts what
Apple's plugin could not compile. Predictions are confirmed or refuted by a METAL run.

Run on 2026-10-26 on Linux with the runner, three pytest-xdist workers; the results, the
predictions and each test's cost (seconds, memory, parameters) are in
`shim/build/jax-suite-cpu/`, which the GPU session compares METAL's results with:

```sh
python shim/tests/jax_suite/run_jax_tests.py re/work/jax-src shim/build/jax-suite-cpu \
  --cpu --workers 3        # JAX_SUITE_TIMEOUT=300 for slow machines
```

The counts below were the same before and after the NaN-propagation rules for max/min
(`re/findings/mps-nan-semantics.md`), which are no-ops where max already propagates NaN.

## What remains, and why

- `linalg_test::testTridiagonal` (10): `lax.linalg.tridiagonal` has no METAL lowering (JAX has
  only LAPACK/cuSOLVER ones). Likewise `eig`, `schur`, `hessenberg`, `geqp3`, `ormqr` and
  `tridiagonal_solve`, whose tests JAX already skips on METAL.
- `lax_test` (13), by design: 11 `CompositeTest`s assert `stablehlo.composite` in the lowered
  program, which METAL replaces by the decomposition (the plugin cannot expand composites);
  `testDotAlgorithmConfig` asserts the dot algorithm attribute and
  `testDotAlgorithmInvalidFloat8Type` expects JAX's algorithm validation, both of which METAL
  skips by dropping the algorithm (it cannot be serialized for StableHLO 0.19.4).
- `lax_control_flow_test::test_platform_dependent*` (6): an emulation artifact.
  `lax.platform_dependent` picks the branch for the platform the program is lowered for,
  `cpu` here, while the tests expect the METAL branch.
- Predicted METAL issues are all documented limits: types MPS lacks (int4/uint4, float8
  variants, float64, complex128; 4 tests with newer float8/float4 types fail to serialize for
  StableHLO 0.19.4 for the same reason), uint64 program inputs and outputs, and
  `lax.dce_sink`'s custom call (a testing aid).
- Not visible on CPU: MPS crashes, MPS accuracy and the plugin's per-pattern limits. Those
  need the METAL run (`shim/tests/jax_suite/run_jax_tests.py`).

| File | passed | failed | skipped |
|---|---|---|---|
| `fft_test` | 109 | 0 | 2 |
| `linalg_test` | 777 | 10 | 193 |
| `random_test` | 238 | 0 | 16 |
| `lax_test` | 2525 | 13 | 76 |
| `lax_autodiff_test` | 744 | 0 | 1 |
| `lax_control_flow_test` | 539 | 6 | 483 |
| `lax_numpy_reducers_test` | 1043 | 0 | 4 |
| `lax_numpy_operators_test` | 1626 | 0 | 2 |
| `nn_test` | 387 | 0 | 56 |
| `lax_scipy_special_functions_test` | 467 | 0 | 0 |
| `lax_numpy_indexing_test` | 390 | 0 | 0 |
| `lax_numpy_einsum_test` | 176 | 0 | 0 |
| `lax_vmap_test` | 318 | 0 | 10 |
| `lax_vmap_op_test` | 405 | 0 | 295 |
| `random_lax_test` | 649 | 0 | 60 |
| `lax_scipy_test` | 103 | 0 | 18 |
| `scipy_signal_test` | 76 | 0 | 24 |
| `ann_test` | 32 | 0 | 14 |
| `lax_numpy_test` | 3311 | 0 | 37 |
| **total** | 13915 | 29 | 1291 |

## Failures on CPU (by test)

| File | Test | Count |
|---|---|---|
| `lax_control_flow_test` | `test_platform_dependent` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_batched` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_grad` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_lowering` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_multiple_identical_branches` | 1 |
| `lax_control_flow_test` | `test_platform_dependent_no_args` | 1 |
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
| `linalg_test` | `testTridiagonal` | 10 |

## Predicted METAL issues (tests that passed or failed on CPU)

| Tests | Passed on CPU | Issue | Example |
|---|---|---|---|
| 18 | 18 | type ui4 | `tests/lax_test.py::LaxTest::testBitcastConvertType2` |
| 15 | 15 | type i4 | `tests/lax_test.py::LaxTest::testBitcastConvertType1` |
| 8 | 8 | type f8E5M2 | `tests/lax_test.py::LaxTest::test_mixed_fp8_dot_general0` |
| 7 | 6 | type f8E4M3FN | `tests/lax_test.py::LaxTest::testDotAlgorithmInvalidFloat8Type` |
| 4 | 4 | uint64 program input or output | `tests/random_test.py::PrngTest::testRngRandomBitsShapeDtype0` |
| 4 | 4 | type f64 | `tests/lax_autodiff_test.py::LaxAutodiffTest::testAsinhAcoshGradLargeValuesFloat640` |
| 4 | 4 | serialize to StableHLO 0.19.4: INVALID_ARGUMENT: Failed to serialize StableHLO to plugin version 0.19.4; | `tests/lax_numpy_operators_test.py::JaxNumpyOperatorTests::testIsinfNonstandardDtypes0` |
| 1 | 1 | custom_call dce_sink | `tests/lax_test.py::LaxTest::test_dce_sink_prevents_xla_dce` |
| 1 | 1 | type f8E4M3B11FNUZ | `tests/lax_numpy_operators_test.py::JaxNumpyOperatorTests::testIsinfNonstandardDtypes1` |
| 1 | 1 | type f8E5M2FNUZ | `tests/lax_numpy_operators_test.py::JaxNumpyOperatorTests::testIsinfNonstandardDtypes2` |
| 1 | 1 | type f8E4M3FNUZ | `tests/lax_numpy_operators_test.py::JaxNumpyOperatorTests::testIsinfNonstandardDtypes8` |
| 1 | 1 | type complex<f64> | `tests/lax_numpy_test.py::LaxBackedNumpyTests::testArrayExplicitDtypes` |
