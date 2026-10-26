# JAX test suite on CPU, emulating METAL

| File | passed | failed | skipped | crashed |
|---|---|---|---|---|
| `tests/ann_test.py` | 32 | 0 | 14 | 0 |
| `tests/api_test.py` | 836 | 4 | 175 | 0 |
| `tests/array_test.py` | 70 | 0 | 168 | 0 |
| `tests/batching_test.py` | 232 | 0 | 0 | 0 |
| `tests/checkify_test.py` | 112 | 2 | 4 | 0 |
| `tests/cholesky_update_test.py` | 2 | 0 | 2 | 0 |
| `tests/custom_linear_solve_test.py` | 13 | 1 | 2 | 0 |
| `tests/custom_root_test.py` | 22 | 1 | 0 | 0 |
| `tests/dtypes_test.py` | 834 | 0 | 31 | 0 |
| `tests/eigh_test.py` | 79 | 5 | 11 | 0 |
| `tests/experimental_rnn_test.py` | 10 | 0 | 12 | 0 |
| `tests/fft_test.py` | 109 | 0 | 2 | 0 |
| `tests/image_test.py` | 57 | 0 | 30 | 0 |
| `tests/lax_autodiff_test.py` | 744 | 0 | 1 | 0 |
| `tests/lax_control_flow_test.py` | 539 | 6 | 483 | 0 |
| `tests/lax_numpy_einsum_test.py` | 176 | 0 | 0 | 0 |
| `tests/lax_numpy_indexing_test.py` | 390 | 0 | 0 | 0 |
| `tests/lax_numpy_operators_test.py` | 1626 | 0 | 2 | 0 |
| `tests/lax_numpy_reducers_test.py` | 1043 | 0 | 4 | 0 |
| `tests/lax_numpy_setops_test.py` | 155 | 0 | 0 | 0 |
| `tests/lax_numpy_test.py` | 3311 | 0 | 37 | 0 |
| `tests/lax_numpy_ufuncs_test.py` | 207 | 0 | 8 | 0 |
| `tests/lax_numpy_vectorize_test.py` | 35 | 0 | 0 | 0 |
| `tests/lax_scipy_sparse_test.py` | 55 | 0 | 36 | 0 |
| `tests/lax_scipy_special_functions_test.py` | 467 | 0 | 0 | 0 |
| `tests/lax_scipy_spectral_dac_test.py` | 3 | 2 | 5 | 0 |
| `tests/lax_scipy_test.py` | 103 | 0 | 18 | 0 |
| `tests/lax_test.py` | 2525 | 13 | 76 | 0 |
| `tests/lax_vmap_op_test.py` | 405 | 0 | 295 | 0 |
| `tests/lax_vmap_test.py` | 318 | 0 | 10 | 0 |
| `tests/linalg_test.py` | 787 | 0 | 193 | 0 |
| `tests/lobpcg_test.py` | 55 | 0 | 0 | 0 |
| `tests/mutable_array_test.py` | 184 | 0 | 34 | 0 |
| `tests/nn_test.py` | 387 | 0 | 56 | 0 |
| `tests/ode_test.py` | 14 | 0 | 0 | 0 |
| `tests/optimizers_test.py` | 24 | 0 | 0 | 0 |
| `tests/pjit_test.py` | 138 | 0 | 750 | 0 |
| `tests/pmap_test.py` | 213 | 0 | 91 | 0 |
| `tests/polynomial_test.py` | 20 | 0 | 0 | 0 |
| `tests/qdwh_test.py` | 46 | 0 | 0 | 0 |
| `tests/random_impl_test.py` | 60 | 0 | 0 | 0 |
| `tests/random_lax_test.py` | 649 | 0 | 60 | 0 |
| `tests/random_test.py` | 238 | 0 | 16 | 0 |
| `tests/scipy_fft_test.py` | 46 | 0 | 1 | 0 |
| `tests/scipy_interpolate_test.py` | 4 | 0 | 0 | 0 |
| `tests/scipy_ndimage_test.py` | 16 | 0 | 0 | 0 |
| `tests/scipy_optimize_test.py` | 13 | 0 | 1 | 0 |
| `tests/scipy_signal_test.py` | 76 | 0 | 24 | 0 |
| `tests/scipy_spatial_test.py` | 71 | 0 | 0 | 0 |
| `tests/scipy_stats_test.py` | 1001 | 0 | 0 | 0 |
| `tests/shard_map_test.py` | 67 | 1 | 420 | 0 |
| `tests/sparse_bcoo_bcsr_test.py` | 420 | 0 | 25 | 0 |
| `tests/sparse_test.py` | 410 | 0 | 54 | 0 |
| `tests/stax_test.py` | 71 | 0 | 0 | 0 |
| `tests/svd_test.py` | 65 | 0 | 27 | 0 |
| `tests/x64_context_test.py` | 23 | 0 | 2 | 0 |
| **total** | 19608 | 35 | 3180 | 0 |

## Failure groups (most common first)

| Count | Outcome | Message (normalized) | Example |
|---|---|---|---|
| 8 | failed | AssertionError:  | `tests/custom_linear_solve_test.py::CustomLinearSolveTest::test_custom_linear_solve_ordered_effects` |
| 3 | failed | RuntimeError: Unknown backend METAL. Available backends are ['cpu'] | `tests/api_test.py::JitTest::test_device_to_device_copy_between_backends` |
| 2 | failed | NotImplementedError: MLIR translation rule for primitive 'check' not found for platform cpu | `tests/checkify_test.py::AssertPrimitiveTests::test_assert_primitive_lowering` |
| 2 | failed | AssertionError: np.floatN(N) not less than or equal to np.floatN(N) | `tests/eigh_test.py::EighTest::testEighRankDeficient2` |
| 2 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my_square_impl attri | `tests/lax_test.py::CompositeTest::test_composite` |
| 1 | failed | AssertionError: assert 'jax:    N.N' in '' | `tests/api_test.py::EnvironmentInfoTest::test_print_environment_info1` |
| 1 | failed | NotImplementedError: MLIR translation rule for primitive 'debug_callback' not found for platform cpu | `tests/custom_root_test.py::CustomRootTest::test_custom_root_solve_runs_once` |
| 1 | failed | NotImplementedError: METAL: Jacobi eigendecomposition of size N exceeds N (larger ones can stall the GPU and WindowServer); complex eigh<path> is limited to thi | `tests/eigh_test.py::EighTest::testEighBatching3` |
| 1 | failed | AssertionError: np.floatN(N) not less than N | `tests/eigh_test.py::EighTest::testEighBatching6` |
| 1 | failed | AssertionError: False != True | `tests/lax_control_flow_test.py::LaxControlFlowTest::test_platform_dependent_grad` |
| 1 | failed | AssertionError: 'stablehlo.sine' unexpectedly found in 'module @jit_f attributes {mhlo.num_partitions = N : iN, mhlo.num_replicas = N : iN} {\n  func.func publi | `tests/lax_control_flow_test.py::LaxControlFlowTest::test_platform_dependent_lowering` |
| 1 | failed | AssertionError: Regex didn't match: 'algorithm = <lhs_precision_type = fN, rhs_precision_type = fN, accumulation_type = fN' not found in 'module @jit_dot attrib | `tests/lax_test.py::LaxTest::testDotAlgorithmConfig` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.tangent" %argN {composite_attributes = {dtype = fN, int = N : iN, str = "bar", tensor = dense<N> : tensor<NxNxfN>, tens | `tests/lax_test.py::CompositeTest::test_composite_attribute_dtypes` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my_square attributes | `tests/lax_test.py::CompositeTest::test_composite_decorator` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.one"' not found in 'module @jit_one attributes {mhlo.num_partitions = N : iN, mhlo.num_replicas = N : iN} {\n  func.fun | `tests/lax_test.py::CompositeTest::test_composite_with_no_args` |
| 1 | failed | AssertionError: '@my.consts(%argN: tensor<NxfN>, %argN: tensor<NxfN>) -> tensor<NxfN>' not found in 'module @jit_my_consts attributes {mhlo.num_partitions = N : | `tests/lax_test.py::CompositeTest::test_composite_with_array_consts` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square, version = N : iN} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my | `tests/lax_test.py::CompositeTest::test_composite_with_non_default_version` |
| 1 | failed | AssertionError: Regex didn't match: 'stablehlo.composite "my.square" %argN, %argN \\{decomposition = @my.square\\} : \\(tensor<i..>, tensor<\\?xfN>\\) -> tensor | `tests/lax_test.py::CompositeTest::test_composite_with_symbolic_shapes` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.top_k" %argN {composite_attributes = {k = N : iN}, decomposition = @my.top_k} : (tensor<NxfN>) -> (tensor<NxfN>, tensor | `tests/lax_test.py::CompositeTest::test_composite_with_attributes` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.top_k" %argN {composite_attributes = {is_stable = false, k = N : iN}, decomposition = @my.top_k} : (tensor<NxfN>) -> (t | `tests/lax_test.py::CompositeTest::test_composite_with_attributes_unstable` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.ident" %argN, %argN, %argN {decomposition = @my.ident} : (tensor<fN>, tensor<fN>, tensor<fN>) -> (tensor<fN>, tensor<fN | `tests/lax_test.py::CompositeTest::test_composite_with_variadic_input_output` |
| 1 | failed | AssertionError: ValueError not raised | `tests/lax_test.py::LaxTest::testDotAlgorithmInvalidFloat8Type` |
| 1 | failed | AssertionError: 'x=[N N]' not found in '' | `tests/shard_map_test.py::ShardMapTest::test_debug_print_eager` |
