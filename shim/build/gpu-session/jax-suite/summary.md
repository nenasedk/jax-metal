# JAX test suite on METAL

| File | passed | failed | skipped | crashed |
|---|---|---|---|---|
| `tests/ann_test.py` | 32 | 0 | 14 | 0 |
| `tests/api_test.py` | 717 | 120 | 177 | 1 |
| `tests/array_test.py` | 71 | 0 | 167 | 0 |
| `tests/batching_test.py` | 226 | 6 | 0 | 0 |
| `tests/checkify_test.py` | 105 | 9 | 4 | 0 |
| `tests/cholesky_update_test.py` | 1 | 1 | 2 | 0 |
| `tests/custom_linear_solve_test.py` | 10 | 4 | 2 | 0 |
| `tests/custom_root_test.py` | 19 | 4 | 0 | 0 |
| `tests/dtypes_test.py` | 795 | 39 | 31 | 0 |
| `tests/eigh_test.py` | 68 | 1 | 11 | 15 |
| `tests/experimental_rnn_test.py` | 9 | 1 | 12 | 0 |
| `tests/fft_test.py` | 109 | 0 | 2 | 0 |
| `tests/image_test.py` | 57 | 0 | 30 | 0 |
| `tests/lax_autodiff_test.py` | 696 | 48 | 1 | 0 |
| `tests/lax_control_flow_test.py` | 525 | 20 | 483 | 0 |
| `tests/lax_numpy_einsum_test.py` | 174 | 2 | 0 | 0 |
| `tests/lax_numpy_indexing_test.py` | 344 | 46 | 0 | 0 |
| `tests/lax_numpy_operators_test.py` | 1617 | 9 | 2 | 0 |
| `tests/lax_numpy_reducers_test.py` | 993 | 50 | 4 | 0 |
| `tests/lax_numpy_setops_test.py` | 128 | 25 | 0 | 2 |
| `tests/lax_numpy_test.py` | 3204 | 106 | 37 | 1 |
| `tests/lax_numpy_ufuncs_test.py` | 204 | 3 | 8 | 0 |
| `tests/lax_numpy_vectorize_test.py` | 34 | 1 | 0 | 0 |
| `tests/lax_scipy_sparse_test.py` | 29 | 3 | 36 | 23 |
| `tests/lax_scipy_special_functions_test.py` | 447 | 15 | 0 | 5 |
| `tests/lax_scipy_spectral_dac_test.py` | 0 | 0 | 5 | 5 |
| `tests/lax_scipy_test.py` | 98 | 0 | 18 | 5 |
| `tests/lax_test.py` | 2347 | 188 | 76 | 3 |
| `tests/lax_vmap_op_test.py` | 402 | 3 | 295 | 0 |
| `tests/lax_vmap_test.py` | 259 | 59 | 10 | 0 |
| `tests/linalg_test.py` | 763 | 20 | 193 | 4 |
| `tests/lobpcg_test.py` | 22 | 31 | 0 | 2 |
| `tests/mutable_array_test.py` | 173 | 11 | 34 | 0 |
| `tests/nn_test.py` | 362 | 25 | 56 | 0 |
| `tests/ode_test.py` | 9 | 4 | 0 | 1 |
| `tests/optimizers_test.py` | 22 | 2 | 0 | 0 |
| `tests/pjit_test.py` | 136 | 2 | 750 | 0 |
| `tests/pmap_test.py` | 208 | 2 | 91 | 3 |
| `tests/polynomial_test.py` | 20 | 0 | 0 | 0 |
| `tests/qdwh_test.py` | 17 | 0 | 0 | 29 |
| `tests/random_impl_test.py` | 59 | 1 | 0 | 0 |
| `tests/random_lax_test.py` | 648 | 0 | 60 | 1 |
| `tests/random_test.py` | 235 | 3 | 16 | 0 |
| `tests/scipy_fft_test.py` | 36 | 10 | 1 | 0 |
| `tests/scipy_interpolate_test.py` | 3 | 1 | 0 | 0 |
| `tests/scipy_ndimage_test.py` | 16 | 0 | 0 | 0 |
| `tests/scipy_optimize_test.py` | 3 | 3 | 1 | 7 |
| `tests/scipy_signal_test.py` | 62 | 12 | 24 | 2 |
| `tests/scipy_spatial_test.py` | 71 | 0 | 0 | 0 |
| `tests/scipy_stats_test.py` | 995 | 6 | 0 | 0 |
| `tests/shard_map_test.py` | 58 | 7 | 420 | 3 |
| `tests/sparse_bcoo_bcsr_test.py` | 330 | 75 | 25 | 15 |
| `tests/sparse_test.py` | 326 | 84 | 54 | 0 |
| `tests/stax_test.py` | 67 | 4 | 0 | 0 |
| `tests/svd_test.py` | 6 | 0 | 27 | 59 |
| `tests/x64_context_test.py` | 9 | 14 | 2 | 0 |
| **total** | 18376 | 1080 | 3181 | 186 |

Failures compared with the CPU run in `../../jax-suite-cpu` (METAL emulation): **METAL only** passed there (MPS behavior or a plugin limit), **also on CPU** failed there too (the shim's rules, or the test).

| Triage | Failures |
|---|---|
| METAL only | 1243 |
| also on CPU | 23 |

## METAL only: failure groups (most common first)

| Count | Outcome | Message (normalized) | Example | Predicted on CPU |
|---|---|---|---|---|
| 183 | failed | AssertionError:  | `tests/custom_linear_solve_test.py::CustomLinearSolveTest::test_custom_linear_solve_aux` |  |
| 125 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: null operand found | `tests/api_test.py::APITest::test_leak_checker_avoids_false_positives_remat3_scan_transpose` |  |
| 106 | crashed | exit -N: Extension modules: jaxlib.cpu_feature_guard, numpy._core._multiarray_umath, numpy.linalg._umath_linalg, numpy.random._common, numpy.random.bit_generato | `tests/api_test.py::JitTest::test_jit_lower_compile_memory_analysis` |  |
| 97 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.real' | `tests/lax_autodiff_test.py::LaxAutodiffTest::testOpGrad131` |  |
| 80 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.reduce_window' | `tests/batching_test.py::BatchingTest::testMinMaxPool_op=max` |  |
| 74 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <unknown>:N: error: 'func.func' op One or more function input<path> data types are not supported. | `tests/api_test.py::JitTest::test_print_token_buffer_error` | op stablehlo.create_token |
| 58 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.reduce' | `tests/lax_autodiff_test.py::LaxAutodiffTest::testReducePairGrad1` |  |
| 53 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.map' | `tests/lax_autodiff_test.py::LaxAutodiffTest::testSortGrad2` |  |
| 48 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.dot_general' | `tests/lax_autodiff_test.py::LaxAutodiffTest::testDotGeneralContractAndBatchGrads3` |  |
| 46 | failed | jax.errors.JaxRuntimeError: INTERNAL: Unable to serialize MPS module | `tests/api_test.py::APITest::test_grad_of_bool` |  |
| 30 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.iota' | `tests/lax_numpy_indexing_test.py::IndexingTest::testIndexingTypePromotion` |  |
| 30 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mhlo.convolution' op Not supported: Integer type convolution. | `tests/lax_numpy_test.py::LaxBackedNumpyTests::testConvolutionsPreferredElementType0` |  |
| 29 | failed | AssertionError: Test changed global config values. Differing values are: {'jax_use_shardy_partitioner': (True, False)} | `tests/batching_test.py::BatchingTest::testAllGatherOfConst` |  |
| 26 | failed | jax.errors.JaxRuntimeError: UNKNOWN: -:N:N: error: unknown vhlo attribute code: N | `tests/lax_numpy_indexing_test.py::IndexingTest::testIndexApply1` |  |
| 25 | crashed | exit -N: Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, _cyutility, scipy._cyutility, scipy._lib._ccallback_c, scipy.linalg._fbla | `tests/eigh_test.py::LaxLinalgEighTest::testRandomUniform0` |  |
| 24 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mhlo.convolution' op Not supported: ConvolutionOp with groups and batch groups. | `tests/lax_autodiff_test.py::LaxAutodiffTest::testConvGeneralDilatedGrad3` |  |
| 24 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to materialize conversion for result #N of operation 'mhlo.dot_general' that remained live after  | `tests/nn_test.py::NNFunctionsTest::testDotProductAttention11` |  |
| 23 | crashed | exit -N: Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, _cyutility, scipy._cyutility, scipy._lib._ccallback_c, scipy.sparse._spar | `tests/lax_scipy_sparse_test.py::LaxBackedScipyTests::test_gmres_matmul` |  |
| 17 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.scatter' | `tests/lax_numpy_indexing_test.py::IndexingTest::testIndexApply0` |  |
| 14 | failed | NotImplementedError: METAL: floatN_eNmN values cannot be program inputs or outputs (StableHLO N.N, which Apple's plugin reads, has no floatN_eNmN); convert them | `tests/dtypes_test.py::TestPromotionTables::testArrayRepr48` | serialize to StableHLO 0.19.4: INVALID_ARGUMENT: Failed to serialize StableHLO to plugin version 0.19.4; |
| 12 | failed | jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Failed to serialize StableHLO to plugin version N.N; | `tests/dtypes_test.py::TestPromotionTables::testArrayRepr56` | serialize to StableHLO 0.19.4: INVALID_ARGUMENT: Failed to serialize StableHLO to plugin version 0.19.4; |
| 10 | crashed | exit -N: Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, _cyutility, scipy._cyutility, scipy._lib._ccallback_c, scipy.special._ufu | `tests/lax_scipy_special_functions_test.py::LaxScipySpecialFunctionsTest::testExpiDisableJit` |  |
| 10 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mps.pooling_max_gradient' op attribute 'window_sizes' failed to satisfy constraint: uiN elements attrib | `tests/lax_vmap_test.py::LaxVmapTest::testSelectAndScatterAdd0` |  |
| 9 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mhlo.convolution' op Not supported: ConvolutionOp with batchGroups. | `tests/batching_test.py::BatchingTest::testConvGeneralDilated` |  |
| 9 | crashed | exit -N: Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, jaxlib.cpu_feature_guard, numpy.random._common, numpy.random.bit_generato | `tests/lax_numpy_setops_test.py::LaxNumpySetopsTest::testUnique0` |  |
| 8 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: expects bounds of the window dimensions of updates to not exceed the bounds of the corresponding dimensi | `tests/lax_numpy_test.py::LaxBackedNumpyTests::testArgpartition3` |  |
| 7 | failed | NotImplementedError: METAL: floatN_eNmNfnu values cannot be program inputs or outputs (StableHLO N.N, which Apple's plugin reads, has no floatN_eNmNfnu); conver | `tests/dtypes_test.py::TestPromotionTables::testArrayRepr52` | serialize to StableHLO 0.19.4: INVALID_ARGUMENT: Failed to serialize StableHLO to plugin version 0.19.4; |
| 7 | failed | NotImplementedError: METAL: floatN_eNmNfn values cannot be program inputs or outputs (StableHLO N.N, which Apple's plugin reads, has no floatN_eNmNfn); convert  | `tests/dtypes_test.py::TestPromotionTables::testArrayRepr54` | serialize to StableHLO 0.19.4: INVALID_ARGUMENT: Failed to serialize StableHLO to plugin version 0.19.4; |
| 7 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to materialize conversion for result #N of operation 'mhlo.convolution' that remained live after  | `tests/lax_autodiff_test.py::LaxAutodiffTest::testConvGeneralDilatedGrad2` |  |
| 6 | failed | jax.errors.JaxRuntimeError: INVALID_ARGUMENT: CopyToDevice cannot accept the same source and destination devices | `tests/api_test.py::JitTest::test_device_put_aliasing` |  |
| 5 | crashed | guard: memory N GB > limit N GB | `tests/pmap_test.py::PythonPmapTest::testSizeOverflow` |  |
| 4 | failed | UserWarning: Some donated buffers were not usable: ~floatN[], ~floatN[]. | `tests/api_test.py::JitTest::test_dict_donation_argnames` |  |
| 4 | failed | AssertionError: JaxRuntimeError not raised by <lambda> | `tests/api_test.py::JitTest::test_jit_lower_compile_with_compiler_options_invalid` |  |
| 4 | failed | AssertionError: Tuples differ: (N,) != (N,) | `tests/lax_numpy_setops_test.py::LaxNumpySetopsTest::testUniqueEqualNan4` |  |
| 3 | failed | UserWarning: Some donated buffers were not usable: floatN[N]. | `tests/api_test.py::JitTest::test_double_donation` |  |
| 3 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'tensor.expand_shape' op expected collapsed rank (N) to equal the number of reassociation maps (N). | `tests/batching_test.py::BatchingTest::testIssue3883` |  |
| 2 | failed | AssertionError: unexpectedly None | `tests/api_test.py::JitTest::test_jit_lower_compile_cost_analysis` |  |
| 2 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mps.matmul' op operand #N must be tensor of floating point values or tensor of complex values, but got  | `tests/lax_test.py::LaxTest::testConvGeneralDilated0D1` |  |
| 2 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mps.strided_slice' op operand #N must be ND tensor of mps index type values or ND tensor of mps index t | `tests/lax_test.py::LaxTest::testDynamicSliceU8Index` |  |
| 2 | crashed | exit -N: Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, _cyutility, scipy._cyutility, scipy._lib._ccallback_c, numpy.random._comm | `tests/scipy_signal_test.py::LaxBackedScipySignalTests::testConvolutions0` |  |
| 2 | failed | AssertionError: N != Array(N, dtype=intN) | `tests/sparse_test.py::SparseObjectTest::test_attrs_CSC` |  |
| 1 | failed | UserWarning: Some donated buffers were not usable: floatN[]. | `tests/api_test.py::JitTest::test_donate_args_info_aot` |  |
| 1 | failed | AssertionError: JaxRuntimeError not raised | `tests/api_test.py::JitTest::test_jit_compile_with_compiler_options_multiple` |  |
| 1 | failed | UserWarning: Some donated buffers were not usable: ~floatN[]. | `tests/api_test.py::JitTest::test_jit_lower_donate_argnums_available` |  |
| 1 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to legalize operation 'mhlo.custom_call' | `tests/api_test.py::APITest::test_cache_miss_explanations_compute_on` | custom_call annotate_device_placement |
| 1 | failed | TypeError: argument of type 'NoneType' is not iterable | `tests/api_test.py::APITest::test_inline_optimized_hlo` |  |
| 1 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mps.cast' op result #N must be tensor of mps native type values or tensor of complex values, but got 't | `tests/dtypes_test.py::ExtendedDTypeTest::test_custom_tangent_dtype` | type f8E5M2 |
| 1 | failed | AssertionError: Array(nan+nanj, dtype=complexN) != Array(N.+N.j, dtype=complexN) | `tests/lax_autodiff_test.py::LaxAutodiffTest::testAsinhDerivatives` |  |
| 1 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: 'mhlo.convolution' op Not Supported: conv transpose with feature groups. | `tests/lax_autodiff_test.py::LaxAutodiffTest::testConvGeneralDilatedGrad0` |  |
| 1 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to materialize conversion for result #N of operation 'mhlo.reshape' that remained live after conv | `tests/lax_autodiff_test.py::LaxAutodiffTest::testDotPreferredElementType` |  |
| 1 | crashed | guard: system memory low (N% free) while the job holds N GB | `tests/lax_numpy_test.py::NumpyUfuncTests::testUfuncInputTypes601` |  |
| 1 | failed | jax.errors.JaxRuntimeError: UNKNOWN: <path>:N:N: error: failed to materialize conversion for result #N of operation 'mhlo.dot' that remained live after conversi | `tests/lax_test.py::LaxTest::testDotPreferredElement2` |  |
| 1 | failed | jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Failed to upgrade versioned StableHLO. | `tests/lax_test.py::LaxTest::test_dce_sink_prevents_xla_dce` | custom_call dce_sink |
| 1 | failed | AssertionError: np.floatN(N) not less than np.floatN(N) | `tests/lobpcg_test.py::F32LobpcgTest::testLobpcgMonotonicityF32cluster_k_1__n100` |  |

## also on CPU: failure groups (most common first)

| Count | Outcome | Message (normalized) | Example | Predicted on CPU |
|---|---|---|---|---|
| 3 | crashed | exit -N: Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, _cyutility, scipy._cyutility, scipy._lib._ccallback_c, scipy.linalg._fbla | `tests/eigh_test.py::EighTest::testEighBatching6` |  |
| 2 | failed | NotImplementedError: MLIR translation rule for primitive 'check' not found for platform METAL | `tests/checkify_test.py::AssertPrimitiveTests::test_assert_primitive_lowering` |  |
| 2 | failed | NotImplementedError: MLIR translation rule for primitive 'debug_callback' not found for platform METAL | `tests/custom_linear_solve_test.py::CustomLinearSolveTest::test_custom_linear_solve_ordered_effects` |  |
| 2 | crashed | exit -N: Extension modules: jaxlib.cpu_feature_guard, numpy._core._multiarray_umath, numpy.linalg._umath_linalg, numpy.random._common, numpy.random.bit_generato | `tests/lax_scipy_spectral_dac_test.py::LaxScipySpectralDacTest::test_spectral_dac_eigh4` |  |
| 2 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my_square_impl attri | `tests/lax_test.py::CompositeTest::test_composite` |  |
| 1 | failed | NotImplementedError: METAL: Jacobi eigendecomposition of size N exceeds N (larger ones can stall the GPU and WindowServer); complex eigh<path> is limited to thi | `tests/eigh_test.py::EighTest::testEighBatching3` |  |
| 1 | failed | AssertionError: Regex didn't match: 'algorithm = <lhs_precision_type = fN, rhs_precision_type = fN, accumulation_type = fN' not found in 'module @jit_dot attrib | `tests/lax_test.py::LaxTest::testDotAlgorithmConfig` |  |
| 1 | failed | AssertionError: ValueError not raised | `tests/lax_test.py::LaxTest::testDotAlgorithmInvalidFloat8Type` |  |
| 1 | failed | AssertionError: 'stablehlo.composite "my.tangent" %argN {composite_attributes = {dtype = fN, int = N : iN, str = "bar", tensor = dense<N> : tensor<NxNxfN>, tens | `tests/lax_test.py::CompositeTest::test_composite_attribute_dtypes` |  |
| 1 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my_square attributes | `tests/lax_test.py::CompositeTest::test_composite_decorator` |  |
| 1 | failed | AssertionError: '@my.consts(%argN: tensor<NxfN>, %argN: tensor<NxfN>) -> tensor<NxfN>' not found in 'module @jit_my_consts attributes {mhlo.num_partitions = N : | `tests/lax_test.py::CompositeTest::test_composite_with_array_consts` |  |
| 1 | failed | AssertionError: 'stablehlo.composite "my.top_k" %argN {composite_attributes = {k = N : iN}, decomposition = @my.top_k} : (tensor<NxfN>) -> (tensor<NxfN>, tensor | `tests/lax_test.py::CompositeTest::test_composite_with_attributes` |  |
| 1 | failed | AssertionError: 'stablehlo.composite "my.top_k" %argN {composite_attributes = {is_stable = false, k = N : iN}, decomposition = @my.top_k} : (tensor<NxfN>) -> (t | `tests/lax_test.py::CompositeTest::test_composite_with_attributes_unstable` |  |
| 1 | failed | AssertionError: 'stablehlo.composite "my.one"' not found in 'module @jit_one attributes {mhlo.num_partitions = N : iN, mhlo.num_replicas = N : iN} {\n  func.fun | `tests/lax_test.py::CompositeTest::test_composite_with_no_args` |  |
| 1 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square, version = N : iN} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my | `tests/lax_test.py::CompositeTest::test_composite_with_non_default_version` |  |
| 1 | failed | AssertionError: Regex didn't match: 'stablehlo.composite "my.square" %argN, %argN \\{decomposition = @my.square\\} : \\(tensor<i..>, tensor<\\?xfN>\\) -> tensor | `tests/lax_test.py::CompositeTest::test_composite_with_symbolic_shapes` |  |
| 1 | failed | AssertionError: 'stablehlo.composite "my.ident" %argN, %argN, %argN {decomposition = @my.ident} : (tensor<fN>, tensor<fN>, tensor<fN>) -> (tensor<fN>, tensor<fN | `tests/lax_test.py::CompositeTest::test_composite_with_variadic_input_output` |  |
