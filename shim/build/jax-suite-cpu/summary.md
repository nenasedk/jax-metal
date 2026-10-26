# JAX test suite on CPU, emulating METAL

| File | passed | failed | skipped | crashed |
|---|---|---|---|---|
| `tests/ann_test.py` | 32 | 0 | 14 | 0 |
| `tests/fft_test.py` | 109 | 0 | 2 | 0 |
| `tests/lax_autodiff_test.py` | 744 | 0 | 1 | 0 |
| `tests/lax_control_flow_test.py` | 539 | 6 | 483 | 0 |
| `tests/lax_numpy_einsum_test.py` | 176 | 0 | 0 | 0 |
| `tests/lax_numpy_indexing_test.py` | 390 | 0 | 0 | 0 |
| `tests/lax_numpy_operators_test.py` | 1626 | 0 | 2 | 0 |
| `tests/lax_numpy_reducers_test.py` | 1043 | 0 | 4 | 0 |
| `tests/lax_numpy_test.py` | 3311 | 0 | 37 | 0 |
| `tests/lax_scipy_special_functions_test.py` | 467 | 0 | 0 | 0 |
| `tests/lax_scipy_test.py` | 103 | 0 | 18 | 0 |
| `tests/lax_test.py` | 2525 | 13 | 76 | 0 |
| `tests/lax_vmap_op_test.py` | 405 | 0 | 295 | 0 |
| `tests/lax_vmap_test.py` | 318 | 0 | 10 | 0 |
| `tests/linalg_test.py` | 777 | 10 | 193 | 0 |
| `tests/nn_test.py` | 387 | 0 | 56 | 0 |
| `tests/random_lax_test.py` | 649 | 0 | 60 | 0 |
| `tests/random_test.py` | 238 | 0 | 16 | 0 |
| `tests/scipy_signal_test.py` | 76 | 0 | 24 | 0 |
| **total** | 13915 | 29 | 1291 | 0 |

## Failure groups (most common first)

| Count | Outcome | Message (normalized) | Example |
|---|---|---|---|
| 10 | failed | NotImplementedError: MLIR translation rule for primitive 'tridiagonal' not found for platform cpu | `tests/linalg_test.py::ScipyLinalgTest::testTridiagonal0` |
| 4 | failed | AssertionError:  | `tests/lax_control_flow_test.py::LaxControlFlowTest::test_platform_dependent` |
| 2 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my_square_impl attri | `tests/lax_test.py::CompositeTest::test_composite` |
| 1 | failed | AssertionError: False != True | `tests/lax_control_flow_test.py::LaxControlFlowTest::test_platform_dependent_grad` |
| 1 | failed | AssertionError: 'stablehlo.sine' unexpectedly found in 'module @jit_f attributes {mhlo.num_partitions = N : iN, mhlo.num_replicas = N : iN} {\n  func.func publi | `tests/lax_control_flow_test.py::LaxControlFlowTest::test_platform_dependent_lowering` |
| 1 | failed | AssertionError: Regex didn't match: 'algorithm = <lhs_precision_type = fN, rhs_precision_type = fN, accumulation_type = fN' not found in 'module @jit_dot attrib | `tests/lax_test.py::LaxTest::testDotAlgorithmConfig` |
| 1 | failed | AssertionError: ValueError not raised | `tests/lax_test.py::LaxTest::testDotAlgorithmInvalidFloat8Type` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.tangent" %argN {composite_attributes = {dtype = fN, int = N : iN, str = "bar", tensor = dense<N> : tensor<NxNxfN>, tens | `tests/lax_test.py::CompositeTest::test_composite_attribute_dtypes` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my_square attributes | `tests/lax_test.py::CompositeTest::test_composite_decorator` |
| 1 | failed | AssertionError: '@my.consts(%argN: tensor<NxfN>, %argN: tensor<NxfN>) -> tensor<NxfN>' not found in 'module @jit_my_consts attributes {mhlo.num_partitions = N : | `tests/lax_test.py::CompositeTest::test_composite_with_array_consts` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.top_k" %argN {composite_attributes = {is_stable = false, k = N : iN}, decomposition = @my.top_k} : (tensor<NxfN>) -> (t | `tests/lax_test.py::CompositeTest::test_composite_with_attributes_unstable` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.top_k" %argN {composite_attributes = {k = N : iN}, decomposition = @my.top_k} : (tensor<NxfN>) -> (tensor<NxfN>, tensor | `tests/lax_test.py::CompositeTest::test_composite_with_attributes` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.ident" %argN, %argN, %argN {decomposition = @my.ident} : (tensor<fN>, tensor<fN>, tensor<fN>) -> (tensor<fN>, tensor<fN | `tests/lax_test.py::CompositeTest::test_composite_with_variadic_input_output` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.one"' not found in 'module @jit_one attributes {mhlo.num_partitions = N : iN, mhlo.num_replicas = N : iN} {\n  func.fun | `tests/lax_test.py::CompositeTest::test_composite_with_no_args` |
| 1 | failed | AssertionError: 'stablehlo.composite "my.square" %argN {decomposition = @my.square, version = N : iN} : (tensor<fN>) -> tensor<fN>' not found in 'module @jit_my | `tests/lax_test.py::CompositeTest::test_composite_with_non_default_version` |
| 1 | failed | AssertionError: Regex didn't match: 'stablehlo.composite "my.square" %argN, %argN \\{decomposition = @my.square\\} : \\(tensor<i..>, tensor<\\?xfN>\\) -> tensor | `tests/lax_test.py::CompositeTest::test_composite_with_symbolic_shapes` |
