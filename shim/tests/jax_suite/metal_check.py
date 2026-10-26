"""Predict whether Apple's plugin can compile a program, without the plugin.

jaxlib hands the plugin each program as a StableHLO 0.19.4 portable artifact (the shim
reports stablehlo_current_version 0.19.4). metal_issues() serializes a module the same way,
reads the artifact back and lists what the plugin's pipeline cannot lower, after
re/findings/mps-op-coverage.md:

- ops without an MPS pattern;
- stablehlo.composite: nothing in the pipeline expands it;
- custom calls other than lapack_sgetrf, which the plugin maps to MPS's LU;
- element types MPS lacks: float64, complex128, 4-, 6- and 8-bit floats, 2- and 4-bit ints;
- reductions, windows, scatters and all-reduces over one operand whose body is more than
  one op: the plugin's converters recognize the reduction by its single op;
- complex values in ops whose MPS kernels or compiler fail on them (the METAL rules split
  these, so one here means a rule missed a path);
- ops MPS lowers but aborts on at run time;
- serialization failures: the program uses something newer than StableHLO 0.19.4.

It predicts; an op on the list can still fail for its attributes or shapes, or at run time.

    python shim/tests/jax_suite/metal_check.py CHECK_LOG.jsonl   # summarize a log
"""

import collections
import json
import re
import sys

STABLEHLO_TARGET = "0.19.4"

# StableHLO ops whose MHLO counterpart has an MPS pattern (re/findings/mps-op-coverage.md).
_PATTERNS = """
abs add all_reduce and atan2 bitcast_convert broadcast_in_dim case ceil clamp compare
concatenate constant convert convolution cosine custom_call divide dot dot_general
dynamic_slice dynamic_update_slice exponential exponential_minus_one floor gather if iota
is_finite log log_plus_one maximum minimum multiply negate not optimization_barrier or pad
power reduce reduce_window remainder reshape reverse rng_bit_generator round_nearest_afz
round_nearest_even rsqrt scatter select select_and_scatter shift_left shift_right_arithmetic
shift_right_logical sign sine slice sort sqrt subtract tan tanh top_k transpose while xor
""".split()
SUPPORTED = ({"stablehlo." + name for name in _PATTERNS}
             | {"builtin.module", "func.func", "func.return", "func.call",  # structure
                "stablehlo.return", "stablehlo.tuple", "stablehlo.get_tuple_element",
                "stablehlo.logistic"})  # no pattern, but jax.nn.sigmoid passes the op probe
# The CPU harness keeps XLA's real/imag/complex; on METAL the rules build them from convert.
CPU_HARNESS_ONLY = {"stablehlo.real", "stablehlo.imag", "stablehlo.complex"}
# lapack_sgetrf: mapped to MPS's LU. Sharding: GSPMD annotations, which fft_test's fftfreq
# tests carry and pass with on METAL.
CUSTOM_CALL_TARGETS = {"lapack_sgetrf", "Sharding"}
RUNTIME_ABORTS = {"stablehlo.rng_bit_generator": "MPS's random kernel aborts the process"}
# Ops that must not see complex values on METAL (no complex MPS kernel, or the plugin's
# compiler crashes on them); the METAL rules split these into real and imaginary parts.
# (Not while: its operands include values the body only reads, and complex ones of those
# work, e.g. complex Cholesky; only some complex loop state crashes MPSGraph at run time.)
NO_COMPLEX = {"stablehlo.pad", "stablehlo.gather", "stablehlo.scatter",
              "stablehlo.dynamic_update_slice", "stablehlo.sort", "stablehlo.fft",
              "stablehlo.dot_general", "stablehlo.convolution"}
# Ops whose body the plugin matches as one combining op (add, maximum, ...), for one operand.
_SINGLE_OP_BODIES = {"stablehlo.reduce", "stablehlo.reduce_window", "stablehlo.scatter",
                     "stablehlo.all_reduce"}
_ELEMENT = re.compile(r"^tensor<(?:[0-9?]+x)*(.+)>$")
_UNSUPPORTED_ELEMENT = re.compile(r"^(f64|complex<f64>|f8E\w+|f6E\w+|f4E\w+|u?i[24])$")


def _element(type_):
    match = _ELEMENT.match(str(type_))
    return match.group(1) if match else None


def _walk(operation, skip_functions):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                child = child.operation
                if child.name == "func.func" and _symbol(child) in skip_functions:
                    continue  # a composite's decomposition: the composite itself is reported
                yield child
                yield from _walk(child, skip_functions)


def _symbol(operation, attribute="sym_name"):
    if attribute not in operation.attributes:
        return None
    return str(operation.attributes[attribute]).strip('"@')


def program_issues(module):
    """Issues in a deserialized (StableHLO 0.19.4) module, as short sorted strings."""
    top = module.operation
    decompositions = set()
    for operation in _walk(top, ()):
        if operation.name == "stablehlo.composite":
            decompositions.add(_symbol(operation, "decomposition"))
    issues = set()
    for operation in _walk(top, decompositions):
        name = operation.name
        if name == "stablehlo.composite":
            issues.add(f"composite {_symbol(operation, 'name')}")
        elif name == "stablehlo.custom_call":
            target = _symbol(operation, "call_target_name")
            if target not in CUSTOM_CALL_TARGETS:
                issues.add(f"custom_call {target}")
        elif name not in SUPPORTED and name not in CPU_HARNESS_ONLY:
            issues.add(f"op {name}")
        if name in RUNTIME_ABORTS:
            issues.add(f"run-time abort: {name} ({RUNTIME_ABORTS[name]})")
        elements = {_element(value.type) for value in (*operation.operands, *operation.results)}
        for element in elements - {None}:
            if _UNSUPPORTED_ELEMENT.match(element):
                issues.add(f"type {element}")
        if name in NO_COMPLEX and any(e and e.startswith("complex") for e in elements):
            issues.add(f"complex {name}")
        if name in _SINGLE_OP_BODIES and len(operation.results) == 1:
            body = [op.operation.name for op in operation.regions[0].blocks[0].operations
                    if op.operation.name != "stablehlo.return"]
            if len(body) > 1:
                issues.add(f"{name} body of {len(body)} ops ({', '.join(sorted(set(body)))})")
    return sorted(issues)


def metal_issues(module):
    """Issues the plugin would hit compiling `module` (an ir.Module JAX is about to compile)."""
    # pylint: disable=import-outside-toplevel
    from jax._src.interpreters import mlir
    from jax._src.lib import _jax
    try:
        artifact = _jax.mlir.serialize_portable_artifact(
            mlir.module_to_bytecode(module), STABLEHLO_TARGET, True)
    except Exception as error:  # pylint: disable=broad-except
        first = (str(error).strip().splitlines() or [type(error).__name__])[0]
        return [f"serialize to StableHLO {STABLEHLO_TARGET}: {first[:160]}"]
    with mlir.make_ir_context() as context:
        return program_issues(_jax.mlir.deserialize_portable_artifact(artifact, context))


def summarize(path):
    """Print each issue with the number of tests it affects, split by CPU outcome."""
    by_issue = collections.defaultdict(collections.Counter)
    examples = {}
    tests = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            tests += 1
            for issue in record["issues"]:
                by_issue[issue][record["outcome"]] += 1
                examples.setdefault(issue, record["nodeid"])
    print(f"{tests} tests with predicted METAL issues\n")
    print("| Tests | Passed on CPU | Issue | Example |\n|---|---|---|---|")
    for issue, outcomes in sorted(by_issue.items(), key=lambda item: -sum(item[1].values())):
        print(f"| {sum(outcomes.values())} | {outcomes['passed']} | {issue.replace('|', '/')} "
              f"| `{examples[issue]}` |")


if __name__ == "__main__":
    summarize(sys.argv[1])
