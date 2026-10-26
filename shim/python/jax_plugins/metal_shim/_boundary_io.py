"""complex64 and uint64 values as METAL program inputs and outputs.

The plugin rejects any compiled function whose signature mentions complex or uint64 types.
For a METAL program with such inputs or outputs this module lowers two versions:

- the original, with its signature, which jaxlib compiles and takes its metadata (output
  types, shapes, layouts) from;
- a boundary version in which each complex64 input/output is a float32[..., 2] (real, imag)
  pair and each uint64 one an int64 with the same bits, converted inside the program (which
  _complex.py and the plugin support), serialized here as a StableHLO 0.19.4 portable
  artifact.

Right before compilation the shim receives the boundary program and the positions and kinds
of the changed values, and compiles the boundary program instead. At execution it passes
complex64 inputs as float32[..., 2] views and returns pair outputs as complex64 views of the
same device memory; uint64 buffers already live in the plugin as int64 and int64 outputs are
reported as uint64. JAX only ever sees genuine complex64 and uint64 arrays, and nothing is
copied.

Hooks into JAX internals: mlir.lower_jaxpr_to_module (lower the boundary version) and
compiler.compile_or_get_cached (hand it to the shim). See shim/DESIGN.md.
"""

import base64
import ctypes

import jax
import numpy as np
from jax import lax
from jax._src import compiler
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lib import _jax
from jax._src.lib.mlir import ir

PLATFORM = "METAL"
STABLEHLO_TARGET = "0.19.4"  # what the plugin deserializes; the shim reports the same
_POSITIONS = "jax_metal.boundary_positions"
_PROGRAM = "jax_metal.boundary_program"

# Kinds of boundary values; the shim (BoundaryKind in pjrt_plugin_metal_shim.cc) uses the
# same numbers.
COMPLEX_PAIRS = 1  # complex64 as float32[..., 2]
UINT64_AS_INT64 = 2  # uint64 as int64

_set_boundary_io = None


def _kind(aval):
    """The boundary kind of a program input/output, or 0 if the plugin takes it as is."""
    # Extended dtypes (PRNG keys) are not NumPy dtypes and need nothing.
    dtype = getattr(aval, "dtype", None)
    if not isinstance(aval, core.ShapedArray) or not isinstance(dtype, np.dtype):
        return 0
    if dtype == np.complex64:
        return COMPLEX_PAIRS
    if dtype == np.uint64:
        return UINT64_AS_INT64
    return 0


def _physical_aval(aval, kind):
    if kind == COMPLEX_PAIRS:
        return core.ShapedArray(tuple(aval.shape) + (2,), np.float32)
    if kind == UINT64_AS_INT64:
        return core.ShapedArray(aval.shape, np.int64)
    return aval


def _to_physical(value, kind):
    if kind == COMPLEX_PAIRS:
        real, imag = lax.real(value), lax.imag(value)
        return lax.concatenate([lax.expand_dims(real, (real.ndim,)),
                                lax.expand_dims(imag, (imag.ndim,))], real.ndim)
    if kind == UINT64_AS_INT64:
        return lax.bitcast_convert_type(value, np.int64)
    return value


def _from_physical(value, kind):
    if kind == COMPLEX_PAIRS:
        return lax.complex(lax.index_in_dim(value, 0, value.ndim - 1, keepdims=False),
                           lax.index_in_dim(value, 1, value.ndim - 1, keepdims=False))
    if kind == UINT64_AS_INT64:
        return lax.bitcast_convert_type(value, np.uint64)
    return value


def _rewrite(closed_jaxpr, in_kinds, out_kinds):
    """A jaxpr like closed_jaxpr whose inputs/outputs of a boundary kind are physical."""
    in_avals = [_physical_aval(a, kind) for a, kind in zip(closed_jaxpr.in_avals, in_kinds)]

    def body(*args):
        args = [_from_physical(a, kind) for a, kind in zip(args, in_kinds)]
        outs = core.jaxpr_as_fun(closed_jaxpr)(*args)
        return [_to_physical(o, kind) for o, kind in zip(outs, out_kinds)]

    shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in in_avals]
    return jax.make_jaxpr(body)(*shapes), in_avals


def _masked(values, flags, replacement):
    if values is None:
        return None
    return type(values)(replacement if flag else v for v, flag in zip(values, flags))


def _lower_jaxpr_to_module(original):
    def lower(module_name, jaxpr, **kwargs):
        in_avals, out_avals = kwargs["in_avals"], kwargs["out_avals"]
        in_kinds = [_kind(a) for a in in_avals]
        out_kinds = [_kind(a) for a in out_avals]
        applicable = (tuple(kwargs["platforms"]) == (PLATFORM,)
                      and (any(in_kinds) or any(out_kinds))
                      and kwargs.get("num_const_args", 0) == 0
                      and len(jaxpr.in_avals) == len(in_avals)
                      and all(core.is_constant_shape(a.shape) for a in [*in_avals, *out_avals]
                              if isinstance(a, core.ShapedArray)))
        if not applicable:
            return original(module_name, jaxpr, **kwargs)
        # The plugin runs the boundary program, which never aliases changed inputs; drop
        # their donation here too so JAX does not warn about unusable donated buffers.
        kwargs = dict(kwargs)
        kwargs["donated_args"] = _masked(kwargs["donated_args"], in_kinds, False)
        if kwargs.get("input_output_aliases") is not None:
            kwargs["input_output_aliases"] = _masked(kwargs["input_output_aliases"], in_kinds, None)
        result = original(module_name, jaxpr, **kwargs)
        boundary_jaxpr, boundary_in_avals = _rewrite(jaxpr, in_kinds, out_kinds)
        boundary_kwargs = dict(kwargs)
        boundary_kwargs["in_avals"] = boundary_in_avals
        boundary_kwargs["out_avals"] = [_physical_aval(a, kind)
                                        for a, kind in zip(out_avals, out_kinds)]
        # Changed positions use default layouts and shardings.
        for key, kinds in (("in_layouts", in_kinds), ("arg_shardings", in_kinds),
                           ("out_layouts", out_kinds), ("result_shardings", out_kinds)):
            boundary_kwargs[key] = _masked(kwargs.get(key), kinds, None)
        boundary_module = original(module_name, boundary_jaxpr, **boundary_kwargs).module
        program = _jax.mlir.serialize_portable_artifact(boundary_module, STABLEHLO_TARGET, True)
        positions = lambda kinds: ",".join(f"{i}:{kind}" for i, kind in enumerate(kinds) if kind)
        with result.module.context:
            attributes = result.module.operation.attributes
            attributes[_POSITIONS] = ir.StringAttr.get(
                f"{positions(in_kinds)};{positions(out_kinds)}")
            attributes[_PROGRAM] = ir.StringAttr.get(base64.b64encode(program).decode())
        return result

    return lower


def _parse_positions(text):
    """"0:1,3:2;1:2" -> ([0, 3], [1, 2], [1], [2]): input positions and kinds, then outputs."""
    parsed = []
    for part in text.split(";"):
        pairs = [tuple(int(n) for n in item.split(":")) for item in part.split(",") if item]
        parsed += [[p for p, _ in pairs], [k for _, k in pairs]]
    return parsed


def _compile_or_get_cached(original):
    def compile_(backend, computation, *args, **kwargs):
        attributes = computation.operation.attributes
        if _POSITIONS not in attributes:
            return original(backend, computation, *args, **kwargs)
        inputs, input_kinds, outputs, output_kinds = _parse_positions(
            ir.StringAttr(attributes[_POSITIONS]).value)
        program = base64.b64decode(ir.StringAttr(attributes[_PROGRAM]).value)
        del attributes[_POSITIONS]  # jaxlib must not serialize these for the plugin
        del attributes[_PROGRAM]
        as_array = lambda values: (ctypes.c_int32 * max(len(values), 1))(*values)
        _set_boundary_io(as_array(inputs), as_array(input_kinds), len(inputs),
                         as_array(outputs), as_array(output_kinds), len(outputs),
                         program, len(program))
        try:
            return original(backend, computation, *args, **kwargs)
        finally:
            _set_boundary_io(None, None, 0, None, None, 0, None, 0)

    return compile_


def register(shim_library):
    global _set_boundary_io  # pylint: disable=global-statement
    library = ctypes.CDLL(str(shim_library))
    _set_boundary_io = library.JaxMetalShim_SetBoundaryIo
    _set_boundary_io.restype = None
    int_array = ctypes.POINTER(ctypes.c_int32)
    _set_boundary_io.argtypes = [int_array, int_array, ctypes.c_size_t,
                                 int_array, int_array, ctypes.c_size_t,
                                 ctypes.c_char_p, ctypes.c_size_t]
    mlir.lower_jaxpr_to_module = _lower_jaxpr_to_module(mlir.lower_jaxpr_to_module)
    compiler.compile_or_get_cached = _compile_or_get_cached(compiler.compile_or_get_cached)
