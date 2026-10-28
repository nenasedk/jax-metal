"""Empty inputs and complex64, uint64 and narrow values at METAL program boundaries.

The plugin rejects any compiled function whose signature mentions complex, uint64, float8 or
4-bit integer types. For a METAL program with such inputs or outputs this module lowers two
versions:

- the original, with its signature, which jaxlib compiles and takes its metadata (output
  types, shapes, layouts) from;
- a boundary version in which each complex64 input/output is a float32[..., 2] (real, imag)
  pair, each uint64 one an int64 with the same bits and each narrow one (float8, int4,
  uint4) its bytes: uint8, or int8 for int4, one value per byte as ml_dtypes stores it. These
  are converted inside the program (which _complex.py, _narrow.py and the plugin support).
  It is serialized here as a StableHLO 0.19.4 portable artifact.

Right before compilation the shim receives the boundary program and the positions and kinds
of the changed values, and compiles the boundary program instead. At execution it passes
complex64 inputs as float32[..., 2] views and returns pair outputs as complex64 views of the
same device memory; uint64 and narrow buffers already live in the plugin as their bytes
(int64, uint8, int8) and such outputs are relabeled with their type. JAX only ever sees
genuine complex64, uint64 and narrow arrays, and nothing is copied.

Tokens (`!mhlo.token`, which the plugin also rejects in a signature) are scalar uint8
values that nothing reads: an output is relabeled as a token buffer, and an input becomes a
fresh create_token inside the program. On METAL's single stream a token orders nothing.

Empty inputs use a scalar uint8 placeholder because MPSGraph rejects zero-sized
input shapes. Their values are reconstructed from shape/type metadata; the shim
allocates a one-byte placeholder and releases it after execution. Empty outputs
already work natively and retain their original shape and type.

The newer narrow types (float8_e4m3, e3m4, e8m0fnu, float4_e2m1fn, int2, uint2) work inside
METAL programs but cannot be their inputs or outputs: jaxlib cannot serialize a signature
with them for the plugin's StableHLO version.

Hooks into JAX internals: mlir.lower_jaxpr_to_module (lower the boundary version) and
compiler.compile_or_get_cached (hand it to the shim). See shim/DESIGN.md.
"""

import base64
import ctypes

import jax
import ml_dtypes
import numpy as np
from jax import lax
from jax._src import compiler
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lib import _jax
from jax._src.lib.mlir import ir
from jax_plugins.metal_shim import _compiler_ir
from jax_plugins.metal_shim import _narrow

PLATFORM = "METAL"
STABLEHLO_TARGET = "0.19.4"  # what the plugin deserializes; the shim reports the same
_POSITIONS = "jax_metal.boundary_positions"
_PROGRAM = "jax_metal.boundary_program"

# Kinds of boundary values; the shim (BoundaryKind in pjrt_plugin_metal_shim.cc) uses the
# same numbers.
COMPLEX_PAIRS = 1  # complex64 as float32[..., 2]
EMPTY_INPUT = 2  # an empty tensor needs no data; pass a scalar uint8 placeholder
RETYPED = 0x100  # | a PJRT_Buffer_Type: a value of that type as the bytes the plugin holds
# PJRT_Buffer_Type numbers (pjrt_c_api.h) of the types passed as their bytes.
_PJRT_TYPES = {np.dtype(np.uint64): 9,
               **{np.dtype(dtype): number for dtype, number in (
                   (ml_dtypes.float8_e5m2, 16), (ml_dtypes.float8_e4m3fn, 17),
                   (ml_dtypes.float8_e4m3b11fnuz, 18), (ml_dtypes.float8_e5m2fnuz, 19),
                   (ml_dtypes.float8_e4m3fnuz, 20), (ml_dtypes.int4, 21), (ml_dtypes.uint4, 22))}}
_PJRT_TOKEN = 23  # PJRT_Buffer_Type_TOKEN: a scalar uint8 that nothing reads

_set_boundary_io = None


def _kind(aval):
    """The boundary kind of a program input/output, or 0 if the plugin takes it as is."""
    if isinstance(aval, core.AbstractToken):  # the plugin rejects !mhlo.token arguments
        return RETYPED | _PJRT_TOKEN
    # Extended dtypes (PRNG keys) are not NumPy dtypes and need nothing.
    dtype = getattr(aval, "dtype", None)
    if not isinstance(aval, core.ShapedArray) or not isinstance(dtype, np.dtype):
        return 0
    if dtype == np.complex64:
        return COMPLEX_PAIRS
    if dtype in _PJRT_TYPES:
        return RETYPED | _PJRT_TYPES[dtype]
    if _narrow.narrow_format(dtype) is not None:
        raise NotImplementedError(
            f"METAL: {dtype} values cannot be program inputs or outputs (StableHLO "
            f"{STABLEHLO_TARGET}, which Apple's plugin reads, has no {dtype}); convert them "
            "inside the jitted function, e.g. to float32, or bitcast float8 to uint8")
    return 0


def _physical_aval(aval, kind):
    if kind == EMPTY_INPUT:
        return core.ShapedArray((), np.uint8)
    if kind == COMPLEX_PAIRS:
        return core.ShapedArray(tuple(aval.shape) + (2,), np.float32)
    if kind == RETYPED | _PJRT_TOKEN:
        return core.ShapedArray((), np.uint8)
    if kind & RETYPED:
        fmt = _narrow.narrow_format(aval.dtype)
        return core.ShapedArray(aval.shape, fmt.storage if fmt else np.int64)
    return aval


def _to_physical(value, kind, aval):
    """The boundary form of an output (narrow values: their carriers, see _narrow.py)."""
    if kind == COMPLEX_PAIRS:
        real, imag = lax.real(value), lax.imag(value)
        return lax.concatenate([lax.expand_dims(real, (real.ndim,)),
                                lax.expand_dims(imag, (imag.ndim,))], real.ndim)
    if kind == RETYPED | _PJRT_TOKEN:
        return lax.full((), 0, np.uint8)
    if kind & RETYPED:
        fmt = _narrow.narrow_format(aval.dtype)
        return _narrow.encode(value, fmt) if fmt else lax.bitcast_convert_type(value, np.int64)
    return value


def _from_physical(value, kind, aval):
    """An input from its boundary form (narrow values: their carriers)."""
    if kind == EMPTY_INPUT:
        return lax.full(aval.shape, 0, aval.dtype)
    if kind == COMPLEX_PAIRS:
        return lax.complex(lax.index_in_dim(value, 0, value.ndim - 1, keepdims=False),
                           lax.index_in_dim(value, 1, value.ndim - 1, keepdims=False))
    if kind == RETYPED | _PJRT_TOKEN:
        return lax.create_token()
    if kind & RETYPED:
        fmt = _narrow.narrow_format(aval.dtype)
        return _narrow.decode(value, fmt) if fmt else lax.bitcast_convert_type(value, np.uint64)
    return value


def _rewrite(jaxpr, in_kinds, out_kinds):
    """A jaxpr like `jaxpr` whose inputs/outputs of a boundary kind are physical."""
    in_avals = [_physical_aval(a, kind) for a, kind in zip(jaxpr.in_avals, in_kinds)]

    def body(*args):
        args = [_from_physical(a, kind, aval)
                for a, kind, aval in zip(args, in_kinds, jaxpr.in_avals)]
        if _narrow.uses_narrow(jaxpr):  # runs on carriers
            outs = _narrow.evaluate(jaxpr, jaxpr.consts or (), args)
        else:
            outs = core.jaxpr_as_fun(jaxpr)(*args)
        return [_to_physical(o, kind, aval)
                for o, kind, aval in zip(outs, out_kinds, jaxpr.out_avals)]

    shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in in_avals]
    with jax.numpy_rank_promotion("allow"), jax.numpy_dtype_promotion("standard"):
        return jax.make_jaxpr(body)(*shapes), in_avals


def _masked(values, flags, replacement):
    if values is None:
        return None
    return type(values)(replacement if flag else v for v, flag in zip(values, flags))


def _lower_jaxpr_to_module(original):
    def lower(module_name, jaxpr, **kwargs):
        in_avals, out_avals = kwargs["in_avals"], kwargs["out_avals"]
        if tuple(kwargs["platforms"]) != (PLATFORM,):
            return original(module_name, jaxpr, **kwargs)
        # Apple's RemoveZeroDim pass removes empty computations/outputs, but
        # leaves empty input arguments. MPSGraph asserts when specializing them.
        # Reconstruct empty values from their static metadata inside the program.
        in_kinds = [EMPTY_INPUT if isinstance(a, core.ShapedArray)
                    and core.is_constant_shape(a.shape) and 0 in a.shape
                    and isinstance(a.dtype, np.dtype) else _kind(a) for a in in_avals]
        out_kinds = [_kind(a) for a in out_avals]
        applicable = ((any(in_kinds) or any(out_kinds))
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
        boundary_module = _compiler_ir.prepare_module(boundary_module)
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
