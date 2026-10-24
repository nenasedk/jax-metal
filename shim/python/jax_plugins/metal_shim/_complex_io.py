"""complex64 values as METAL program inputs and outputs.

The plugin rejects any compiled function whose signature mentions complex types. For a
METAL program with complex64 inputs or outputs this module lowers two versions:

- the original, with its complex signature, which jaxlib compiles and takes its metadata
  (output types, shapes, layouts) from;
- a pair version in which each complex64 input/output is a float32[..., 2] (real, imag)
  pair, converted inside the program (which _complex.py supports), serialized here as a
  StableHLO 0.19.4 portable artifact.

Right before compilation the shim receives the pair program and the complex positions and
compiles the pair program instead. At execution it passes complex64 inputs as
float32[..., 2] views and returns pair outputs as complex64 views of the same device
memory, so JAX only ever sees genuine complex64 arrays and nothing is copied.

Hooks into JAX internals: mlir.lower_jaxpr_to_module (lower the pair version) and
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
_POSITIONS = "jax_metal.complex_positions"
_PROGRAM = "jax_metal.pair_program"

_set_complex_io = None


def _is_complex(aval):
    # Extended dtypes (PRNG keys) are not NumPy dtypes and never complex.
    dtype = getattr(aval, "dtype", None)
    return isinstance(aval, core.ShapedArray) and isinstance(dtype, np.dtype) and dtype == np.complex64


def _pair_aval(aval):
    return core.ShapedArray(tuple(aval.shape) + (2,), np.float32)


def _to_pair(z):
    real, imag = lax.real(z), lax.imag(z)
    return lax.concatenate([lax.expand_dims(real, (real.ndim,)),
                            lax.expand_dims(imag, (imag.ndim,))], real.ndim)


def _from_pair(pair):
    return lax.complex(lax.index_in_dim(pair, 0, pair.ndim - 1, keepdims=False),
                       lax.index_in_dim(pair, 1, pair.ndim - 1, keepdims=False))


def _rewrite(closed_jaxpr, in_flags, out_flags):
    """A jaxpr like closed_jaxpr whose flagged inputs/outputs are float32[..., 2] pairs."""
    in_avals = [_pair_aval(a) if flag else a for a, flag in zip(closed_jaxpr.in_avals, in_flags)]

    def body(*args):
        args = [_from_pair(a) if flag else a for a, flag in zip(args, in_flags)]
        outs = core.jaxpr_as_fun(closed_jaxpr)(*args)
        return [_to_pair(o) if flag else o for o, flag in zip(outs, out_flags)]

    shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in in_avals]
    return jax.make_jaxpr(body)(*shapes), in_avals


def _masked(values, flags, replacement):
    if values is None:
        return None
    return type(values)(replacement if flag else v for v, flag in zip(values, flags))


def _lower_jaxpr_to_module(original):
    def lower(module_name, jaxpr, **kwargs):
        in_avals, out_avals = kwargs["in_avals"], kwargs["out_avals"]
        in_flags = [_is_complex(a) for a in in_avals]
        out_flags = [_is_complex(a) for a in out_avals]
        applicable = (tuple(kwargs["platforms"]) == (PLATFORM,)
                      and (any(in_flags) or any(out_flags))
                      and kwargs.get("num_const_args", 0) == 0
                      and len(jaxpr.in_avals) == len(in_avals)
                      and all(core.is_constant_shape(a.shape) for a in [*in_avals, *out_avals]
                              if isinstance(a, core.ShapedArray)))
        if not applicable:
            return original(module_name, jaxpr, **kwargs)
        # The plugin runs the pair program, which never aliases complex inputs; drop their
        # donation here too so JAX does not warn about unusable donated buffers.
        kwargs = dict(kwargs)
        kwargs["donated_args"] = _masked(kwargs["donated_args"], in_flags, False)
        if kwargs.get("input_output_aliases") is not None:
            kwargs["input_output_aliases"] = _masked(kwargs["input_output_aliases"], in_flags, None)
        result = original(module_name, jaxpr, **kwargs)
        pair_jaxpr, pair_in_avals = _rewrite(jaxpr, in_flags, out_flags)
        pair_kwargs = dict(kwargs)
        pair_kwargs["in_avals"] = pair_in_avals
        pair_kwargs["out_avals"] = [_pair_aval(a) if flag else a
                                    for a, flag in zip(out_avals, out_flags)]
        # Pair positions use default layouts and shardings.
        for key, flags in (("in_layouts", in_flags), ("arg_shardings", in_flags),
                           ("out_layouts", out_flags), ("result_shardings", out_flags)):
            pair_kwargs[key] = _masked(kwargs.get(key), flags, None)
        pair_module = original(module_name, pair_jaxpr, **pair_kwargs).module
        program = _jax.mlir.serialize_portable_artifact(pair_module, STABLEHLO_TARGET, True)
        positions = lambda flags: ",".join(str(i) for i, flag in enumerate(flags) if flag)
        with result.module.context:
            attributes = result.module.operation.attributes
            attributes[_POSITIONS] = ir.StringAttr.get(
                f"{positions(in_flags)};{positions(out_flags)}")
            attributes[_PROGRAM] = ir.StringAttr.get(base64.b64encode(program).decode())
        return result

    return lower


def _compile_or_get_cached(original):
    def compile_(backend, computation, *args, **kwargs):
        attributes = computation.operation.attributes
        if _POSITIONS not in attributes:
            return original(backend, computation, *args, **kwargs)
        inputs, outputs = ([int(p) for p in part.split(",") if p] for part in
                           ir.StringAttr(attributes[_POSITIONS]).value.split(";"))
        program = base64.b64decode(ir.StringAttr(attributes[_PROGRAM]).value)
        del attributes[_POSITIONS]  # jaxlib must not serialize these for the plugin
        del attributes[_PROGRAM]
        as_array = lambda values: (ctypes.c_int32 * max(len(values), 1))(*values)
        _set_complex_io(as_array(inputs), len(inputs), as_array(outputs), len(outputs),
                        program, len(program))
        try:
            return original(backend, computation, *args, **kwargs)
        finally:
            _set_complex_io(None, 0, None, 0, None, 0)

    return compile_


def register(shim_library):
    global _set_complex_io  # pylint: disable=global-statement
    library = ctypes.CDLL(str(shim_library))
    _set_complex_io = library.JaxMetalShim_SetComplexIo
    _set_complex_io.restype = None
    _set_complex_io.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                                ctypes.c_char_p, ctypes.c_size_t]
    mlir.lower_jaxpr_to_module = _lower_jaxpr_to_module(mlir.lower_jaxpr_to_module)
    compiler.compile_or_get_cached = _compile_or_get_cached(compiler.compile_or_get_cached)
