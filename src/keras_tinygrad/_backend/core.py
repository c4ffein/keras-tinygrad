import builtins
import contextlib
import os
import math
import threading
import warnings

import numpy as np
from tinygrad import Tensor
from tinygrad import dtypes as tg_dtypes
from tinygrad.dtype import DType

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = False
IS_THREAD_SAFE = True

TINYGRAD_DTYPES = {
    "float16": tg_dtypes.float16,
    "float32": tg_dtypes.float32,
    "float64": tg_dtypes.float64,
    "bfloat16": tg_dtypes.bfloat16,
    "uint8": tg_dtypes.uint8,
    "uint16": tg_dtypes.uint16,
    "uint32": tg_dtypes.uint32,
    "uint64": tg_dtypes.uint64,
    "int8": tg_dtypes.int8,
    "int16": tg_dtypes.int16,
    "int32": tg_dtypes.int32,
    "int64": tg_dtypes.int64,
    "bool": tg_dtypes.bool,
    "float8_e4m3fn": tg_dtypes.fp8e4m3,
    "float8_e5m2": tg_dtypes.fp8e5m2,
}
KERAS_DTYPES = {v: k for k, v in TINYGRAD_DTYPES.items()}

# Keras dtypes with an exactly matching native/ml_dtypes numpy dtype we can
# `astype` to before wrapping in a Tensor. bfloat16/float8 are excluded: they
# must be built as float32 buffers and cast on-device (tinygrad ignores the
# `dtype=` kwarg when constructing from a numpy buffer).
_NUMPY_NATIVE_DTYPES = frozenset(
    k for k in TINYGRAD_DTYPES if k != "bfloat16" and not k.startswith("float8")
)

# tinygrad's DType.__repr__ uses tinygrad spellings ("dtypes.half",
# "dtypes.fp8e4m3"). Keras user code (and keras' own tests) expect
# `str(tensor.dtype)` to contain the Keras dtype name ("float16",
# "float8_e4m3fn") like every other backend. Give DType a __str__ using the
# Keras names for the dtypes we map; __repr__ stays untouched so tinygrad's
# own debug output is unchanged.
_TG_DTYPE_REPR = DType.__repr__


def _dtype_str(self):
    keras_name = KERAS_DTYPES.get(self)
    if keras_name is not None:
        return f"dtypes.{keras_name}"
    return _TG_DTYPE_REPR(self)


DType.__str__ = _dtype_str


def to_tinygrad_dtype(dtype):
    dtype = standardize_dtype(dtype)
    if dtype not in TINYGRAD_DTYPES:
        raise ValueError(f"Unsupported dtype for tinygrad backend: {dtype}")
    return TINYGRAD_DTYPES[dtype]


def to_keras_dtype(tg_dtype):
    if tg_dtype == "complex64":
        # Complex-lite: `ComplexTensor.dtype` is already the Keras name
        # (there is no tinygrad complex DType to map).
        return "complex64"
    if tg_dtype not in KERAS_DTYPES:
        raise ValueError(f"Cannot map tinygrad dtype {tg_dtype} to Keras")
    return KERAS_DTYPES[tg_dtype]


def standardize_dtype_hook(dtype):
    """Plugin-protocol hook (see keras.src.backend.plugins): map tinygrad
    DType objects to Keras dtype names inside keras-core's
    `standardize_dtype`. tinygrad `DType.name` spellings ("float", "half",
    ...) don't match the Keras names, so the generic name-based fallbacks
    there would produce wrong strings. Returns None for non-tinygrad
    values (declining lets the generic handling proceed)."""
    if type(dtype).__module__.split(".")[0] == "tinygrad":
        return to_keras_dtype(dtype)
    return None


# --- complex-lite interop ---------------------------------------------------
# tinygrad has no complex dtypes. keras-core's `view_as_complex` /
# `view_as_real` (implemented in keras/src/ops/math.py, not
# backend-dispatched) only need a complex VALUE that can exist, enter, and
# leave the backend — not complex math. `ComplexTensor` is that value: a pair
# of float32 Tensors plus exactly the closed op set the keras-core
# implementations use (pairwise `+`, python-scalar `*` for the literal `1j`,
# `real`/`imag` accessors). Everything else raises NotImplementedError with
# one canonical message — a complex value must never silently produce float
# math (invariant: loud errors over wrong answers). Complex ARITHMETIC
# (tensor multiply, matmul, reductions, conjugate, ...) is deliberately out
# of scope; `SUPPORTS_COMPLEX_DTYPES` stays False.
_COMPLEX_INTEROP_MSG = (
    "complex64 tensors support only view_as_complex/view_as_real interop; "
    "complex arithmetic is not implemented in the tinygrad backend"
)


def _complex_guard(*_args, **_kwargs):
    raise NotImplementedError(_COMPLEX_INTEROP_MSG)


class ComplexTensor:
    """Complex-lite value: `.real` / `.imag` float32 Tensors, equal shape.

    Interop container, not a number type: exactly the closed op set that
    view_as_complex/view_as_real need; everything else raises loudly.
    Extending the op set is a tier-boundary crossing — see the pkg repo's
    docs/complex-support.md for the rule before adding anything here.
    """

    def __init__(self, real, imag):
        if not isinstance(real, Tensor) or not isinstance(imag, Tensor):
            raise TypeError(
                "ComplexTensor components must be tinygrad Tensors. "
                f"Received: {type(real)}, {type(imag)}"
            )
        if tuple(real.shape) != tuple(imag.shape):
            raise ValueError(
                "ComplexTensor components must have equal shapes. "
                f"Received: {tuple(real.shape)} and {tuple(imag.shape)}"
            )
        f32 = tg_dtypes.float32
        self.real = real if real.dtype == f32 else real.cast(f32)
        self.imag = imag if imag.dtype == f32 else imag.cast(f32)

    @property
    def shape(self):
        return tuple(self.real.shape)

    @property
    def ndim(self):
        return len(self.real.shape)

    @property
    def dtype(self):
        # A plain Keras dtype name; `standardize_dtype` validates strings
        # as-is, so no shim work is needed for the wrapper.
        return "complex64"

    def __repr__(self):
        return f"<ComplexTensor shape={self.shape} dtype=complex64>"

    # -- the closed op set (what keras-core's view_as_complex composes) ------
    def __add__(self, other):
        if isinstance(other, ComplexTensor):
            return ComplexTensor(
                self.real + other.real, self.imag + other.imag
            )
        if isinstance(other, complex):  # np.complex128 subclasses complex
            return ComplexTensor(
                self.real + other.real, self.imag + other.imag
            )
        if isinstance(other, (builtins.int, builtins.float)) and not isinstance(
            other, builtins.bool
        ):
            return ComplexTensor(self.real + other, self.imag)
        raise NotImplementedError(_COMPLEX_INTEROP_MSG)

    __radd__ = __add__  # addition commutes; wrapper+wrapper hits __add__

    def __mul__(self, other):
        # Python-scalar multiply only — needed for the literal `1j * x` in
        # keras-core's view_as_complex. (a+bi)(c+di) with (c+di) a scalar.
        if isinstance(other, complex):
            return ComplexTensor(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )
        if isinstance(other, (builtins.int, builtins.float)) and not isinstance(
            other, builtins.bool
        ):
            return ComplexTensor(self.real * other, self.imag * other)
        raise NotImplementedError(_COMPLEX_INTEROP_MSG)

    __rmul__ = __mul__  # scalar multiplication commutes

    # -- everything else: one loud, uniform refusal --------------------------
    # (No AttributeError leaks, no silent float math on a complex value.)
    __array__ = _complex_guard
    __bool__ = _complex_guard
    __float__ = _complex_guard
    __int__ = _complex_guard
    __index__ = _complex_guard
    __iter__ = _complex_guard
    __len__ = _complex_guard
    __getitem__ = _complex_guard
    __neg__ = _complex_guard
    __pos__ = _complex_guard
    __abs__ = _complex_guard
    __sub__ = _complex_guard
    __rsub__ = _complex_guard
    __truediv__ = _complex_guard
    __rtruediv__ = _complex_guard
    __matmul__ = _complex_guard
    __rmatmul__ = _complex_guard
    __pow__ = _complex_guard
    __rpow__ = _complex_guard

    def __getattr__(self, name):
        if name.startswith("_"):
            # Normal missing-attribute protocol for private/dunder probes so
            # hasattr()-based machinery (copy, pytest repr, numpy protocol
            # sniffing) degrades gracefully instead of exploding mid-probe.
            raise AttributeError(name)
        raise NotImplementedError(_COMPLEX_INTEROP_MSG)


def _is_complex_value(x):
    if isinstance(x, complex):  # covers np.complex128 scalars too
        return True
    return getattr(getattr(x, "dtype", None), "kind", None) == "c"


def _convert_to_complex(x, dtype):
    """Complex-lite entry: build or route a `ComplexTensor`.

    `dtype` is already standardized, or None. Only complex64 is
    representable; complex128 host data lands as complex64 (the backend is
    single-precision-default, like the float side).
    """
    if dtype == "complex128":
        raise NotImplementedError(
            "complex128 is not supported by the tinygrad backend; "
            + _COMPLEX_INTEROP_MSG
        )
    if isinstance(x, ComplexTensor):
        if dtype in (None, "complex64"):
            return x
        # numpy would warn and drop the imaginary part; we refuse loudly.
        raise NotImplementedError(
            f"Cannot cast a complex64 tensor to '{dtype}' (this would drop "
            "the imaginary part); use view_as_real/real/imag explicitly. "
            + _COMPLEX_INTEROP_MSG
        )
    if isinstance(x, Variable):
        x = x.value
    if isinstance(x, Tensor):
        # The `cast(real_part, "complex64")` half of view_as_complex: a real
        # tensor becomes the real component, imaginary part zero. The real
        # component stays a lazy Tensor graph — differentiable w.r.t. x.
        real = x.cast(tg_dtypes.float32)
        imag = Tensor.zeros(
            *[builtins.int(d) for d in x.shape], dtype=tg_dtypes.float32
        )
        return ComplexTensor(real, imag)
    arr = np.asarray(x)
    if arr.dtype.kind == "c":
        # complex64/complex128 host data -> complex64 wrapper. Components go
        # through convert_to_tensor for copy-on-convert (`.real`/`.imag` are
        # views into the complex buffer) and shape-() scalar handling.
        real = convert_to_tensor(
            np.ascontiguousarray(arr.real), dtype="float32"
        )
        imag = convert_to_tensor(
            np.ascontiguousarray(arr.imag), dtype="float32"
        )
        return ComplexTensor(real, imag)
    # Explicit complex64 request on real host-side data: nothing in the
    # interop paths needs it; refuse rather than guess.
    raise NotImplementedError(
        "convert_to_tensor(..., dtype='complex64') on real host data is not "
        "supported; " + _COMPLEX_INTEROP_MSG
    )


class Variable(KerasVariable):
    def _initialize(self, value):
        value = convert_to_tensor(value, dtype=self._dtype)
        # `.contiguous().realize()` keeps the stored value a concrete buffer
        # rather than an unbounded lazy graph accumulating across training
        # steps. The `.contiguous()` is load-bearing for scalar-initialized
        # variables (an SGD learning rate): convert_to_tensor gives a CONST
        # uop, realize alone is a no-op on it, and a const-backed Variable
        # bakes into traces as an immediate — the exported lr could never be
        # changed again. ORDER matters: contiguous() BEFORE detach() — a
        # DETACH in front defeats tinygrad's buffer-identity short-circuit,
        # turning every assign of an already-realized tensor into a full
        # copy kernel plus a fresh allocation (measured; see
        # tests/test_backend_regressions.py).
        self._value = value.contiguous().detach().realize()
        self._value.requires_grad = bool(self.trainable)

    def _direct_assign(self, value):
        value = convert_to_tensor(value, dtype=self._dtype)
        self._value = value.contiguous().detach().realize()
        self._value.requires_grad = bool(self.trainable)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor. Full numpy protocol signature (numpy 2.x
    # passes `dtype`/`copy`), matching the Tensor-level `__array__` patch.
    def __array__(self, dtype=None, copy=None):
        arr = convert_to_numpy(self._value)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with tinygrad backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with tinygrad backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if (
        isinstance(x, ComplexTensor)
        or dtype in ("complex64", "complex128")
        or _is_complex_value(x)
    ):
        return _convert_to_complex(x, dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return x.value.cast(to_tinygrad_dtype(dtype))
        return x.value
    if isinstance(x, Tensor):
        if dtype and dtype != to_keras_dtype(x.dtype):
            return x.cast(to_tinygrad_dtype(dtype))
        return x
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
        if dtype in ("complex64", "complex128"):
            # e.g. a python list of complex scalars.
            return _convert_to_complex(x, dtype)
    if isinstance(x, (list, tuple)) and builtins.any(
        isinstance(e, (Tensor, Variable)) for e in x
    ):
        # np.asarray chokes on sequences containing tinygrad Tensors, and a
        # numpy round-trip would detach gradients — stack natively instead.
        elems = [convert_to_tensor(e, dtype=dtype) for e in x]
        if builtins.all(tuple(e.shape) == tuple(elems[0].shape) for e in elems):
            out = Tensor.stack(*elems)
            return out
        # Ragged shapes: fall through and let np.asarray raise its usual
        # error via the numpy interop path.
        x = tree.map_structure(
            lambda e: convert_to_numpy(e)
            if isinstance(e, (Tensor, Variable))
            else e,
            x,
        )
    if isinstance(x, (bool, int, float)):
        # Python scalars MUST become tinygrad CONST UOps, never buffer-backed
        # tensors. The numpy path below promotes 0-d to shape (1,) (that's
        # np.ascontiguousarray) and wraps the buffer — an anonymous input
        # buffer that export_model cannot save: SGD's momentum=0.9 exported
        # as a zeroed createEmptyBuf and the browser bundle silently trained
        # plain SGD. A const folds into kernels as an immediate.
        return Tensor(x, dtype=to_tinygrad_dtype(dtype))
    arr = np.asarray(x)
    tg_dtype = to_tinygrad_dtype(dtype)
    if dtype in _NUMPY_NATIVE_DTYPES:
        if arr.dtype != dtype:
            arr = arr.astype(dtype)  # astype copies
        else:
            # MUST copy: tinygrad wraps numpy buffers zero-copy and reads them
            # lazily at realize time, but Keras assumes value semantics at
            # conversion (data adapters recycle batch buffers — without the
            # copy, the first post-fit predict computes on clobbered memory).
            arr = arr.copy()
    else:
        # bfloat16/float8: build a float32 buffer and cast on-device.
        # (Constructing a Tensor from a numpy buffer silently ignores a
        # mismatched `dtype=` kwarg, so the cast must be explicit.)
        arr = arr.astype("float32")  # astype copies
    out = Tensor(np.ascontiguousarray(arr))
    if out.dtype != tg_dtype:
        out = out.cast(tg_dtype)
    if arr.shape == () and tuple(out.shape) != ():
        # tinygrad wraps 0-d numpy input as shape (1,); Keras needs real
        # scalars (metric variables are shape ()).
        out = out.reshape(())
    return out


def convert_to_numpy(x):
    if isinstance(x, ComplexTensor):
        # Terminal exit for complex-lite values (never a gradient path).
        return (
            x.real.detach().numpy() + 1j * x.imag.detach().numpy()
        ).astype(np.complex64)
    if isinstance(x, Variable):
        x = x.value
    if isinstance(x, Tensor):
        keras_dtype = KERAS_DTYPES.get(x.dtype)
        if keras_dtype is not None and keras_dtype.startswith("float8"):
            # tinygrad's `.numpy()` materializes fp8 tensors as plain float32
            # arrays; every other backend hands back an ml_dtypes float8
            # array, so re-quantize the buffer to the true storage dtype.
            # (Terminal conversion — never a gradient path.)
            import ml_dtypes

            ml_dtype = (
                ml_dtypes.float8_e4m3fn
                if keras_dtype == "float8_e4m3fn"
                else ml_dtypes.float8_e5m2
            )
            return x.detach().cast(tg_dtypes.float32).numpy().astype(ml_dtype)
        return x.detach().numpy()
    return np.array(x)


def is_tensor(x):
    # ComplexTensor counts: it is a backend-native value, and the test
    # harness routes anything `is_tensor` through `convert_to_numpy`.
    return isinstance(x, (Tensor, ComplexTensor))


def shape(x):
    return tuple(x.shape)


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


def cond(pred, true_fn, false_fn):
    if isinstance(pred, Tensor):
        pred = pred.numpy().item()
    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    if not isinstance(elements, (list, tuple)):
        return Tensor.stack(
            *[function(elements[i]) for i in range(elements.shape[0])]
        )
    batch_size = elements[0].shape[0]
    outputs = [
        function([x[index] for x in elements]) for index in range(batch_size)
    ]
    return Tensor.stack(*outputs)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope(), SymbolicScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )

        def convert_keras_tensor_to_tinygrad(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                return Tensor.zeros(
                    *shape, dtype=to_tinygrad_dtype(x.dtype)
                )
            return x

        args_1, kwargs_1 = tree.map_structure(
            lambda x: convert_keras_tensor_to_tinygrad(x, fill_value=83),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        if none_in_shape:
            args_2, kwargs_2 = tree.map_structure(
                lambda x: convert_keras_tensor_to_tinygrad(x, fill_value=89),
                (args, kwargs),
            )
            outputs_2 = fn(*args_2, **kwargs_2)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != shape[i]:
                        shape[i] = None
                flat_out.append(
                    KerasTensor(shape, to_keras_dtype(x1.dtype))
                )
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        def convert_tinygrad_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(tuple(x.shape), to_keras_dtype(x.dtype))
            return x

        output_spec = tree.map_structure(
            convert_tinygrad_to_keras_tensor, outputs
        )
    return output_spec


def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    # Python-loop implementation, mirroring the numpy backend.
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    if xs is None and length is None:
        raise ValueError("Got no `xs` to scan over and `length` not provided.")

    input_is_sequence = tree.is_nested(xs)
    output_is_sequence = tree.is_nested(init)

    def pack_input(x):
        return tree.pack_sequence_as(xs, x) if input_is_sequence else x[0]

    def pack_output(x):
        return tree.pack_sequence_as(init, x) if output_is_sequence else x[0]

    if xs is None:
        xs_flat = []
        n = int(length)
    else:
        xs_flat = tree.flatten(xs)
        xs_flat = [convert_to_tensor(elem) for elem in xs_flat]
        n = int(length) if length is not None else shape(xs_flat[0])[0]

    init_flat = tree.flatten(init)
    init_flat = [convert_to_tensor(init) for init in init_flat]
    init = pack_output(init_flat)
    dummy_y = [Tensor.zeros(*init.shape, dtype=init.dtype) for init in init_flat]

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    stacked_y = tree.map_structure(
        lambda *ys: Tensor.stack(*ys), *maybe_reversed(ys)
    )
    return carry, stacked_y


def _slice_along_axis(x, start=0, stop=None, step=1, axis=0):
    index = [builtins.slice(None)] * x.ndim
    index[axis] = builtins.slice(start, stop, step)
    return x[tuple(index)]


def associative_scan(f, elems, reverse=False, axis=0):
    # Ref: jax.lax.associative_scan (mirrors the numpy backend).
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    elems_flat = tree.flatten(elems)
    elems_flat = [convert_to_tensor(elem) for elem in elems_flat]
    if axis < 0:
        axis += elems_flat[0].ndim
    if reverse:
        elems_flat = [elem.flip(axis) for elem in elems_flat]

    def _combine(a_flat, b_flat):
        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        return tree.flatten(f(a, b))

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [tuple(elem.shape) for elem in elems_flat]
            )
        )

    def _dilate(x):
        # [x1, x2, ..., xn] -> [x1, 0, x2, 0, ..., xn] (length 2n-1) along
        # `axis`, via interleave-with-zeros through an extra dim (tinygrad
        # has no item assignment to copy into a strided view).
        n = int(x.shape[axis])
        x = x.unsqueeze(axis + 1)
        pads = [(0, 0)] * x.ndim
        pads[axis + 1] = (0, 1)
        x = x.pad(tuple(pads))
        merged_shape = list(x.shape)
        merged_shape[axis] = 2 * n
        del merged_shape[axis + 1]
        x = x.reshape(tuple(int(d) for d in merged_shape))
        return _slice_along_axis(x, 0, 2 * n - 1, axis=axis)

    def _interleave(a, b):
        """Given two Tensors of static shape, interleave them along axis."""
        n_a = int(a.shape[axis])
        n_b = int(b.shape[axis])
        if not (n_a == n_b or n_a == n_b + 1):
            raise ValueError(
                "Shapes are incompatible for associative_scan interleaving. "
                f"a.shape[{axis}]={n_a}, b.shape[{axis}]={n_b}"
            )
        # we want to get a: [a1, a2], b: [b1, b2]
        # to a: [a1, 0, a2, 0], b: [0, b1, 0, b2]
        a_dil = _dilate(a)
        b_dil = _dilate(b)
        a_pad = [(0, 0)] * a.ndim
        a_pad[axis] = (0, 1 if n_a == n_b else 0)
        b_pad = [(0, 0)] * b.ndim
        b_pad[axis] = (1, 0) if n_a == n_b else (1, 1)
        a_dil = a_dil.pad(tuple(a_pad))
        b_dil = b_dil.pad(tuple(b_pad))
        if a.dtype == tg_dtypes.bool:
            return a_dil | b_dil
        return a_dil + b_dil

    def _scan(elems):
        num_elems = int(elems[0].shape[axis])
        if num_elems < 2:
            return elems

        reduced_elems = _combine(
            [
                _slice_along_axis(elem, 0, -1, step=2, axis=axis)
                for elem in elems
            ],
            [
                _slice_along_axis(elem, 1, None, step=2, axis=axis)
                for elem in elems
            ],
        )

        odd_elems = _scan(reduced_elems)
        if num_elems % 2 == 0:
            even_elems = _combine(
                [_slice_along_axis(e, 0, -1, axis=axis) for e in odd_elems],
                [
                    _slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )
        else:
            even_elems = _combine(
                odd_elems,
                [
                    _slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )

        even_elems = [
            _slice_along_axis(elem, 0, 1, axis=axis).cat(result, dim=axis)
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(builtins.map(_interleave, even_elems, odd_elems))

    scans = _scan(elems_flat)
    if reverse:
        scans = [scanned.flip(axis) for scanned in scans]

    return tree.pack_sequence_as(elems, scans)


def _index_int(i):
    """Static python int from an index that may be a Tensor or numpy scalar."""
    if isinstance(i, Tensor):
        return int(i.item())
    return int(i)


def _flat_index_parts(inputs_shape, indices, index_length):
    """Shared scatter helper: flatten index vectors against `inputs_shape`.

    Returns `(flat_idx, m)` where `flat_idx` is a 1-D int32 tensor of
    flattened positions into the first `index_length` dims and `m` is the
    number of such positions.
    """
    lead = [int(d) for d in inputs_shape[:index_length]]
    m = math.prod(lead)
    strides = []
    acc = 1
    for d in reversed(lead):
        strides.insert(0, acc)
        acc *= d
    indices = indices.reshape(-1, index_length).cast(tg_dtypes.int32)
    flat_idx = (indices * Tensor(strides, dtype=tg_dtypes.int32)).sum(axis=-1)
    return flat_idx, m


def scatter(indices, values, shape):
    # numpy reference: zeros of `shape`, then `np.add.at` (duplicate indices
    # accumulate). Built here as one-hot matmul: rows sum per target slot.
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    index_length = int(indices.shape[-1])
    value_shape = tuple(int(d) for d in shape[index_length:])
    v_size = math.prod(value_shape)
    flat_idx, m = _flat_index_parts(shape, indices, index_length)
    num_updates = int(flat_idx.shape[0])
    values = values.reshape(num_updates, v_size)
    onehot = Tensor.arange(m, dtype=tg_dtypes.int32).reshape(
        m, 1
    ) == flat_idx.reshape(1, num_updates)
    out_dtype = values.dtype
    if out_dtype == tg_dtypes.bool:
        result = (onehot.cast(tg_dtypes.int32) @ values.cast(tg_dtypes.int32)) > 0
    else:
        result = onehot.cast(out_dtype) @ values
    return result.reshape(tuple(int(d) for d in shape))


def scatter_update(inputs, indices, updates, reduction=None):
    # tinygrad has no in-place item assignment: apply each update as a
    # masked `where` over the (flattened) input, in order, so duplicate
    # indices behave exactly like the numpy reference (`inputs[idx] = ...`
    # is last-write-wins; `np.<op>.at` applies once per occurrence).
    if reduction not in (None, "add", "max", "min", "mul"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices)
    updates = convert_to_tensor(updates)
    if updates.dtype != inputs.dtype:
        updates = updates.cast(inputs.dtype)
    index_length = int(indices.shape[-1])
    in_shape = tuple(int(d) for d in inputs.shape)
    v_size = math.prod(in_shape[index_length:])
    flat_idx, m = _flat_index_parts(in_shape, indices, index_length)
    num_updates = int(flat_idx.shape[0])
    out = inputs.reshape(m, v_size)
    updates = updates.reshape(num_updates, v_size)
    rows = Tensor.arange(m, dtype=tg_dtypes.int32).reshape(m, 1)
    for i in range(num_updates):
        mask = rows == flat_idx[i]
        u = updates[i].reshape(1, v_size)
        if reduction is None:
            out = mask.where(u, out)
        elif reduction == "add":
            out = mask.where(out + u, out)
        elif reduction == "max":
            out = mask.where(out.maximum(u), out)
        elif reduction == "min":
            out = mask.where(out.minimum(u), out)
        else:  # "mul"
            out = mask.where(out * u, out)
    return out.reshape(in_shape)


def slice(inputs, start_indices, shape):
    if len(start_indices) != len(shape):
        raise ValueError(
            "Length of `start_indices` must match length of `shape`. "
            f"Received: start_indices={start_indices}, shape={shape}"
        )
    inputs = convert_to_tensor(inputs)
    index = tuple(
        builtins.slice(_index_int(start), _index_int(start) + _index_int(length))
        for start, length in zip(start_indices, shape)
    )
    return inputs[index]


def slice_update(inputs, start_indices, updates):
    # No in-place assignment in tinygrad: pad the update block (and a
    # same-shaped True mask) out to the full input shape, then `where`.
    inputs = convert_to_tensor(inputs)
    updates = convert_to_tensor(updates)
    if updates.dtype != inputs.dtype:
        updates = updates.cast(inputs.dtype)
    starts = [_index_int(s) for s in start_indices]
    pads = tuple(
        (start, int(dim) - start - int(usize))
        for start, dim, usize in zip(starts, inputs.shape, updates.shape)
    )
    mask = Tensor.ones(*updates.shape, dtype=tg_dtypes.bool).pad(pads)
    return mask.where(updates.pad(pads), inputs)


def switch(index, branches, *operands):
    index = _index_int(index)
    index = builtins.min(builtins.max(index, 0), len(branches) - 1)
    return branches[index](*operands)


def stop_gradient(variable):
    if isinstance(variable, Variable):
        variable = variable.value
    return variable.detach()


def unstack(x, num=None, axis=0):
    x = convert_to_tensor(x)
    if axis != 0:
        order = [axis] + [i for i in range(x.ndim) if i != axis]
        x = x.permute(order)
    return [x[i] for i in range(x.shape[0])]


def while_loop(cond, body, loop_vars, maximum_iterations=None):
    current_iter = 0
    iteration_check = lambda iter: (
        maximum_iterations is None or iter < maximum_iterations
    )
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    loop_vars = tree.map_structure(convert_to_tensor, loop_vars)

    def eval_cond(*args):
        result = cond(*args)
        if isinstance(result, Tensor):
            result = result.numpy().item()
        return result

    while eval_cond(*loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars,)
        loop_vars = tuple(loop_vars)
        current_iter += 1
    return loop_vars if is_tuple else loop_vars[0]


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def random_seed_dtype():
    return "uint32"


# --- custom gradients -------------------------------------------------------
# tinygrad has no native custom-VJP hook, but `Tensor.gradient` can compute
# the gradient of any tensor w.r.t. any other tensors in its graph, accepts an
# explicit seed (`gradient=`), and returns zeros for disconnected targets.
# That is enough to compose VJPs by hand:
#
#   * While a "tape" is active (the trainer opens one around the forward
#     pass), each `custom_gradient` block detaches its output — cutting
#     tinygrad's autograd path through the block internals — and records
#     `(args, detached_output, grad_fn)`.
#   * `compute_gradients` then runs the chain rule around the recorded
#     blocks: walking blocks in reverse creation order, it accumulates the
#     upstream gradient at each block's detached output (from the loss and
#     from later blocks' argument flows), calls the user `grad_fn` on it —
#     torch-backend calling convention: `grad_fn(*args, upstream=up)` — and
#     turns the returned per-argument gradients into new VJP seed points.
#     Reverse creation order is sufficient: a block's output can only feed
#     computations created after it.
#
# Without an active tape (predict/evaluate/symbolic build), blocks act as a
# forward passthrough, which is all those paths need.
_custom_gradient_tape = threading.local()

# Device-RNG scope: random ops draw on-device (threefry, in-graph) ONLY
# inside this scope — entered by the trainer around the train step, and by
# export drivers around their traced step. Outside it (initializers, the
# global seed generator, direct keras.random.* calls) sampling stays on
# the host, bit-identical to the reference backend. docs/device-rng.md.
def device_rng_enabled():
    """Single authority for the KERAS_TINYGRAD_DEVICE_RNG flag (default on).
    Read it through this function only — independent env reads in different
    modules invite drift between the trainer's JIT gate and the sampling
    path."""
    return os.environ.get("KERAS_TINYGRAD_DEVICE_RNG", "1") == "1"


_device_rng_scope = threading.local()


@contextlib.contextmanager
def device_rng_scope():
    prev = getattr(_device_rng_scope, "active", False)
    _device_rng_scope.active = True
    try:
        yield
    finally:
        _device_rng_scope.active = prev


def in_device_rng_scope():
    return getattr(_device_rng_scope, "active", False)


@contextlib.contextmanager
def custom_gradient_tape():
    """Record `custom_gradient` blocks for `compute_gradients`."""
    prev = getattr(_custom_gradient_tape, "blocks", None)
    _custom_gradient_tape.blocks = []
    try:
        yield _custom_gradient_tape.blocks
    finally:
        _custom_gradient_tape.blocks = prev


class custom_gradient:
    """Decorator for custom gradients.

    Inside a `custom_gradient_tape` the custom backward function is honored
    (via VJP composition in `compute_gradients`); outside one, the block is
    a forward passthrough.
    """

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        outputs, grad_fn = self.fun(*args, **kwargs)
        blocks = getattr(_custom_gradient_tape, "blocks", None)
        if blocks is None:
            return outputs
        if not isinstance(outputs, Tensor):
            raise NotImplementedError(
                "The tinygrad backend only supports `custom_gradient` "
                "functions returning a single tensor output. "
                f"Received: {type(outputs)}"
            )
        proxy = outputs.detach()
        # tinygrad hash-conses structurally identical graphs, so two
        # identical blocks over the same inputs would share ONE proxy UOp —
        # and `compute_gradients` would then hand each grad_fn the
        # accumulated upstream of BOTH uses (silent double-counting).
        # De-alias by adding a zero from a fresh realized buffer: distinct
        # buffers never merge (same trick as linalg's jvp dummy cotangents),
        # and x + 0.0 is exact. Re-detach so the add itself cannot leak
        # gradient from this proxy back into the aliased one.
        if builtins.any(p.uop is proxy.uop for _, p, _ in blocks):
            zero = Tensor.zeros(
                (), dtype=proxy.dtype, device=proxy.device
            ).realize()
            proxy = (proxy + zero).detach()
        blocks.append((args, proxy, grad_fn))
        return proxy


def _vjp(root, targets, seed):
    """Gradients of `root` w.r.t. `targets`, seeded with `seed` (or 1.0)."""
    if seed is None:
        return root.gradient(*targets)
    return root.gradient(*targets, gradient=seed)


def compute_gradients(loss, targets, blocks=None):
    """Gradients of `loss` w.r.t. `targets`, honoring recorded custom blocks.

    `blocks` is the list yielded by `custom_gradient_tape` (in forward
    creation order). With no blocks this is exactly
    `loss.gradient(*targets)`.
    """
    targets = list(targets)
    if not blocks:
        return list(loss.gradient(*targets))
    # (root, seed) pairs to backpropagate from; seed None means scalar 1.0.
    sources = [(loss, None)]
    for args, proxy, grad_fn in reversed(blocks):
        upstream = None
        for root, seed in sources:
            g = _vjp(root, [proxy], seed)[0]
            upstream = g if upstream is None else upstream + g
        arg_grads = grad_fn(*args, upstream=upstream)
        if not isinstance(arg_grads, (list, tuple)):
            arg_grads = (arg_grads,)
        if len(arg_grads) != len(args):
            raise ValueError(
                "custom_gradient function returned "
                f"{len(arg_grads)} gradients for {len(args)} arguments."
            )
        for arg, g in builtins.zip(args, arg_grads):
            if g is None or not isinstance(arg, Tensor):
                continue
            if not isinstance(g, Tensor):
                g = convert_to_tensor(g)
            sources.append((arg, g))
    totals = [None] * len(targets)
    for root, seed in sources:
        grads = _vjp(root, targets, seed)
        for i, g in enumerate(grads):
            totals[i] = g if totals[i] is None else totals[i] + g
    return totals


@contextlib.contextmanager
def device_scope(device_name):
    yield


def remat(f):
    warnings.warn(
        "Rematerialization memory optimization is not supported by the "
        "tinygrad backend. It has no effect."
    )
    return f
