"""tinygrad implementations of the `keras.ops.numpy` surface.

Implemented ops map directly onto tinygrad Tensor methods. Anything not yet
ported raises `NotImplementedError` via the module-level `__getattr__` (PEP
562) or an explicit raise — there are NO silent numpy fallbacks, because a
fallback would silently detach gradients. The op-coverage tally against
Keras' own test suite is only honest if missing means loud.
"""

import builtins
import math as _math
import operator

import numpy as np
from tinygrad import Tensor

from keras.src import tree
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.config import floatx
from keras.src.backend.tinygrad.core import _COMPLEX_INTEROP_MSG
from keras.src.backend.tinygrad.core import ComplexTensor
from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.backend.tinygrad.core import to_keras_dtype
from keras.src.backend.tinygrad.core import to_tinygrad_dtype


# tinygrad's `Tensor.__bool__` raises unconditionally, but Keras — and its
# test suite, e.g. `assertEqual(loss, 0.0)` in losses_test — relies on
# numpy/torch-style scalar truthiness: `bool(t)` on a one-element tensor
# realizes it and converts. Install the numpy-matching behavior once, at
# backend import time (this module is imported by the backend package
# `__init__`). Multi-element tensors still raise, mirroring numpy's
# ambiguity error. This patch would sit more naturally in `core.py`; it
# lives here because this module owns the numpy-surface semantics.
def _tensor_bool(self):
    n = 1
    for d in self.shape:
        n *= d
    if n != 1:
        raise TypeError(
            "The truth value of a Tensor with more than one element is "
            "ambiguous. Use `.any()` or `.all()`."
        )
    return builtins.bool(self.item())


Tensor.__bool__ = _tensor_bool


# numpy interop for tuple-returning ops (`nonzero`, `unravel_index`) and the
# test harness: `np.array(tuple_of_tensors)` needs each element to expose
# `__array__`. Conversion realizes the tensor (numpy semantics, like torch's
# tensor `__array__`) — op implementations must still never round-trip
# through numpy themselves.
def _tensor_array(self, dtype=None, copy=None):
    out = self.detach().numpy()
    return out.astype(dtype) if dtype is not None else out


Tensor.__array__ = _tensor_array


# Scalar interop for the rest of the numeric protocol: Keras user code and
# the test suite do `float(t)` / `int(t)` / `assertAlmostEqual(t, x)` on
# 0-d tensors, matching numpy/torch backends. Guarded like `__bool__`:
# added only, never overriding an existing tinygrad implementation.
def _tensor_scalar(convert):
    def method(self):
        n = 1
        for d in self.shape:
            n *= d
        if n != 1:
            raise TypeError(
                "Only one-element tensors can be converted to Python scalars"
            )
        return convert(self.item())

    return method


if not hasattr(Tensor, "__float__"):
    Tensor.__float__ = _tensor_scalar(builtins.float)
if not hasattr(Tensor, "__int__"):
    Tensor.__int__ = _tensor_scalar(builtins.int)
if not hasattr(Tensor, "__index__"):
    Tensor.__index__ = _tensor_scalar(builtins.int)


def _dtype_of(x):
    if isinstance(x, Tensor):
        return to_keras_dtype(x.dtype)
    if hasattr(x, "dtype"):
        return standardize_dtype(x.dtype)
    return type(x)


def _pair(x1, x2):
    """Convert a binary-op operand pair, honoring Keras dtype promotion.

    Python scalars stay scalars (tinygrad broadcasts them without changing
    the tensor operand's dtype, which matches Keras' weak-type behavior).
    """
    s1 = isinstance(x1, (builtins.int, builtins.float, builtins.bool))
    s2 = isinstance(x2, (builtins.int, builtins.float, builtins.bool))
    if s1 and s2:
        return convert_to_tensor(x1), x2
    if s1:
        return x1, convert_to_tensor(x2)
    if s2:
        return convert_to_tensor(x1), x2
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    return convert_to_tensor(x1, dtype), convert_to_tensor(x2, dtype)


def _float(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        dtype = result_type(to_keras_dtype(x.dtype), float)
        x = x.cast(to_tinygrad_dtype(dtype))
    return x


def _norm_axis(axis, ndim):
    if axis is None:
        return None
    if isinstance(axis, (list, tuple)):
        return tuple(a % ndim if a < 0 else a for a in axis)
    return axis % ndim if axis < 0 else axis


# ---- arithmetic -------------------------------------------------------------


def add(x1, x2):
    a, b = _pair(x1, x2)
    return a + b


def subtract(x1, x2):
    a, b = _pair(x1, x2)
    return a - b


def multiply(x1, x2):
    a, b = _pair(x1, x2)
    return a * b


def divide(x1, x2):
    a, b = _pair(x1, x2)
    if isinstance(a, Tensor) and "float" not in to_keras_dtype(a.dtype):
        a = _float(a)
    if isinstance(b, Tensor) and "float" not in to_keras_dtype(b.dtype):
        b = _float(b)
    return a / b


true_divide = divide


def floor_divide(x1, x2):
    a, b = _pair(x1, x2)
    return a // b


def divide_no_nan(x1, x2):
    a, b = _pair(x1, x2)
    a, b = _float(a) if isinstance(a, Tensor) else a, b
    quotient = a / b
    bt = b if isinstance(b, Tensor) else convert_to_tensor(b)
    return (bt == 0).where(0.0, quotient)


def matmul(x1, x2):
    # The int8-to-int32 jax rule keys on the ORIGINAL operand dtypes: an
    # int8 x bool product promotes to int8 and stays int8.
    both_int8 = _dtype_of(x1) == "int8" and _dtype_of(x2) == "int8"
    a, b = _pair(x1, x2)
    if isinstance(a, Tensor) and isinstance(b, Tensor) and both_int8:
        # int8 matmul must accumulate in int32 (numpy-backend semantics);
        # int8 accumulation silently wraps and corrupts quantized inference.
        return a.cast(to_tinygrad_dtype("int32")).matmul(
            b.cast(to_tinygrad_dtype("int32"))
        )
    out = a.matmul(b)
    # tinygrad widens small-int accumulation internally; the numpy reference
    # keeps the promoted operand dtype (int8-int8 above is the one exception).
    return out.cast(a.dtype) if out.dtype != a.dtype else out


def negative(x):
    return -convert_to_tensor(x)


def positive(x):
    return convert_to_tensor(x)


def power(x1, x2):
    # numpy-reference promotion: BOTH operands are converted to
    # result_type(dtype1, dtype2) — python scalars participate as weak types
    # via `type(x)` (so `power(int8_tensor, 1.0)` promotes to floatx, but
    # `power(float16_tensor, 2)` stays float16).
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    if "int" in dtype and not isinstance(x2, Tensor):
        # Free host-side check: numpy raises for negative integer
        # exponents (float pow + int cast would silently truncate to 0).
        # NOTE deviation: lazy Tensor exponents are not checked (that
        # would force a realize) and follow the truncate-to-0 behavior.
        e_np = np.asarray(x2)
        if e_np.size and builtins.int(e_np.min()) < 0:
            raise ValueError(
                "Integers to negative integer powers are not allowed."
            )
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    out = a**b
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def mod(x1, x2):
    a, b = _pair(x1, x2)
    return a % b


def fmod(x1, x2):
    # C-style remainder: sign follows the DIVIDEND (numpy fmod), unlike
    # `mod`, whose sign follows the divisor. tinygrad's `%` is floor-mod
    # (verified: matches np.mod on ints and floats), so adjust: where the
    # operand signs differ and the remainder is nonzero, floor-mod sits one
    # divisor away from the C remainder.
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    if dtype == "bool":
        dtype = "int32"  # numpy-backend semantics: bool fmod computes in int32
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    r = a % b
    return (((a < 0) != (b < 0)) & (r != 0)).where(r - b, r)


def gcd(x1, x2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    if "int" not in dtype:
        # numpy: gcd is integer-only (TypeError on floats/bools).
        raise TypeError(
            f"gcd is only defined for integer dtypes. Received: {dtype}"
        )
    a = convert_to_tensor(x1, dtype).abs()
    b = convert_to_tensor(x2, dtype).abs()
    # Euclid with a shape-stable iteration count: the worst case (Fibonacci
    # pairs) needs ~log_phi(2^width) ≈ 1.44 * width steps for width-bit ints.
    width = builtins.int("".join(c for c in dtype if c.isdigit()))
    steps = builtins.int(1.5 * width) + 2
    for _ in range(steps):
        nz = b != 0
        safe_b = nz.where(b, 1)
        a, b = nz.where(b, a), nz.where(a % safe_b, 0)
    tg_dtype = to_tinygrad_dtype(dtype)
    return a.cast(tg_dtype) if a.dtype != tg_dtype else a


def lcm(x1, x2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    g = gcd(a, b)  # raises TypeError for non-integer dtypes
    nz = g != 0
    safe_g = nz.where(g, 1)
    # (|a| // g) * |b| instead of |a*b| // g to reduce overflow exposure.
    out = nz.where((a.abs() // safe_g) * b.abs(), 0)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def maximum(x1, x2):
    a, b = _pair(x1, x2)
    if not isinstance(a, Tensor):
        a, b = b, a
    return a.maximum(b)


def minimum(x1, x2):
    a, b = _pair(x1, x2)
    if not isinstance(a, Tensor):
        a, b = b, a
    return a.minimum(b)


def _nan_preferring(extremum):
    """numpy fmax/fmin: prefer the NON-nan operand (both nan stays nan);
    otherwise defer to maximum/minimum semantics."""

    def op(x1, x2):
        a, b = _pair(x1, x2)
        if not isinstance(a, Tensor):
            a, b = b, a
        if not isinstance(b, Tensor):
            # b is a weak python scalar; keep it weak for promotion.
            if b != b:  # scalar nan: numpy prefers the tensor operand
                return _float(a)  # weak float nan still promotes ints
            out = getattr(a, extremum)(b)
            return (a != a).where(b, out)
        # Swap each nan for the OTHER operand before the extremum: a lone
        # nan loses, a nan pair stays nan, finite pairs are untouched.
        a2 = (a != a).where(b, a)
        b2 = (b != b).where(a, b)
        return getattr(a2, extremum)(b2)

    return op


fmax = _nan_preferring("maximum")
fmin = _nan_preferring("minimum")


def reciprocal(x):
    return 1.0 / _float(x)


def absolute(x):
    return convert_to_tensor(x).abs()


abs = absolute


def sign(x):
    return convert_to_tensor(x).sign()


def square(x):
    x = convert_to_tensor(x)
    if to_keras_dtype(x.dtype) == "bool":
        # jax lattice: square(bool) is int32 (numpy-backend referee dtype).
        x = x.cast(to_tinygrad_dtype("int32"))
    return x * x


def sqrt(x):
    return _float(x).sqrt()


def exp(x):
    return _float(x).exp()


def exp2(x):
    return _float(x).exp2()


def expm1(x):
    x = _float(x)
    u = x.exp()
    d = u - 1.0
    # Kahan's compensated form: `exp(x) - 1` loses all precision for
    # |x| << 1 (u rounds to 1). `d * x / log(u)` evaluates the identity
    # expm1(x) = (u - 1) * x / log(u) at the x the rounded u actually
    # corresponds to. Edges: u == 1 -> x; u == inf -> inf (the compensated
    # form would be inf/inf = nan); u == 0 -> -1 (would be x/-inf = 0).
    out = (d == 0.0).where(x, d * x / u.log())
    out = (u == builtins.float("inf")).where(u, out)
    return (u == 0.0).where(-1.0, out)


def log(x):
    return _float(x).log()


def log2(x):
    return _float(x).log2()


def log10(x):
    return _float(x).log() / _math.log(10.0)


def log1p(x):
    x = _float(x)
    # The contiguous() is load-bearing: without the buffer barrier,
    # tinygrad's simplifier folds `(1.0 + x) - 1.0` back into `x`,
    # turning the compensated form below into plain log(1+x).
    u = (1.0 + x).contiguous()
    d = u - 1.0
    # Kahan's compensated form: `log(1 + x)` loses all precision for
    # |x| << 1 (u rounds to 1). `log(u) * x / d` evaluates the log at the
    # x that survived the rounding. Edges: u == 1 -> x (preserves -0.0);
    # u == inf -> inf (the compensated form would be inf/inf = nan).
    out = (d == 0.0).where(x, u.log() * x / d)
    return (u == builtins.float("inf")).where(u, out)


def logaddexp(x1, x2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2), builtins.float)
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    m = a.maximum(b)
    # Overflow-safe: log(e^a + e^b) = m + log(e^(a-m) + e^(b-m)); the
    # shifted exponents are <= 0, so exp cannot overflow.
    out = m + ((a - m).exp() + (b - m).exp()).log()
    inf = builtins.float("inf")
    # inf - inf = nan poisons the shifted form at the infinities; numpy:
    # either +inf -> +inf, both -inf -> -inf, any nan -> nan.
    out = ((a == -inf) & (b == -inf)).where(-inf, out)
    out = ((a == inf) | (b == inf)).where(inf, out)
    out = ((a != a) | (b != b)).where(a + b, out)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def logaddexp2(x1, x2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2), builtins.float)
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    m = a.maximum(b)
    # Overflow-safe: log2(2^a + 2^b) = m + log2(2^(a-m) + 2^(b-m)); the
    # shifted exponents are <= 0, so exp2 cannot overflow.
    out = m + ((a - m).exp2() + (b - m).exp2()).log2()
    inf = builtins.float("inf")
    # inf - inf = nan poisons the shifted form at the infinities; numpy:
    # either +inf -> +inf, both -inf -> -inf, any nan -> nan.
    out = ((a == -inf) & (b == -inf)).where(-inf, out)
    out = ((a == inf) | (b == inf)).where(inf, out)
    out = ((a != a) | (b != b)).where(a + b, out)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def _binary_float_dtype(x1, x2):
    """numpy-reference result dtype for heaviside/hypot: small ints go to
    floatx, int64 to float64, floats stay."""
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    if dtype in ("int8", "int16", "int32", "uint8", "uint16", "uint32"):
        dtype = floatx()
    elif dtype == "int64":
        dtype = "float64"
    return dtype


def heaviside(x1, x2):
    dtype = _binary_float_dtype(x1, x2)
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    out = (a > 0).where(1.0, (a < 0).where(0.0, b))
    # NaN in x1 passes through (numpy semantics); NaN is neither >0 nor <0,
    # so the where chain above would have picked b.
    out = (a != a).where(a, out)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def hypot(x1, x2):
    dtype = _binary_float_dtype(x1, x2)
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    out = (a * a + b * b).sqrt()
    # numpy: hypot is inf whenever either input is infinite, even if the
    # other is NaN (nan + inf would otherwise poison the sum).
    inf = builtins.float("inf")
    out = ((a.abs() == inf) | (b.abs() == inf)).where(inf, out)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def ldexp(x1, x2):
    x2_dtype = _dtype_of(x2)
    if not (
        x2_dtype is builtins.int
        or (isinstance(x2_dtype, str) and "int" in x2_dtype)
    ):
        raise TypeError(
            f"ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2_dtype}"
        )
    dtype = result_type(_dtype_of(x1), x2_dtype, float)
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2)
    out = a * (2.0 ** b.cast(a.dtype))
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


# ---- trig / hyperbolic ------------------------------------------------------


def sin(x):
    return _float(x).sin()


def cos(x):
    return _float(x).cos()


def tan(x):
    return _float(x).tan()


def arcsin(x):
    return _float(x).asin()


def arccos(x):
    return _float(x).acos()


def arctan(x):
    return _float(x).atan()


def arctan2(x1, x2):
    # numpy-reference dtype: result_type(x1, x2, float). Computed in float32
    # (the intermediate bool-times-scalar terms would silently promote halfs
    # to float32 anyway) and cast to the lattice dtype at the end.
    dtype = result_type(_dtype_of(x1), _dtype_of(x2), builtins.float)
    compute = "float64" if dtype == "float64" else "float32"
    tg_compute = to_tinygrad_dtype(compute)
    y = convert_to_tensor(x1, dtype).cast(tg_compute)
    x = convert_to_tensor(x2, dtype).cast(tg_compute)
    at = (y / (x + ((x == 0) * 1e-30))).atan()
    at = at + (x < 0) * ((y >= 0) * 2.0 - 1.0) * _math.pi
    at = (x != 0).where(at, (y > 0) * _math.pi / 2 + (y < 0) * -_math.pi / 2)
    tg_dtype = to_tinygrad_dtype(dtype)
    return at.cast(tg_dtype) if at.dtype != tg_dtype else at


def sinh(x):
    return _float(x).sinh()


def cosh(x):
    return _float(x).cosh()


def tanh(x):
    return _float(x).tanh()


def arcsinh(x):
    return _float(x).asinh()


def arccosh(x):
    return _float(x).acosh()


def arctanh(x):
    return _float(x).atanh()


def deg2rad(x):
    return _float(x) * (_math.pi / 180.0)


def rad2deg(x):
    return _float(x) * (180.0 / _math.pi)


# ---- reductions -------------------------------------------------------------


def mean(x, axis=None, keepdims=False):
    x = _float(x)
    return x.mean(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if to_keras_dtype(x.dtype) == "bool":
        x = x.cast(to_tinygrad_dtype("int32"))
    return x.sum(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        # numpy-reference accumulation dtype for small ints/bool.
        dtype = to_keras_dtype(x.dtype)
        if dtype in ("bool", "int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"
    else:
        dtype = standardize_dtype(dtype)
    tg_dtype = to_tinygrad_dtype(dtype)
    if x.dtype != tg_dtype:
        x = x.cast(tg_dtype)
    out = x.prod(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    out = x.max(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)
    if initial is not None:
        out = out.maximum(initial)
    # tinygrad's bool reduce comes back int32; numpy keeps the input dtype.
    return out.cast(x.dtype) if out.dtype != x.dtype else out


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    out = x.min(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)
    if initial is not None:
        out = out.minimum(initial)
    # tinygrad's bool reduce comes back int32; numpy keeps the input dtype.
    return out.cast(x.dtype) if out.dtype != x.dtype else out


amax = max
amin = min


def all(x, axis=None, keepdims=False):
    x = convert_to_tensor(x).cast(to_tinygrad_dtype("bool"))
    return x.all(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)


def any(x, axis=None, keepdims=False):
    x = convert_to_tensor(x).cast(to_tinygrad_dtype("bool"))
    return x.any(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)


def var(x, axis=None, keepdims=False):
    x = _float(x)
    return x.var(axis=_norm_axis(axis, x.ndim), keepdim=keepdims, correction=0)


def std(x, axis=None, keepdims=False):
    x = _float(x)
    return x.std(axis=_norm_axis(axis, x.ndim), keepdim=keepdims, correction=0)


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(dtype) if dtype else to_keras_dtype(x.dtype)
    if dtype == "bool":
        # numpy reference: bool cumsum accumulates in int32.
        dtype = "int32"
    tg_dtype = to_tinygrad_dtype(dtype)
    if x.dtype != tg_dtype:
        x = x.cast(tg_dtype)
    if axis is None:
        out = x.reshape(-1).cumsum(0)
    else:
        out = x.cumsum(_norm_axis(axis, x.ndim))
    # tinygrad accumulates small ints / halfs in a wider dtype; the numpy
    # reference keeps the requested dtype.
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x, dtype=dtype)
    if to_keras_dtype(x.dtype) == "bool":
        # numpy reference: bool cumprod accumulates in int32.
        x = x.cast(to_tinygrad_dtype("int32"))
    if axis is None:
        out = x.reshape(-1).cumprod(0)
    else:
        out = x.cumprod(_norm_axis(axis, x.ndim))
    # tinygrad accumulates small ints / halfs in a wider dtype; the numpy
    # reference keeps the input dtype.
    return out.cast(x.dtype) if out.dtype != x.dtype else out


def median(x, axis=None, keepdims=False):
    x = _float(x)
    if x.ndim == 0:
        return x
    if axis is None:
        red = tuple(range(x.ndim))
    elif isinstance(axis, (list, tuple)):
        red = tuple(a % x.ndim for a in axis)
    else:
        red = (axis % x.ndim,)
    keep = [i for i in range(x.ndim) if i not in red]
    # Collapse all reduced axes into one trailing axis, sort it, then
    # average the two middle elements (numpy semantics for even counts).
    n = 1
    for i in red:
        n *= x.shape[i]
    xt = x.permute(keep + builtins.list(red)).reshape(
        [x.shape[i] for i in keep] + [n]
    )
    s = xt.sort(dim=-1)[0]
    lead = (builtins.slice(None),) * len(keep)
    lo = s[lead + ((n - 1) // 2,)]
    hi = s[lead + (n // 2,)]
    out = (lo + hi) * 0.5
    if keepdims:
        out = out.reshape(
            [1 if i in red else x.shape[i] for i in range(x.ndim)]
        )
    # tinygrad's sort promotes halfs to float32 internally; restore the
    # numpy-reference dtype result_type(x.dtype, float) (== _float's dtype).
    return out.cast(x.dtype) if out.dtype != x.dtype else out


def nancumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x, dtype=dtype)
    if to_keras_dtype(x.dtype) == "bool":
        x = x.cast(to_tinygrad_dtype("int32"))
    if "float" in to_keras_dtype(x.dtype):
        x = (x != x).where(1, x)
    return cumprod(x, axis=axis)


def nanmedian(x, axis=None, keepdims=False):
    x = _float(x)
    if x.ndim == 0:
        return x
    if axis is None:
        red = tuple(range(x.ndim))
    elif isinstance(axis, (list, tuple)):
        red = tuple(a % x.ndim for a in axis)
    else:
        red = (axis % x.ndim,)
    keep = [i for i in range(x.ndim) if i not in red]
    n = 1
    for i in red:
        n *= x.shape[i]
    xt = x.permute(keep + builtins.list(red)).reshape(
        [x.shape[i] for i in keep] + [n]
    )
    # NaNs are pushed to the tail by replacing them with +inf before the
    # sort; per-slice valid counts pick the true middle element(s), and
    # all-NaN slices come back as NaN.
    nan_mask = xt != xt
    s = nan_mask.where(builtins.float("inf"), xt).sort(dim=-1)[0]
    c = nan_mask.logical_not().cast(to_tinygrad_dtype("int32")).sum(axis=-1)
    lo_i = (c - 1).maximum(0) // 2
    hi_i = c // 2
    lead = builtins.list(s.shape[:-1])
    lo = s.gather(-1, lo_i.reshape(lead + [1]))
    hi = s.gather(-1, hi_i.reshape(lead + [1]))
    out = ((lo + hi) * 0.5).reshape(lead)
    out = (c == 0).where(builtins.float("nan"), out)
    if keepdims:
        out = out.reshape(
            [1 if i in red else x.shape[i] for i in range(x.ndim)]
        )
    return out.cast(x.dtype) if out.dtype != x.dtype else out


def nansum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = to_keras_dtype(x.dtype)
    # numpy reference accumulation dtype for small ints/bool.
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    if "float" in dtype:
        x = (x != x).where(0, x)
    x = x.cast(to_tinygrad_dtype(dtype))
    out = x.sum(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = result_type(to_keras_dtype(x.dtype), float)
    x = x.cast(to_tinygrad_dtype(dtype))
    axis = _norm_axis(axis, x.ndim)
    valid = x == x
    total = valid.where(x, 0).sum(axis=axis, keepdim=keepdims)
    count = valid.cast(x.dtype).sum(axis=axis, keepdim=keepdims)
    out = total / count
    # All-NaN slices divide by zero; numpy returns NaN there.
    out = (count == 0).where(builtins.float("nan"), out)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = _norm_axis(axis, x.ndim)
    if "float" not in to_keras_dtype(x.dtype):
        return x.max(axis=axis, keepdim=keepdims)
    nan_mask = x != x
    out = nan_mask.where(builtins.float("-inf"), x).max(
        axis=axis, keepdim=keepdims
    )
    # numpy returns NaN (with a warning we don't reproduce) for all-NaN
    # slices; the masked max alone would say -inf.
    return nan_mask.all(axis=axis, keepdim=keepdims).where(
        builtins.float("nan"), out
    )


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = _norm_axis(axis, x.ndim)
    if "float" not in to_keras_dtype(x.dtype):
        return x.min(axis=axis, keepdim=keepdims)
    nan_mask = x != x
    out = nan_mask.where(builtins.float("inf"), x).min(
        axis=axis, keepdim=keepdims
    )
    return nan_mask.all(axis=axis, keepdim=keepdims).where(
        builtins.float("nan"), out
    )


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = _norm_axis(axis, x.ndim)
    return x.max(axis=axis, keepdim=keepdims) - x.min(
        axis=axis, keepdim=keepdims
    )


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is None:
        out = x.reshape(-1).argmax(0)
        if keepdims:
            out = out.reshape(*([1] * x.ndim))
        return out.cast(to_tinygrad_dtype("int32"))
    return x.argmax(axis=axis, keepdim=keepdims).cast(
        to_tinygrad_dtype("int32")
    )


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is None:
        out = x.reshape(-1).argmin(0)
        if keepdims:
            out = out.reshape(*([1] * x.ndim))
        return out.cast(to_tinygrad_dtype("int32"))
    return x.argmin(axis=axis, keepdim=keepdims).cast(
        to_tinygrad_dtype("int32")
    )


def average(x, axis=None, weights=None):
    # numpy-reference promotion: result_type(x, weights, float) — 1-D int
    # weights on a float16 input stay float16 (jax lattice sends ints to the
    # float operand's dtype), NOT floatx.
    dtypes_to_resolve = [_dtype_of(x), builtins.float]
    if weights is not None:
        dtypes_to_resolve.append(_dtype_of(weights))
    dtype = result_type(*dtypes_to_resolve)
    tg_dtype = to_tinygrad_dtype(dtype)
    x = convert_to_tensor(x, dtype)
    if isinstance(axis, (builtins.list, tuple)):
        if len(axis) == 0:
            # numpy: reducing over no axes is the identity (as float).
            if weights is not None:
                raise NotImplementedError(
                    "tinygrad backend: average with axis=() and weights "
                    "is not implemented"
                )
            return x
        if len(axis) == 1:
            axis = axis[0]
        else:
            axis = tuple(axis)
    if weights is None:
        out = x.mean(axis=_norm_axis(axis, x.ndim))
        return out.cast(tg_dtype) if out.dtype != tg_dtype else out
    w = convert_to_tensor(weights, dtype)
    if w.ndim == 1 and x.ndim != 1:
        # numpy's validation for 1-D weights on an N-D input.
        if axis is None:
            raise TypeError(
                "Axis must be specified when shapes of x and weights differ."
            )
        if isinstance(axis, tuple):
            raise ValueError(
                "1-D weights can only be used with a single reduction axis."
            )
        if w.shape[0] != x.shape[_norm_axis(axis, x.ndim)]:
            raise ValueError(
                "Length of weights not compatible with specified axis."
            )
        shape = [1] * x.ndim
        shape[_norm_axis(axis, x.ndim)] = w.shape[0]
        w = w.reshape(shape)
    out = (x * w).sum(axis=_norm_axis(axis, x.ndim)) / w.sum(
        axis=_norm_axis(axis, x.ndim)
    )
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def count_nonzero(x, axis=None):
    x = convert_to_tensor(x)
    return (
        (x != 0)
        .cast(to_tinygrad_dtype("int32"))
        .sum(axis=_norm_axis(axis, x.ndim))
    )


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError(
            "Unsupported value `sparse=True` with tinygrad backend"
        )
    x = convert_to_tensor(x)
    if weights is not None:
        w = convert_to_tensor(weights)
        dtype = result_type(to_keras_dtype(x.dtype), to_keras_dtype(w.dtype))
    else:
        w = None
        dtype = "int32"
    # numpy raises for negative values; the one-hot match below would
    # silently drop them. This op already realizes a scalar for the
    # output length, so the min check costs one more scalar read.
    if builtins.int(x.min().numpy()) < 0:
        raise ValueError("bincount: input must be non-negative")
    # The output length is data-dependent (max value + 1): realize just
    # that scalar — the counts themselves stay lazy tinygrad ops.
    length = builtins.max(
        builtins.int(x.max().numpy()) + 1, builtins.int(minlength)
    )
    ids = Tensor.arange(length, dtype=to_tinygrad_dtype("int32"))
    onehot = x.cast(to_tinygrad_dtype("int32")).reshape(
        *x.shape, 1
    ) == ids.reshape(*([1] * x.ndim), length)
    if w is None:
        out = onehot.cast(to_tinygrad_dtype("int32")).sum(axis=x.ndim - 1)
    else:
        tg_dtype = to_tinygrad_dtype(dtype)
        out = (onehot.cast(tg_dtype) * w.cast(tg_dtype).reshape(
            *w.shape, 1
        )).sum(axis=x.ndim - 1)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


# ---- comparison / logic -----------------------------------------------------


def equal(x1, x2):
    a, b = _pair(x1, x2)
    return a == b


def not_equal(x1, x2):
    a, b = _pair(x1, x2)
    return a != b


def greater(x1, x2):
    a, b = _pair(x1, x2)
    return a > b


def greater_equal(x1, x2):
    a, b = _pair(x1, x2)
    return a >= b


def less(x1, x2):
    a, b = _pair(x1, x2)
    return a < b


def less_equal(x1, x2):
    a, b = _pair(x1, x2)
    return a <= b


def logical_and(x1, x2):
    a = convert_to_tensor(x1, "bool")
    b = convert_to_tensor(x2, "bool")
    return a & b


def logical_or(x1, x2):
    a = convert_to_tensor(x1, "bool")
    b = convert_to_tensor(x2, "bool")
    return a | b


def logical_xor(x1, x2):
    a = convert_to_tensor(x1, "bool")
    b = convert_to_tensor(x2, "bool")
    return a ^ b


def logical_not(x):
    return convert_to_tensor(x, "bool").logical_not()


def where(condition, x1=None, x2=None):
    if x1 is None and x2 is None:
        # numpy's single-argument form: indices of the nonzero elements.
        return nonzero(condition)
    condition = convert_to_tensor(condition, "bool")
    if isinstance(
        x1, (builtins.bool, builtins.int, builtins.float)
    ) and isinstance(x2, (builtins.bool, builtins.int, builtins.float)):
        return condition.where(x1, x2)
    # _pair converts both branches (raw numpy arrays included) with Keras
    # dtype promotion; python scalars stay weak-typed.
    a, b = _pair(x1, x2)
    return condition.where(a, b)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    a, b = _pair(x1, x2)
    a, b = _float(a), _float(b)
    # The tolerance formula only applies to finite pairs (np.isclose):
    # anything non-finite is close iff exactly equal — same-signed inf yes,
    # inf vs finite / inf vs -inf / NaN vs anything no.
    close = (a - b).abs() <= (atol + rtol * b.abs())
    close = (isfinite(a) & isfinite(b)).where(close, a == b)
    if equal_nan:
        close = close | (isnan(a) & isnan(b))
    return close


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    return isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan).all()


def isnan(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return Tensor.zeros(*x.shape, dtype=to_tinygrad_dtype("bool"))
    return x != x


def isinf(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return Tensor.zeros(*x.shape, dtype=to_tinygrad_dtype("bool"))
    return x.abs() == float("inf")


def isfinite(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return Tensor.ones(*x.shape, dtype=to_tinygrad_dtype("bool"))
    return (x == x) & (x.abs() != float("inf"))


# ---- rounding ---------------------------------------------------------------


def floor(x):
    return _float(x).floor()


def ceil(x):
    return _float(x).ceil()


def round(x, decimals=0):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        # numpy reference: rounding an integer tensor keeps its dtype and,
        # for non-negative decimals, is the identity.
        if decimals >= 0:
            return x
        scale = 10 ** (-decimals)
        return (
            (x.cast(to_tinygrad_dtype("float32")) / scale).round() * scale
        ).cast(x.dtype)
    if decimals == 0:
        return x.round()  # tinygrad rounds half-to-even, like numpy
    scale = 10.0**decimals
    return (x * scale).round() / scale


def trunc(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return x
    return x.trunc()


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    if isinstance(x_min, Tensor) or isinstance(x_max, Tensor):
        return x.maximum(x_min).minimum(x_max)
    # tinygrad's ufix refuses numpy scalars (np.float32(...)) as bounds.
    if isinstance(x_min, np.generic):
        x_min = x_min.item()
    if isinstance(x_max, np.generic):
        x_max = x_max.item()
    return x.clip(x_min, x_max)


# ---- shape manipulation -----------------------------------------------------


def reshape(x, newshape):
    x = convert_to_tensor(x)
    if isinstance(newshape, int):
        newshape = (newshape,)
    # Plain tuple: tinygrad's argfix checks `.__class__ in (tuple, list)`
    # exactly, so list subclasses (keras' TrackedList) are mis-parsed as a
    # single dimension.
    return x.reshape(tuple(newshape))


def ravel(x):
    return convert_to_tensor(x).reshape(-1)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    if axes is None:
        axes = list(range(x.ndim))[::-1]
    return x.permute(tuple(axes))


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    order = list(range(x.ndim))
    order[axis1], order[axis2] = order[axis2], order[axis1]
    return x.permute(order)


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    source = [s % x.ndim for s in source]
    destination = [d % x.ndim for d in destination]
    order = [i for i in range(x.ndim) if i not in source]
    for d, s in sorted(zip(destination, source)):
        order.insert(d, s)
    return x.permute(order)


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    if isinstance(axis, int):
        axis = (axis,)
    out_ndim = x.ndim + len(axis)
    axis = sorted(a % out_ndim for a in axis)
    shape = list(x.shape)
    for a in axis:
        shape.insert(a, 1)
    return x.reshape(shape)


def squeeze(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        shape = [d for d in x.shape if d != 1]
        return x.reshape(shape)
    if isinstance(axis, int):
        axis = (axis,)
    axis = [a % x.ndim for a in axis]
    for a in axis:
        if x.shape[a] != 1:
            raise ValueError(
                f"Cannot squeeze axis {a} with dim {x.shape[a]} != 1"
            )
    shape = [d for i, d in enumerate(x.shape) if i not in axis]
    return x.reshape(shape)


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    # Plain tuple: tinygrad's argfix checks `.__class__ in (tuple, list)`
    # exactly, so list subclasses (keras' TrackedList, e.g. Normalization's
    # `_broadcast_shape` attribute) are mis-parsed as a single dimension.
    shape = tuple(shape)
    if x.ndim < len(shape):
        x = x.reshape([1] * (len(shape) - x.ndim) + list(x.shape))
    return x.expand(shape)


def concatenate(xs, axis=0):
    dtype = result_type(*[_dtype_of(x) for x in xs])
    xs = [convert_to_tensor(x, dtype) for x in xs]
    return xs[0].cat(*xs[1:], dim=axis)


def stack(x, axis=0):
    dtype = result_type(*[_dtype_of(e) for e in x])
    x = [convert_to_tensor(e, dtype) for e in x]
    return Tensor.stack(*x, dim=axis)


def hstack(xs):
    first = convert_to_tensor(xs[0])
    return concatenate(xs, axis=0 if first.ndim == 1 else 1)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = [x.reshape(1, *x.shape) if x.ndim == 1 else x for x in xs]
    return concatenate(xs, axis=0)


def append(x1, x2, axis=None):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    if axis is None:
        a, b = a.reshape(-1), b.reshape(-1)
        axis = 0
    return a.cat(b, dim=axis)


def column_stack(xs):
    dtype = result_type(*[_dtype_of(x) for x in xs])
    xs = [convert_to_tensor(x, dtype) for x in xs]
    # numpy: sub-2-D inputs become columns; 2-D+ inputs join as-is.
    xs = [x.reshape(-1, 1) if x.ndim < 2 else x for x in xs]
    return xs[0].cat(*xs[1:], dim=1)


def dstack(xs):
    dtype = result_type(*[_dtype_of(x) for x in xs])
    xs = [convert_to_tensor(x, dtype) for x in xs]

    def atleast_3d(x):
        # numpy atleast_3d: () -> (1,1,1); (N,) -> (1,N,1); (M,N) -> (M,N,1)
        if x.ndim == 0:
            return x.reshape(1, 1, 1)
        if x.ndim == 1:
            return x.reshape(1, x.shape[0], 1)
        if x.ndim == 2:
            return x.reshape(*x.shape, 1)
        return x

    xs = [atleast_3d(x) for x in xs]
    return xs[0].cat(*xs[1:], dim=2)


def split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    axis = _norm_axis(axis, x.ndim)
    dim = x.shape[axis]
    # Split points are shape metadata, not a differentiable path: tensors /
    # numpy arrays of indices are read out as host ints (like repeat counts).
    if isinstance(indices_or_sections, Tensor):
        indices_or_sections = indices_or_sections.numpy().tolist()
    elif hasattr(indices_or_sections, "tolist"):
        indices_or_sections = indices_or_sections.tolist()
    if isinstance(indices_or_sections, int):
        if dim % indices_or_sections != 0:
            raise ValueError(
                f"Cannot split axis of size {dim} into "
                f"{indices_or_sections} equal parts"
            )
        if dim == 0:
            # chunk() collapses empty tensors; numpy returns n empty parts.
            return [x] * indices_or_sections
        return builtins.list(x.chunk(indices_or_sections, dim=axis))
    starts = [0] + builtins.list(indices_or_sections) + [dim]
    out = []
    for i in range(len(starts) - 1):
        index = [builtins.slice(None)] * x.ndim
        index[axis] = builtins.slice(starts[i], starts[i + 1])
        out.append(x[tuple(index)])
    return out


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    if isinstance(repeats, Tensor):
        # Per-element repeat counts make the output shape data-dependent;
        # the counts are structural ints (no gradient path), so realizing
        # them here is acceptable — x itself stays lazy.
        repeats = repeats.numpy().tolist()
    elif hasattr(repeats, "tolist"):  # numpy array / scalar
        repeats = repeats.tolist()
    if isinstance(repeats, (builtins.list, tuple)):
        if len(repeats) == 1:
            repeats = repeats[0]  # numpy broadcasts a length-1 repeats
    if not isinstance(repeats, (builtins.list, tuple)):
        return x.repeat_interleave(builtins.int(repeats), dim=axis)
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    axis = _norm_axis(axis, x.ndim)
    if len(repeats) != x.shape[axis]:
        raise ValueError(
            f"repeats length {len(repeats)} does not match axis "
            f"{axis} size {x.shape[axis]}"
        )
    idx = []
    for i, r in enumerate(repeats):
        idx.extend([i] * builtins.int(r))
    return take(x, convert_to_tensor(idx, "int32"), axis=axis)


def tile(x, repeats):
    x = convert_to_tensor(x)
    if isinstance(repeats, int):
        repeats = (repeats,)
    repeats = builtins.list(repeats)
    if len(repeats) < x.ndim:
        repeats = [1] * (x.ndim - len(repeats)) + repeats
    if len(repeats) > x.ndim:
        x = x.reshape([1] * (len(repeats) - x.ndim) + builtins.list(x.shape))
    return x.repeat(repeats)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        axis = tuple(range(x.ndim))
    return x.flip(axis)


def rot90(array, k=1, axes=(0, 1)):
    x = convert_to_tensor(array)
    if x.ndim < 2:
        raise ValueError(
            "Input array must have at least 2 dimensions. "
            f"Received: array.ndim={x.ndim}"
        )
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple "
            "of two different dimensions."
        )
    ax0, ax1 = (a % x.ndim for a in axes)
    if ax0 == ax1:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple "
            "of two different dimensions."
        )
    k %= 4
    if k == 0:
        return x
    if k == 2:
        return x.flip((ax0, ax1))
    perm = builtins.list(range(x.ndim))
    perm[ax0], perm[ax1] = perm[ax1], perm[ax0]
    if k == 1:
        return x.flip(ax1).permute(perm)
    return x.permute(perm).flip(ax1)  # k == 3


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    # Normalize numpy arrays / tensors of shifts and axes to python values.
    if isinstance(shift, Tensor):
        shift = shift.numpy().tolist()
    elif hasattr(shift, "tolist"):
        shift = shift.tolist()
    if isinstance(axis, Tensor):
        axis = axis.numpy().tolist()
    elif hasattr(axis, "tolist") and axis is not None:
        axis = axis.tolist()
    if isinstance(shift, (builtins.list, tuple)) or isinstance(
        axis, (builtins.list, tuple)
    ):
        if axis is None:
            raise ValueError(
                "roll with a sequence of shifts requires `axis` to be given"
            )
        shifts = (
            builtins.list(shift)
            if isinstance(shift, (builtins.list, tuple))
            else [shift]
        )
        axes = (
            builtins.list(axis)
            if isinstance(axis, (builtins.list, tuple))
            else [axis]
        )
        # numpy broadcasting of shift against axis.
        if len(shifts) == 1:
            shifts = shifts * len(axes)
        if len(axes) == 1:
            axes = axes * len(shifts)
        if len(shifts) != len(axes):
            raise ValueError(
                "`shift` and `axis` must be broadcastable: got "
                f"{len(shifts)} shifts and {len(axes)} axes"
            )
        out = x
        for s, a in zip(shifts, axes):
            out = roll(out, builtins.int(s), builtins.int(a))
        return out
    if axis is None:
        flat = x.reshape(-1)
        out = roll(flat, shift, axis=0)
        return out.reshape(x.shape)
    shift = shift % x.shape[axis]
    if shift == 0:
        return x
    index_a = [builtins.slice(None)] * x.ndim
    index_b = [builtins.slice(None)] * x.ndim
    index_a[axis] = builtins.slice(-shift, None)
    index_b[axis] = builtins.slice(None, -shift)
    return x[tuple(index_a)].cat(x[tuple(index_b)], dim=axis)


def pad(x, pad_width, mode="constant", constant_values=None):
    x = convert_to_tensor(x)
    if mode != "constant" and constant_values is not None:
        raise ValueError(
            "Argument `constant_values` can only be "
            "provided when `mode == 'constant'`. "
            f"Received: mode={mode}"
        )
    if mode not in ("constant", "reflect", "symmetric"):
        raise NotImplementedError(
            f"tinygrad backend: pad mode '{mode}' is not implemented"
        )
    if isinstance(pad_width[0], int):
        pad_width = [pad_width] * x.ndim

    def _pad_amount(v):
        # Pad widths are shape metadata, not a differentiable path; a 0-d
        # tensor here (e.g. MaxNumBoundingBoxes' ops-computed pad size) is
        # read out, matching the numpy backend where np.pad materializes
        # widths as host integers.
        if isinstance(v, Tensor):
            return builtins.int(v.item())
        return v

    pads = [tuple(_pad_amount(v) for v in p) for p in pad_width]
    if mode == "constant":
        value = 0 if constant_values is None else constant_values
        return x.pad(pads, value=value)
    # reflect / symmetric: gather along each padded axis with a host-built
    # reflected index table (np.pad on arange — a creation-time constant,
    # sanctioned numpy role). This reproduces numpy's semantics exactly,
    # including multi-bounce reflection when a pad exceeds the axis length,
    # and gradients flow through the gather.
    out = x
    for axis, (before, after) in enumerate(pads):
        if before == 0 and after == 0:
            continue
        idx = np.pad(
            np.arange(out.shape[axis], dtype=np.int32),
            (before, after),
            mode=mode,
        )
        out = take(out, convert_to_tensor(idx, "int32"), axis=axis)
    return out


# ---- creation ---------------------------------------------------------------


def _shape_tuple(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def zeros(shape, dtype=None):
    dtype = standardize_dtype(dtype) if dtype else "float32"
    return Tensor.zeros(*_shape_tuple(shape), dtype=to_tinygrad_dtype(dtype))


def ones(shape, dtype=None):
    dtype = standardize_dtype(dtype) if dtype else "float32"
    return Tensor.ones(*_shape_tuple(shape), dtype=to_tinygrad_dtype(dtype))


def zeros_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(dtype) if dtype else to_keras_dtype(x.dtype)
    return Tensor.zeros(*x.shape, dtype=to_tinygrad_dtype(dtype))


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(dtype) if dtype else to_keras_dtype(x.dtype)
    return Tensor.ones(*x.shape, dtype=to_tinygrad_dtype(dtype))


def full(shape, fill_value, dtype=None):
    # Reference-backend default: dtype is floatx when not given, regardless
    # of the fill value's dtype.
    dtype = standardize_dtype(dtype) if dtype else floatx()
    if isinstance(fill_value, (Tensor, np.ndarray, builtins.list, tuple)):
        fill = convert_to_tensor(fill_value, dtype)
        if fill.ndim > 0:
            # Array-valued fill broadcasts against the target shape.
            return broadcast_to(fill, _shape_tuple(shape))
        fill_value = fill.numpy().item()
    elif isinstance(fill_value, np.generic):
        fill_value = fill_value.item()
    return Tensor.full(
        _shape_tuple(shape), fill_value, dtype=to_tinygrad_dtype(dtype)
    )


def full_like(x, fill_value, dtype=None):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(dtype) if dtype else to_keras_dtype(x.dtype)
    return full(tuple(x.shape), fill_value, dtype=dtype)


def empty(shape, dtype=None):
    return zeros(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


def arange(start, stop=None, step=None, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [_dtype_of(start)]
        if stop is not None:
            dtypes_to_resolve.append(_dtype_of(stop))
        if step is not None:
            dtypes_to_resolve.append(_dtype_of(step))
        dtype = result_type(*dtypes_to_resolve)
    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1

    def _scalar(v):
        # Range bounds are structural (no gradient path); live Tensors or
        # numpy scalars here would be embedded as raw UOp srcs and crash
        # tinygrad's graph_rewrite.
        if isinstance(v, Tensor):
            return v.item()
        if isinstance(v, np.generic):
            return v.item()
        return v

    return Tensor.arange(
        _scalar(start), _scalar(stop), _scalar(step),
        dtype=to_tinygrad_dtype(dtype),
    )


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if dtype is None:
        dtype = result_type(_dtype_of(start), _dtype_of(stop), float)
    dtype = standardize_dtype(dtype)
    tg_dtype = to_tinygrad_dtype(dtype)
    num = builtins.int(num)
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    div = (num - 1) if endpoint else num
    if isinstance(start, (builtins.int, builtins.float)) and isinstance(
        stop, (builtins.int, builtins.float)
    ):
        # Scalar endpoints: host-side float64 grid (see logspace), the
        # tensor stays lazy.
        s, e = builtins.float(start), builtins.float(stop)
        step = (e - s) / div if div > 0 else builtins.float("nan")
        if num == 1:
            vals = [s]  # a NaN step (endpoint, div=0) must not poison s
        else:
            vals = [s + i * step for i in range(num)]
            if endpoint and num > 1:
                vals[-1] = e
        out = (
            convert_to_tensor(vals, dtype)
            if num > 0
            else Tensor.zeros(0, dtype=tg_dtype)
        )
        if axis != 0:
            out = moveaxis(out, 0, axis)
        return (out, step) if retstep else out
    # Tensor endpoints: sample grid stacked along `axis`.
    s, e = _broadcast_endpoints(start, stop)
    if num == 0:
        out = Tensor.zeros(0, *s.shape, dtype=tg_dtype)
        step = builtins.float("nan")
    else:
        out = _linspace_grid(s, e, num, endpoint)
        out = out.cast(tg_dtype) if out.dtype != tg_dtype else out
        step = (
            (e - s) / div if div > 0 else (e - s) * builtins.float("nan")
        )
    if axis != 0:
        out = moveaxis(out, 0, axis)
    return (out, step) if retstep else out


def _broadcast_endpoints(start, stop):
    """Convert start/stop to float tensors broadcast to a common shape."""
    s = _float(convert_to_tensor(start))
    e = _float(convert_to_tensor(stop))
    nd = builtins.max(s.ndim, e.ndim)
    sshape = (1,) * (nd - s.ndim) + tuple(s.shape)
    eshape = (1,) * (nd - e.ndim) + tuple(e.shape)
    bshape = tuple(builtins.max(a, b) for a, b in zip(sshape, eshape))
    return s.reshape(sshape).expand(bshape), e.reshape(eshape).expand(bshape)


def _linspace_grid(s, e, num, endpoint):
    """(num, *s.shape) linear grid between same-shape tensors s and e; the
    last row is pinned exactly to e when endpoint (numpy semantics)."""
    div = (num - 1) if endpoint else num
    if div <= 0:
        div = 1
    t = Tensor.arange(num, dtype=to_tinygrad_dtype("float32")).reshape(
        [num] + [1] * s.ndim
    )
    s1 = s.reshape([1, *s.shape])
    e1 = e.reshape([1, *e.shape])
    out = s1 + (e1 - s1) * (t / div)
    if endpoint and num > 1:
        out = (t == (num - 1)).where(e1, out)
    return out


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if axis != 0:
        raise NotImplementedError(
            "tinygrad backend: logspace with axis != 0 is not implemented"
        )
    if dtype is None:
        dtype = result_type(_dtype_of(start), _dtype_of(stop), float)
    dtype = standardize_dtype(dtype)
    if isinstance(start, (builtins.int, builtins.float)) and isinstance(
        stop, (builtins.int, builtins.float)
    ):
        # Creation op with scalar endpoints (no gradient path): computed
        # host-side in float64 for numpy-matching precision (same precedent
        # as `tri`/`eye(k != 0)`), the tensor stays lazy.
        s, e = builtins.float(start), builtins.float(stop)
        if num <= 0:
            return Tensor.zeros(0, dtype=to_tinygrad_dtype(dtype))
        div = (num - 1) if endpoint else num
        step = (e - s) / div if div else 0.0
        exps = [s + i * step for i in range(num)]
        if endpoint and num > 1:
            exps[-1] = e
        vals = [builtins.float(base) ** v for v in exps]
        return convert_to_tensor(vals, dtype)
    # Tensor endpoints: output stacks along axis 0.
    s, e = _broadcast_endpoints(start, stop)
    tg_dtype = to_tinygrad_dtype(dtype)
    if num <= 0:
        return Tensor.zeros(0, *s.shape, dtype=tg_dtype)
    grid = _linspace_grid(s, e, num, endpoint)
    out = (grid * _math.log(builtins.float(base))).exp()
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if axis != 0:
        raise NotImplementedError(
            "tinygrad backend: geomspace with axis != 0 is not implemented"
        )
    if dtype is None:
        dtype = result_type(_dtype_of(start), _dtype_of(stop), float)
    dtype = standardize_dtype(dtype)
    tg_dtype = to_tinygrad_dtype(dtype)
    if isinstance(start, (builtins.int, builtins.float)) and isinstance(
        stop, (builtins.int, builtins.float)
    ):
        # Scalar endpoints: host-side float64 (see logspace).
        s, e = builtins.float(start), builtins.float(stop)
        if s == 0 or e == 0 or (s < 0) != (e < 0):
            raise ValueError(
                "geomspace requires nonzero start/stop of the same sign"
            )
        if num <= 0:
            return Tensor.zeros(0, dtype=tg_dtype)
        sign = -1.0 if s < 0 else 1.0
        ls, le = _math.log10(builtins.abs(s)), _math.log10(builtins.abs(e))
        div = (num - 1) if endpoint else num
        step = (le - ls) / div if div else 0.0
        vals = [sign * 10.0 ** (ls + i * step) for i in range(num)]
        # numpy pins the endpoints exactly.
        vals[0] = s
        if endpoint and num > 1:
            vals[-1] = e
        return convert_to_tensor(vals, dtype)
    # Tensor endpoints: per-element sign, geometric grid along axis 0.
    s, e = _broadcast_endpoints(start, stop)
    if num <= 0:
        return Tensor.zeros(0, *s.shape, dtype=tg_dtype)
    grid = _linspace_grid(s.abs().log(), e.abs().log(), num, endpoint)
    out = s.sign().reshape([1, *s.shape]) * grid.exp()
    # Pin both endpoints exactly, like numpy.
    t = Tensor.arange(num, dtype=to_tinygrad_dtype("int32")).reshape(
        [num] + [1] * s.ndim
    )
    out = (t == 0).where(s.reshape([1, *s.shape]), out)
    if endpoint and num > 1:
        out = (t == (num - 1)).where(e.reshape([1, *e.shape]), out)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def _eye_dim(v):
    # numpy validates eye's dimensions with operator.index: floats (python,
    # numpy, or float-dtype tensors) are a TypeError, never truncated.
    if isinstance(v, Tensor):
        if "int" not in to_keras_dtype(v.dtype):
            raise TypeError(
                "eye dimensions must be integers. Received: tensor of "
                f"dtype {to_keras_dtype(v.dtype)}"
            )
        return builtins.int(v.item())
    if isinstance(v, np.generic):
        v = v.item()
    return operator.index(v)


def eye(N, M=None, k=0, dtype=None):
    dtype = standardize_dtype(dtype) if dtype else "float32"
    N = _eye_dim(N)
    M = N if M is None else _eye_dim(M)
    if k == 0:
        return Tensor.eye(N, M, dtype=to_tinygrad_dtype(dtype))
    return convert_to_tensor(np.eye(N, M, k=k), dtype)


def identity(n, dtype=None):
    return eye(n, dtype=dtype)


def tri(N, M=None, k=0, dtype=None):
    dtype = standardize_dtype(dtype) if dtype else "float32"
    return convert_to_tensor(np.tri(N, M, k), dtype)


def tril(x, k=0):
    return convert_to_tensor(x).tril(diagonal=k)


def triu(x, k=0):
    return convert_to_tensor(x).triu(diagonal=k)


def copy(x):
    return convert_to_tensor(x).contiguous()


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


# ---- indexing ---------------------------------------------------------------


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    axis = _norm_axis(axis, x.ndim)
    dim = x.shape[axis]
    if not isinstance(indices, Tensor):
        # Free host-side bounds check: numpy raises IndexError, while
        # tinygrad's indexing would silently zero-fill out-of-range rows.
        idx_np = np.asarray(indices)
        if idx_np.size and (
            builtins.int(idx_np.min()) < -dim
            or builtins.int(idx_np.max()) >= dim
        ):
            raise IndexError(
                f"take: index out of range for axis {axis} with size {dim}"
            )
    # NOTE deviation: lazy Tensor indices are NOT bounds-checked (that
    # would force a realize); out-of-range entries follow tinygrad's
    # silent zero-fill instead of numpy's IndexError.
    indices = convert_to_tensor(indices, "int32")
    # Wrap negative indices explicitly (numpy semantics) rather than relying
    # on the indexing engine.
    indices = (indices < 0).where(indices + dim, indices)
    index = [builtins.slice(None)] * x.ndim
    index[axis] = indices
    return x[tuple(index)]


def take_along_axis(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, "int32")
    if axis is None:
        return take_along_axis(x.reshape(-1), indices.reshape(-1), axis=0)
    axis = _norm_axis(axis, x.ndim)
    # numpy broadcasts x and indices against each other on every dim except
    # `axis` (tinygrad's gather wants exact shapes).
    x_shape = builtins.list(x.shape)
    i_shape = builtins.list(indices.shape)
    for i in range(x.ndim):
        if i == axis:
            continue
        d = builtins.max(x_shape[i], i_shape[i])
        x_shape[i] = d
        i_shape[i] = d
    x = x.expand(x_shape)
    indices = indices.expand(i_shape)
    dim = x.shape[axis]
    # NOTE deviation: indices are lazy Tensors here, so out-of-range
    # entries are NOT bounds-checked (that would force a realize) and
    # follow tinygrad's silent zero-fill instead of numpy's IndexError.
    indices = (indices < 0).where(indices + dim, indices)
    return x.gather(axis, indices)


def get_item(x, key):
    return convert_to_tensor(x)[key]


def nonzero(x):
    """Indices of nonzero elements, numpy-style tuple-per-dimension.

    The output shape is data-dependent, so the nonzero COUNT is realized
    eagerly here (`.numpy()` on the mask sum) — acceptable for this op only,
    the index computation itself stays in tinygrad ops. All-zero inputs
    return zero-length tensors.
    """
    x = convert_to_tensor(x)
    if x.ndim == 0:
        x = x.reshape(1)
    flat = x.reshape(-1)
    n = flat.shape[0]
    mask = flat != 0
    count = builtins.int(mask.sum().numpy())  # data-dependent shape
    # Stable extraction without argsort-stability assumptions: nonzero
    # positions keep their own flat index as sort key, zeros get sentinel
    # `n`, so ascending sort puts nonzero positions first, in order.
    key = mask.where(
        Tensor.arange(n, dtype=to_tinygrad_dtype("int32")), n
    )
    pos = key.sort(dim=0)[0][:count]
    strides = []
    stride = 1
    for d in reversed(x.shape):
        strides.append(stride)
        stride *= d
    strides = strides[::-1]
    return tuple(
        ((pos // s) % d).cast(to_tinygrad_dtype("int32"))
        for s, d in zip(strides, x.shape)
    )


def isin(x1, x2, assume_unique=False, invert=False):
    # Comparison-matrix membership test: O(size(x1) * size(x2)) but fully
    # lazy — fine for the moderate sizes Keras uses. `assume_unique` needs
    # no special handling in this formulation.
    a = convert_to_tensor(x1)
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    af = convert_to_tensor(x1, dtype).reshape(-1)
    bf = convert_to_tensor(x2, dtype).reshape(-1)
    hits = (af.reshape(-1, 1) == bf.reshape(1, -1)).any(axis=1)
    if invert:
        hits = hits.logical_not()
    return hits.reshape(a.shape)


def unravel_index(indices, shape):
    x = convert_to_tensor(indices)
    dtype = x.dtype
    out = []
    for d in reversed(tuple(shape)):
        out.append((x % d).cast(dtype))
        x = x // d
    return tuple(reversed(out))


def diag(x, k=0):
    x = convert_to_tensor(x)
    if x.ndim == 1:
        n = x.shape[0] + builtins.abs(k)
        base = Tensor.eye(n, dtype=x.dtype)
        if k != 0:
            mask = convert_to_tensor(
                np.eye(n, k=k), to_keras_dtype(x.dtype)
            )
        else:
            mask = base
        # Both broadcast directions place x[i] at diagonal position i, so
        # the padded vector needs |k| leading zeros for either sign of k
        # (k > 0 shifts columns, k < 0 shifts rows).
        pad_before = builtins.abs(k)
        padded = x.pad(
            ((pad_before, n - x.shape[0] - pad_before),)
        )
        return mask * padded.reshape(1, n).expand(n, n).permute(1, 0) if (
            k < 0
        ) else mask * padded.reshape(1, n).expand(n, n)
    if x.ndim == 2:
        # Diagonal EXTRACTION (numpy semantics for 2-D input): shift the
        # columns by k, mask with eye, and sum rows.
        rows, cols = x.shape
        n = builtins.min(rows, cols - k) if k >= 0 else builtins.min(
            rows + k, cols
        )
        if n <= 0:
            return Tensor.zeros(0, dtype=x.dtype)
        row0 = builtins.max(0, -k)
        col0 = builtins.max(0, k)
        window = x[row0 : row0 + n, col0 : col0 + n]
        eye = Tensor.eye(n, dtype=x.dtype)
        return (window * eye).sum(axis=1)
    raise NotImplementedError(
        "tinygrad backend: diag of a >2-D tensor is not implemented"
    )


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return x.diagonal(
        offset=offset,
        dim1=_norm_axis(axis1, x.ndim),
        dim2=_norm_axis(axis2, x.ndim),
    )


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    dtype = to_keras_dtype(x.dtype)
    # numpy reference dtype promotion for the accumulation.
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    x = x.cast(to_tinygrad_dtype(dtype))
    return diagonal(x, offset=offset, axis1=axis1, axis2=axis2).sum(
        axis=-1
    ).cast(to_tinygrad_dtype(dtype))


# ---- sorting ----------------------------------------------------------------
# tinygrad's native sort/argsort are bitonic (inputs padded to a power of two
# internally) and stable — fine for moderate sizes; very large sorts pay the
# O(n log^2 n) bitonic cost.


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        out = x.reshape(-1).sort(dim=0)[0]
    else:
        out = x.sort(dim=_norm_axis(axis, x.ndim))[0]
    # tinygrad's bitonic sort promotes small dtypes internally; sorting must
    # preserve the input dtype.
    return out.cast(x.dtype) if out.dtype != x.dtype else out


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if x.ndim == 0:
        # numpy: argsort of a scalar is [0].
        return Tensor.zeros(1, dtype=to_tinygrad_dtype("int32"))
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    return x.argsort(dim=_norm_axis(axis, x.ndim)).cast(
        to_tinygrad_dtype("int32")
    )


# ---- linear algebra helpers -------------------------------------------------


def dot(x1, x2):
    a, b = _pair(x1, x2)
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        return a * b
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    if a.ndim == 1 and b.ndim == 1:
        return (a * b).sum()
    if b.ndim <= 2:
        # matmul agrees with numpy dot here (dot keeps the promoted dtype;
        # the int8-to-int32 rule is matmul-specific).
        return a.matmul(b)
    # numpy dot with a >2-D right operand contracts a's last axis with b's
    # SECOND-TO-LAST axis and outer-stacks the remaining axes — result shape
    # a.shape[:-1] + b.shape[:-2] + b.shape[-1:]. matmul would broadcast
    # batch dims instead, producing a silently wrong shape/values.
    return tensordot(a, b, axes=[[a.ndim - 1], [b.ndim - 2]])


def outer(x1, x2):
    a, b = _pair(x1, x2)
    a = a if isinstance(a, Tensor) else convert_to_tensor(a)
    b = b if isinstance(b, Tensor) else convert_to_tensor(b)
    return a.reshape(-1, 1) * b.reshape(1, -1)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa = axisb = axisc = axis
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)
    da, db = a.shape[-1], b.shape[-1]
    if da not in (2, 3) or db not in (2, 3):
        raise ValueError(
            "incompatible dimensions for cross product (dimension must "
            f"be 2 or 3). Received: x1 last dim={da}, x2 last dim={db}"
        )
    a0, a1 = a[..., 0], a[..., 1]
    b0, b1 = b[..., 0], b[..., 1]
    if da == 2 and db == 2:
        # numpy: 2-element cross is the scalar z-component; no vector axis
        # remains, so axisc does not apply.
        return a0 * b1 - a1 * b0
    # A missing third component is an implicit zero (2-vector in the plane).
    a2 = a[..., 2] if da == 3 else 0
    b2 = b[..., 2] if db == 3 else 0
    c0 = a1 * b2 - a2 * b1
    c1 = a2 * b0 - a0 * b2
    c2 = a0 * b1 - a1 * b0
    c = Tensor.stack(c0, c1, c2, dim=-1)
    return moveaxis(c, -1, axisc)


def kron(x1, x2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    ndim = builtins.max(a.ndim, b.ndim)
    a_shape = (1,) * (ndim - a.ndim) + a.shape
    b_shape = (1,) * (ndim - b.ndim) + b.shape
    # Interleave: a as (a1,1,a2,1,...), b as (1,b1,1,b2,...); the broadcast
    # product is (a1,b1,a2,b2,...), collapsed pairwise to (a1*b1, a2*b2, ...).
    a_r = a.reshape([d for da_ in a_shape for d in (da_, 1)])
    b_r = b.reshape([d for db_ in b_shape for d in (1, db_)])
    out = a_r * b_r
    return out.reshape([da_ * db_ for da_, db_ in zip(a_shape, b_shape)])


def tensordot(x1, x2, axes=2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    if isinstance(axes, int):
        a_axes = builtins.list(range(a.ndim - axes, a.ndim))
        b_axes = builtins.list(range(axes))
    else:
        a_ax, b_ax = axes
        a_axes = [a_ax] if isinstance(a_ax, int) else builtins.list(a_ax)
        b_axes = [b_ax] if isinstance(b_ax, int) else builtins.list(b_ax)
        a_axes = [ax % a.ndim for ax in a_axes]
        b_axes = [ax % b.ndim for ax in b_axes]
    a_free = [i for i in range(a.ndim) if i not in a_axes]
    b_free = [i for i in range(b.ndim) if i not in b_axes]
    k = 1
    for ax in a_axes:
        k *= a.shape[ax]
    # Contracted axes are moved (in pair order) to the tail of `a` and the
    # head of `b`, then the contraction is one matmul.
    ap = (
        a.permute(a_free + a_axes).reshape(-1, k) if a.ndim else
        a.reshape(1, 1)
    )
    bp = (
        b.permute(b_axes + b_free).reshape(k, -1) if b.ndim else
        b.reshape(1, 1)
    )
    if dtype == "bool":
        # bool contraction is any-of-products (numpy bool algebra).
        out = ap.cast(to_tinygrad_dtype("int32")).matmul(
            bp.cast(to_tinygrad_dtype("int32"))
        ) != 0
    else:
        out = ap.matmul(bp).cast(to_tinygrad_dtype(dtype))
    out_shape = [a.shape[i] for i in a_free] + [b.shape[i] for i in b_free]
    return out.reshape(out_shape)


def vdot(x1, x2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    a = convert_to_tensor(x1, dtype).reshape(-1)
    b = convert_to_tensor(x2, dtype).reshape(-1)
    if dtype == "bool":
        return (
            a.cast(to_tinygrad_dtype("int32"))
            * b.cast(to_tinygrad_dtype("int32"))
        ).sum() != 0
    return (a * b).sum().cast(to_tinygrad_dtype(dtype))


def inner(x1, x2):
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    a = convert_to_tensor(x1, dtype)
    b = convert_to_tensor(x2, dtype)
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    return tensordot(a, b, axes=[[a.ndim - 1], [b.ndim - 1]])


def correlate(x1, x2, mode="valid"):
    # 1-D cross-correlation via a gathered sliding-window matrix:
    # O(output_len * len(x2)) memory — cheap for the moderate sizes Keras
    # sees. dtype policy copied from the numpy reference backend.
    dtype = result_type(_dtype_of(x1), _dtype_of(x2))
    if dtype == "int64":
        dtype = "float64"
    elif dtype not in ("bfloat16", "float16", "float64"):
        dtype = "float32"
    a = convert_to_tensor(x1, dtype).reshape(-1)
    v = convert_to_tensor(x2, dtype).reshape(-1)
    n, m = a.shape[0], v.shape[0]
    if m > n:
        # numpy computes with the longer input first and reverses the output.
        return correlate(x2, x1, mode).flip(0)
    apad = a.pad(((m - 1, m - 1),)) if m > 1 else a
    k_full = n + m - 1
    idx = Tensor.arange(k_full, dtype=to_tinygrad_dtype("int32")).reshape(
        k_full, 1
    ) + Tensor.arange(m, dtype=to_tinygrad_dtype("int32")).reshape(1, m)
    full = (apad[idx] * v.reshape(1, m)).sum(axis=1)
    if mode == "full":
        return full
    if mode == "same":
        start = (m - 1) // 2
        return full[start:start + n]
    if mode == "valid":
        return full[m - 1:n]
    raise ValueError(f"Invalid correlate mode: {mode}")


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(op) for op in operands]
    kdtypes = sorted({to_keras_dtype(op.dtype) for op in operands})
    if len(kdtypes) == 1 and kdtypes[0] == "int8":
        # jax rule (numpy-backend referee): all-int8 einsum returns int32.
        result_dtype = "int32"
    else:
        result_dtype = result_type(*kdtypes)
    # bool contraction is any-of-products (numpy bool algebra): compute in
    # int32, compare back. Everything else computes in the result dtype and
    # is cast back (tinygrad widens small-int accumulation internally).
    compute_dtype = "int32" if result_dtype == "bool" else result_dtype
    tg_compute = to_tinygrad_dtype(compute_dtype)
    operands = [
        op.cast(tg_compute) if op.dtype != tg_compute else op
        for op in operands
    ]
    out = Tensor.einsum(subscripts, *operands)
    if result_dtype == "bool":
        return out != 0
    tg_result = to_tinygrad_dtype(result_dtype)
    return out.cast(tg_result) if out.dtype != tg_result else out


def ndim(x):
    return convert_to_tensor(x).ndim


def size(x):
    x = convert_to_tensor(x)
    out = 1
    for d in x.shape:
        out *= d
    return out


def meshgrid(*x, indexing="xy"):
    # Native reshape+expand formulation (each output keeps its own input's
    # dtype, like numpy). For "xy", the first two axes are swapped relative
    # to "ij".
    if indexing not in ("xy", "ij"):
        raise ValueError(f"Invalid meshgrid indexing: {indexing}")
    tensors = [convert_to_tensor(t).reshape(-1) for t in x]
    n = len(tensors)

    def out_axis(i):
        if indexing == "xy" and n >= 2 and i in (0, 1):
            return 1 - i
        return i

    shape = [1] * n
    for i, t in enumerate(tensors):
        shape[out_axis(i)] = t.shape[0]
    outs = []
    for i, t in enumerate(tensors):
        sh = [1] * n
        sh[out_axis(i)] = t.shape[0]
        outs.append(t.reshape(sh).expand(shape))
    return outs


def select(condlist, choicelist, default=0):
    # np.select promotes the result dtype across every choice (a weak-typed
    # python-scalar default doesn't participate).
    default_is_scalar = isinstance(
        default, (builtins.bool, builtins.int, builtins.float)
    )
    dtypes_to_resolve = [_dtype_of(c) for c in choicelist]
    if not default_is_scalar:
        dtypes_to_resolve.append(_dtype_of(default))
    dtype = result_type(*dtypes_to_resolve)
    out = default if default_is_scalar else convert_to_tensor(default, dtype)
    for cond, choice in zip(condlist[::-1], choicelist[::-1]):
        cond = convert_to_tensor(cond, "bool")
        out = cond.where(convert_to_tensor(choice, dtype), out)
    return out


def searchsorted(sorted_sequence, values, side="left"):
    # No native searchsorted in tinygrad: comparison-matrix formulation
    # (fine for the moderate bin counts Keras uses it for).
    s = convert_to_tensor(sorted_sequence)
    v = convert_to_tensor(values)
    if side == "left":
        hits = (s.reshape(1, -1) < v.reshape(-1, 1)).cast(
            to_tinygrad_dtype("int32")
        )
    else:
        hits = (s.reshape(1, -1) <= v.reshape(-1, 1)).cast(
            to_tinygrad_dtype("int32")
        )
    return hits.sum(axis=1).reshape(v.shape)


def digitize(x, bins):
    return searchsorted(bins, x, side="right").cast(
        to_tinygrad_dtype("int32")
    )


def histogram(x, bins=10, range=None):
    # numpy semantics: `bins` equal-width buckets, all half-open except the
    # last which is closed; values outside [lo, hi] are dropped. Everything
    # stays lazy: the degenerate lo == hi case is handled with tensor ops.
    x = _float(convert_to_tensor(x)).reshape(-1)
    nbins = builtins.int(bins)
    if range is not None:
        lo = convert_to_tensor(builtins.float(range[0]), "float32")
        hi = convert_to_tensor(builtins.float(range[1]), "float32")
    else:
        lo, hi = x.min(), x.max()
    degenerate = hi == lo
    lo = degenerate.where(lo - 0.5, lo)
    hi = degenerate.where(hi + 0.5, hi)
    steps = Tensor.arange(
        nbins + 1, dtype=to_tinygrad_dtype("float32")
    ) / builtins.float(nbins)
    edges = lo + steps * (hi - lo)
    xs = x.reshape(-1, 1)
    ge = xs >= edges[:-1].reshape(1, nbins)
    lt = xs < edges[1:].reshape(1, nbins)
    # Close the last bin: x == hi belongs to it.
    is_last = Tensor.arange(nbins, dtype=to_tinygrad_dtype("int32")).reshape(
        1, nbins
    ) == (nbins - 1)
    lt = is_last.where(xs <= edges[-1], lt)
    counts = (ge & lt).cast(to_tinygrad_dtype("int32")).sum(axis=0)
    return counts, edges


def diff(a, n=1, axis=-1):
    a = convert_to_tensor(a)
    if n == 0:
        return a
    out = a
    for _ in range(n):
        nd = out.ndim
        ax = axis % nd
        upper = [builtins.slice(None)] * nd
        lower = [builtins.slice(None)] * nd
        upper[ax] = builtins.slice(1, None)
        lower[ax] = builtins.slice(None, -1)
        out = out[tuple(upper)] - out[tuple(lower)]
    return out


def real(x):
    x = convert_to_tensor(x)
    if isinstance(x, ComplexTensor):
        # Complex-lite interop: the held float32 component (differentiable).
        return x.real
    return x


def imag(x):
    x = convert_to_tensor(x)
    if isinstance(x, ComplexTensor):
        # Complex-lite interop: the held float32 component (differentiable).
        return x.imag
    return zeros_like(x)


def conjugate(x):
    x = convert_to_tensor(x)
    if isinstance(x, ComplexTensor):
        # Passthrough would be silently wrong (conj negates the imaginary
        # part); conjugate is complex ARITHMETIC, outside the interop set.
        raise NotImplementedError(_COMPLEX_INTEROP_MSG)
    return x


conj = conjugate


def signbit(x):
    x = convert_to_tensor(x)
    dtype = to_keras_dtype(x.dtype)
    if "float" not in dtype:
        # Ints/bool have no negative zero; the comparison is exact.
        return x < 0
    # `x < 0` cannot see the sign bit of -0.0 (or negative NaNs): bitcast to
    # the same-width signed int and test the sign bit, like np.signbit.
    int_dtype = {
        "float16": "int16",
        "bfloat16": "int16",
        "float32": "int32",
        "float64": "int64",
    }[dtype]
    return x.bitcast(to_tinygrad_dtype(int_dtype)) < 0


# Same-width signed-int spelling of each float dtype, for bit-level ops
# (bitcast works on all four widths in tinygrad 0.13 — proven by signbit).
_FLOAT_BITS_INT = {
    "float16": "int16",
    "bfloat16": "int16",
    "float32": "int32",
    "float64": "int64",
}


def nextafter(x1, x2):
    """Bit-exact ulp step toward x2, via same-width int bitcast.

    For IEEE floats, adding/subtracting 1 to/from the raw bit pattern moves
    one ulp away from / toward zero (sign-magnitude ordering), and two's-
    complement integer +-1 performs exactly that bit-pattern step. The
    inf edges fall out naturally (max_finite + 1 ulp = inf and back);
    equal / zero / nan cases are handled by explicit masks.
    """
    dtype = result_type(_dtype_of(x1), _dtype_of(x2), builtins.float)
    x = convert_to_tensor(x1, dtype)
    y = convert_to_tensor(x2, dtype)
    int_dtype = to_tinygrad_dtype(_FLOAT_BITS_INT[dtype])
    tg_dtype = to_tinygrad_dtype(dtype)
    xi = x.bitcast(int_dtype)
    # Step +1 grows the magnitude (toward y on x's side of zero), -1 shrinks
    # it: for x > 0 grow iff y > x, for x < 0 grow iff y < x.
    step = ((y > x) == (x > 0)).where(1, -1).cast(int_dtype)
    out = (xi + step).bitcast(tg_dtype)
    # +-0 steps to the smallest subnormal with y's sign (both-zero pairs are
    # caught by the x == y mask below, which returns y, preserving its sign).
    tiny = Tensor(1, dtype=int_dtype).bitcast(tg_dtype)
    out = (x == 0).where((y > 0).where(tiny, -tiny), out)
    out = (x == y).where(y, out)
    out = ((x != x) | (y != y)).where(x + y, out)
    return out


def view(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        return x
    dtype = standardize_dtype(dtype)
    tg_dtype = to_tinygrad_dtype(dtype)
    old_size, new_size = x.dtype.itemsize, tg_dtype.itemsize
    if old_size != new_size:
        if x.ndim == 0:
            raise ValueError(
                "Cannot view a scalar as a different dtype if item sizes "
                "are different."
            )
        if (x.shape[-1] * old_size) % new_size != 0:
            raise ValueError(
                f"Cannot view array of shape {x.shape} and dtype "
                f"{to_keras_dtype(x.dtype)} as dtype {dtype}: the last axis "
                f"is not divisible by the new itemsize."
            )
    # tinygrad bitcast reinterprets in place for same-width dtypes and
    # repacks the last axis for size-changing views (numpy view semantics).
    return x.bitcast(tg_dtype)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    dtype = to_keras_dtype(x.dtype)
    if "float" not in dtype:
        # Integer/bool tensors cannot hold NaN or inf: identity, dtype kept.
        return x
    # numpy substitutes the dtype's own max/min for the infinities.
    big = {
        "float16": 65504.0,
        "bfloat16": 3.3895313892515355e38,
        "float32": 3.4028234663852886e38,
        "float64": 1.7976931348623157e308,
    }.get(dtype, 3.4028234663852886e38)
    posinf = big if posinf is None else posinf
    neginf = -big if neginf is None else neginf
    x = (x != x).where(nan, x)
    x = (x == float("inf")).where(posinf, x)
    x = (x == float("-inf")).where(neginf, x)
    return x


# ---- bitwise (numpy-backend semantics: promote via result_type; a plain
# python int shift count passes through unpromoted) -------------------------


def _bitwise_pair(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = result_type(to_keras_dtype(x.dtype), to_keras_dtype(y.dtype))
    tg = to_tinygrad_dtype(dtype)
    return x.cast(tg), y.cast(tg)


def bitwise_and(x, y):
    x, y = _bitwise_pair(x, y)
    return x & y


def bitwise_invert(x):
    x = convert_to_tensor(x)
    if to_keras_dtype(x.dtype) == "bool":
        return x.logical_not()
    return x ^ -1


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x, y = _bitwise_pair(x, y)
    return x | y


def bitwise_xor(x, y):
    x, y = _bitwise_pair(x, y)
    return x ^ y


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, builtins.int):
        x, y = _bitwise_pair(x, y)
    return x << y


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, builtins.int):
        x, y = _bitwise_pair(x, y)
    return x >> y


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    result_dtype = result_type(to_keras_dtype(y.dtype), builtins.float)
    compute = "float64" if result_dtype == "float64" else "float32"
    tg_compute = to_tinygrad_dtype(compute)
    yt = y.cast(tg_compute) if y.dtype != tg_compute else y
    ax = _norm_axis(axis, yt.ndim)

    def _slc(t, sl, a):
        index = [builtins.slice(None)] * t.ndim
        index[a] = sl
        return t[tuple(index)]

    y0 = _slc(yt, builtins.slice(None, -1), ax)
    y1 = _slc(yt, builtins.slice(1, None), ax)
    if x is not None:
        xt = convert_to_tensor(x).cast(tg_compute)
        if xt.ndim == 1:
            shape = [1] * yt.ndim
            shape[ax] = xt.shape[0] - 1
            d = (xt[1:] - xt[:-1]).reshape(shape)
        else:
            d = _slc(xt, builtins.slice(1, None), ax) - _slc(
                xt, builtins.slice(None, -1), ax
            )
    elif isinstance(dx, Tensor):
        d = dx.cast(tg_compute)
    else:
        d = builtins.float(_host_scalar(dx))
    out = (d * (y0 + y1) * 0.5).sum(axis=ax)
    tg_result = to_tinygrad_dtype(result_dtype)
    return out.cast(tg_result) if out.dtype != tg_result else out


def corrcoef(x):
    x = convert_to_tensor(x)
    in_dtype = to_keras_dtype(x.dtype)
    # numpy-reference dtype rule (matches deg2rad/rad2deg's family).
    if in_dtype in ("int64", "float64"):
        dtype = "float64"
    elif in_dtype in ("bfloat16", "float16"):
        dtype = in_dtype
    else:
        dtype = floatx()
    compute = "float64" if dtype == "float64" else "float32"
    xt = x.cast(to_tinygrad_dtype(compute))
    one_dim = xt.ndim == 1
    if one_dim:
        xt = xt.reshape(1, -1)
    if xt.ndim != 2:
        raise ValueError(
            "corrcoef expects a 1-D or 2-D input. "
            f"Received: x.ndim={x.ndim}"
        )
    m = xt - xt.mean(axis=1, keepdim=True)
    # The 1/(N-1) covariance normalization cancels in the correlation ratio.
    c = m.matmul(m.permute(1, 0))
    d = (m * m).sum(axis=1).sqrt()
    out = c / (d.reshape(-1, 1) * d.reshape(1, -1))
    # numpy clips rounding excursions to the valid correlation range.
    out = out.clip(-1.0, 1.0)
    if one_dim:
        # np.corrcoef of a 1-D input is the scalar self-correlation.
        out = out.reshape(())
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


# ---- quantile family --------------------------------------------------------
# Composition: move the reduced axes to one trailing axis (like median), sort
# it with tinygrad's bitonic sort, and gather/interpolate at the virtual
# index q * (n - 1). Quantile POSITIONS are structural parameters (like split
# indices or pad widths): tensor/array q values are read out host-side; the
# data path (sort + gather + lerp) is pure Tensor ops and differentiable
# w.r.t. x.


def _host_q(q):
    if isinstance(q, Tensor):
        if to_keras_dtype(q.dtype) == "bfloat16":
            q = q.cast(to_tinygrad_dtype("float32"))
        qn = np.asarray(q.numpy(), dtype=np.float64)
    else:
        qn = np.asarray(q, dtype=np.float64)
    shape = builtins.list(qn.shape)
    vals = [builtins.float(v) for v in qn.reshape(-1)]
    return vals, shape


def _quantile_impl(x, q_vals, q_shape, axis, method, keepdims, nan_aware):
    """x: float Tensor in the compute dtype. Returns the stacked result
    (q dims leading, numpy layout)."""
    if method not in ("linear", "lower", "higher", "midpoint", "nearest"):
        raise ValueError(f"Invalid method: {method}")
    if x.ndim == 0:
        x = x.reshape(1)
    if axis is None:
        red = tuple(range(x.ndim))
    elif isinstance(axis, (builtins.list, tuple)):
        red = tuple(a % x.ndim for a in axis)
    else:
        red = (axis % x.ndim,)
    keep = [i for i in range(x.ndim) if i not in red]
    n = 1
    for i in red:
        n *= x.shape[i]
    xt = x.permute(keep + builtins.list(red)).reshape(
        [x.shape[i] for i in keep] + [n]
    )
    lead = builtins.list(xt.shape[:-1])
    nan_mask = xt != xt
    if nan_aware:
        # NaNs sort to the tail as +inf; per-slice valid counts move the
        # virtual index, and all-NaN slices answer NaN.
        s = nan_mask.where(builtins.float("inf"), xt).sort(dim=-1)[0]
        c = (
            nan_mask.logical_not()
            .cast(to_tinygrad_dtype("int32"))
            .sum(axis=-1)
        )
        cf = c.cast(to_tinygrad_dtype(s.dtype))
        has_all_nan = c == 0
    else:
        s = xt.sort(dim=-1)[0]
        # np.quantile propagates NaN per slice (nanquantile is the one that
        # skips them).
        has_nan = nan_mask.any(axis=-1)

    def _gather(idx):
        # idx: int32 Tensor of shape `lead` — one sorted position per slice.
        return s.gather(-1, idx.reshape(lead + [1])).reshape(lead)

    int32 = to_tinygrad_dtype("int32")
    outs = []
    for qv in q_vals:
        if nan_aware:
            v = ((cf - 1.0) * qv).maximum(0.0)
            lo = v.floor()
            hi = v.ceil()
            if method == "linear":
                frac = v - lo
                out = _gather(lo.cast(int32)) * (1.0 - frac) + _gather(
                    hi.cast(int32)
                ) * frac
            elif method == "lower":
                out = _gather(lo.cast(int32))
            elif method == "higher":
                out = _gather(hi.cast(int32))
            elif method == "midpoint":
                out = (
                    _gather(lo.cast(int32)) + _gather(hi.cast(int32))
                ) * 0.5
            else:  # nearest (round-half-even, like np.round)
                out = _gather(v.round().cast(int32))
            out = has_all_nan.where(builtins.float("nan"), out)
        else:
            v = qv * (n - 1)
            lo_i = builtins.min(builtins.int(_math.floor(v)), n - 1)
            hi_i = builtins.min(builtins.int(_math.ceil(v)), n - 1)
            lead_slice = (builtins.slice(None),) * len(lead)
            if method == "linear":
                frac = v - lo_i
                out = (
                    s[lead_slice + (lo_i,)] * (1.0 - frac)
                    + s[lead_slice + (hi_i,)] * frac
                )
            elif method == "lower":
                out = s[lead_slice + (lo_i,)]
            elif method == "higher":
                out = s[lead_slice + (hi_i,)]
            elif method == "midpoint":
                out = (
                    s[lead_slice + (lo_i,)] + s[lead_slice + (hi_i,)]
                ) * 0.5
            else:  # nearest (round-half-even, like np.round)
                out = s[lead_slice + (builtins.int(builtins.round(v)),)]
            out = has_nan.where(builtins.float("nan"), out)
        if keepdims:
            out = out.reshape(
                [1 if i in red else x.shape[i] for i in range(x.ndim)]
            )
        outs.append(out)
    slice_shape = builtins.list(outs[0].shape)
    if len(q_shape) == 0:
        return outs[0]
    stacked = Tensor.stack(*outs, dim=0)
    return stacked.reshape(q_shape + slice_shape)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = to_keras_dtype(x.dtype)
    if ori_dtype == "bool":
        x = x.cast(to_tinygrad_dtype(floatx()))
        ori_dtype = floatx()
    if ori_dtype == "int64":
        dtype = floatx()
    else:
        dtype = result_type(ori_dtype, builtins.float)
    compute = "float64" if dtype == "float64" else "float32"
    x = x.cast(to_tinygrad_dtype(compute))
    q_vals, q_shape = _host_q(q)
    out = _quantile_impl(x, q_vals, q_shape, axis, method, keepdims, False)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def percentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = to_keras_dtype(x.dtype)
    if ori_dtype == "bool":
        x = x.cast(to_tinygrad_dtype(floatx()))
        ori_dtype = floatx()
    dtype = result_type(ori_dtype, builtins.float)
    compute = "float64" if dtype == "float64" else "float32"
    x = x.cast(to_tinygrad_dtype(compute))
    q_vals, q_shape = _host_q(q)
    q_vals = [v / 100.0 for v in q_vals]
    out = _quantile_impl(x, q_vals, q_shape, axis, method, keepdims, False)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = to_keras_dtype(x.dtype)
    if ori_dtype == "bool":
        x = x.cast(to_tinygrad_dtype(floatx()))
        ori_dtype = floatx()
    dtype = result_type(ori_dtype, builtins.float)
    compute = "float64" if dtype == "float64" else "float32"
    x = x.cast(to_tinygrad_dtype(compute))
    q_vals, q_shape = _host_q(q)
    out = _quantile_impl(x, q_vals, q_shape, axis, method, keepdims, True)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    # numpy-reference dtype rule (differs from nanquantile's): non-float
    # inputs answer floatx, float inputs keep their own dtype.
    ori_dtype = to_keras_dtype(x.dtype)
    dtype = ori_dtype if "float" in ori_dtype else floatx()
    compute = "float64" if dtype == "float64" else "float32"
    x = x.cast(to_tinygrad_dtype(compute))
    q_vals, q_shape = _host_q(q)
    q_vals = [v / 100.0 for v in q_vals]
    out = _quantile_impl(x, q_vals, q_shape, axis, method, keepdims, True)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


# ---- nan-family tail --------------------------------------------------------


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return argmax(x, axis=axis, keepdims=keepdims)
    nan_mask = x != x
    out = argmax(
        nan_mask.where(builtins.float("-inf"), x),
        axis=axis,
        keepdims=keepdims,
    )
    # numpy-backend semantics: all-NaN slices answer -1, not an error.
    all_nan = nan_mask.all(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)
    return all_nan.where(-1, out)


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return argmin(x, axis=axis, keepdims=keepdims)
    nan_mask = x != x
    out = argmin(
        nan_mask.where(builtins.float("inf"), x),
        axis=axis,
        keepdims=keepdims,
    )
    all_nan = nan_mask.all(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)
    return all_nan.where(-1, out)


def nancumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x, dtype=dtype)
    if "float" in to_keras_dtype(x.dtype):
        x = (x != x).where(0, x)
    return cumsum(x, axis=axis)


def nanprod(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = to_keras_dtype(x.dtype)
    # numpy reference accumulation dtype for small ints/bool.
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    if "float" in dtype:
        x = (x != x).where(1, x)
    tg_dtype = to_tinygrad_dtype(dtype)
    if x.dtype != tg_dtype:
        x = x.cast(tg_dtype)
    out = x.prod(axis=_norm_axis(axis, x.ndim), keepdim=keepdims)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def _nanvar(x, axis, keepdims):
    """Masked variance in result_type(x, float32); NaNs excluded per slice,
    all-NaN slices answer NaN (numpy nanvar/nanstd semantics)."""
    x = convert_to_tensor(x)
    compute_dtype = result_type(to_keras_dtype(x.dtype), "float32")
    result_dtype = result_type(to_keras_dtype(x.dtype), builtins.float)
    tg_compute = to_tinygrad_dtype(compute_dtype)
    x = x.cast(tg_compute) if x.dtype != tg_compute else x
    axis = _norm_axis(axis, x.ndim)
    valid = x == x
    count_keep = valid.cast(x.dtype).sum(axis=axis, keepdim=True)
    mean_ = valid.where(x, 0).sum(axis=axis, keepdim=True) / count_keep
    dev = valid.where(x - mean_, 0)
    sq = (dev * dev).sum(axis=axis, keepdim=keepdims)
    count = valid.cast(x.dtype).sum(axis=axis, keepdim=keepdims)
    out = sq / count
    out = (count == 0).where(builtins.float("nan"), out)
    return out, to_tinygrad_dtype(result_dtype)


def nanvar(x, axis=None, keepdims=False):
    out, tg_result = _nanvar(x, axis, keepdims)
    return out.cast(tg_result) if out.dtype != tg_result else out


def nanstd(x, axis=None, keepdims=False):
    out, tg_result = _nanvar(x, axis, keepdims)
    out = out.sqrt()
    return out.cast(tg_result) if out.dtype != tg_result else out


# ---- structure tail ---------------------------------------------------------


def fliplr(x):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "Input must be >= 2-d. Received: input.ndim=" f"{x.ndim}"
        )
    return x.flip(1)


def flipud(x):
    x = convert_to_tensor(x)
    if x.ndim < 1:
        raise ValueError(
            "Input must be >= 1-d. Received: input.ndim=" f"{x.ndim}"
        )
    return x.flip(0)


def array_split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    axis = _norm_axis(axis, x.ndim)
    if isinstance(indices_or_sections, Tensor):
        indices_or_sections = indices_or_sections.numpy().tolist()
    elif hasattr(indices_or_sections, "tolist"):
        indices_or_sections = indices_or_sections.tolist()
    if not isinstance(indices_or_sections, builtins.int):
        return split(x, indices_or_sections, axis=axis)
    n = indices_or_sections
    if n <= 0:
        raise ValueError("Number sections must be larger than 0.")
    dim = x.shape[axis]
    base, extra = divmod(dim, n)
    sizes = [base + 1] * extra + [base] * (n - extra)
    out = []
    start = 0
    for size_ in sizes:
        index = [builtins.slice(None)] * x.ndim
        index[axis] = builtins.slice(start, start + size_)
        out.append(x[tuple(index)])
        start += size_
    return out


def hsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim == 0:
        raise ValueError(
            "hsplit only works on arrays of 1 or more dimensions"
        )
    return split(x, indices_or_sections, axis=1 if x.ndim > 1 else 0)


def vsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "vsplit only works on arrays of 2 or more dimensions"
        )
    return split(x, indices_or_sections, axis=0)


def dsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim < 3:
        raise ValueError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return split(x, indices_or_sections, axis=2)


def diagflat(x, k=0):
    x = convert_to_tensor(x)
    return diag(x.reshape(-1), k=k)


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    if x.ndim != 1:
        raise ValueError(
            "x must be a one-dimensional array or sequence. "
            f"Received: x.ndim={x.ndim}"
        )
    result_dtype = to_keras_dtype(x.dtype)
    # numpy reference: powers are computed in result_type(x, floatx) and the
    # result cast back to x's own dtype.
    compute_dtype = result_type(result_dtype, floatx())
    n = x.shape[0]
    N = n if N is None else builtins.int(N)
    tg_result = to_tinygrad_dtype(result_dtype)
    if N == 0:
        return Tensor.zeros(n, 0, dtype=tg_result)
    xf = x.cast(to_tinygrad_dtype(compute_dtype))
    # Iterated products instead of pow: exact integer powers and a correct
    # x^0 == 1 at x == 0 (a float pow would produce NaN there).
    cols = []
    cur = Tensor.ones(n, dtype=to_tinygrad_dtype(compute_dtype))
    for _ in range(N):
        cols.append(cur)
        cur = cur * xf
    if not increasing:
        cols = cols[::-1]
    out = Tensor.stack(*cols, dim=1)
    return out.cast(tg_result) if out.dtype != tg_result else out


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    # A full stable argsort is a valid partition at every kth (all smaller
    # elements precede it, all larger follow), and tinygrad's bitonic
    # argsort is stable — this reproduces np.argpartition's introselect
    # output on the referee's arrays and jax's own sort-based behavior.
    return argsort(x, axis=axis)


def slogdet(x):
    from keras.src.backend.tinygrad.linalg import det

    d = det(convert_to_tensor(x))
    return (d.sign(), _float(d).abs().log())


# ---- window functions -------------------------------------------------------
# Window lengths are structural metadata (like shapes): tensor/numpy inputs
# are read out host-side, and the table itself is a host-built creation-time
# constant (sanctioned numpy role, same as the DFT matrices). Output dtype is
# floatx, per the numpy reference backend.


def _host_scalar(x):
    """Read a structural scalar out of a Tensor / numpy value host-side."""
    if isinstance(x, Tensor):
        if to_keras_dtype(x.dtype) == "bfloat16":
            # tinygrad can't read bfloat16 buffers directly (no host fmt).
            x = x.cast(to_tinygrad_dtype("float32"))
        return x.item()
    if isinstance(x, (np.generic, np.ndarray)):
        return x.item()
    return x


def _window_length(x):
    return builtins.int(_host_scalar(x))


def bartlett(x):
    return convert_to_tensor(np.bartlett(_window_length(x)), floatx())


def blackman(x):
    return convert_to_tensor(np.blackman(_window_length(x)), floatx())


def hamming(x):
    return convert_to_tensor(np.hamming(_window_length(x)), floatx())


def hanning(x):
    return convert_to_tensor(np.hanning(_window_length(x)), floatx())


def kaiser(x, beta):
    return convert_to_tensor(
        np.kaiser(_window_length(x), builtins.float(_host_scalar(beta))),
        floatx(),
    )


# ---- elementwise tail -------------------------------------------------------


def fabs(x):
    x = convert_to_tensor(x)
    dtype = to_keras_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        # numpy reference: fabs is a float op; ints/bool go to floatx.
        x = x.cast(to_tinygrad_dtype(floatx()))
    return x.abs()


def angle(x):
    x = convert_to_tensor(x)
    if to_keras_dtype(x.dtype) == "int64":
        dtype = floatx()
    else:
        dtype = result_type(to_keras_dtype(x.dtype), builtins.float)
    tg_dtype = to_tinygrad_dtype(dtype)
    x = x.cast(tg_dtype) if x.dtype != tg_dtype else x
    # np.angle of a real input is arctan2(0, x): pi where the sign bit is
    # set (including -0.0 and -inf), 0 elsewhere; NaN passes through.
    out = signbit(x).where(_math.pi, 0.0)
    out = (x != x).where(x, out)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def cbrt(x):
    x = convert_to_tensor(x)
    dtype = to_keras_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16", "int32", "uint8", "uint16",
                 "uint32"):
        dtype = floatx()
    elif dtype == "int64":
        dtype = "float64"
    compute = "float64" if dtype == "float64" else "float32"
    x = x.cast(to_tinygrad_dtype(compute))
    # Odd real root: sign(x) * |x|^(1/3).
    out = x.sign() * x.abs().pow(1.0 / 3.0)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def sinc(x):
    x = convert_to_tensor(x)
    if to_keras_dtype(x.dtype) == "int64":
        dtype = floatx()
    else:
        dtype = result_type(to_keras_dtype(x.dtype), builtins.float)
    compute = "float64" if dtype == "float64" else "float32"
    x = x.cast(to_tinygrad_dtype(compute))
    xp = x * _math.pi
    zero = x == 0
    safe = zero.where(1.0, xp)
    out = zero.where(1.0, safe.sin() / safe)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def i0(x):
    x = convert_to_tensor(x)
    dtype = (
        "float64"
        if to_keras_dtype(x.dtype) in ("int64", "float64")
        else result_type(to_keras_dtype(x.dtype), builtins.float)
    )
    compute = "float64" if dtype == "float64" else "float32"
    a = x.abs().cast(to_tinygrad_dtype(compute))
    # Abramowitz & Stegun 9.8.1 / 9.8.2 polynomial approximations (abs error
    # < 1.6e-7 small branch, rel error < 1.9e-7 large branch) — the same
    # classic rational fits numpy's own i0 predecessor used.
    t2 = (a / 3.75) * (a / 3.75)
    small = 1.0 + t2 * (
        3.5156229
        + t2 * (
            3.0899424
            + t2 * (
                1.2067492
                + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813))
            )
        )
    )
    safe = a.maximum(1e-30)
    r = 3.75 / safe
    big = (safe.exp() / safe.sqrt()) * (
        0.39894228
        + r * (
            0.01328592
            + r * (
                0.00225319
                + r * (
                    -0.00157565
                    + r * (
                        0.00916281
                        + r * (
                            -0.02057706
                            + r * (
                                0.02635537
                                + r * (-0.01647633 + r * 0.00392377)
                            )
                        )
                    )
                )
            )
        )
    )
    out = (a <= 3.75).where(small, big)
    tg_dtype = to_tinygrad_dtype(dtype)
    return out.cast(tg_dtype) if out.dtype != tg_dtype else out


def isneginf(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return Tensor.zeros(*x.shape, dtype=to_tinygrad_dtype("bool"))
    return x == builtins.float("-inf")


def isposinf(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        return Tensor.zeros(*x.shape, dtype=to_tinygrad_dtype("bool"))
    return x == builtins.float("inf")


def isreal(x):
    x = convert_to_tensor(x)
    if isinstance(x, ComplexTensor):
        # Complex-lite interop: reading a component is inside the closed set.
        return x.imag == 0
    return Tensor.ones(*x.shape, dtype=to_tinygrad_dtype("bool"))


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise NotImplementedError(
        f"tinygrad backend: `keras.ops.{name}` is not implemented yet"
    )
