"""`keras.ops.math` for the tinygrad backend — minimal subset, loud stubs."""

import math as _math

# numpy is used ONLY for creation-time host-side constant tables (DFT
# cos/sin matrices, scipy windows). Constants are not gradient paths into
# the inputs; every op on actual data is a pure tinygrad Tensor op.
import numpy as np
from tinygrad import Tensor
from tinygrad import dtypes as _tg_dtypes

from keras.src.backend import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.backend.tinygrad.core import to_keras_dtype
from keras.src.backend.tinygrad.core import to_tinygrad_dtype
from keras.src.backend.tinygrad.numpy import _float
from keras.src.utils.module_utils import scipy


def rsqrt(x):
    return _float(x).rsqrt()


def erf(x):
    return _float(x).erf()


# Single-precision polynomial approximation from M. Giles, "Approximating
# the erfinv function" (2010) — max error ~1e-7 in float32, well inside the
# tests' 1e-4 tolerance. Central branch covers |x| <~ 0.99998 (w < 5), the
# tail branch the rest of the open interval (-1, 1).
_ERFINV_CENTRAL = (
    2.81022636e-08,
    3.43273939e-07,
    -3.5233877e-06,
    -4.39150654e-06,
    0.00021858087,
    -0.00125372503,
    -0.00417768164,
    0.246640727,
    1.50140941,
)
_ERFINV_TAIL = (
    -0.000200214257,
    0.000100950558,
    0.00134934322,
    -0.00367342844,
    0.00573950773,
    -0.0076224613,
    0.00943887047,
    1.00167406,
    2.83297682,
)


def _horner(w, coeffs):
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + p * w
    return p


def erfc(x):
    return 1.0 - _float(x).erf()


def erfinv(x):
    x = _float(x)
    out_dtype = x.dtype
    xf = (
        x.cast(_tg_dtypes.float32)
        if out_dtype in (_tg_dtypes.float16, _tg_dtypes.bfloat16)
        else x
    )
    # w is nan outside [-1, 1] and +inf at |x| == 1; both fixed up below.
    w = -((1.0 - xf) * (1.0 + xf)).log()
    p = (w < 5.0).where(
        _horner(w - 2.5, _ERFINV_CENTRAL),
        _horner(w.sqrt() - 3.0, _ERFINV_TAIL),
    )
    out = p * xf
    ax = xf.abs()
    out = (ax == 1.0).where(xf * _math.inf, out)
    out = (ax > 1.0).where(_math.nan, out)
    return out.cast(out_dtype)


def logsumexp(x, axis=None, keepdims=False):
    x = _float(x)
    m = x.max(axis=axis, keepdim=True)
    # An infinite max would turn exp(x - m) into nan; shift by 0 instead so
    # all-(-inf) folds to -inf and any +inf to +inf (scipy semantics).
    m = m.isinf().where(0.0, m)
    out = (x - m).exp().sum(axis=axis, keepdim=True).log() + m
    if not keepdims:
        if axis is None:
            out = out.reshape(())
        elif isinstance(axis, (list, tuple)):
            for a in sorted((a % x.ndim for a in axis), reverse=True):
                out = out.squeeze(a)
        else:
            out = out.squeeze(axis)
    return out


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    # numpy-backend semantics: k is clipped to the last-axis size instead of
    # erroring. tinygrad only produces sorted output, which is also a valid
    # answer for sorted=False (any order is acceptable then).
    values, indices = x.topk(min(int(k), x.shape[-1]))
    return values, indices


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets, "int32")
    predictions = convert_to_tensor(predictions)
    topk_values = top_k(predictions, k)[0]
    targets_values = predictions.gather(-1, targets.unsqueeze(-1))
    # tinygrad comparisons are not IEEE for nan (nan >= x is True), so a nan
    # prediction has to be masked out on both sides explicitly.
    mask = (targets_values >= topk_values) & (
        targets_values.isnan() | topk_values.isnan()
    ).logical_not()
    return mask.any(axis=-1)


def qr(x, mode="reduced"):
    raise NotImplementedError(
        "tinygrad backend: `keras.ops.qr` is not implemented yet"
    )


def cdist(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    # numpy-reference dtype rule: result_type(x, y, float).
    dtype = result_type(to_keras_dtype(x.dtype), to_keras_dtype(y.dtype), float)
    x = x.cast(to_tinygrad_dtype(dtype))
    y = y.cast(to_tinygrad_dtype(dtype))
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("`cdist` inputs must have rank >= 2")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Last dimension of inputs to `cdist` must match")
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    return (diff * diff).sum(-1).sqrt()


def _segment_prepare(data, segment_ids, num_segments):
    """Shared segment-op front door: convert, resolve num_segments.

    `num_segments=None` means "read it off the ids" (numpy-reference
    `np.amax(segment_ids) + 1`) — a host read of the *ids*, which are an
    integer index tensor, never a gradient path. The `sorted` flag every
    segment op receives only promises an input ordering; the composition
    below is order-independent, so it is accepted and ignored.
    """
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids, "int32")
    if num_segments is None:
        num_segments = int(segment_ids.max().numpy()) + 1
    return data, segment_ids, int(num_segments)


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data, segment_ids, num_segments = _segment_prepare(
        data, segment_ids, num_segments
    )
    # one_hot maps out-of-range ids (including the -1 "ignore" marker) to an
    # all-zero row, which is exactly the numpy-reference drop semantics.
    one_hot = segment_ids.one_hot(num_segments).cast(data.dtype)
    # [n, segments]^T @ [n, ...] via einsum-free matmul on flattened tails
    flat = data.reshape(data.shape[0], -1)
    out = one_hot.T.matmul(flat)
    return out.reshape(num_segments, *data.shape[1:])


def _segment_reduce_masked(data, segment_ids, num_segments, mode, fill):
    """max/min/prod segment reductions as a masked broadcast-reduce.

    Builds the [num_segments, n] membership mask, broadcasts data to
    [num_segments, n, *tail], fills non-members with the reduction's
    identity, and reduces over the n axis — pure Tensor ops, differentiable
    w.r.t. data. O(num_segments * n * tail) intermediate, the price of a
    scatter-free composition. Ids of -1 (and any out-of-range id) match no
    segment row, so they are dropped like the numpy reference drops them;
    empty segments come out as `fill`, matching the numpy reference's
    identity-initialized output (-inf / +inf / 1).
    """
    n = data.shape[0]
    tail = data.shape[1:]
    seg_range = Tensor.arange(num_segments, dtype=segment_ids.dtype)
    mask = segment_ids.reshape(1, n) == seg_range.reshape(num_segments, 1)
    mask = mask.reshape(num_segments, n, *((1,) * len(tail)))
    filled = mask.where(data.reshape(1, n, *tail), fill)
    return getattr(filled, mode)(axis=1)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    data, segment_ids, num_segments = _segment_prepare(
        data, segment_ids, num_segments
    )
    return _segment_reduce_masked(
        data, segment_ids, num_segments, "max", -_math.inf
    )


def segment_min(data, segment_ids, num_segments=None, sorted=False):
    data, segment_ids, num_segments = _segment_prepare(
        data, segment_ids, num_segments
    )
    return _segment_reduce_masked(
        data, segment_ids, num_segments, "min", _math.inf
    )


def segment_prod(data, segment_ids, num_segments=None, sorted=False):
    data, segment_ids, num_segments = _segment_prepare(
        data, segment_ids, num_segments
    )
    return _segment_reduce_masked(
        data, segment_ids, num_segments, "prod", 1
    )


def logdet(x):
    # numpy-reference semantics: `slogdet(x)[1]`, i.e. log|det(x)| — computed
    # off linalg's det (float64 internally, shape-stable partial pivoting).
    from keras.src.backend.tinygrad.linalg import det

    return _float(det(convert_to_tensor(x))).abs().log()


# ---- FFT family (matmul-DFT) ------------------------------------------------
#
# All transforms are computed as matmuls against precomputed cos/sin constant
# matrices (O(n^2) per transform — fine for the audio-feature sizes Keras
# uses, n_fft <= 2048). The matrices are host-side constant tables (allowed:
# constants are not gradient paths into the inputs); everything applied to
# data is a differentiable tinygrad op.

_DFT_CACHE = {}


def _dft_consts(kind, n, keras_dtype):
    """Constant DFT matrices as tinygrad Tensors, cached per (kind, n, dtype).

    kind "full": (cos, sin), each [n, n] with entry [j, k] = f(2*pi*j*k/n).
    kind "rfft": (cos, -sin) sliced to the n//2+1 one-sided bins, [n, K].
    kind "irfft": (Mr, Mi), [K, n], real/imag synthesis weights including the
        1/n normalization and the doubling of the interior bins. The sin term
        is identically zero at k=0 and (n even) k=n/2, so the imaginary parts
        of the DC/Nyquist bins are ignored exactly like `np.fft.irfft`.
    """
    key = (kind, n, keras_dtype)
    if key not in _DFT_CACHE:
        j = np.arange(n, dtype=np.float64)
        if kind == "full":
            ang = 2.0 * np.pi * np.outer(j, j) / n
            mats = (np.cos(ang), np.sin(ang))
        elif kind == "rfft":
            num_bins = n // 2 + 1
            ang = 2.0 * np.pi * np.outer(j, j[:num_bins]) / n
            mats = (np.cos(ang), -np.sin(ang))
        elif kind == "irfft":
            num_bins = n // 2 + 1
            ang = 2.0 * np.pi * np.outer(j[:num_bins], j) / n
            w = np.full((num_bins, 1), 2.0)
            w[0, 0] = 1.0
            if n % 2 == 0:
                w[-1, 0] = 1.0
            mats = (w * np.cos(ang) / n, -w * np.sin(ang) / n)
        else:
            raise AssertionError(kind)
        _DFT_CACHE[key] = tuple(
            convert_to_tensor(m, dtype=keras_dtype) for m in mats
        )
    return _DFT_CACHE[key]


def _get_real_imag_tuple(x):
    """Mirror of the numpy backend's `_get_complex_tensor_from_tuple`, except
    it keeps (real, imag) as separate tensors — tinygrad has no complex
    dtype, and the DFT-by-matmul never needs one."""
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    real, imag = x
    if tuple(real.shape) != tuple(imag.shape):
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    if "float" not in to_keras_dtype(real.dtype) or "float" not in (
        to_keras_dtype(imag.dtype)
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    return real, imag


def _fft_last_axis(real, imag, inverse=False):
    n = real.shape[-1]
    cos_m, sin_m = _dft_consts("full", n, to_keras_dtype(real.dtype))
    if not inverse:
        # F[k] = sum_j (r + i*m)(cos - i*sin)
        return real @ cos_m + imag @ sin_m, imag @ cos_m - real @ sin_m
    # x[j] = (1/n) sum_k (r + i*m)(cos + i*sin)
    return (
        (real @ cos_m - imag @ sin_m) / n,
        (real @ sin_m + imag @ cos_m) / n,
    )


def fft(x):
    real, imag = _get_real_imag_tuple(x)
    return _fft_last_axis(real, imag)


def fft2(x):
    real, imag = _get_real_imag_tuple(x)
    real, imag = _fft_last_axis(real, imag)
    real, imag = real.transpose(-1, -2), imag.transpose(-1, -2)
    real, imag = _fft_last_axis(real, imag)
    return real.transpose(-1, -2), imag.transpose(-1, -2)


def ifft2(x):
    real, imag = _get_real_imag_tuple(x)
    real, imag = _fft_last_axis(real, imag, inverse=True)
    real, imag = real.transpose(-1, -2), imag.transpose(-1, -2)
    real, imag = _fft_last_axis(real, imag, inverse=True)
    return real.transpose(-1, -2), imag.transpose(-1, -2)


def _fit_last_axis(x, size):
    """Truncate or zero-pad the last axis to exactly `size` (np.fft n=...)."""
    cur = x.shape[-1]
    if cur > size:
        return x[..., :size]
    if cur < size:
        pad = [(0, 0)] * (x.ndim - 1) + [(0, size - cur)]
        return x.pad(tuple(pad))
    return x


def rfft(x, fft_length=None):
    x = _float(convert_to_tensor(x))
    n = int(fft_length) if fft_length is not None else x.shape[-1]
    x = _fit_last_axis(x, n)
    cos_m, neg_sin_m = _dft_consts("rfft", n, to_keras_dtype(x.dtype))
    return x @ cos_m, x @ neg_sin_m


def irfft(x, fft_length=None):
    real, imag = _get_real_imag_tuple(x)
    n = (
        int(fft_length)
        if fft_length is not None
        else 2 * (real.shape[-1] - 1)
    )
    num_bins = n // 2 + 1
    real = _fit_last_axis(real, num_bins)
    imag = _fit_last_axis(imag, num_bins)
    mat_r, mat_i = _dft_consts("irfft", n, to_keras_dtype(real.dtype))
    return real @ mat_r + imag @ mat_i


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)
    return x.unfold(-1, int(sequence_length), int(sequence_stride))


def _stft_window(window, sequence_length, l_pad, r_pad, keras_dtype):
    """Shared stft/istft window construction: validate, pad to fft_length."""
    if window is not None:
        if isinstance(window, str):
            # Host-side constant table (scipy), like the numpy backend.
            win = convert_to_tensor(
                scipy.signal.get_window(window, sequence_length),
                dtype=keras_dtype,
            )
        else:
            win = convert_to_tensor(window, dtype=keras_dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = win.pad(((l_pad, r_pad),))
    else:
        win = Tensor.ones(
            sequence_length + l_pad + r_pad,
            dtype=convert_to_tensor(0.0, dtype=keras_dtype).dtype,
        )
    return win


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    if standardize_dtype(x.dtype) not in {"float32", "float64"}:
        raise TypeError(
            "Invalid input type. Expected `float32` or `float64`. "
            f"Received: input type={x.dtype}"
        )
    if fft_length < sequence_length:
        raise ValueError(
            "`fft_length` must equal or larger than `sequence_length`. "
            f"Received: sequence_length={sequence_length}, "
            f"fft_length={fft_length}"
        )
    if isinstance(window, str):
        if window not in {"hann", "hamming"}:
            raise ValueError(
                "If a string is passed to `window`, it must be one of "
                f'`"hann"`, `"hamming"`. Received: window={window}'
            )
    x = convert_to_tensor(x)

    if center:
        pad = [(0, 0)] * (x.ndim - 1) + [(fft_length // 2, fft_length // 2)]
        x = x.pad(tuple(pad), mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    win = _stft_window(
        window, sequence_length, l_pad, r_pad, to_keras_dtype(x.dtype)
    )

    # The scipy pipeline the numpy backend uses (frame, window, rfft, scale
    # by 1/win.sum(), then un-scale by win.sum()) reduces to a plain
    # windowed-frame rfft; nperseg == nfft == fft_length after the l/r pads.
    frames = x.unfold(-1, fft_length, sequence_stride) * win
    return rfft(frames, fft_length)


def _overlap_sequences(x, sequence_stride):
    """Overlap-add along the last two axes — port of the librosa/TF-style
    `_overlap_sequences` (pure reshape/pad/permute/sum tinygrad ops)."""
    *batch_shape, num_sequences, sequence_length = x.shape
    flat_batchsize = _math.prod(batch_shape)
    x = x.reshape(flat_batchsize, num_sequences, sequence_length)
    output_size = sequence_stride * (num_sequences - 1) + sequence_length
    nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride
    padded_segment_len = nstep_per_segment * sequence_stride
    x = x.pad(((0, 0), (0, 0), (0, padded_segment_len - sequence_length)))
    x = x.reshape(
        flat_batchsize, num_sequences, nstep_per_segment, sequence_stride
    )
    x = x.permute(0, 2, 1, 3)
    x = x.pad(((0, 0), (0, 0), (0, num_sequences), (0, 0)))
    shrinked = x.shape[2] - 1
    x = x.reshape(flat_batchsize, -1)
    x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
    x = x.reshape(
        flat_batchsize, nstep_per_segment, shrinked * sequence_stride
    )
    x = x.sum(1)[:, :output_size]
    return x.reshape(*batch_shape, -1)


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    real, imag = _get_real_imag_tuple(x)
    keras_dtype = to_keras_dtype(real.dtype)

    # Per-frame inverse rfft: (..., num_sequences, fft_length)
    x = irfft((real, imag), fft_length)

    expected_output_len = fft_length + sequence_stride * (x.shape[-2] - 1)
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        win = _stft_window(
            window, sequence_length, l_pad, r_pad, keras_dtype
        )
        # Inverse-STFT window normalization (librosa/TF semantics): divide
        # the synthesis window by the sum of its own squared overlapped
        # copies, which is periodic with period `sequence_stride`.
        _sequence_length = sequence_length + l_pad + r_pad
        overlaps = -(-_sequence_length // sequence_stride)
        denom = win.square()
        denom = denom.pad(
            ((0, overlaps * sequence_stride - _sequence_length),)
        )
        denom = denom.reshape(overlaps, sequence_stride).sum(0, keepdim=True)
        denom = denom.expand(overlaps, sequence_stride).reshape(
            overlaps * sequence_stride
        )
        win = win / denom[:_sequence_length]
        x = x * win

    x = _overlap_sequences(x, sequence_stride)

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = expected_output_len - (fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise NotImplementedError(
        f"tinygrad backend: `keras.ops.{name}` is not implemented yet"
    )
