"""tinygrad implementations of the `keras.ops.nn` surface.

Semantics are copied from the numpy backend (the reference for exact
epsilon-clipping and normalization behavior); execution is tinygrad, so
everything here is differentiable. Missing ops raise via module
`__getattr__` — no silent fallbacks.
"""

from tinygrad import Tensor
from tinygrad import dtypes

from keras.src import backend
from keras.src.backend.common.backend_utils import (
    compute_adaptive_pooling_window_sizes,
)
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_output_crops_for_torch,
)
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.backend.tinygrad.core import to_keras_dtype
from keras.src.backend.tinygrad.core import to_tinygrad_dtype


def _f(x):
    x = convert_to_tensor(x)
    if "float" not in to_keras_dtype(x.dtype):
        x = x.cast(to_tinygrad_dtype(backend.floatx()))
    return x


def relu(x):
    return _f(x).relu()


def relu6(x):
    return _f(x).relu6()


def sigmoid(x):
    return _f(x).sigmoid()


def tanh(x):
    return _f(x).tanh()


def softplus(x):
    return _f(x).softplus()


def softsign(x):
    x = _f(x)
    return x / (1 + x.abs())


def silu(x):
    return _f(x).silu()


def log_sigmoid(x):
    x = _f(x)
    return -(-x).softplus()


def leaky_relu(x, negative_slope=0.2):
    return _f(x).leaky_relu(neg_slope=negative_slope)


def hard_sigmoid(x):
    x = _f(x)
    return (x / 6.0 + 0.5).clip(0.0, 1.0)


def hard_silu(x):
    x = _f(x)
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    return _f(x).elu(alpha=alpha)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = _f(x)
    return scale * elu(x, alpha)


def gelu(x, approximate=True):
    x = _f(x)
    if approximate:
        return x.gelu()
    # Exact erf-based gelu via tanh-free formulation is unavailable in
    # tinygrad; erf exists though:
    return 0.5 * x * (1 + (x / 1.4142135623730951).erf())


def celu(x, alpha=1.0):
    x = _f(x)
    return x.maximum(0.0) + alpha * ((x.minimum(0.0) / alpha).exp() - 1)


def glu(x, axis=-1):
    x = _f(x)
    if x.shape[axis] % 2 != 0:
        raise ValueError(
            "axis size must be divisible by 2. "
            f"Received: x.shape={x.shape} with axis={axis}"
        )
    a, b = x.chunk(2, dim=axis)
    return a * b.sigmoid()


def hard_tanh(x):
    return _f(x).clip(-1.0, 1.0)


def hard_shrink(x, threshold=0.5):
    x = _f(x)
    return (x.abs() > threshold).where(x, 0.0)


def soft_shrink(x, threshold=0.5):
    x = _f(x)
    return (x > threshold).where(
        x - threshold, (x < -threshold).where(x + threshold, 0.0)
    )


def tanh_shrink(x):
    x = _f(x)
    return x - x.tanh()


def squareplus(x, b=4):
    x = _f(x)
    return (x + (x * x + b).sqrt()) / 2


def sparse_plus(x):
    x = _f(x)
    return (x <= -1).where(0.0, (x < 1).where(0.25 * (x + 1) * (x + 1), x))


def sparse_sigmoid(x):
    x = _f(x)
    return (x <= -1).where(0.0, (x >= 1).where(1.0, 0.5 * (x + 1)))


def threshold(x, threshold, default_value):
    x = _f(x)
    return (x > threshold).where(x, default_value)


def sparsemax(x, axis=-1):
    # Mirrors the numpy backend: sort descending, find the support set,
    # derive the threshold tau, project onto the simplex.
    logits = _f(x)
    axis = axis % logits.ndim
    logits_sorted = logits.sort(dim=axis, descending=True)[0]
    logits_cumsum = logits_sorted.cumsum(axis=axis)
    r_shape = [1] * logits.ndim
    r_shape[axis] = -1
    r = Tensor.arange(1, logits.shape[axis] + 1, dtype=logits.dtype).reshape(
        r_shape
    )
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    k = support.sum(axis=axis, keepdim=True).cast(logits.dtype)
    logits_cumsum_safe = support.where(logits_cumsum, 0.0)
    tau = (logits_cumsum_safe.sum(axis=axis, keepdim=True) - 1) / k
    return (logits - tau).maximum(0.0)


def softmax(x, axis=-1):
    return _f(x).softmax(axis=axis)


def log_softmax(x, axis=-1):
    return _f(x).log_softmax(axis=axis)


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError(
            "Unsupported value `sparse=True` with tinygrad backend"
        )
    x = convert_to_tensor(x, "int32")
    out = x.one_hot(num_classes)
    dtype = dtype or backend.floatx()
    out = out.cast(to_tinygrad_dtype(backend.standardize_dtype(dtype)))
    if axis != -1 and axis != out.ndim - 1:
        order = list(range(out.ndim - 1))
        order.insert(axis % out.ndim, out.ndim - 1)
        out = out.permute(order)
    return out


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError(
            "Unsupported value `sparse=True` with tinygrad backend"
        )
    x = convert_to_tensor(x, "int32")
    reduction_axis = 1 if x.ndim > 1 else 0
    outputs = one_hot(x, num_classes, axis=-1, dtype=dtype).max(
        axis=reduction_axis
    )
    if axis != -1:
        order = list(range(outputs.ndim - 1))
        order.insert(axis % outputs.ndim, outputs.ndim - 1)
        outputs = outputs.permute(order)
    return outputs


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = _f(target)
    output = _f(output)

    if tuple(target.shape) != tuple(output.shape):
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = output.log_softmax(axis=axis)
    else:
        output = output / output.sum(axis=axis, keepdim=True)
        output = output.clip(
            backend.epsilon(), 1.0 - backend.epsilon()
        )
        log_prob = output.log()
    return -(target * log_prob).sum(axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, "int32")
    output = _f(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = target.reshape(target.shape[:-1])

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if tuple(target.shape) != tuple(output.shape[:-1]):
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if from_logits:
        log_prob = output.log_softmax(axis=axis)
    else:
        output = output / output.sum(axis=axis, keepdim=True)
        output = output.clip(
            backend.epsilon(), 1.0 - backend.epsilon()
        )
        log_prob = output.log()
    target_oh = one_hot(target, output.shape[axis], axis=axis)
    return -(target_oh * log_prob).sum(axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = _f(target)
    output = _f(output)

    if tuple(target.shape) != tuple(output.shape):
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = output.sigmoid()

    output = output.clip(backend.epsilon(), 1.0 - backend.epsilon())
    bce = target * output.log()
    bce = bce + (1.0 - target) * (1.0 - output).log()
    return -bce


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with tinygrad."
        )
    x = convert_to_tensor(x)
    axes = tuple(axes) if isinstance(axes, list) else axes

    need_cast = False
    ori_dtype = backend.standardize_dtype(to_keras_dtype(x.dtype))
    if ori_dtype == "float16":
        need_cast = True
        x = x.cast(to_tinygrad_dtype("float32"))
    else:
        x = _f(x)

    mean = x.mean(axis=axes, keepdim=True)
    # Var = E[x^2] - E[x]^2 (matches the numpy backend's formulation).
    variance = (x * x).mean(axis=axes, keepdim=True) - mean * mean

    if not keepdims:
        shape = [
            d
            for i, d in enumerate(x.shape)
            if i not in tuple(a % x.ndim for a in axes)
        ]
        mean = mean.reshape(shape)
        variance = variance.reshape(shape)
    if need_cast:
        mean = mean.clip(-65504.0, 65504.0).cast(to_tinygrad_dtype(ori_dtype))
        variance = variance.clip(-65504.0, 65504.0).cast(
            to_tinygrad_dtype(ori_dtype)
        )
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = _f(x)
    mean = _f(mean)
    variance = _f(variance)
    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = mean.reshape(shape)
    variance = variance.reshape(shape)

    inv = (variance + epsilon).rsqrt()
    if scale is not None:
        scale = _f(scale).reshape(shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = _f(offset).reshape(shape)
        res = res + offset

    return x * inv + res


def _nchw(x, data_format):
    # Keras is channels_last by default; tinygrad convs/pools are NCHW.
    if backend.standardize_data_format(data_format) == "channels_last":
        order = [0, x.ndim - 1] + list(range(1, x.ndim - 1))
        return x.permute(order), True
    return x, False


def _from_nchw(x, was_channels_last):
    if was_channels_last:
        order = [0] + list(range(2, x.ndim)) + [1]
        return x.permute(order)
    return x


def _spatial_tuple(value, spatial):
    return (value,) * spatial if isinstance(value, int) else tuple(value)


def _same_pads(x, sizes, strides, dilations=None):
    # TF-style "same": out = ceil(in / stride); pad_before = total // 2
    # (the extra cell, if any, goes at the end). x is NCHW here.
    pads = []
    for i, size in enumerate(sizes):
        k = (size - 1) * (dilations[i] if dilations else 1) + 1
        in_size = x.shape[2 + i]
        out_size = -(-in_size // strides[i])
        total = max(0, (out_size - 1) * strides[i] + k - in_size)
        pads.append((total // 2, total - total // 2))
    return pads


def _flat_pads(pads):
    # tinygrad conv/pool `padding` sequences are torch-flat: LAST spatial
    # dim first, (before, after) within each pair.
    flat = []
    for before, after in reversed(pads):
        flat.extend((before, after))
    return flat


def max_pool(inputs, pool_size, strides=None, padding="valid", data_format=None):
    x = _f(inputs)
    x, restore = _nchw(x, data_format)
    spatial = x.ndim - 2
    pool_size = _spatial_tuple(pool_size, spatial)
    strides = _spatial_tuple(strides, spatial) if strides else pool_size
    pads = (
        _flat_pads(_same_pads(x, pool_size, strides))
        if padding == "same"
        else 0
    )
    # tinygrad pads with dtype.min (= -inf for floats): "same" never
    # selects a padded cell, matching the reference semantics.
    out = x.max_pool2d(kernel_size=pool_size, stride=strides, padding=pads)
    # tinygrad's max_pool2d promotes half to float; the op is
    # dtype-preserving, so cast back.
    return _from_nchw(out.cast(x.dtype), restore)


def average_pool(
    inputs, pool_size, strides=None, padding="valid", data_format=None
):
    x = _f(inputs)
    x, restore = _nchw(x, data_format)
    spatial = x.ndim - 2
    pool_size = _spatial_tuple(pool_size, spatial)
    strides = _spatial_tuple(strides, spatial) if strides else pool_size
    pads = (
        _flat_pads(_same_pads(x, pool_size, strides))
        if padding == "same"
        else 0
    )
    # count_include_pad=False: padded cells are excluded from the divisor,
    # matching the numpy backend's window_counts division.
    out = x.avg_pool2d(
        kernel_size=pool_size,
        stride=strides,
        padding=pads,
        count_include_pad=False,
    )
    return _from_nchw(out.cast(x.dtype), restore)


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    x = _f(inputs)
    kernel = _f(kernel)
    x, restore = _nchw(x, data_format)
    spatial = x.ndim - 2
    # Keras kernel layout: spatial... , in_channels, out_channels
    # tinygrad wants: out_channels, in_channels, spatial...
    korder = [kernel.ndim - 1, kernel.ndim - 2] + list(range(spatial))
    w = kernel.permute(korder)
    strides = _spatial_tuple(strides, spatial)
    dilation_rate = _spatial_tuple(dilation_rate, spatial)
    channels = x.shape[1]
    kernel_in_channels = w.shape[1]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    pads = (
        _flat_pads(_same_pads(x, w.shape[2:], strides, dilation_rate))
        if padding == "same"
        else 0
    )
    out = x.conv2d(
        w,
        stride=strides,
        dilation=dilation_rate,
        padding=pads,
        groups=channels // kernel_in_channels,
    )
    return _from_nchw(out, restore)


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    x = _f(inputs)
    kernel = _f(kernel)
    x, restore = _nchw(x, data_format)
    spatial = x.ndim - 2
    in_channels = x.shape[1]
    # Keras depthwise kernel: spatial..., in_channels, channel_multiplier
    multiplier = kernel.shape[-1]
    korder = [kernel.ndim - 2, kernel.ndim - 1] + list(range(spatial))
    w = kernel.permute(korder)  # in, mult, spatial...
    w = w.reshape(in_channels * multiplier, 1, *w.shape[2:])
    strides = _spatial_tuple(strides, spatial)
    dilation_rate = _spatial_tuple(dilation_rate, spatial)
    pads = (
        _flat_pads(_same_pads(x, w.shape[2:], strides, dilation_rate))
        if padding == "same"
        else 0
    )
    out = x.conv2d(
        w,
        stride=strides,
        dilation=dilation_rate,
        padding=pads,
        groups=in_channels,
    )
    return _from_nchw(out, restore)


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    # Mirrors the numpy backend: depthwise then 1x1 pointwise.
    depthwise_output = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )
    return conv(
        depthwise_output,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=dilation_rate,
    )


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    x = _f(inputs)
    kernel = _f(kernel)
    x, restore = _nchw(x, data_format)
    spatial = x.ndim - 2
    strides = _spatial_tuple(strides, spatial)
    dilation_rate = _spatial_tuple(dilation_rate, spatial)
    # Same trick as the torch backend: run the transpose conv with zero
    # padding (largest "natural" output), then asymmetrically crop — or
    # zero-pad where the crop is negative — to the window Keras expects.
    # The helper only reads ndim from input_shape and the KERAS-layout
    # kernel spatial dims, so it is layout-safe here.
    crops = compute_conv_transpose_output_crops_for_torch(
        input_shape=x.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )
    # Keras conv_transpose kernel layout: spatial..., out_channels,
    # in_channels; tinygrad (torch-style) wants: in, out, spatial...
    korder = [kernel.ndim - 1, kernel.ndim - 2] + list(range(spatial))
    w = kernel.permute(korder)
    out = x.conv_transpose2d(
        w, stride=strides, dilation=dilation_rate, padding=0, output_padding=0
    )
    slices = [slice(None), slice(None)]
    for crop_left, crop_right in crops:
        start = max(0, crop_left)
        end = -crop_right if crop_right > 0 else None
        slices.append(slice(start, end))
    out = out[tuple(slices)]
    pad_spec = [
        (max(0, -crop_left), max(0, -crop_right))
        for crop_left, crop_right in crops
    ]
    if any(before or after for before, after in pad_spec):
        out = out.pad([(0, 0), (0, 0)] + pad_spec)
    return _from_nchw(out, restore)


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def psnr(x1, x2, max_val):
    x1 = _f(x1)
    x2 = _f(x2)
    if tuple(x1.shape) != tuple(x2.shape):
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )
    max_val = convert_to_tensor(max_val).cast(x2.dtype)
    mse = ((x1 - x2) * (x1 - x2)).mean()
    ln10 = 2.302585092994046
    return 20.0 * max_val.log() / ln10 - 10.0 * mse.log() / ln10


def _get_large_negative(keras_dtype):
    # Mirrors the numpy backend: a large-but-finite negative so softmax
    # zeroes masked positions without producing nan.
    val = 65500.0 if keras_dtype == "float16" else 3.38953e38
    return val * -0.7


def _apply_masks(logits, mask, is_causal, logits_keras_dtype):
    if mask is None and not is_causal:
        return logits
    bool_dt = to_tinygrad_dtype("bool")
    combined = None
    if mask is not None:
        combined = convert_to_tensor(mask).cast(bool_dt)
    if is_causal:
        T, S = logits.shape[2], logits.shape[3]
        causal = Tensor.ones(T, S).tril().cast(bool_dt).reshape(1, 1, T, S)
        combined = causal if combined is None else combined & causal
    return combined.where(logits, _get_large_negative(logits_keras_dtype))


def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None,
):
    # `flash_attention` is a performance hint; the standard path computes
    # the same values, so it is accepted and served by the XLA-style path
    # below (mirroring the numpy backend's math exactly).
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    compute_dtype = result_type(
        to_keras_dtype(query.dtype),
        to_keras_dtype(key.dtype),
        to_keras_dtype(value.dtype),
    )
    compute_tg = to_tinygrad_dtype(compute_dtype)
    query = query.cast(compute_tg)
    key = key.cast(compute_tg)
    value = value.cast(compute_tg)

    _, _, _, H = key.shape
    scale = (1.0 / (float(H) ** 0.5)) if scale is None else scale

    # Softmax (and bfloat16 einsum) run in at-least-float32, matching the
    # numpy backend.
    logits_dtype = result_type(compute_dtype, "float32")
    logits_tg = to_tinygrad_dtype(logits_dtype)
    # np.einsum accumulates float16 in float32 internally; tinygrad would
    # accumulate in float16 and drift a ulp. Match the reference by doing
    # the matmuls in the (>= float32) logits dtype for reduced-precision
    # compute dtypes, casting the results back.
    reduced = compute_dtype in ("float16", "bfloat16")
    if reduced:
        q_mm, k_mm = query.cast(logits_tg), key.cast(logits_tg)
    else:
        q_mm, k_mm = query, key
    logits = Tensor.einsum("BTNH,BSNH->BNTS", q_mm, k_mm).cast(logits_tg)
    logits = logits * scale

    if bias is not None:
        bias = convert_to_tensor(bias).cast(logits_tg)
        logits = logits + bias

    padded_logits = _apply_masks(logits, mask, is_causal, logits_dtype)

    # Softmax is always carried out in fp32.
    probs = (
        padded_logits.cast(to_tinygrad_dtype("float32"))
        .softmax(axis=-1)
        .cast(compute_tg)
    )
    if reduced:
        p_mm, v_mm = probs.cast(logits_tg), value.cast(logits_tg)
    else:
        p_mm, v_mm = probs, value
    encoded = Tensor.einsum("BNTS,BSNH->BTNH", p_mm, v_mm)
    return encoded.cast(compute_tg)


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    # Extract sliding local blocks from an NCHW tensor -> (N, C*kH*kW, L).
    # Pure tensor ops: one strided slice per kernel offset, stacked.
    x = convert_to_tensor(input)
    k = _pair(kernel_size)
    d = _pair(dilation)
    p = _pair(padding)
    s = _pair(stride)
    N, C, _, _ = x.shape
    if p[0] > 0 or p[1] > 0:
        x = x.pad(((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])))
    oH = (x.shape[2] - (k[0] - 1) * d[0] - 1) // s[0] + 1
    oW = (x.shape[3] - (k[1] - 1) * d[1] - 1) // s[1] + 1
    patches = []
    for i in range(k[0]):
        for j in range(k[1]):
            h0, w0 = i * d[0], j * d[1]
            patches.append(
                x[
                    :,
                    :,
                    h0 : h0 + (oH - 1) * s[0] + 1 : s[0],
                    w0 : w0 + (oW - 1) * s[1] + 1 : s[1],
                ]
            )
    out = Tensor.stack(*patches, dim=2)  # (N, C, kH*kW, oH, oW)
    return out.reshape(N, C * k[0] * k[1], oH * oW)


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    # col2im: (N, C*kH*kW, L) -> (N, C, oH, oW). Implemented as a grouped
    # transposed conv with a constant one-hot (identity) kernel, so the
    # overlap-add stays a differentiable tensor op.
    x = convert_to_tensor(x)
    oH, oW = _pair(output_size)
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
    pH, pW = _pair(padding)
    sH, sW = _pair(stride)
    N, CKK, L = x.shape
    C = CKK // (kH * kW)
    nH = (oH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    nW = (oW + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    cols = x.reshape(N, CKK, nH, nW)
    # Weight (torch layout: in, out//groups, kH, kW): for group c, input
    # channel m = i*kW + j scatters onto kernel position (i, j).
    w = (
        Tensor.eye(kH * kW, dtype=x.dtype)
        .repeat((C, 1))
        .reshape(C * kH * kW, 1, kH, kW)
    )
    out = cols.conv_transpose2d(
        w, groups=C, stride=(sH, sW), dilation=(dH, dW)
    )
    oH_pad, oW_pad = oH + 2 * pH, oW + 2 * pW
    natH = (nH - 1) * sH + dH * (kH - 1) + 1
    natW = (nW - 1) * sW + dW * (kW - 1) + 1
    if natH < oH_pad or natW < oW_pad:
        out = out.pad(
            ((0, 0), (0, 0), (0, oH_pad - natH), (0, oW_pad - natW))
        )
    if pH > 0 or pW > 0:
        out = out[:, :, pH : oH_pad - pH, pW : oW_pad - pW]
    return out


def depth_to_space(x, block_size, data_format="channels_last"):
    x = convert_to_tensor(x)
    if data_format == "channels_last":
        n, h, w, c = x.shape
        new_c = c // (block_size**2)
        x = x.reshape(n, h, w, block_size, block_size, new_c)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(n, h * block_size, w * block_size, new_c)
    else:
        n, c, h, w = x.shape
        new_c = c // (block_size**2)
        x = x.reshape(n, new_c, block_size, block_size, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(n, new_c, h * block_size, w * block_size)
    return x


def space_to_depth(x, block_size, data_format="channels_last"):
    x = convert_to_tensor(x)
    if data_format == "channels_last":
        n, h, w, c = x.shape
        new_h = h // block_size
        new_w = w // block_size
        x = x.reshape(n, new_h, block_size, new_w, block_size, c)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(n, new_h, new_w, c * block_size**2)
    else:
        n, c, h, w = x.shape
        new_h = h // block_size
        new_w = w // block_size
        x = x.reshape(n, c, new_h, block_size, new_w, block_size)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(n, c * block_size**2, new_h, new_w)
    return x


def _adaptive_gather_indices(input_dim, output_size, big_window):
    # Structural python-int math (no gradient path): which pooled window
    # (small- or big-sized) each output cell reads from. Mirrors the numpy
    # backend's _compute_adaptive_pooling_gather_indices.
    small_window = big_window - 1
    small_pool_len = input_dim - small_window + 1
    gather = []
    for o in range(output_size):
        start = (o * input_dim) // output_size
        end = -((-(o + 1) * input_dim) // output_size)  # ceil
        gather.append(
            start + small_pool_len if end - start == big_window else start
        )
    return gather


def _adaptive_pool_axis1(x, output_size, mode):
    # x: (m, l, c) -> (m, output_size, c), pooling along axis 1.
    _, l, _ = x.shape
    small, big = compute_adaptive_pooling_window_sizes(l, output_size)
    gather = _adaptive_gather_indices(l, output_size, big)

    def pool(window):
        count = l - window + 1
        if count <= 0:
            return None
        sv = Tensor.stack(
            *[x[:, off : off + count, :] for off in range(window)], dim=2
        )  # (m, count, window, c)
        return sv.mean(axis=2) if mode == "average" else sv.max(axis=2)

    small_pool = pool(small)
    big_pool = pool(big)
    combined = (
        small_pool if big_pool is None else small_pool.cat(big_pool, dim=1)
    )
    idx = Tensor(gather, dtype=to_tinygrad_dtype("int32"))
    return combined[:, idx, :]


def _adaptive_pool(inputs, output_size, mode, data_format):
    x = _f(inputs)
    dims = x.ndim - 2
    if dims not in (1, 2, 3):
        raise ValueError(
            f"adaptive_{mode}_pool supports only 1D/2D/3D inputs. "
            f"Received: inputs.shape={x.shape}"
        )
    output_size = (
        (output_size,) * dims
        if isinstance(output_size, int)
        else tuple(output_size)
    )
    if data_format == "channels_first":
        x = x.permute([0] + list(range(2, x.ndim)) + [1])
    # x is now channels_last: (n, *spatial, c). Pool one spatial axis at a
    # time by rotating it into position 1 of an (m, l, c) view — the same
    # choreography as the numpy backend's per-axis strided views.
    n = x.shape[0]
    c = x.shape[-1]
    for axis in range(dims):
        spatial = list(x.shape[1:-1])
        # Move the target axis in front, flatten the rest into the batch.
        order = (
            [0]
            + [i + 1 for i in range(dims) if i != axis]
            + [axis + 1, x.ndim - 1]
        )
        others = [spatial[i] for i in range(dims) if i != axis]
        flat = 1
        for d in others:
            flat *= d
        v = x.permute(order).reshape(n * flat, spatial[axis], c)
        v = _adaptive_pool_axis1(v, output_size[axis], mode)
        v = v.reshape([n] + others + [output_size[axis], c])
        # Move the pooled axis back to its original position.
        inv = list(range(v.ndim))
        inv.remove(dims)  # index of the pooled axis in v
        inv.insert(axis + 1, dims)
        x = v.permute(inv)
    if data_format == "channels_first":
        x = x.permute([0, x.ndim - 1] + list(range(1, x.ndim - 1)))
    return x


def adaptive_average_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    return _adaptive_pool(inputs, output_size, "average", data_format)


def adaptive_max_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    return _adaptive_pool(inputs, output_size, "max", data_format)


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    """CTC loss via the log-space forward algorithm.

    Port of the numpy backend's implementation (itself lifted from
    `optax.ctc_loss_with_forward_probs`). The time dimension is a python
    loop, but every step stays in tinygrad Tensors so gradients flow
    through `output`.
    """
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length, dtype="int32")
    output_length = convert_to_tensor(output_length, dtype="int32")
    batch_size, max_input_length, num_classes = output.shape
    _, max_label_length = target.shape
    log_epsilon = -1e5

    # Ensure that the dtype promotion behavior matches that of `tf.nn.ctc_loss`
    dtype = result_type(to_keras_dtype(output.dtype), "float32")
    tg_dtype = to_tinygrad_dtype(dtype)
    output = output.cast(tg_dtype)

    def _lengths_to_paddings(lengths, max_length):
        # padding[b, t] == 1.0 where t >= lengths[b]
        indices = Tensor.arange(max_length).reshape(1, max_length)
        return (indices >= lengths.reshape(-1, 1)).cast(tg_dtype)

    target_paddings = _lengths_to_paddings(target_length, max_label_length)
    output_paddings = _lengths_to_paddings(output_length, max_input_length)

    logprobs = output.log_softmax(-1)
    label_lengths = (
        max_label_length - target_paddings.sum(axis=1).cast(dtypes.int32)
    )

    # repeat[b, n] == 1.0 when target[b, n] == target[b, n + 1]
    repeat = (target[:, :-1] == target[:, 1:]).cast(tg_dtype)
    repeat = repeat.pad(((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = logprobs_phi.permute(1, 0, 2)  # [T, B, 1]

    # One-hot is a constant gather matrix: gradients flow via `logprobs`.
    _one_hot_target = one_hot(target, num_classes=num_classes).cast(tg_dtype)
    # einsum("btk,bnk->btn"): [B, T, K] @ [B, K, N] -> [B, T, N]
    logprobs_emit = logprobs.matmul(_one_hot_target.permute(0, 2, 1))
    logprobs_emit = logprobs_emit.permute(1, 0, 2)  # [T, B, N]

    # logalpha_phi_init: 0.0 in column 0, log_epsilon elsewhere. [B, N + 1]
    logalpha_phi = Tensor.zeros(batch_size, 1, dtype=tg_dtype).cat(
        Tensor.full((batch_size, max_label_length), log_epsilon, dtype=tg_dtype),
        dim=1,
    )
    logalpha_emit = Tensor.full(
        (batch_size, max_label_length), log_epsilon, dtype=tg_dtype
    )

    def _update_phi_score(phi, added_score):
        # Update `phi[:, 1:]` with adding `added_score` in log space.
        return phi[:, :1].cat(phi[:, 1:].logaddexp(added_score), dim=1)

    for t in range(max_input_length):
        logprob_emit = logprobs_emit[t]  # [B, N]
        logprob_phi = logprobs_phi[t]  # [B, 1]
        pad_t = output_paddings[:, t].reshape(batch_size, 1)  # [B, 1]

        prev_phi_orig = logalpha_phi
        prev_emit = logalpha_emit
        # emit-to-phi epsilon transition, except if the next label repeats
        prev_phi = _update_phi_score(
            prev_phi_orig, prev_emit + log_epsilon * repeat
        )
        # phi-to-emit transition
        next_emit = (prev_phi[:, :-1] + logprob_emit).logaddexp(
            prev_emit + logprob_emit
        )
        # self-loop transition
        next_phi = prev_phi + logprob_phi
        # emit-to-phi blank transition only when the next label repeats
        next_phi = _update_phi_score(
            next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat)
        )
        logalpha_emit = pad_t * prev_emit + (1.0 - pad_t) * next_emit
        logalpha_phi = pad_t * prev_phi_orig + (1.0 - pad_t) * next_phi

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = _update_phi_score(logalpha_phi, logalpha_emit)

    # einsum("bn,bn->b") against a constant one-hot of the label lengths
    _one_hot_len = one_hot(
        label_lengths, num_classes=max_label_length + 1
    ).cast(tg_dtype)
    per_seq_loss = -(logalpha_phi_last * _one_hot_len).sum(axis=1)
    return per_seq_loss


def _ctc_greedy_decode(
    inputs,
    sequence_lengths,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    batch_size, max_length, num_classes = inputs.shape

    if mask_index is None:
        mask_index = num_classes - 1

    indices = inputs.argmax(axis=-1).cast(dtypes.int32)
    scores = inputs.max(axis=-1)

    seqlen_mask = Tensor.arange(max_length).reshape(1, max_length)
    seqlen_mask = seqlen_mask >= sequence_lengths.reshape(-1, 1)

    indices = seqlen_mask.where(mask_index, indices).cast(dtypes.int32)
    scores = seqlen_mask.where(0.0, scores).cast(inputs.dtype)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = repeat_mask.pad(((0, 0), (1, 0)))
        indices = repeat_mask.where(mask_index, indices).cast(dtypes.int32)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = invalid_mask.where(-1, indices).cast(dtypes.int32)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = Tensor.arange(max_length).reshape(1, max_length)
    order = order.expand(batch_size, max_length)
    order = invalid_mask.where(max_length, order)
    order = order.argsort(dim=-1)  # tinygrad's sort is stable
    indices = indices.gather(1, order)

    scores = -scores.sum(axis=1).reshape(batch_size, 1)
    indices = indices.unsqueeze(0)
    return indices, scores


# Knuth's multiplicative hash constant, used as the polynomial base for the
# row-hash in `_ctc_unique_padded` (same scheme as the torch backend).
_KNUTH_HASH_CONSTANT = 2654435769

# Large-but-finite stand-in for -inf inside beam search. tinygrad's
# logaddexp/exp produce nan once two -inf values meet, so all "impossible"
# scores use this finite floor instead; `exp(x - max)` underflows to exactly
# 0.0 for it, which is the behavior the log-space merges need. Real log-probs
# are many orders of magnitude above this, so ranking is unaffected.
_CTC_NEG_INF = -1e30


def _ctc_unique_padded(paths, size, pad):
    """Row-dedup of `paths`, padded to a fixed leading `size` with `pad` rows.

    Fixed-shape tinygrad port of the torch backend's `_unique_padded`
    (hash + stable sort + adjacent-equality collision check). Returns
    `(unique, inverse)` where `unique` has the deduped rows first (in
    hash-sorted order) followed by all-`pad` rows, and `inverse` maps each
    input row to its row index in `unique`. Boolean-mask compaction would
    need data-dependent shapes, so first-occurrence rows are scattered to
    their compacted slots and duplicates are parked in a discarded extra row.
    """
    n, t = paths.shape

    # Polynomial hash in int64 with two's-complement wraparound (python-side
    # constants emulate the wrap; the on-device multiply-add wraps natively).
    # Values are shifted by +1 so all-`pad` rows don't collapse to zero.
    # Collisions are detected below, so correctness doesn't rely on the hash.
    powers, v = [], 1
    for _ in range(t):
        powers.append(v - (1 << 64) if v >= (1 << 63) else v)
        v = (v * _KNUTH_HASH_CONSTANT) % (1 << 64)
    powers = Tensor(powers, dtype=dtypes.int64)
    hashes = ((paths.cast(dtypes.int64) + 1) * powers).sum(axis=1)

    order = hashes.argsort()  # stable
    sorted_paths = paths[order]
    sorted_hashes = hashes[order]

    adj_hash_eq = sorted_hashes[1:] == sorted_hashes[:-1]
    adj_row_eq = (sorted_paths[1:] == sorted_paths[:-1]).all(axis=1)
    is_dup = adj_hash_eq & adj_row_eq
    is_first = Tensor.ones(1, dtype=dtypes.bool).cat(~is_dup)

    cum_first = is_first.cast(dtypes.int32).cumsum(0) - 1
    # inverse[order[i]] = cum_first[i]
    inverse = (
        Tensor.zeros(n, dtype=dtypes.int32)
        .contiguous()
        .scatter(0, order, cum_first)
    )

    # Scatter first occurrences to their compacted slots; every duplicate
    # goes to the extra row `size`, which is sliced away.
    dest = is_first.where(cum_first, size)
    unique = (
        Tensor.full((size + 1, t), pad, dtype=paths.dtype)
        .contiguous()
        .scatter(0, dest.reshape(n, 1).expand(n, t), sorted_paths)
    )
    return unique[:size], inverse


def _ctc_merge_scores(inverse, scores, num_buckets):
    """Log-space scatter-add of `scores` into `num_buckets` buckets."""
    scores_max = scores.max()
    scores_exp = (scores - scores_max).exp()
    out = (
        Tensor.zeros(num_buckets, dtype=scores.dtype)
        .contiguous()
        .scatter_reduce(0, inverse, scores_exp, reduce="sum")
    )
    # Empty buckets (sum == 0) get the finite -inf stand-in, not log(0).
    return (out > 0).where(out.log() + scores_max, _CTC_NEG_INF)


def _ctc_beam_extend(paths, scores, masked, x, num_classes, mask_index, _pad):
    """Extend each beam with every possible class for a single timestep."""
    paths = paths.repeat_interleave(num_classes, dim=0)
    scores = scores.repeat_interleave(num_classes, dim=0)
    masked = masked.repeat_interleave(num_classes, dim=0)
    n, max_seq_len = paths.shape

    is_pad = paths == _pad
    path_tail_index = is_pad.cast(dtypes.int32).argmax(axis=1)
    tails_at = (path_tail_index - 1).maximum(0).reshape(n, 1)
    path_tails = paths.gather(1, tails_at).reshape(n)
    path_tails = (path_tail_index == 0).where(_pad, path_tails)

    classes = Tensor.arange(num_classes).cast(dtypes.int32)
    classes = (classes == mask_index).where(_pad, classes)
    classes = classes.reshape(1, num_classes)
    classes = classes.expand(n // num_classes, num_classes).reshape(n)

    prev_masked = masked
    masked = classes == _pad

    masked_repeat = (~prev_masked) & (path_tails == classes)
    classes = masked_repeat.where(_pad, classes).cast(dtypes.int32)

    # paths[i, path_tail_index[i]] = classes[i], as a functional update
    col_mask = Tensor.arange(max_seq_len).reshape(1, max_seq_len)
    col_mask = col_mask == path_tail_index.reshape(n, 1)
    paths = col_mask.where(classes.reshape(n, 1), paths).cast(dtypes.int32)

    x_tiled = x.reshape(1, num_classes)
    x_tiled = x_tiled.expand(n // num_classes, num_classes).reshape(n)
    scores = scores + x_tiled
    return paths, scores, masked


def _ctc_beam_prune(paths, scores, masked, num_classes, beam_width, _pad):
    """Dedup + score-merge + keep top `beam_width` (emit, blank) tracks."""
    size = 2 * num_classes * beam_width
    paths_unique, inverse = _ctc_unique_padded(paths, size=size, pad=_pad)

    emit_scores = masked.where(_CTC_NEG_INF, scores)
    mask_scores = masked.where(scores, _CTC_NEG_INF)

    emit_scores = _ctc_merge_scores(inverse, emit_scores, size)
    mask_scores = _ctc_merge_scores(inverse, mask_scores, size)

    total_scores = emit_scores.logaddexp(mask_scores)
    top_indices = total_scores.argsort()[-beam_width:]  # stable

    paths_top = paths_unique[top_indices]
    emit_scores_top = emit_scores[top_indices]
    mask_scores_top = mask_scores[top_indices]

    paths = paths_top.repeat((2, 1))
    scores = emit_scores_top.cat(mask_scores_top)
    masked_out = Tensor.zeros(beam_width, dtype=dtypes.bool).cat(
        Tensor.ones(beam_width, dtype=dtypes.bool)
    )
    return paths, scores, masked_out


def _ctc_beam_search_decode(
    inputs,
    sequence_lengths,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    """Beam search CTC decoding, ported from the torch backend (itself a
    port of the jax reference). Decoding is a discrete search — there is no
    gradient contract through it in any backend — but everything stays in
    tinygrad ops except the per-batch sequence lengths, which are structural
    python ints controlling the timestep loop (rule-2 style exception, same
    as the torch backend's `.cpu().tolist()`).
    """
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")

    batch_size, max_seq_len, num_classes = inputs.shape
    inputs = inputs.log_softmax(-1)

    if mask_index is None:
        mask_index = num_classes - 1

    # Tie-breaking parity with the reference implementations: flip classes so
    # the desired ordering falls out of the default ascending argsort.
    inputs = inputs.flip(2)
    mask_index = num_classes - mask_index - 1

    _pad = -1
    seqlen_host = sequence_lengths.tolist()  # structural loop bounds
    num_init_paths = min(num_classes, beam_width)

    paths_per_batch = []
    scores_per_batch = []
    for b in range(batch_size):
        x = inputs[b]  # [T, K]
        seq_len_b = int(seqlen_host[b])

        max_classes = x[0].argsort()[-num_init_paths:]
        init_classes = (
            (max_classes == mask_index)
            .where(_pad, max_classes)
            .cast(dtypes.int32)
        )
        # paths: init_classes down column 0 (padded to 2 * beam_width rows),
        # `_pad` everywhere else.
        col0 = init_classes.cat(
            Tensor.full(
                (2 * beam_width - num_init_paths,), _pad, dtype=dtypes.int32
            )
        )
        paths = col0.reshape(2 * beam_width, 1).cat(
            Tensor.full(
                (2 * beam_width, max_seq_len - 1), _pad, dtype=dtypes.int32
            ),
            dim=1,
        )
        scores = x[0].gather(0, max_classes).cat(
            Tensor.full(
                (2 * beam_width - num_init_paths,),
                _CTC_NEG_INF,
                dtype=inputs.dtype,
            )
        )
        masked = paths[:, 0] == _pad

        # Only iterate timesteps within this sequence's length.
        for t in range(1, seq_len_b):
            paths, scores, masked = _ctc_beam_extend(
                paths, scores, masked, x[t], num_classes, mask_index, _pad
            )
            paths, scores, masked = _ctc_beam_prune(
                paths, scores, masked, num_classes, beam_width, _pad
            )

        # Final dedup + top_paths selection.
        size = 2 * num_classes * beam_width
        paths_unique, inverse = _ctc_unique_padded(paths, size=size, pad=_pad)
        scores = _ctc_merge_scores(inverse, scores, size)

        top_indices = scores.argsort()[-top_paths:].flip(0)
        paths_per_batch.append(paths_unique[top_indices])
        scores_per_batch.append(scores[top_indices])

    paths = Tensor.stack(*paths_per_batch, dim=0)
    scores = Tensor.stack(*scores_per_batch, dim=0)

    # Convert classes back from the flipped representation.
    paths = (
        (paths == _pad).where(_pad, num_classes - paths - 1).cast(dtypes.int32)
    )
    paths = paths.permute(1, 0, 2)
    return paths, scores


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    inputs = convert_to_tensor(inputs)
    dtype = result_type(to_keras_dtype(inputs.dtype), "float32")
    inputs = inputs.cast(to_tinygrad_dtype(dtype))

    if strategy == "greedy":
        return _ctc_greedy_decode(
            inputs,
            sequence_lengths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    elif strategy == "beam_search":
        return _ctc_beam_search_decode(
            inputs,
            sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=mask_index,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise NotImplementedError(
        f"tinygrad backend: `keras.ops.{name}` is not implemented yet"
    )
