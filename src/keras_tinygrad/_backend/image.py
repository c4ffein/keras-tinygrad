"""`keras.ops.image` for the tinygrad backend.

Implemented with pure tinygrad Tensor ops so gradients flow w.r.t. the
IMAGE through every sampling path (gather-based indexing, weight-matrix
matmuls). Host-side numpy appears ONLY to build constant lookup tables
(resize weight matrices, nearest-neighbor index tables, coordinate
meshgrids): those are functions of static shapes/scales, not of the image,
so they are not a gradient path into any input. Coordinate-side inputs
(`transform`, `scale`, `translation`) get zero/no gradients, matching the
numpy reference backend.

Anything not implemented below still raises loudly via the module-level
`__getattr__`.
"""

import functools
import itertools
import operator

import ml_dtypes
import numpy as np
from tinygrad import Tensor
from tinygrad import dtypes as tg_dtypes

from keras.src import backend
from keras.src.backend.tinygrad.core import convert_to_numpy
from keras.src.random.seed_generator import draw_seed
from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.backend.tinygrad.core import to_keras_dtype
from keras.src.backend.tinygrad.core import to_tinygrad_dtype

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)
AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
AFFINE_TRANSFORM_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
MAP_COORDINATES_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
SCALE_AND_TRANSLATE_METHODS = {
    "linear",
    "bilinear",
    "trilinear",
    "cubic",
    "bicubic",
    "tricubic",
    "lanczos3",
    "lanczos5",
}


def _keras_dtype(x):
    return to_keras_dtype(x.dtype)


def _is_float_tensor(x):
    return backend.is_float_dtype(_keras_dtype(x))


def _split_channels(images, channels_axis):
    ndim = len(images.shape)
    axis = channels_axis % ndim
    planes = []
    for i in range(3):
        idx = [slice(None)] * ndim
        idx[axis] = i  # integer index squeezes the axis, like numpy
        planes.append(images[tuple(idx)])
    return planes


def rgb_to_grayscale(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if images.shape[channels_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )
    if images.shape[channels_axis] == 1:
        return images
    original_dtype = _keras_dtype(images)
    compute_dtype = backend.result_type(original_dtype, float)
    images = images.cast(to_tinygrad_dtype(compute_dtype))

    # Ref: tf.image.rgb_to_grayscale. Constant weight table.
    rgb_weights = Tensor(
        np.array([0.2989, 0.5870, 0.1140], dtype="float32")
    ).cast(images.dtype)
    ndim = len(images.shape)
    axis = channels_axis % ndim
    bshape = [1] * ndim
    bshape[axis] = 3
    grayscales = (images * rgb_weights.reshape(bshape)).sum(
        axis=axis, keepdim=True
    )
    return grayscales.cast(to_tinygrad_dtype(original_dtype))


def rgb_to_hsv(images, data_format=None):
    # Ref: dm_pix (mirrors the numpy backend)
    images = convert_to_tensor(images)
    dtype = _keras_dtype(images)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    if images.shape[channels_axis] != 3:
        # The numpy backend raises here too (np.split needs an equal split).
        raise ValueError(
            "Invalid channel size: expected 3 (RGB). Received input with "
            f"shape: images.shape={images.shape}"
        )
    eps = float(ml_dtypes.finfo(dtype).eps)
    images = (images.abs() < eps).where(0.0, images)
    red, green, blue = _split_channels(images, channels_axis)

    value = red.maximum(green).maximum(blue)
    minimum = red.minimum(green).minimum(blue)
    range_ = value - minimum

    safe_value = (value > 0).where(value, 1.0)
    safe_range = (range_ > 0).where(range_, 1.0)

    saturation = (value > 0).where(range_ / safe_value, 0.0)
    norm = 1.0 / (6.0 * safe_range)

    hue = (value == green).where(
        norm * (blue - red) + 2.0 / 6.0,
        norm * (red - green) + 4.0 / 6.0,
    )
    hue = (value == red).where(norm * (green - blue), hue)
    hue = (range_ > 0).where(hue, 0.0) + (hue < 0.0).cast(hue.dtype)

    out = Tensor.stack(hue, saturation, value, dim=channels_axis)
    return out.cast(to_tinygrad_dtype(dtype))


def hsv_to_rgb(images, data_format=None):
    # Ref: dm_pix (mirrors the numpy backend)
    images = convert_to_tensor(images)
    dtype = _keras_dtype(images)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    if images.shape[channels_axis] != 3:
        # The numpy backend raises here too (np.split needs an equal split).
        raise ValueError(
            "Invalid channel size: expected 3 (HSV). Received input with "
            f"shape: images.shape={images.shape}"
        )
    hue, saturation, value = _split_channels(images, channels_axis)

    dh = (hue - hue.floor()) * 6.0  # np.mod(hue, 1.0) * 6.0
    dr = ((dh - 3.0).abs() - 1.0).clamp(0.0, 1.0)
    dg = (2.0 - (dh - 2.0).abs()).clamp(0.0, 1.0)
    db = (2.0 - (dh - 4.0).abs()).clamp(0.0, 1.0)
    one_minus_s = 1.0 - saturation

    red = value * (one_minus_s + saturation * dr)
    green = value * (one_minus_s + saturation * dg)
    blue = value * (one_minus_s + saturation * db)

    out = Tensor.stack(red, green, blue, dim=channels_axis)
    return out.cast(to_tinygrad_dtype(dtype))


def resize(
    images,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
        )
    if fill_mode != "constant":
        raise ValueError(
            "Invalid value for argument `fill_mode`. Only `'constant'` "
            f"is supported. Received: fill_mode={fill_mode}"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    images = convert_to_tensor(images)
    size = tuple(size)
    target_height, target_width = size
    if len(images.shape) == 4:
        if data_format == "channels_last":
            size = (images.shape[0],) + size + (images.shape[-1],)
        else:
            size = (images.shape[0], images.shape[1]) + size
    elif len(images.shape) == 3:
        if data_format == "channels_last":
            size = size + (images.shape[-1],)
        else:
            size = (images.shape[0],) + size
    else:
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if crop_to_aspect_ratio:
        shape = images.shape
        if data_format == "channels_last":
            height, width = shape[-3], shape[-2]
        else:
            height, width = shape[-2], shape[-1]
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        if data_format == "channels_last":
            if len(images.shape) == 4:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
            else:
                images = images[
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
        else:
            if len(images.shape) == 4:
                images = images[
                    :,
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
            else:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
    elif pad_to_aspect_ratio:
        shape = images.shape
        batch_size = images.shape[0]
        if data_format == "channels_last":
            height, width, channels = shape[-3], shape[-2], shape[-1]
        else:
            channels, height, width = shape[-3], shape[-2], shape[-1]
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)

        def _fill_block(shape_):
            return Tensor.full(
                tuple(int(s) for s in shape_), float(fill_value)
            ).cast(images.dtype)

        # Mirrors the numpy backend: pad symmetrically on whichever spatial
        # axis needs it (height takes precedence).
        if data_format == "channels_last":
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    block = _fill_block(
                        (batch_size, img_box_hstart, width, channels)
                    )
                    padded_img = Tensor.cat(block, images, block, dim=1)
                else:
                    block = _fill_block((img_box_hstart, width, channels))
                    padded_img = Tensor.cat(block, images, block, dim=0)
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    block = _fill_block(
                        (batch_size, height, img_box_wstart, channels)
                    )
                    padded_img = Tensor.cat(block, images, block, dim=2)
                else:
                    block = _fill_block((height, img_box_wstart, channels))
                    padded_img = Tensor.cat(block, images, block, dim=1)
            else:
                padded_img = images
        else:
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    block = _fill_block(
                        (batch_size, channels, img_box_hstart, width)
                    )
                    padded_img = Tensor.cat(block, images, block, dim=2)
                else:
                    block = _fill_block((channels, img_box_hstart, width))
                    padded_img = Tensor.cat(block, images, block, dim=1)
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    block = _fill_block(
                        (batch_size, channels, height, img_box_wstart)
                    )
                    padded_img = Tensor.cat(block, images, block, dim=3)
                else:
                    block = _fill_block((channels, height, img_box_wstart))
                    padded_img = Tensor.cat(block, images, block, dim=2)
            else:
                padded_img = images
        images = padded_img

    return _resize(images, size, method=interpolation, antialias=antialias)


def _compute_weight_mat(
    input_size, output_size, scale, translation, kernel, antialias
):
    # Host-side constant table: a pure function of static sizes and
    # scale/translation scalars — not a gradient path into the image.
    # Byte-for-byte the numpy backend's `_compute_weight_mat`.
    dtype = np.result_type(scale, translation)
    inv_scale = 1.0 / scale
    kernel_scale = np.maximum(inv_scale, 1.0) if antialias else 1.0

    sample_f = (
        (np.arange(output_size, dtype=dtype) + 0.5) * inv_scale
        - translation * inv_scale
        - 0.5
    )

    x = (
        np.abs(
            sample_f[np.newaxis, :]
            - np.arange(input_size, dtype=dtype)[:, np.newaxis]
        )
        / kernel_scale
    )

    weights = kernel(x)

    total_weight_sum = np.sum(weights, axis=0, keepdims=True)
    weights = np.where(
        np.abs(total_weight_sum) > 1000.0 * np.finfo(np.float32).eps,
        np.divide(
            weights, np.where(total_weight_sum != 0, total_weight_sum, 1)
        ),
        0,
    )

    input_size_minus_0_5 = input_size - 0.5
    return np.where(
        np.logical_and(sample_f >= -0.5, sample_f <= input_size_minus_0_5)[
            np.newaxis, :
        ],
        weights,
        0,
    )


def _resize(image, shape, method, antialias):
    if method == "nearest":
        return _resize_nearest(image, shape)
    else:
        kernel = _kernels.get(method, None)
    if kernel is None:
        raise ValueError("Unknown resize method")

    spatial_dims = tuple(
        i for i in range(len(shape)) if image.shape[i] != shape[i]
    )
    scale = [
        shape[d] / image.shape[d] if image.shape[d] != 0 else 1.0
        for d in spatial_dims
    ]

    return _scale_and_translate(
        image,
        shape,
        spatial_dims,
        scale,
        [0.0] * len(spatial_dims),
        kernel,
        antialias,
    )


def _resize_nearest(x, output_shape):
    input_shape = x.shape
    spatial_dims = tuple(
        i for i in range(len(input_shape)) if input_shape[i] != output_shape[i]
    )

    for d in spatial_dims:
        m, n = input_shape[d], output_shape[d]
        # Constant index table (structural, no gradient path); the gather
        # itself is a tinygrad op so image gradients flow.
        offsets = (np.arange(n, dtype=np.float32) + 0.5) * m / n
        offsets = np.floor(offsets).astype(np.int32)
        indices = [slice(None)] * len(input_shape)
        indices[d] = Tensor(offsets)
        x = x[tuple(indices)]
    return x


def _fill_triangle_kernel(x):
    return np.maximum(0, 1 - np.abs(x))


def _fill_keys_cubic_kernel(x):
    out = ((1.5 * x - 2.5) * x) * x + 1.0
    out = np.where(x >= 1.0, ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0, out)
    return np.where(x >= 2.0, 0.0, out)


def _fill_lanczos_kernel(radius, x):
    y = radius * np.sin(np.pi * x) * np.sin(np.pi * x / radius)
    out = np.where(
        x > 1e-3, np.divide(y, np.where(x != 0, np.pi**2 * x**2, 1)), 1
    )
    return np.where(x > radius, 0.0, out)


_kernels = {
    "linear": _fill_triangle_kernel,
    "bilinear": _fill_triangle_kernel,  # For `resize`.
    "cubic": _fill_keys_cubic_kernel,
    "bicubic": _fill_keys_cubic_kernel,  # For `resize`.
    "lanczos3": lambda x: _fill_lanczos_kernel(3.0, x),
    "lanczos5": lambda x: _fill_lanczos_kernel(5.0, x),
}


def _scale_and_translate(
    x, output_shape, spatial_dims, scale, translation, kernel, antialias
):
    input_shape = x.shape

    if len(spatial_dims) == 0:
        return x

    if not _is_float_tensor(x):
        output = x.cast(tg_dtypes.float32)
        use_rounding = True
    else:
        output = x
        use_rounding = False

    ndim = len(input_shape)
    np_dtype = (
        np.float64 if output.dtype == tg_dtypes.float64 else np.float32
    )
    for i, d in enumerate(spatial_dims):
        d = d % ndim
        m, n = input_shape[d], output_shape[d]

        w_np = _compute_weight_mat(
            m, n, scale[i], translation[i], kernel, antialias
        ).astype(np_dtype)
        w = Tensor(w_np).cast(output.dtype)
        # tensordot(output, w, axes=(d, 0)) then moveaxis(-1, d):
        others = [j for j in range(ndim) if j != d]
        out_p = output.permute(others + [d]) @ w  # (..., n)
        current = others + [d]
        inv = [current.index(j) for j in range(ndim)]
        output = out_p.permute(inv)

    if use_rounding:
        xf = x.cast(tg_dtypes.float32)
        output = output.round().maximum(xf.min()).minimum(xf.max())
        output = output.cast(x.dtype)
    return output


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )

    images = convert_to_tensor(images)
    transform = convert_to_tensor(transform)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if len(transform.shape) not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    input_dtype = _keras_dtype(images)
    compute_dtype = backend.result_type(input_dtype, "float32")
    tg_compute = to_tinygrad_dtype(compute_dtype)
    images = images.cast(tg_compute)
    transform = transform.cast(tg_compute)

    # unbatched case
    need_squeeze = False
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = transform.unsqueeze(0)

    if data_format == "channels_first":
        images = images.permute(0, 2, 3, 1)

    batch_size = images.shape[0]
    h, w, c = images.shape[1:]

    # Constant coordinate grid (host-built, structural: pure function of
    # the static spatial shape).
    mesh = np.stack(
        np.meshgrid(*[np.arange(s) for s in (h, w, c)], indexing="ij"),
        axis=-1,
    ).astype(
        np.float64 if tg_compute == tg_dtypes.float64 else np.float32
    )  # (h, w, c, 3)
    indices = Tensor(mesh).cast(tg_compute)

    # The numpy backend swaps the transform entries in place and pads to a
    # 3x3 matrix with the offsets zeroed out. Build the same matrix and
    # offset directly:
    #   matrix = [[t4, t1, 0], [t3, t0, 0], [t6, t7, 1]]
    #   offset = [t5, t2, 0]
    t = [transform[:, i] for i in range(8)]
    zeros = Tensor.zeros_like(t[0])
    ones = Tensor.ones_like(t[0])
    matrix = Tensor.stack(
        Tensor.stack(t[4], t[1], zeros, dim=-1),
        Tensor.stack(t[3], t[0], zeros, dim=-1),
        Tensor.stack(t[6], t[7], ones, dim=-1),
        dim=1,
    )  # (B, 3, 3)
    offset = Tensor.stack(t[5], t[2], zeros, dim=-1)  # (B, 3)

    # einsum("Bhwij, Bjk -> Bhwik", indices, matrix) as a batched matmul
    coordinates = indices.reshape(1, h * w * c, 3).expand(
        batch_size, h * w * c, 3
    ) @ matrix
    coordinates = coordinates.reshape(batch_size, h, w, c, 3)
    coordinates = coordinates.permute(0, 4, 1, 2, 3)  # (B, 3, h, w, c)
    coordinates = coordinates + offset.reshape(batch_size, 3, 1, 1, 1)

    # apply affine transformation
    order = AFFINE_TRANSFORM_INTERPOLATIONS[interpolation]
    affined = Tensor.stack(
        *[
            map_coordinates(
                images[i],
                coordinates[i],
                order=order,
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
            for i in range(batch_size)
        ],
        dim=0,
    )

    if data_format == "channels_first":
        affined = affined.permute(0, 3, 1, 2)
    if need_squeeze:
        affined = affined.squeeze(0)
    return affined.cast(to_tinygrad_dtype(input_dtype))


def _mirror_index_fixer(index, size):
    s = size - 1  # Half-wavelength of triangular wave
    if s <= 0:
        return index * 0
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return ((index + s) % (2 * s) - s).abs()


def _reflect_index_fixer(index, size):
    return (_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1) // 2


_INDEX_FIXERS = {
    # out-of-bound indices must be fixed before gathering
    "constant": lambda index, size: index.clamp(0, size - 1),
    "nearest": lambda index, size: index.clamp(0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _nearest_indices_and_weights(coordinate):
    if _is_float_tensor(coordinate):
        # scipy's order-0 rounding is floor(x + 0.5) (round half up), which
        # is what the numpy backend (scipy-based) produces.
        index = (coordinate + 0.5).floor().cast(tg_dtypes.int32)
    else:
        index = coordinate.cast(tg_dtypes.int32)
    return [(index, 1)]


def _linear_indices_and_weights(coordinate):
    lower = coordinate.floor()
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.cast(tg_dtypes.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    input_arr = convert_to_tensor(inputs)
    if isinstance(coordinates, Tensor):
        coordinate_arrs = [
            coordinates[i] for i in range(coordinates.shape[0])
        ]
    else:
        coordinate_arrs = [convert_to_tensor(c) for c in coordinates]

    if len(coordinate_arrs) != len(input_arr.shape):
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {input_arr.shape} and coordinate "
            f"leading dim of {len(coordinate_arrs)}"
        )
    if len(coordinate_arrs[0].shape) < 1:
        dim = len(coordinate_arrs)
        shape = (dim,) + tuple(coordinate_arrs[0].shape)
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {shape}"
        )

    index_fixer = _INDEX_FIXERS.get(fill_mode)
    if index_fixer is None:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{set(_INDEX_FIXERS.keys())}. Received: fill_mode={fill_mode}"
        )
    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError(
            "tinygrad backend: map_coordinates currently requires order<=1"
        )

    input_is_int = not _is_float_tensor(input_arr)
    if isinstance(fill_value, (int, float)) and input_is_int:
        fill_value = int(fill_value)

    if fill_mode == "constant":

        def is_valid(index, size):
            return (0 <= index) & (index < size)

    else:

        def is_valid(index, size):
            return True

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        if all(valid is True for valid in validities):
            # fast path
            contribution = input_arr[tuple(indices)]
        else:
            all_valid = functools.reduce(operator.and_, validities)
            contribution = all_valid.where(
                input_arr[tuple(indices)], fill_value
            )
        outputs.append(functools.reduce(operator.mul, weights) * contribution)
    result = functools.reduce(operator.add, outputs)
    if input_is_int and _is_float_tensor(result):
        result = result.round()
    return result.cast(input_arr.dtype)


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    if method not in SCALE_AND_TRANSLATE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{SCALE_AND_TRANSLATE_METHODS}. Received: method={method}"
        )
    if method in ("linear", "bilinear", "trilinear", "triangle"):
        method = "linear"
    elif method in ("cubic", "bicubic", "tricubic"):
        method = "cubic"

    images = convert_to_tensor(images)
    # scale/translation feed the host-built constant weight matrices only;
    # they carry no image gradient (numpy reference behaves identically).
    scale = np.asarray(convert_to_numpy(convert_to_tensor(scale)))
    translation = np.asarray(convert_to_numpy(convert_to_tensor(translation)))
    kernel = _kernels[method]
    dtype = np.result_type(scale.dtype, translation.dtype)
    scale = scale.astype(dtype)
    translation = translation.astype(dtype)
    return _scale_and_translate(
        images,
        output_shape,
        spatial_dims,
        scale,
        translation,
        kernel,
        antialias,
    )


def _host_array(x):
    # Realize a coordinate-side input (scale/sigma/points/...) to a host
    # numpy array. These feed constant tables only and carry no image
    # gradient (numpy-reference behavior).
    if isinstance(x, Tensor):
        return np.asarray(convert_to_numpy(x))
    return np.asarray(x)


def _compute_homography_matrix_np(start_points, end_points):
    # Same row layout as the numpy backend's 8x8 system: for each of the 4
    # correspondences p, two rows
    #   [ex, ey, 1, 0, 0, 0, -sx*ex, -sx*ey] -> sx
    #   [0, 0, 0, ex, ey, 1, -sy*ex, -sy*ey] -> sy
    ones = np.ones_like(end_points[:, 0, 0])
    zeros = np.zeros_like(ones)
    rows, targets = [], []
    for p in range(4):
        sx, sy = start_points[:, p, 0], start_points[:, p, 1]
        ex, ey = end_points[:, p, 0], end_points[:, p, 1]
        rows.append(
            np.stack(
                [ex, ey, ones, zeros, zeros, zeros, -sx * ex, -sx * ey],
                axis=-1,
            )
        )
        rows.append(
            np.stack(
                [zeros, zeros, zeros, ex, ey, ones, -sy * ex, -sy * ey],
                axis=-1,
            )
        )
        targets.extend([sx, sy])
    coefficient_matrix = np.stack(rows, axis=1)  # (N, 8, 8)
    target_vector = np.stack(targets, axis=-1)[..., np.newaxis]  # (N, 8, 1)
    homography_matrix = np.linalg.solve(coefficient_matrix, target_vector)
    return np.reshape(homography_matrix, (-1, 8))


def compute_homography_matrix(start_points, end_points):
    # The homography is a pure function of the (coordinate-side) point sets,
    # which get no gradients (numpy-reference behavior), so the 8x8 solve
    # runs in host numpy on realized values — a creation-time constant
    # table. np.linalg.solve raises loudly on singular systems.
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)
    dtype = backend.result_type(
        _keras_dtype(start_points), _keras_dtype(end_points), float
    )
    # Like the numpy backend: solve in >=float32 so low-precision points do
    # not create an ill-conditioned (or singular) homography.
    compute_dtype = backend.result_type(dtype, "float32")
    np_dtype = np.dtype(compute_dtype)
    start_np = _host_array(start_points).astype(np_dtype)
    end_np = _host_array(end_points).astype(np_dtype)
    homography = _compute_homography_matrix_np(start_np, end_np)
    return convert_to_tensor(homography.astype(np_dtype), compute_dtype)


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)

    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if start_points.ndim not in (2, 3) or start_points.shape[-2:] != (4, 2):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.ndim not in (2, 3) or end_points.shape[-2:] != (4, 2):
        raise ValueError(
            "Invalid end_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {end_points.shape}"
        )
    if start_points.shape != end_points.shape:
        raise ValueError(
            "start_points and end_points must have the same shape."
            f" Received start_points.shape={start_points.shape}, "
            f"end_points.shape={end_points.shape}"
        )

    images = convert_to_tensor(images)
    input_dtype = _keras_dtype(images)
    compute_dtype = backend.result_type(input_dtype, "float32")
    tg_compute = to_tinygrad_dtype(compute_dtype)
    images = images.cast(tg_compute)

    need_squeeze = False
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
        need_squeeze = True
    if len(start_points.shape) == 2:
        start_points = start_points.unsqueeze(0)
    if len(end_points.shape) == 2:
        end_points = end_points.unsqueeze(0)

    if data_format == "channels_first":
        images = images.permute(0, 2, 3, 1)

    batch_size, height, width, channels = images.shape

    transforms = _host_array(
        compute_homography_matrix(start_points, end_points)
    )
    if transforms.ndim == 1:
        transforms = transforms[np.newaxis, :]
    if transforms.shape[0] == 1 and batch_size > 1:
        transforms = np.tile(transforms, (batch_size, 1))

    # Host-built constant sampling grid: homography scalars applied to the
    # static meshgrid. Image gradients flow through the map_coordinates
    # gathers below, not through the coordinates.
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    order = AFFINE_TRANSFORM_INTERPOLATIONS[interpolation]
    input_is_int = not backend.is_float_dtype(input_dtype)

    batch_results = []
    for i in range(batch_size):
        a0, a1, a2, a3, a4, a5, a6, a7 = transforms[i]
        denom = a6 * x + a7 * y + 1.0
        x_in = (a0 * x + a1 * y + a2) / denom
        y_in = (a3 * x + a4 * y + a5) / denom
        coords = [Tensor(y_in.ravel()), Tensor(x_in.ravel())]

        mapped_channels = []
        for channel in range(channels):
            mapped = map_coordinates(
                images[i, :, :, channel],
                coords,
                order=order,
                fill_mode="constant",
                fill_value=fill_value,
            )
            mapped_channels.append(mapped.reshape(height, width))
        batch_results.append(Tensor.stack(*mapped_channels, dim=-1))
    output = Tensor.stack(*batch_results, dim=0)

    if data_format == "channels_first":
        output = output.permute(0, 3, 1, 2)
    if need_squeeze:
        output = output.squeeze(0)
    if input_is_int:
        # The reference samples int images through scipy (which rounds);
        # we compute in float, so round before the terminal int cast.
        output = output.round()
    return output.cast(to_tinygrad_dtype(input_dtype))


def _gaussian_kernel_np(kernel_size, sigma, input_dtype, compute_dtype):
    # Host-side constant table: verbatim port of the numpy backend's kernel
    # builder (1-D gaussians from kernel_size/sigma, outer product), incl.
    # the reference's final quantize-to-input-dtype step.
    np_compute = np.dtype(compute_dtype)

    def _kernel1d(size, s):
        x = np.arange(int(size), dtype=np_compute) - (int(size) - 1) / 2
        kernel1d = np.exp(-0.5 * (x / np_compute.type(float(s))) ** 2)
        return kernel1d / np.sum(kernel1d)

    kernel2d = np.outer(
        _kernel1d(kernel_size[1], sigma[1]),
        _kernel1d(kernel_size[0], sigma[0]),
    )
    return kernel2d.astype(np.dtype(input_dtype)).astype(np_compute)


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    # NOTE: like the numpy reference, data_format is compared raw
    # (None => channels_last), not standardized.
    images = convert_to_tensor(images)
    input_dtype = _keras_dtype(images)
    compute_dtype = backend.result_type(input_dtype, "float32")
    tg_compute = to_tinygrad_dtype(compute_dtype)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    # kernel_size/sigma feed the host-built constant kernel only; they
    # carry no image gradient (numpy reference behaves identically).
    kernel_size_np = _host_array(kernel_size)
    sigma_np = _host_array(sigma)

    images = images.cast(tg_compute)
    need_squeeze = False
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
        need_squeeze = True
    if data_format == "channels_first":
        images = images.permute(0, 2, 3, 1)

    num_channels = images.shape[-1]
    kernel_np = _gaussian_kernel_np(
        kernel_size_np, sigma_np, input_dtype, compute_dtype
    )
    kernel_h, kernel_w = kernel_np.shape
    pad_h = (kernel_h - 1) // 2
    pad_h_after = kernel_h - 1 - pad_h
    pad_w = (kernel_w - 1) // 2
    pad_w_after = kernel_w - 1 - pad_w

    # Depthwise conv (the gaussian kernel is symmetric, so the reference's
    # convolve2d equals this cross-correlation) over a zero-padded input.
    x = images.permute(0, 3, 1, 2)  # NCHW
    x = x.pad(((0, 0), (0, 0), (pad_h, pad_h_after), (pad_w, pad_w_after)))
    weight = (
        Tensor(kernel_np)
        .cast(tg_compute)
        .reshape(1, 1, kernel_h, kernel_w)
        .expand(num_channels, 1, kernel_h, kernel_w)
    )
    blurred = x.conv2d(weight, groups=num_channels)
    blurred = blurred.permute(0, 2, 3, 1)  # back to NHWC

    if data_format == "channels_first":
        blurred = blurred.permute(0, 3, 1, 2)
    if need_squeeze:
        blurred = blurred.squeeze(0)
    return blurred.cast(to_tinygrad_dtype(input_dtype))


def elastic_transform(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    images = convert_to_tensor(images)
    input_dtype = _keras_dtype(images)
    np_input = np.dtype(input_dtype)
    # Displacement-field dtype: the reference works in the input dtype; we
    # quantize to it, then compute in >=float32 (sub-tolerance deviation
    # for low-precision floats, bit-identical for float32).
    np_host = np.dtype(backend.result_type(input_dtype, "float32"))

    # alpha/sigma and the drawn noise are coordinate-side constants (no
    # image gradient; numpy reference behaves identically).
    alpha_np = _host_array(alpha).astype(np_input)
    sigma_np = _host_array(sigma).astype(np_input)
    kernel_size = (
        int(6 * float(sigma_np)) | 1,
        int(6 * float(sigma_np)) | 1,
    )

    need_squeeze = False
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
        need_squeeze = True

    if data_format == "channels_last":
        batch_size, height, width, channels = images.shape
        channel_axis = -1
    else:
        batch_size, channels, height, width = images.shape
        channel_axis = 1

    # numpy-Generator sampling under Keras seeding — identical bits to the
    # reference backend; samples are autograd constants.
    seed = draw_seed(seed)
    if isinstance(seed, Tensor):
        seed = convert_to_numpy(seed)
    rng = np.random.default_rng(seed)
    dx = (
        rng.normal(size=(batch_size, height, width), loc=0.0, scale=1.0)
        .astype(np_input)
        .astype(np_host)
        * sigma_np.astype(np_host)
    )
    dy = (
        rng.normal(size=(batch_size, height, width), loc=0.0, scale=1.0)
        .astype(np_input)
        .astype(np_host)
        * sigma_np.astype(np_host)
    )

    sigma_f = float(sigma_np)
    dx = gaussian_blur(
        np.expand_dims(dx, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma_f, sigma_f),
        data_format=data_format,
    )
    dy = gaussian_blur(
        np.expand_dims(dy, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma_f, sigma_f),
        data_format=data_format,
    )

    # The reference squeezes all unit axes and re-broadcasts; squeezing the
    # channel axis is equivalent here.
    dx = dx.squeeze(channel_axis).cast(to_tinygrad_dtype(str(np_host)))
    dy = dy.squeeze(channel_axis).cast(to_tinygrad_dtype(str(np_host)))

    # Constant coordinate grid + random displacement field.
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    xg = Tensor(x[np.newaxis, :, :].astype(np_host)).cast(dx.dtype)
    yg = Tensor(y[np.newaxis, :, :].astype(np_host)).cast(dy.dtype)
    alpha_f = float(alpha_np)
    distorted_x = xg + alpha_f * dx  # (B, H, W)
    distorted_y = yg + alpha_f * dy

    order = AFFINE_TRANSFORM_INTERPOLATIONS[interpolation]
    channel_results = []
    for i in range(channels):
        per_batch = []
        for b in range(batch_size):
            plane = (
                images[b, :, :, i]
                if data_format == "channels_last"
                else images[b, i]
            )
            per_batch.append(
                map_coordinates(
                    plane,
                    [distorted_y[b], distorted_x[b]],
                    order=order,
                    fill_mode=fill_mode,
                    fill_value=fill_value,
                )
            )
        channel_results.append(Tensor.stack(*per_batch, dim=0))
    dim = -1 if data_format == "channels_last" else 1
    transformed_images = Tensor.stack(*channel_results, dim=dim)

    if need_squeeze:
        transformed_images = transformed_images.squeeze(0)
    return transformed_images.cast(to_tinygrad_dtype(input_dtype))


def sobel_edges(images, data_format=None):
    # NOTE: like the numpy reference, data_format is compared raw
    # (None => channels_last), not standardized.
    images = convert_to_tensor(images)
    if len(images.shape) != 4:
        # The numpy reference also fails on non-4D input (a bare unpacking
        # ValueError); raise the keras-style message instead.
        raise ValueError(
            "Invalid images rank: expected rank 4 (batch of images). "
            f"Received input with shape: images.shape={images.shape}"
        )
    if data_format == "channels_first":
        images = images.permute(0, 2, 3, 1)

    input_dtype = _keras_dtype(images)
    compute_dtype = backend.result_type(input_dtype, "float32")
    tg_compute = to_tinygrad_dtype(compute_dtype)
    num_channels = images.shape[-1]

    # scipy.ndimage.sobel semantics: correlate with [-1, 0, 1] along the
    # derivative axis and [1, 2, 1] along the other, boundary mode
    # "reflect" — which at pad width 1 is edge replication. Integer-weight
    # depthwise convs on a replicate-padded input reproduce it exactly,
    # with image gradients flowing through the convs.
    np_dtype = (
        np.float64 if tg_compute == tg_dtypes.float64 else np.float32
    )
    k_dy = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np_dtype
    )  # d/dy (axis 0)
    k_dx = np.ascontiguousarray(k_dy.T)  # d/dx (axis 1)

    x = images.cast(tg_compute).permute(0, 3, 1, 2)  # NCHW
    x = x.pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="replicate")
    planes = []
    for k in (k_dy, k_dx):
        weight = (
            Tensor(k)
            .cast(tg_compute)
            .reshape(1, 1, 3, 3)
            .expand(num_channels, 1, 3, 3)
        )
        planes.append(x.conv2d(weight, groups=num_channels))
    edges = Tensor.stack(*planes, dim=-1)  # (B, C, H, W, 2), [dy, dx]

    if data_format != "channels_first":
        edges = edges.permute(0, 2, 3, 1, 4)  # (B, H, W, C, 2)
    return edges.cast(to_tinygrad_dtype(input_dtype))


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise NotImplementedError(
        f"tinygrad backend: `keras.ops.image.{name}` is not implemented yet"
    )
