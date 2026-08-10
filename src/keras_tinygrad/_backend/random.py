"""Random ops for the tinygrad backend.

Sampling itself runs on numpy's Generator (seeded through Keras'
SeedGenerator machinery, exactly like the numpy backend) and the samples
are wrapped as tinygrad Tensors. This keeps seeding semantics identical to
the numpy backend; masks and samples are constants w.r.t. autograd, so
gradients flow through the arithmetic they participate in (e.g. dropout's
multiply), never through the sampling.
"""

import math

import numpy as np
from tinygrad import Tensor

from keras.src.backend.config import floatx
from keras.src.backend.tinygrad.core import convert_to_numpy
from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def _rng(seed):
    seed = draw_seed(seed)
    if isinstance(seed, Tensor):
        seed = convert_to_numpy(seed)
    return np.random.default_rng(seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = _rng(seed)
    sample = rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    return convert_to_tensor(sample, dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = _rng(seed)
    sample = rng.uniform(size=shape, low=minval, high=maxval).astype(dtype)
    return convert_to_tensor(sample, dtype)


def categorical(logits, num_samples, dtype="int64", seed=None):
    rng = _rng(seed)
    logits = convert_to_numpy(logits)
    output = []
    for logits_instance in logits:
        exp = np.exp(logits_instance - np.max(logits_instance))
        probabilities = exp / np.sum(exp)
        classes = np.arange(logits_instance.shape[-1])
        samples = rng.choice(classes, size=num_samples, p=probabilities)
        output.append(samples)
    return convert_to_tensor(np.array(output).astype(dtype), dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    rng = _rng(seed)
    output = rng.integers(low=minval, high=maxval, size=shape, dtype=dtype)
    return convert_to_tensor(output, dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = _rng(seed)

    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev

    flat_shape = math.prod(shape)
    random_numbers = np.empty(0)

    while random_numbers.shape[0] < flat_shape:
        batch = rng.normal(loc=mean, scale=stddev, size=flat_shape)
        valid = batch[(batch >= lower_bound) & (batch <= upper_bound)]
        random_numbers = np.append(random_numbers, valid)

    sample = random_numbers[:flat_shape].astype(dtype).reshape(shape)
    return convert_to_tensor(sample, dtype)


def dropout(inputs, rate, noise_shape=None, seed=None):
    inputs = convert_to_tensor(inputs)
    if rate == 1.0:
        return Tensor.zeros(*inputs.shape, dtype=inputs.dtype)
    if rate == 0.0:
        return inputs

    keep_prob = 1.0 - rate
    if noise_shape is None:
        noise_shape = tuple(inputs.shape)
    else:
        noise_shape = tuple(
            n if n is not None else inputs.shape[i]
            for i, n in enumerate(noise_shape)
        )

    rng = _rng(seed)
    mask = rng.uniform(size=noise_shape) < keep_prob
    mask = np.broadcast_to(mask, tuple(inputs.shape))
    mask_t = convert_to_tensor(mask.astype(np.bool_), "bool")
    return mask_t.where(inputs / keep_prob, 0.0)


def shuffle(x, axis=0, seed=None):
    x = convert_to_tensor(x)
    rng = _rng(seed)
    axis = axis % x.ndim
    n = x.shape[axis]
    bshape = [1] * x.ndim
    bshape[axis] = n
    idx = np.broadcast_to(
        np.arange(n, dtype=np.int32).reshape(bshape), tuple(x.shape)
    )
    # Draws the same bits as the reference's `rng.permuted(x, axis)` — the
    # Fisher-Yates draws depend only on the shape (verified bit-identical)
    # — but permutes an INDEX table so the values move through a
    # differentiable gather instead of a host round-trip (which would
    # silently detach gradients from x).
    idx = rng.permuted(idx, axis=axis)
    return x.gather(axis, convert_to_tensor(idx, "int32"))


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = _rng(seed)
    return convert_to_tensor(
        rng.gamma(alpha, scale=1.0, size=shape).astype(dtype), dtype
    )


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = _rng(seed)
    sample = rng.binomial(n=counts, p=probabilities, size=shape).astype(dtype)
    return convert_to_tensor(sample, dtype)


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = _rng(seed)
    sample = rng.beta(a=alpha, b=beta, size=shape).astype(dtype)
    return convert_to_tensor(sample, dtype)
