"""Random ops for the tinygrad backend.

Two sampling paths (docs/device-rng.md):

* HOST (the reference path): numpy's Generator, seeded through Keras'
  SeedGenerator machinery exactly like the numpy backend, samples wrapped
  as tinygrad Tensors — bit-identical to the reference backend. Used by
  initializers, build-time draws, and every op without a device port.
* DEVICE: tinygrad's threefry stream, computed in-graph with its counter
  advance part of the graph, so TinyJit replays and exported WebGPU
  bundles draw fresh every step. Used ONLY inside the trainer's
  `device_rng_scope` (the train step) by `dropout`, `normal` and
  `uniform`; deterministic under Keras seeding, NOT numpy-bit-parity.

Masks and samples are constants w.r.t. autograd on both paths: gradients
flow through the arithmetic they participate in (e.g. dropout's multiply),
never through the sampling.
"""

import math

import numpy as np
from tinygrad import Tensor

from keras.src.backend.config import floatx
from keras.src.backend.tinygrad.core import convert_to_numpy
from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.backend.tinygrad.core import to_tinygrad_dtype
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def _rng(seed):
    seed = draw_seed(seed)
    if isinstance(seed, Tensor):
        seed = convert_to_numpy(seed)
    return np.random.default_rng(seed)


_device_stream_seeded = False


def reset_device_stream():
    """Forget the device stream's seeding: the next device draw re-seeds it
    from its SeedGenerator (`_ensure_device_stream_seeded`). The trainer
    calls this whenever it builds a train function, so every training run
    starts a stream determined by the Keras seed state — that is what makes
    `set_random_seed(s); build; fit` reproducible twice in ONE process
    (tinygrad's stream is process-global; without the reset the second run
    continued the first run's stream). Exposed as
    `keras_tinygrad.reset_device_rng()` for loops that never rebuild a
    train function (`train_on_batch` after a manual `set_random_seed`).

    SAFETY: `Tensor.manual_seed` REPLACES tinygrad's per-device seed/counter
    buffers, so the re-seed must land in an eager step, never inside a
    TinyJit capture (the capture would keep kernels bound to the old
    buffers). The trainer's first step after a build is the eager probe —
    the re-seed always happens there. Exporters must seed BEFORE capturing
    those buffers as state (keras_tinygrad.webgpu does)."""
    global _device_stream_seeded
    _device_stream_seeded = False


def _ensure_device_stream_seeded(seed_generator):
    """Seed tinygrad's threefry stream from a SeedGenerator, once per
    `reset_device_stream()` (i.e. once per train function build).

    tinygrad's default `Tensor._seed` is wall-clock, which made device-RNG
    training irreproducible across processes even under
    `keras.utils.set_random_seed`. The stream seed is derived from the
    generator's state `[seed, counter]`, and the generator is ADVANCED by
    the draw (`next()`), exactly as a host draw would advance it: same
    seeds, same script ⇒ same stream, same masks, same run; a second run
    on the same (not rebuilt) generators gets a different, still
    deterministic stream — the same shape as the numpy backend, whose
    generators also advance across fits. Deterministic stream, NOT
    numpy-bit-parity: the documented deviation (docs/device-rng.md).
    Later generators in the same run do not re-seed: one run, one stream,
    fully determined by the first device draw's generator."""
    global _device_stream_seeded
    if _device_stream_seeded:
        return
    state = convert_to_numpy(seed_generator.next())
    seed, counter = int(state[0]), int(state[1])
    # Fold the counter in so successive seedings of one generator differ;
    # counter 0 leaves the seed untouched (first run == the bare seed).
    Tensor.manual_seed((seed ^ (counter * 0x9E3779B1)) & 0xFFFFFFFF)
    _device_stream_seeded = True


def _use_device_rng(seed):
    """Device (threefry, in-graph) sampling applies to per-step draws
    inside the trainer's `device_rng_scope` seeded by a `SeedGenerator` or
    by None (= Keras' global generator; `draw_seed(None)` resolves to it).
    NOTE: Keras wraps EVERY layer seed in a SeedGenerator —
    `Dropout(0.5, seed=42)` included — so explicit layer seeds take this
    path too; determinism is preserved by seeding the device stream from
    the first generator's state (`_ensure_device_stream_seeded`). A None
    seed inside the scope (a custom layer calling `keras.random.*` with no
    seed) takes the device path so a TinyJit capture replays it fresh; on
    the host path the capture-time draw would be baked (tinygrad's
    JitError makes that loud, and the step falls back to eager forever).
    Draws OUTSIDE the scope (initializers, build-time draws, direct
    keras.random calls outside a train step) and raw-int seeds keep the
    host/numpy path, bit-identical to the reference backend
    (docs/device-rng.md)."""
    from keras.src.backend.tinygrad.core import device_rng_enabled, in_device_rng_scope

    if not (device_rng_enabled() and in_device_rng_scope()):
        return False
    if seed is None:
        from keras.src.random.seed_generator import global_seed_generator

        seed = global_seed_generator()
    if not isinstance(seed, SeedGenerator):
        return False
    _ensure_device_stream_seeded(seed)
    return True


def _device_cast(t, dtype):
    tg = to_tinygrad_dtype(dtype)
    return t.cast(tg) if t.dtype != tg else t


def _device_shape(shape):
    # tinygrad's `rand` rejects numpy integer dims (an exact-class check —
    # the same argfix class HANDOFF lists for reshape/broadcast_to); the
    # host path's numpy accepted them, so the device path must too.
    return tuple(int(d) for d in shape)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    if _use_device_rng(seed):
        # In-graph Box-Muller over threefry uniforms (verified fresh per
        # JIT replay, mean~0/std~1 on tinygrad 0.13).
        return _device_cast(Tensor.randn(*_device_shape(shape)) * stddev + mean, dtype)
    rng = _rng(seed)
    sample = rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    return convert_to_tensor(sample, dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    if _use_device_rng(seed):
        sample = Tensor.rand(*_device_shape(shape)) * (maxval - minval) + minval
        return _device_cast(sample, dtype)
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

    if _use_device_rng(seed):
        # The tinygrad-native path: threefry RNG computed ON DEVICE, with
        # its counter advance part of the graph — so TinyJit replays (and
        # exported WebGPU bundles) draw a fresh mask every step instead of
        # freezing the capture-time one. Trade-off vs invariant 9: masks
        # are no longer bit-identical to the numpy reference backend; they
        # ARE deterministic — the stream is seeded from the Keras seed
        # state (`_ensure_device_stream_seeded`). Decision record:
        # docs/device-rng.md.
        mask = Tensor.rand(*_device_shape(noise_shape)) < keep_prob
        if noise_shape != tuple(inputs.shape):
            mask = mask.expand(tuple(inputs.shape))
        return mask.where(inputs / keep_prob, 0.0)

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
