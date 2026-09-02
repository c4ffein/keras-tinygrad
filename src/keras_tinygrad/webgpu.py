"""Export ONE Keras training step as a self-contained WebGPU runner.

THE single implementation of the browser-export recipe (previously
hand-copied across experiments — three copies of subtle, load-bearing code
is where the momentum-class bugs hide; see
experiments/m0-keras-trainstep/README.md, "three bugs"):

    os.environ["DEV"] = "NULL:WGSL"          # BEFORE tinygrad is imported
    os.environ["NULL_ALLOW_COPYOUT"] = "1"
    os.environ["KERAS_TINYGRAD_TRAINER_JIT"] = "0"   # trace eagerly
    import keras_tinygrad, keras
    from keras_tinygrad.webgpu import export_train_step
    out = export_train_step(model, keras.optimizers.SGD(0.01), loss_fn,
                            batch_size=32, input_shape=(784,),
                            learning_rate=0.01)   # repeats the optimizer's lr, see below
    # out["js"]: ES module exposing setupNet(device, weights) -> step(x, y)
    # out["weights"]: safetensors bytes (each weight's Keras initializer
    #                 re-run on the host, the lr, zeroed optimizer slots)
    # out["meta"]: kernels, state entries, sizes

The runner's weight buffers update in place, so looping `step(x, y)` from
JavaScript IS SGD training. The load-bearing pieces, each learned the hard
way: realize loss+grads BEFORE `optimizer.apply` (or the scheduler
recomputes the loss from moved weights); pin every Variable — `iterations`
included — so updates land in the buffers the capture read; run the forward
inside `device_rng_scope` so covered layers' randomness is on-device
threefry with its counter advance in the graph (fresh masks per replayed
step); and `validate_runner_js`, which fails loudly if any compute pass
reads a never-written empty buffer — the NULL device fake-executes, so a
const that leaks into a buffer exports as zeros (this shipped once, as
momentum=0.9 becoming 0.0: the bundles silently trained plain SGD).
"""

import json
import math
import random as _random
import re
import struct

from keras_tinygrad._vendor import export_model as _em

from tinygrad import Device, Tensor

if "NULL" not in _em.EXPORT_SUPPORTED_DEVICE:
    _em.EXPORT_SUPPORTED_DEVICE.append("NULL")


class KerasTrainStep:
    """One Keras SGD step per call, shaped for NULL-device capture."""

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.opt = optimizer
        self.opt.build(model.trainable_variables)
        self.loss_fn = loss_fn

    def float_variables(self):
        out, seen = [], set()
        for group in (
            self.model.trainable_variables,
            self.model.non_trainable_variables,
            self.opt.variables,
        ):
            for v in group:
                if id(v) in seen or "float" not in str(v.dtype):
                    continue
                seen.add(id(v))
                out.append(v)
        return out

    def __call__(self, x, y):
        from keras.src.backend.tinygrad.core import (
            compute_gradients,
            custom_gradient_tape,
            device_rng_scope,
        )

        # iterations (int32) is pinned like the floats: its +1 kernel is in
        # every trace, and without a pinned state buffer it reads garbage.
        variables = self.float_variables() + [self.opt.iterations]
        pinned = [v._value for v in variables]
        with custom_gradient_tape() as blocks, device_rng_scope():
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
            if len(loss.shape):  # reduction=None: fold the mean as an immediate
                loss = loss.mean()
        grads = compute_gradients(loss, [v.value for v in self.model.trainable_variables], blocks)
        Tensor.realize(loss, *grads)  # pre-update loss; wrong order is unbuildable
        self.opt.apply(grads, self.model.trainable_variables)
        changed = []
        for v, pin in zip(variables, pinned):
            if v._value is pin:
                continue
            pin.assign(v._value.detach())
            v._value = pin
            changed.append(pin)
        Tensor.realize(*changed)
        return loss


def _host_initial_values(name, shape, count, initializer, rng):
    """Re-run a weight's Keras initializer on the host with Python's RNG.

    The trace runs on the NULL device, where every tensor read is fake
    (zeros), so the real initial values cannot be read back — they are
    recomputed here from the initializer the owning layer declared. Only
    initializers with a port below are accepted; anything else is a loud
    NotImplementedError. The alternative shipped once: "Glorot for kernels,
    zeros elsewhere" exported BatchNormalization's gamma and
    moving_variance as 0.0 — a dead network, silently.
    """
    from keras.src import initializers as ki
    from keras.src.initializers.random_initializers import compute_fans

    if isinstance(initializer, ki.Zeros):
        return [0.0] * count
    if isinstance(initializer, ki.Ones):
        return [1.0] * count
    if isinstance(initializer, ki.Constant):
        return [float(initializer.value)] * count
    if isinstance(initializer, ki.GlorotUniform):
        fan_in, fan_out = compute_fans(shape)
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return [rng.uniform(-limit, limit) for _ in range(count)]
    if isinstance(initializer, ki.RandomUniform):
        return [rng.uniform(float(initializer.minval), float(initializer.maxval)) for _ in range(count)]
    if isinstance(initializer, ki.RandomNormal):
        return [rng.gauss(float(initializer.mean), float(initializer.stddev)) for _ in range(count)]
    raise NotImplementedError(
        f"export: no host-side port for the initializer of {name!r} "
        f"({type(initializer).__name__ if initializer is not None else 'unresolved'}); "
        "add one to keras_tinygrad.webgpu._host_initial_values rather than exporting zeros."
    )


def _initializer_map(model, optimizer):
    """Variable path -> the Keras initializer that produced it.

    Resolved from the owning layer's `<weight>_initializer` attribute
    (Keras' own convention: `Dense.kernel` <-> `kernel_initializer`,
    `BatchNormalization.moving_variance` <-> `moving_variance_initializer`).
    Optimizer slots are zeros by construction
    (`add_variable_from_reference`); the learning rate is special-cased by
    name in `build_safetensors`. A weight without a matching attribute maps
    to None, which `build_safetensors` refuses loudly."""
    from keras.src import initializers as ki

    out = {}
    for layer in model._flatten_layers():
        for v in list(layer._trainable_variables) + list(layer._non_trainable_variables):
            out[v.path] = getattr(layer, f"{v.path.rsplit('/', 1)[-1]}_initializer", None)
    for v in optimizer.variables:
        out.setdefault(v.path, ki.Zeros())
    return out


def build_safetensors(state, learning_rate, initializers, seed=1337):
    """Real initial values for the NULL-traced (all-zeros) state: each float
    weight's Keras initializer re-run on the host (`_host_initial_values`,
    keyed by `initializers`, a path -> initializer map from
    `_initializer_map`), the real lr; U32 rng seed/counter and the I32
    iteration counter start at zero."""
    rng = _random.Random(seed)
    header, blobs, offset = {}, [], 0
    for name, tensor in state.items():
        shape = [int(d) for d in tensor.shape]
        count = math.prod(shape) if shape else 1
        dt = str(tensor.dtype)
        if "uint" in dt or "unsigned" in dt:
            blob, dtype = struct.pack(f"<{count}I", *([0] * count)), "U32"
        elif "int" in dt:
            blob, dtype = struct.pack(f"<{count}i", *([0] * count)), "I32"
        else:
            assert dt in ("dtypes.float", "dtypes.float32"), (name, dt)
            path = name[len("state.") :] if name.startswith("state.") else name
            if "learning_rate" in name:
                values = [learning_rate] * count
            else:
                values = _host_initial_values(name, shape, count, initializers.get(path), rng)
            blob, dtype = struct.pack(f"<{count}f", *values), "F32"
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    hj = json.dumps(header).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def validate_runner_js(js):
    """No pass may read a createEmptyBuf nothing wrote earlier (a NULL-baked
    const exported as zeros). Anchored on `infinityBuf, [` — the naive
    `addComputePass\\([^[]*\\[` regex stops at `pipelines[N]`'s bracket and
    validates pipeline indices, i.e. nothing (that vacuous guard is how the
    zeroed momentum shipped)."""
    empty = set(re.findall(r"const (\w+) = createEmptyBuf\(", js))
    written = set()
    passes = re.findall(r"addComputePass\([^)]*?infinityBuf, \[([^\]]*)\]", js)
    assert passes, "no addComputePass calls matched — runner format changed?"
    assert all("[" not in p for p in passes)
    for arglist in passes:
        args = [a.strip() for a in arglist.split(",")]
        out, reads = args[0], args[1:]
        for name in reads:
            if name in empty and name not in written and not name.startswith("input"):
                raise AssertionError(f"pass reads never-written empty buffer {name}")
        written.add(out)


def export_train_step(
    model,
    optimizer,
    loss_fn,
    *,
    batch_size,
    input_shape,
    learning_rate,
    label_shape=None,
    label_dtype=None,
    model_name="kerasstep",
    weight_seed=1337,
):
    """Trace one training step of `model` on NULL:WGSL and emit the runner.

    Returns {"js": str, "weights": bytes, "meta": dict}. Requires
    DEV=NULL:WGSL to have been set before tinygrad was imported, and the
    trainer JIT disabled for the trace (KERAS_TINYGRAD_TRAINER_JIT=0).
    `learning_rate` must repeat the optimizer's fixed lr — see the comment
    at its use for why the exporter cannot read it itself.

    The traced label batch `y` follows the loss: a `Sparse*` loss gets
    int32 class ids shaped like the model output minus its last axis
    (`(batch_size,)` for a classifier), anything else gets float32 labels
    shaped exactly like the model output (a regression on
    MeanSquaredError). Pass `label_shape` (WITHOUT the batch axis) and/or
    `label_dtype` to override — the trace bakes the label layout, so a
    mismatch here is a wrong bundle, not a runtime error."""
    assert "NULL" in Device.DEFAULT, (
        f"export requires DEV=NULL:WGSL before importing tinygrad (device is {Device.DEFAULT})"
    )
    from keras.src.backend.tinygrad.core import device_rng_enabled

    if getattr(loss_fn, "reduction", None) not in (None, "none"):
        # Keras' default "sum_over_batch_size" divides by
        # `convert_to_tensor(shape)` — a host TUPLE, which becomes a
        # numpy-backed 4-byte buffer inside the trace: an anonymous input
        # the bundle cannot carry (it exported as an unwritten buffer, i.e.
        # zeros, i.e. a NaN loss; validate_runner_js catches it, but the
        # message named a buffer, not the cause). KerasTrainStep folds the
        # mean over a reduction=None loss as an immediate instead.
        raise ValueError(
            "export_train_step: build the loss with reduction=None (got "
            f"reduction={loss_fn.reduction!r}); the step folds the batch mean itself."
        )

    step = KerasTrainStep(model, optimizer, loss_fn)

    def target(x, y):
        return step(x, y)

    has_seed_layers = any(getattr(layer, "_seed_generators", None) for layer in model._flatten_layers())
    if has_seed_layers and device_rng_enabled():
        # Seed the device stream BEFORE creating/capturing any rng state
        # buffer: Tensor.manual_seed (inside _ensure_device_stream_seeded)
        # RESETS tinygrad's per-device seed/counter dicts, so a mid-trace
        # first draw would replace the very buffers saved in `state` — the
        # captured kernels then read fresh, never-written buffers
        # (validate_runner_js catches it as an empty-buffer read).
        from keras.src.backend.tinygrad import random as _backend_random

        generators = [
            gen for layer in model._flatten_layers() for gen in (getattr(layer, "_seed_generators", None) or [])
        ]
        if generators:
            _backend_random._ensure_device_stream_seeded(generators[0])
        Tensor.rand(1).realize()  # ensures the device rng seed/counter buffers exist
    target.state = {v.path: v._value for v in step.float_variables()}
    target.state[step.opt.iterations.path] = step.opt.iterations._value
    if has_seed_layers and device_rng_enabled():
        dev = Device.DEFAULT
        target.state["rng/seed"] = Tensor._device_seeds[dev]
        target.state["rng/counter"] = Tensor._device_rng_counters[dev]
    Tensor.realize(*target.state.values())  # materialize BEFORE capture
    output_shape = tuple(int(d) for d in model.output_shape[1:])
    sparse = type(loss_fn).__name__.startswith("Sparse")
    if label_shape is None:
        label_shape = output_shape[:-1] if sparse else output_shape
    if label_dtype is None:
        label_dtype = "int32" if sparse else "float32"
    # Placeholder batches: input BUFFERS (the exporter needs buffers, not
    # consts) whose contents the NULL trace never reads. Not Tensor.randn —
    # that draws from the device RNG stream and would advance the very
    # counter captured as state above.
    x = Tensor.empty(int(batch_size), *[int(d) for d in input_shape])
    y = Tensor.empty(int(batch_size), *[int(d) for d in label_shape], dtype=label_dtype)
    js, inp_sizes, out_sizes, state = _em.export_model(target, "webgpu", x, y, model_name=model_name)
    validate_runner_js(js)
    # `learning_rate` is a REQUIRED argument because the optimizer's own
    # value is unrecoverable here: on the NULL device every tensor read is
    # fake (zeros) — `convert_to_numpy(optimizer.learning_rate)` AND
    # `optimizer.get_config()` (which serializes via `.numpy()`) both baked
    # lr=0.0 into the weights blob. NULL-fakery bug class, fourth
    # occurrence; caught by the hub's bundle-vs-in-tab identity check and
    # pinned by tests/test_backend_regressions.py.
    lr = float(learning_rate)
    weights = build_safetensors(state, lr, _initializer_map(model, optimizer), seed=weight_seed)
    meta = {
        "batchSize": int(batch_size),
        "inputShape": [int(d) for d in input_shape],
        "labelShape": [int(d) for d in label_shape],
        "labelDtype": label_dtype,
        "learningRate": lr,
        "stateEntries": list(state.keys()),
        "kernels": sum(1 for line in js.splitlines() if "@compute" in line),
        "jsBytes": len(js),
        "weightBytes": len(weights),
    }
    return {"js": js, "weights": weights, "meta": meta}
