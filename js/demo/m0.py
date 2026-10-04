"""M0 spike: export a KERAS train step as a standalone WebGPU runner.

The last unproven link between the two tracks (see ../../docs/browser-training.md):
nnvp's TinygradRuntime already traces RAW tinygrad train steps on the GPU-less
NULL:WGSL device and loops the emitted runner in the browser (2-4x faster than
tfjs-webgl on Chrome WebGPU). This spike does the same trace on a step built
from KERAS APIs through keras-tinygrad: keras layers forward, keras loss,
backend compute_gradients, keras optimizer.apply. If it exports and the runner
trains, "pip install keras-tinygrad" is a browser-training story, and NNVP can
feed its generated Keras Python (a target it already emits) straight in.

The two prior arts being married:
- nnvp .../TinygradRuntime/py/driver.py — the trace/export recipe (realize
  weights BEFORE capture; realize loss WITH the updates; NULL fake-executes so
  real initial weights are substituted into the safetensors afterwards).
- keras_tinygrad/_backend/trainer.py (_TrainStepJit) — the pinning scheme:
  Keras Variables rebind `_value` on every assign, so inside the traced call
  each new value is copied back into the buffer the capture read (in-place
  Tensor.assign) and `_value` repointed. That turns Keras' functional update
  into the stable-buffer, in-place form a replayed/exported graph needs.

Usage (from this directory, fork-venv python):
  python m0.py export out            # NULL:WGSL trace -> out.js + out.safetensors + out.meta.json
  CC=~/.local/bin/zigcc python m0.py cpu   # numeric proof: SAME step, real exec, loss falls
"""

import json
import math
import os
import random
import struct
import sys

MODE = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else None
if MODE == "export":
    # Must be set before tinygrad import (run_local.py precedent).
    os.environ["DEV"] = "NULL:WGSL"
    os.environ["NULL_ALLOW_COPYOUT"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import keras_tinygrad  # noqa: F401, E402  (must precede keras)

import keras  # noqa: E402

# isort: split
# AFTER keras: keras_tinygrad.src's alias registration imports keras itself,
# so pulling it in mid-keras-import leaves the backend star-import empty
# (NameError name_scope). isort's keras_tinygrad section would hoist it
# above keras — keep it in its own block, below.
from keras_tinygrad.src.ops.core import compute_gradients  # noqa: E402

import numpy as np  # noqa: E402
from tinygrad import Tensor  # noqa: E402

BATCH_SIZE = 32
LEARNING_RATE = 0.05
MOMENTUM = 0.9
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10


def build_model():
    return keras.Sequential(
        [
            keras.layers.Input(shape=INPUT_SHAPE),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(NUM_CLASSES),  # logits; softmax folded into the loss
        ]
    )


class KerasTrainStep:
    """One SGD step of a real Keras model, callable as step(x, y) -> loss."""

    def __init__(self):
        self.model = build_model()
        self.opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
        self.opt.build(self.model.trainable_variables)
        # reduction=None + tinygrad-side .mean(), NOT the stock
        # sum_over_batch_size: Keras' own reduction realizes its 1/batch
        # normalization as a scalar const BUFFER, which the NULL device
        # fake-computes to zero and export_model_webgpu then emits as an
        # empty buffer (bufs_to_save is only honored for get_state_dict
        # names) — the exported loss reads ×0. tinygrad's mean folds the
        # same 1/batch as a kernel immediate, which survives export. Same
        # math either way; validate_runner_js guards the general hazard.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)

    def float_variables(self):
        """Every float Variable the step assigns, deduplicated, stable order.
        Int state is handled separately: optimizer.iterations (int32) rides
        in the export state so its +1 kernel reads a real initialized buffer
        — its assign does NOT drop out of the trace (the exported runners
        always contained the increment kernel, reading garbage)."""
        out, seen = [], {}
        groups = [self.model.trainable_variables, self.model.non_trainable_variables, self.opt.variables]
        for group in groups:
            for v in group:
                if id(v) in seen or "float" not in str(v.dtype):
                    continue
                seen[id(v)] = True
                out.append(v)
        return out

    def __call__(self, x, y):
        # iterations is pinned too, so its +1 lands back in the state buffer
        # and the exported counter actually counts.
        variables = self.float_variables() + [self.opt.iterations]
        pinned = [v._value for v in variables]
        logits = self.model(x, training=True)
        loss = self.loss_fn(y, logits).mean()
        grads = compute_gradients(loss, [v.value for v in self.model.trainable_variables])
        # Realize loss AND grads before the optimizer builds any update: the
        # returned loss must be the PRE-update value, and scheduling it
        # together with the pin-assigns below let the scheduler recompute it
        # from moved weights (measured: 2.927 became 0.378 on step one — the
        # same hazard driver.py documents). With no update graph in existence
        # yet, the wrong order is unbuildable.
        Tensor.realize(loss, *grads)
        self.opt.apply(grads, self.model.trainable_variables)
        # The _TrainStepJit pinning: land every update as an in-place assign
        # on the buffer the capture read.
        changed = []
        for v, pin in zip(variables, pinned):
            if v._value is pin:
                continue
            pin.assign(v._value.detach())
            v._value = pin
            changed.append(pin)
        Tensor.realize(*changed)
        return loss


def make_export_target(step):
    """A bare function so export_model's get_state_dict never walks the Keras
    object graph (parent references cycle); only fn.__dict__ is traversed, and
    it holds exactly the pinned buffers under their Keras variable paths."""

    def target(x, y):
        return step(x, y)

    target.state = {v.path: v._value for v in step.float_variables()}
    # iterations (int32) too: its increment kernel is part of every trace,
    # and without a state entry it reads a zeroed createEmptyBuf.
    target.state[step.opt.iterations.path] = step.opt.iterations._value
    return target


def build_safetensors(state, seed=1337):
    """Real initial values for the NULL-traced (all-zeros) weights — the
    driver.py trick, keyed on Keras names: Glorot for kernels, the actual lr
    for learning_rate, zeros for biases and momentum slots."""
    rng = random.Random(seed)
    header, blobs, offset = {}, [], 0
    for name, tensor in state.items():
        shape = [int(d) for d in tensor.shape]
        count = math.prod(shape) if shape else 1
        if "int" in str(tensor.dtype):  # optimizer.iterations: starts at 0
            blob = struct.pack(f"<{count}i", *([0] * count))
            header[name] = {"dtype": "I32", "shape": shape, "data_offsets": [offset, offset + len(blob)]}
            blobs.append(blob)
            offset += len(blob)
            continue
        assert str(tensor.dtype) in ("dtypes.float", "dtypes.float32"), f"unexpected dtype for {name}: {tensor.dtype}"
        if name.endswith("/kernel") and len(shape) >= 2:
            fan_in, fan_out = shape[0], shape[-1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            values = [rng.uniform(-limit, limit) for _ in range(count)]
        elif "learning_rate" in name:
            values = [LEARNING_RATE] * count
        else:  # biases and momentum slots start at zero
            values = [0.0] * count
        blob = struct.pack(f"<{count}f", *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header_json = json.dumps(header).encode("utf8")
    return struct.pack("<Q", len(header_json)) + header_json + b"".join(blobs)


def synthetic_batch(rng, prototypes):
    """Ten noisy class prototypes — separable, so a working step visibly
    learns within ~100 steps, with no dataset download."""
    y = rng.integers(0, NUM_CLASSES, size=BATCH_SIZE)
    x = prototypes[y] + 0.35 * rng.standard_normal((BATCH_SIZE, *INPUT_SHAPE)).astype("float32")
    return x.astype("float32"), y.astype("int32")


def validate_runner_js(js):
    """No pass may read a createEmptyBuf that nothing wrote earlier.

    On NULL, any value the trace computed OUTSIDE the captured call (a
    realized const chain — e.g. a loss-reduction scalar) is fake-executed to
    zeros AND emitted as createEmptyBuf (export_model_webgpu honors
    bufs_to_save only for get_state_dict names), so the runner silently
    multiplies by zero. Fail loudly instead, naming the buffer."""
    import re

    empty = set(re.findall(r"const (\w+) = createEmptyBuf\(", js))
    written = set()
    # Anchor on "infinityBuf, [" — the bufs list is the bracket group AFTER
    # it. r"addComputePass\([^[]*\[" is WRONG: pipelines[N] contains a '[',
    # so it captures the pipeline index and the check passes vacuously —
    # which is how the zeroed momentum scalar shipped (found 2026-08-30 via
    # the hub's vs-tfjs mode: the bundle trained plain SGD).
    passes = re.findall(r"addComputePass\([^)]*?infinityBuf, \[([^\]]*)\]", js)
    assert passes, "no addComputePass calls found in the emitted runner"
    assert all("[" not in p for p in passes), "unexpected bracket in bufs list"
    for arglist in passes:
        args = [a.strip() for a in arglist.split(",")]
        out, reads = args[0], args[1:]
        for name in reads:
            if name in empty and name not in written and not name.startswith("input"):
                raise AssertionError(
                    f"pass reads {name}, a never-written empty buffer — a "
                    "baked const was dropped by the export (see the "
                    "reduction=None note in KerasTrainStep)"
                )
        written.add(out)


def run_export(prefix):
    from keras_tinygrad._vendor import export_model as em

    if "NULL" not in em.EXPORT_SUPPORTED_DEVICE:
        em.EXPORT_SUPPORTED_DEVICE.append("NULL")
    step = KerasTrainStep()
    target = make_export_target(step)
    Tensor.realize(*target.state.values())  # materialize BEFORE capture
    x = Tensor.randn(BATCH_SIZE, *INPUT_SHAPE)
    y = Tensor.randint(BATCH_SIZE, low=0, high=NUM_CLASSES)
    js, inp_sizes, out_sizes, state = em.export_model(target, "webgpu", x, y, model_name="kerastrainstep")
    weights = build_safetensors(state)
    meta = {
        "batchSize": BATCH_SIZE,
        "inputShape": list(INPUT_SHAPE),
        "numClasses": NUM_CLASSES,
        "learningRate": LEARNING_RATE,
        "inputSizes": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in inp_sizes.items()},
        "outputSizes": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in out_sizes.items()},
        "stateEntries": list(state.keys()),
        "stateShapes": {k: [int(d) for d in v.shape] for k, v in state.items()},
        "kernels": sum(1 for line in js.splitlines() if "@compute" in line),
    }
    with open(f"{prefix}.js", "w") as f:
        f.write(js)
    with open(f"{prefix}.safetensors", "wb") as f:
        f.write(weights)
    with open(f"{prefix}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("state entries:", json.dumps(meta["stateShapes"], indent=2))
    print(f"kernels: {meta['kernels']}  js bytes: {len(js)}  weights bytes: {len(weights)}")
    assert "setupNet" in js and "@compute" in js, "emitted JS is not the expected runner"
    assert any(name.endswith("/kernel") for name in state), "no Keras kernel in exported state"
    validate_runner_js(js)
    print(f"OK — Keras train step exported to {prefix}.js / .safetensors / .meta.json")


def run_cpu():
    rng = np.random.default_rng(0)
    prototypes = rng.standard_normal((NUM_CLASSES, *INPUT_SHAPE)).astype("float32")
    step = KerasTrainStep()
    losses = []
    for i in range(120):
        x, y = synthetic_batch(rng, prototypes)
        loss = step(Tensor(x), Tensor(y))
        losses.append(float(loss.numpy()))
        if i % 20 == 0:
            print(f"step {i:3d}  loss {losses[-1]:.4f}")
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    print(f"mean loss first 10: {first:.4f}  last 10: {last:.4f}")
    assert last < first * 0.5, "loss did not fall — the step is not training"
    # And the trained model actually classifies the synthetic classes:
    x, y = synthetic_batch(rng, prototypes)
    pred = np.asarray(step.model(Tensor(x))).argmax(axis=1)
    acc = float((pred == y).mean())
    print(f"accuracy on a fresh batch: {acc:.2f}")
    assert acc > 0.8, "trained model does not classify the separable classes"
    print("OK — the Keras step trains (real execution)")


if MODE == "export":
    run_export(sys.argv[2] if len(sys.argv) > 2 else "out")
elif MODE == "cpu":
    run_cpu()
elif MODE is not None:
    raise SystemExit(f"unknown mode {MODE!r} (export|cpu)")
