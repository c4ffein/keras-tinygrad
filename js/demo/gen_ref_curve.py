"""CPU reference curve for the hub's dense/easy training run — the
"comparison with local keras-tinygrad" seal, consumed by tests/test_webgpu_export.py.

Runs the SAME model (init loaded from the exported build/out.safetensors —
the exact bytes the browser starts from) on the SAME LCG batch stream
(BatchGen, the byte-exact twin of the npm package's lcg32; parity itself is
asserted in js/test/run.mjs), through local Python keras-tinygrad on CPU,
and writes build/ref_curve.json. The e2e suite then asserts the browser's
curve against it: steps 0-1 must match tightly (pre-update computation
identity), and both tails must fall — post-update float32 chaos is allowed
to diverge, convergence is not (tolerance model: docs/browser-training.md).

  python gen_ref_curve.py      # run by `make browser-assets`

Extracted 2026-09-23 from gen_demo.py (whose demo.html page the hub
superseded); the numbers here mirror gen_hub.py's easy task.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
STEPS = 300
BATCH, DIM, CLASSES = 32, 28 * 28, 10
LR, MOM = 0.05, 0.9  # keep in sync with m0.py / gen_hub.py's easy task


# ---- byte-exact twin of the npm package's lcg32 (js/src/index.mjs) ----------
class BatchGen:
    def __init__(self):
        self.seed = 1234567 & 0xFFFFFFFF
        self.spare = None

    def rand(self):
        self.seed = (self.seed * 1103515245 + 12345) & 0xFFFFFFFF
        return self.seed / 4294967296.0

    def gauss(self):
        if self.spare is not None:
            v, self.spare = self.spare, None
            return v
        u = max(self.rand(), 1e-12)
        v = self.rand()
        r = math.sqrt(-2.0 * math.log(u))
        self.spare = r * math.sin(2.0 * math.pi * v)
        return r * math.cos(2.0 * math.pi * v)


def make_stream():
    import numpy as np

    g = BatchGen()
    protos = np.empty(CLASSES * DIM, dtype=np.float32)
    for i in range(CLASSES * DIM):
        protos[i] = g.gauss()  # float64 -> float32 store, like Float32Array
    for _ in range(STEPS):
        x = np.empty((BATCH, DIM), dtype=np.float32)
        y = np.empty((BATCH,), dtype=np.int32)
        for b in range(BATCH):
            y[b] = int(g.rand() * CLASSES)
            base = int(y[b]) * DIM
            for i in range(DIM):
                x[b, i] = float(protos[base + i]) + 0.35 * g.gauss()
        yield x, y


def main():
    os.environ.setdefault("KERAS_BACKEND", "tinygrad")
    import keras_tinygrad  # noqa: F401

    import keras

    # isort: split
    # AFTER keras (its alias shim imports keras; see m0.py)
    from keras_tinygrad.src.ops.core import compute_gradients, custom_gradient_tape

    import numpy as np
    from tinygrad import Tensor
    from tinygrad.nn.state import safe_load

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(CLASSES),
        ]
    )
    model.build((None, 28, 28))
    opt = keras.optimizers.SGD(learning_rate=LR, momentum=MOM)
    opt.build(model.trainable_variables)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)

    # Load the EXPORTED initial weights (same bytes the browser starts from).
    state = safe_load(os.path.join(BUILD, "out.safetensors"))
    by_shape = {}
    for name, t in state.items():
        by_shape.setdefault(tuple(t.shape), []).append((name, t))
    for var in model.trainable_variables:
        shape = tuple(int(d) for d in var.shape)
        name, t = by_shape[shape].pop(0)
        var.assign(np.asarray(t.numpy(), dtype="float32"))

    losses = []
    for i, (x, y) in enumerate(make_stream()):
        with custom_gradient_tape() as blocks:
            logits = model(Tensor(x).reshape(BATCH, 28, 28), training=True)
            loss = loss_fn(y, logits).mean()
        grads = compute_gradients(loss, [v.value for v in model.trainable_variables], blocks)
        loss_val = float(loss.numpy())  # pre-update, like the export
        opt.apply(grads, model.trainable_variables)
        losses.append(loss_val)
        if i % 50 == 0:
            print(f"step {i:3d}  loss {loss_val:.4f}")
    with open(os.path.join(BUILD, "ref_curve.json"), "w") as f:
        json.dump({"steps": STEPS, "losses": losses}, f)
    print(f"ref_curve.json written  (first10 {sum(losses[:10]) / 10:.4f}, last10 {sum(losses[-10:]) / 10:.4f})")


if __name__ == "__main__":
    main()
