"""Probe for the intermittent wrong `model.evaluate` on tinygrad's CL device
(PoCL via scripts/with_cl.sh), found 2026-09-28: over 10 runs of this
script, 3 returned an `evaluate` that disagreed with predict-MSE and with
the loss function applied to `model(x)` — values 224.0 (= 7 x 32, one
batch short of the 256 samples) and 0.0 — while fit, predict, the loss
function and 60 isolated rounds of raw sum / Mean metric / Variable.assign
/ MeanSquaredError were all exact. So the trigger is inside the eager
test_step loop (per-batch metric update + reset), on CL only; CPU never
shows it. Root-caused and fixed 2026-10-02 (`IS_THREAD_SAFE = False` in ops/core.py:
keras ran the per-batch metric read on a callback worker thread, racing
tinygrad's in-place kernel arguments on the main thread; see
tests/test_device_cl.py). Kept as the standalone repro — with the flag set
back to True it fails again. Run:

    scripts/with_cl.sh uv run python scripts/probe_cl_evaluate.py   # repeat; ~1 in 3 disagrees

Every line prints five numbers that must agree.
"""

import os

import keras_tinygrad  # noqa: F401  (must precede keras)

import keras
import numpy as np
from keras import ops

print("DEV", os.environ.get("DEV", "CPU"), "JIT", os.environ.get("KERAS_TINYGRAD_TRAINER_JIT", "1"))
rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)).astype("float32")
y = x @ rng.normal(size=(8, 1)).astype("float32")
m = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
m.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
h = m.fit(x, y, epochs=3, batch_size=32, verbose=0)
p = m.predict(x, verbose=0, batch_size=32)
mse_np = float(np.mean((p - y) ** 2))
ev = float(m.evaluate(x, y, batch_size=32, verbose=0))
ev_full = float(m.evaluate(x, y, batch_size=256, verbose=0))
loss_fn = float(ops.convert_to_numpy(keras.losses.mean_squared_error(y, m(x))).mean())
print(
    f"fit last {h.history['loss'][-1]:.4f} | predict-mse {mse_np:.4f} | evaluate(bs32) {ev:.4f}"
    f" | evaluate(bs256) {ev_full:.4f} | loss_fn on m(x) {loss_fn:.4f}"
)
agree = all(abs(v - mse_np) < 1e-3 * max(1.0, mse_np) for v in (ev, ev_full, loss_fn))
print("AGREE" if agree else "DISAGREE — evaluate is wrong on this device")
raise SystemExit(0 if agree else 1)
