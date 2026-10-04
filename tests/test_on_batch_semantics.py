"""`train_on_batch` / `test_on_batch` return THIS batch's metrics.

The tensorflow, jax and torch trainers call `reset_metrics()` before each
*_on_batch call; the numpy trainer (this backend's port source) does not,
so ours returned the loss tracker's running mean: at step k exactly the
mean of the per-batch losses 1..k (found 2026-09-28 comparing
`train_on_batch` curves with tensorflow — the "Adam diverges after step 1"
mystery was this). Weights were always right; the report was wrong.
"""

import os
import pathlib
import subprocess
import sys

from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent


def run(code):
    env = dict(os.environ, KERAS_BACKEND="tinygrad")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900, preexec_fn=child_limits
    )
    assert proc.returncode == 0, f"child failed\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr[-4000:]}"
    return proc.stdout


def test_on_batch_reports_this_batch_not_a_running_mean():
    out = run(
        """
import keras_tinygrad  # noqa: F401
import keras, numpy as np
rng = np.random.default_rng(0)
m = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(16, activation="tanh"), keras.layers.Dense(1)])
m.compile(optimizer=keras.optimizers.SGD(0.05), loss="mse", metrics=["mae"])
xs = rng.normal(size=(4, 32, 8)).astype("float32"); ys = rng.normal(size=(4, 32, 1)).astype("float32")
reported, fresh = [], []
for i in range(4):
    # the loss of THIS batch on the pre-update weights, from an independent path
    fresh.append(float(np.mean((m.predict(xs[i], verbose=0) - ys[i]) ** 2)))
    reported.append(float(m.train_on_batch(xs[i], ys[i])[0]))
np.testing.assert_allclose(reported, fresh, rtol=1e-5, atol=1e-6)
running = np.cumsum(fresh) / np.arange(1, 5)
assert not np.allclose(reported[1:], running[1:]), "still a running mean"
# test_on_batch: evaluating the same batch twice reports the same number
a = m.test_on_batch(xs[0], ys[0]); b = m.test_on_batch(xs[0], ys[0])
np.testing.assert_allclose(a, b, rtol=1e-6)
# and it is that batch's loss, not the mean with the previous call's batch
c = m.test_on_batch(xs[1], ys[1])
d = float(np.mean((m.predict(xs[1], verbose=0) - ys[1]) ** 2))
np.testing.assert_allclose(c[0], d, rtol=1e-5, atol=1e-6)
print("ON_BATCH OK", np.round(reported, 4).tolist())
"""
    )
    assert "ON_BATCH OK" in out, out
