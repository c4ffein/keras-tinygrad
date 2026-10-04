"""The recurrent train step: bounded cost, jitted, and gradients that survive
the forward's kernel barriers. Three receipts for the 2026-09-28 fix, each
a subprocess under tests/_limits.py (a Keras process is locked to one
backend at import).

What was found (keras.io's functional_api guide timing out on its
Embedding+LSTM section): LSTM(32) over 10 steps, batch 32, CPU — 451
kernels and 21-36 s per eager train step, superlinear in the sequence
length, and the JIT never captured (the recurrent cells own a SeedGenerator,
so the "seed generators present" gate forced eager without a word). Fixed
in three places: the scan cuts every timestep (`rnn._cut`), the cells are
covered by the device-RNG gate (their only draw is `random.dropout`), and
the trainer builds the gradient graph BEFORE the loss tracker realizes the
loss — with the cuts in place, the old order zeroed every gradient upstream
of the barriers (Embedding + LSTM weights did not move; the Dense head did).
"""

import json
import math
import os
import pathlib
import subprocess
import sys
import textwrap

from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent


def _run(code, env=None):
    full_env = dict(os.environ, KERAS_BACKEND="tinygrad", KERAS_TINYGRAD_NO_VERSION_WARNING="1", **(env or {}))
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=REPO,
        timeout=900,
        preexec_fn=child_limits,
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout[-4000:]}\n{out.stderr[-4000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])


_LSTM_STEPS = """
import json, os, time
import keras_tinygrad, keras
import numpy as np
from tinygrad.helpers import GlobalCounters
keras.utils.set_random_seed(0)
T, B = 10, 32
rate = float(os.environ.get("KTG_TEST_RNN_DROPOUT", "0"))
m = keras.Sequential([keras.Input((T,)), keras.layers.Embedding(1000, 64),
                      keras.layers.LSTM(32, dropout=rate, recurrent_dropout=rate), keras.layers.Dense(1)])
m.compile(optimizer="adam", loss="mse")
rng = np.random.default_rng(0)
x = rng.integers(0, 1000, size=(B, T)).astype("int32"); y = rng.normal(size=(B, 1)).astype("float32")
times, kernels, losses = [], [], []
for i in range(6):
    GlobalCounters.reset(); t0 = time.time()
    losses.append(float(m.train_on_batch(x, y)))
    times.append(time.time() - t0); kernels.append(GlobalCounters.kernel_count)
# the _TrainStepJit lives in the train function's closure
jit = next((c.cell_contents for c in (m.train_function.__closure__ or ())
            if type(c.cell_contents).__name__ == "_TrainStepJit"), None)
weights = [np.asarray(w, "float64").ravel().tolist() for w in m.get_weights()]
print(json.dumps({"times": times, "kernels": kernels, "losses": losses, "weights": weights,
                  "jit": None if jit is None else {"disabled": jit._disabled, "captures": len(jit._jits)}}))
"""


def test_lstm_eager_train_step_is_bounded():
    """Eager path (the one every fallback lands on). Measured 2026-09-28,
    this box, zig cc, T=10, batch 32, no dropout: before the fix 451
    kernels and 21-36 s per steady step (load ~5-9); after, 297 kernels and
    6.4 s at load ~13 — the recurrence is cut per timestep so the scheduler
    no longer re-walks it, forward and backward. The bound is a loud guard
    (~4x the measurement), not a benchmark. (An earlier 59-kernel / 0.6 s
    reading was the trainer's old order: gradients zeroed upstream of the
    cuts, so the backward graph was empty — never trust a fast RNN step
    without the gradient receipt below.)"""
    r = _run(_LSTM_STEPS, {"KERAS_TINYGRAD_TRAINER_JIT": "0"})
    steady = r["times"][2:]
    assert max(r["kernels"][2:]) < 400, f"kernels per eager step: {r['kernels']} (uncut?), times {r['times']}"
    assert max(steady) < 25.0, f"eager LSTM step too slow: {r['times']}"
    assert all(math.isfinite(v) for v in r["losses"]), r["losses"]


def test_lstm_train_step_jits_and_matches_eager():
    """The gate lets recurrent cells through (their randomness is
    `random.dropout` on the device stream) and the capture is exact: with
    dropout AND recurrent dropout on, six steps JIT vs eager leave every
    weight bit-for-bit identical, and the JIT object reports one capture.
    Before: `_disabled` was True from the probe on, silently."""
    env = {"KTG_TEST_RNN_DROPOUT": "0.2"}
    jit = _run(_LSTM_STEPS, {"KERAS_TINYGRAD_TRAINER_JIT": "1", **env})
    eager = _run(_LSTM_STEPS, {"KERAS_TINYGRAD_TRAINER_JIT": "0", **env})
    assert jit["jit"] is not None, "no _TrainStepJit found in the train function"
    assert not jit["jit"]["disabled"], "LSTM train step was gated to eager"
    assert jit["jit"]["captures"] == 1, jit["jit"]
    assert jit["losses"] == eager["losses"], (jit["losses"], eager["losses"])
    for i, (a, b) in enumerate(zip(jit["weights"], eager["weights"])):
        assert a == b, f"weight {i} differs between JIT and eager"


_BARRIER_GRADS = """
import json
import keras_tinygrad, keras
import numpy as np
from keras_tinygrad.src.trainer import compute_gradients
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
x = rng.normal(size=(32, 6)).astype("float32"); y = rng.normal(size=(32, 1)).astype("float32")
def build():
    # log1p holds a `.contiguous()` barrier (numpy.py: the compensated form
    # needs it) — the same shape as the RNN scan's per-step cuts.
    return keras.Sequential([keras.Input((6,)), keras.layers.Dense(5, activation="tanh"),
                             keras.layers.Lambda(lambda t: keras.ops.log1p(keras.ops.square(t))),
                             keras.layers.Dense(1)])
m = build(); w0 = [np.asarray(w, "float64") for w in m.get_weights()]
# the reference: gradients built and read before anything else realizes the loss
loss = keras.ops.mean(keras.losses.mean_squared_error(y, m(x, training=True)))
grads = compute_gradients(loss, [v.value for v in m.trainable_weights], [])
ref = [np.asarray(keras.ops.convert_to_numpy(g), "float64") for g in grads]
lr = 0.1
m2 = build(); m2.set_weights(w0); m2.compile(optimizer=keras.optimizers.SGD(lr), loss="mse")
m2.train_on_batch(x, y)
w1 = [np.asarray(w, "float64") for w in m2.get_weights()]
applied = [(a - b) / lr for a, b in zip(w0, w1)]
print(json.dumps({"ref": [g.ravel().tolist() for g in ref], "applied": [g.ravel().tolist() for g in applied]}))
"""


def test_train_step_gradients_survive_forward_barriers():
    """The trainer's order receipt: one SGD step must apply exactly the
    gradient a fresh graph gives, THROUGH a forward barrier. With the loss
    tracker realizing the loss before `compute_gradients`, every weight
    upstream of the log1p barrier got a zero gradient (the first Dense did
    not move) while the head trained — silently."""
    r = _run(_BARRIER_GRADS, {"KERAS_TINYGRAD_TRAINER_JIT": "0"})
    for i, (ref, applied) in enumerate(zip(r["ref"], r["applied"])):
        scale = max(1e-6, max(abs(v) for v in ref))
        assert scale > 1e-4, f"reference gradient {i} is ~zero — the probe is not exercising the path"
        worst = max(abs(a - b) for a, b in zip(ref, applied))
        assert worst <= 1e-4 * scale, (
            f"weight {i}: applied gradient differs from the fresh graph's by {worst:.3e} (scale {scale:.3e})"
        )
