"""The backend is not thread-safe, and keras must know it.

keras' CallbackList runs batch-end hooks on a ThreadPoolExecutor when the
backend declares `IS_THREAD_SAFE = True`; `pythonify_logs` in that worker
realizes the metric result tensors (a kernel launch + a copyout) while the
main thread launches the next batch on the same device and recycles its
buffers. tinygrad shares one kernel object per program and sets its args
in place, and its LRU allocator has no lock, so on an asynchronous device
the worker's launch runs with the main thread's arguments and the read
returns a freed buffer's bytes: `model.evaluate` on the CL device (PoCL via
scripts/with_cl.sh) returned the previous batch's `count` (224.0 = 7 x 32)
or garbage about one run in three (2026-10-02). Found by tracing every
launch and copyout with its Python stack: the bad reads came from
`concurrent.futures.thread` frames. The CPU device never showed it.

Two receipts: the flag reaches keras (runs everywhere), and the evaluate
loop on the CL device (SKIPPED unless `KERAS_TINYGRAD_TEST_CL=1` and the
CL environment from scripts/with_cl.sh is present — the CL device is not a
default device and needs PoCL + the system ICD loader; this skip names
the only reason the test does not run). When it runs it is loud: 40
evaluates must all agree with the predict-MSE.
"""

import os
import pathlib
import subprocess
import sys

import pytest
from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent


def run(code, env=None):
    env = dict(os.environ, KERAS_BACKEND="tinygrad", **(env or {}))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=1200, preexec_fn=child_limits
    )
    assert proc.returncode == 0, f"child failed\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr[-4000:]}"
    return proc.stdout


def test_backend_declares_itself_not_thread_safe_and_keras_runs_callbacks_inline():
    out = run(
        """
import keras_tinygrad  # noqa: F401
import keras, numpy as np, threading
from keras.src import backend
from keras.src.callbacks.callback_list import CallbackList
assert backend.IS_THREAD_SAFE is False, backend.IS_THREAD_SAFE
threads = set()
class Spy(keras.callbacks.Callback):
    def on_test_batch_end(self, batch, logs=None): threads.add(threading.get_ident())
    def on_train_batch_end(self, batch, logs=None): threads.add(threading.get_ident())
cl = CallbackList([Spy()], model=None)
assert not (cl._async_train or cl._async_test or cl._async_predict), "keras would dispatch callbacks to a thread pool"
x = np.random.default_rng(0).normal(size=(64, 4)).astype("float32"); y = x[:, :1]
m = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(1)]); m.compile(optimizer="sgd", loss="mse")
m.fit(x, y, epochs=1, batch_size=16, verbose=0, callbacks=[Spy()])
m.evaluate(x, y, batch_size=16, verbose=0, callbacks=[Spy()])
assert threads == {threading.get_ident()}, f"callbacks ran on other threads: {threads}"
print("INLINE OK")
"""
    )
    assert "INLINE OK" in out, out


def test_evaluate_on_the_cl_device_is_deterministic():
    cl_env = os.environ.get("DEV") == "CL" and "OCL_ICD_VENDORS" in os.environ
    if os.environ.get("KERAS_TINYGRAD_TEST_CL") != "1" or not cl_env:
        pytest.skip(
            "CL device only: run `KERAS_TINYGRAD_TEST_CL=1 scripts/with_cl.sh uv run --group dev pytest "
            "tests/test_device_cl.py` — the CL device is not a default device (needs PoCL + the ICD loader)"
        )
    out = run(
        """
import keras_tinygrad  # noqa: F401
import keras, numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)).astype("float32"); y = x @ rng.normal(size=(8, 1)).astype("float32")
m = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
m.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
m.fit(x, y, epochs=2, batch_size=32, verbose=0)
p = m.predict(x, verbose=0, batch_size=32); ref = float(np.mean((p - y) ** 2))
bad = []
for i in range(40):
    ev = float(m.evaluate(x, y, batch_size=32, verbose=0))
    if abs(ev - ref) > 1e-4 * max(1.0, ref): bad.append((i, ev))
assert not bad, f"evaluate disagreed with predict-MSE {ref:.4f} in {len(bad)}/40 runs: {bad[:5]}"
print("CL EVALUATE OK", round(ref, 4))
"""
    )
    assert "CL EVALUATE OK" in out, out
