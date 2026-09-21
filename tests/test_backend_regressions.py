"""Receipts for backend bugs found 2026-08-30 — each test is the check that
would have caught its bug before it shipped (m0 README "three bugs" + the
code-review findings). Subprocess-based where process state matters."""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent


def _run(code, *args):
    env = dict(os.environ, KERAS_BACKEND="tinygrad", KERAS_TINYGRAD_NO_VERSION_WARNING="1")
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=600,
        preexec_fn=child_limits,
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout.strip().splitlines()[-1]


_FIT = """
import os, sys
import keras_tinygrad
import numpy as np
import keras

keras.utils.set_random_seed(0)
seed = int(sys.argv[1])
model = keras.Sequential([keras.layers.Input((16,)), keras.layers.Dense(8, activation="relu"),
                          keras.layers.Dropout(0.5, seed=seed), keras.layers.Dense(1)])
model.compile(optimizer="sgd", loss="mse")
rng = np.random.default_rng(123)
x, y = rng.normal(size=(64, 16)).astype("float32"), rng.normal(size=(64, 1)).astype("float32")
h = model.fit(x, y, epochs=3, batch_size=16, verbose=0)
import json; print(json.dumps([round(v, 8) for v in h.history["loss"]]))
"""


def test_dropout_training_is_reproducible_across_processes():
    """The device-RNG stream must derive from Keras' seed state, not the
    wall clock: same seeds, two fresh processes, identical loss history.
    (tinygrad's default Tensor._seed is time-based — this exact test fails
    on the unseeded version of the device path.)"""
    a = json.loads(_run(_FIT, "42"))
    b = json.loads(_run(_FIT, "42"))
    assert a == b, f"same seeds, different runs: {a} vs {b}"


def test_dropout_seed_changes_the_run():
    """Dropout(seed=N) must influence the masks — the device path once
    silently discarded explicit layer seeds (they arrive wrapped in a
    SeedGenerator, not as ints)."""
    a = json.loads(_run(_FIT, "42"))
    b = json.loads(_run(_FIT, "43"))
    assert a != b, f"different Dropout seeds produced identical runs: {a}"


def test_assign_of_realized_tensor_schedules_no_copy():
    """Variable assign order must be contiguous().detach().realize():
    detach-first defeats tinygrad's buffer-identity short-circuit and turns
    every assign into a weight-sized copy kernel + allocation."""
    out = _run("""
import keras_tinygrad, keras
import numpy as np
from keras_tinygrad.src.ops import core
from tinygrad.helpers import GlobalCounters
v = core.Variable(initializer=np.zeros((64, 64), dtype="float32"), trainable=True)
x = core.convert_to_tensor(np.ones((64, 64), dtype="float32")).realize()
k0 = GlobalCounters.kernel_count
v.assign(x)
print(GlobalCounters.kernel_count - k0)
""")
    assert out == "0", f"assign of a realized tensor scheduled {out} kernel(s)"


def test_python_scalars_convert_to_const_uops():
    """Scalars must fold as CONST uops (immediates in traced kernels): the
    buffer-backed alternative exported SGD's momentum as zeroed memory and
    the bundles silently trained plain SGD (m0 README bug #3)."""
    out = _run("""
import keras_tinygrad, keras
from keras_tinygrad.src.ops import core
print(core.convert_to_tensor(0.9).uop.base.op)
""")
    assert "CONST" in out, f"python scalar became {out}, not a CONST uop"


def test_export_train_step_bakes_the_real_lr():
    """The whole export path, end to end, on the NULL device — asserting the
    values that NULL fakery once silently zeroed: the baked learning rate
    (read through NULL it comes back 0.0 — fourth occurrence of the bug
    class) and a validator-clean runner."""
    out = _run(
        """
import os, sys, json, struct
os.environ["DEV"] = "NULL:WGSL"; os.environ["NULL_ALLOW_COPYOUT"] = "1"
os.environ["KERAS_TINYGRAD_TRAINER_JIT"] = "0"
import keras_tinygrad, keras
from keras_tinygrad.webgpu import export_train_step
model = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(4, activation="relu"),
                          keras.layers.Dropout(0.5), keras.layers.Dense(2)])
o = export_train_step(model, keras.optimizers.SGD(learning_rate=0.037, momentum=0.9),
                      keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None),
                      batch_size=4, input_shape=(8,), learning_rate=0.037)
w = o["weights"]; n = struct.unpack("<Q", w[:8])[0]; h = json.loads(w[8:8+n])
off = next(v for k, v in h.items() if "learning_rate" in k)["data_offsets"][0]
baked = struct.unpack("<f", w[8+n+off:8+n+off+4])[0]
print(json.dumps({"lr": round(baked, 6), "kernels": o["meta"]["kernels"],
                  "rng": any("rng/" in k for k in o["meta"]["stateEntries"])}))
"""
    )
    got = json.loads(out)
    assert got["lr"] == 0.037, f"baked lr {got['lr']} != 0.037 (NULL fake-read?)"
    assert got["kernels"] > 0 and got["rng"], got


def test_dropout_training_is_reproducible_in_one_process():
    """set_random_seed + build + fit, twice in ONE process, must agree: the
    device stream re-seeds from the Keras seed state whenever a train
    function is built (`random.reset_device_stream`). tinygrad's stream is
    process-global, and the once-per-process seeding this replaced carried
    the first run's stream into the second (found in the 2026-09-01 review).
    A CONTINUED fit on the first model must NOT replay the first run's
    masks: its generator advanced, so its stream moved on."""
    out = _run("""
import keras_tinygrad, keras, numpy as np, json
rng = np.random.default_rng(1)
x, y = rng.normal(size=(64, 16)).astype("float32"), rng.normal(size=(64, 1)).astype("float32")
def run():
    keras.utils.set_random_seed(0)
    layers = [keras.layers.Input((16,)), keras.layers.Dense(8), keras.layers.Dropout(0.5), keras.layers.Dense(1)]
    m = keras.Sequential(layers)
    m.compile(optimizer="sgd", loss="mse")
    return m.fit(x, y, epochs=2, batch_size=16, verbose=0).history["loss"], m
a, m = run()
b, _ = run()
c = m.fit(x, y, epochs=2, batch_size=16, verbose=0).history["loss"]
print(json.dumps({"same": a == b, "continued_differs": c != a}))
""")
    got = json.loads(out)
    assert got == {"same": True, "continued_differs": True}, got


def test_seedless_random_op_in_step_jits_and_stays_fresh():
    """A custom layer calling `keras.random.dropout` with NO seed draws
    through Keras' global generator. Inside the train step that draw now
    takes the device path, so the step JITs (one capture) and every replay
    draws a fresh mask: six identical batches at lr=0 give six distinct
    losses. (On the host path the capture-time draw would be baked;
    tinygrad's JitError made that loud and the step fell back to eager.)"""
    out = _run("""
import keras_tinygrad, keras, numpy as np, json
from keras_tinygrad.src import trainer as T
inst = []
orig = T._TrainStepJit.__init__
T._TrainStepJit.__init__ = lambda self, tr: (orig(self, tr), inst.append(self))[0]
class SeedlessDrop(keras.layers.Layer):
    def call(self, x, training=None):
        return keras.random.dropout(x, 0.5) if training else x
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.layers.Input((16,)), keras.layers.Dense(8), SeedlessDrop(), keras.layers.Dense(1)])
m.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0), loss="mse")
x = np.ones((4, 16), "float32"); y = np.ones((4, 1), "float32")
losses = [float(m.train_on_batch(x, y)) for _ in range(6)]
print(json.dumps({"jit": not inst[-1]._disabled, "captures": len(inst[-1]._jits), "distinct": len(set(losses))}))
""")
    got = json.loads(out)
    assert got == {"jit": True, "captures": 1, "distinct": 6}, got


def test_device_rng_accepts_numpy_integer_dims():
    """The host path's numpy accepted numpy-integer dims; tinygrad's `rand`
    rejects them (exact-class argfix check), so the device path coerces."""
    out = _run("""
import keras_tinygrad, keras, numpy as np
from keras_tinygrad.src import random as R
from keras_tinygrad.src.ops.core import device_rng_scope
from tinygrad import Tensor
with device_rng_scope():
    a = R.normal((np.int64(3),), seed=keras.random.SeedGenerator(1))
    b = R.uniform([np.int32(2), 2], seed=keras.random.SeedGenerator(2))
    c = R.dropout(Tensor.ones(2, 3), 0.5, noise_shape=(np.int64(2), 1), seed=keras.random.SeedGenerator(3))
print(tuple(a.shape), tuple(b.shape), tuple(c.shape))
""")
    assert out == "(3,) (2, 2) (2, 3)", out


_EXPORT_PRELUDE = """
import os, sys, json, struct
os.environ["DEV"] = "NULL:WGSL"; os.environ["NULL_ALLOW_COPYOUT"] = "1"
os.environ["KERAS_TINYGRAD_TRAINER_JIT"] = "0"
import keras_tinygrad, keras
from keras_tinygrad.webgpu import export_train_step
def floats(out, needle):
    w = out["weights"]; n = struct.unpack("<Q", w[:8])[0]; h = json.loads(w[8:8+n])
    return {k[len("state."):]: list(struct.unpack(f"<{(b-a)//4}f", w[8+n+a:8+n+b]))
            for k, e in h.items() for a, b in [e["data_offsets"]] if needle in k and e["dtype"] == "F32"}
sgd = lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
"""


def test_export_initial_values_follow_the_keras_initializers():
    """BatchNormalization's gamma and moving_variance are ONES. The "Glorot
    for kernels, zeros elsewhere" exporter shipped them as zeros — a dead
    network, silently. Values now come from each weight's declared
    initializer (`webgpu._initializer_map` + `_host_initial_values`)."""
    out = _run(
        _EXPORT_PRELUDE
        + """
L = keras.layers
m = keras.Sequential([L.Input((8,)), L.Dense(4), L.BatchNormalization(), L.Dense(2)])
o = export_train_step(m, sgd(), keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None),
                      batch_size=4, input_shape=(8,), learning_rate=0.01)
v = floats(o, "batch_normalization/")
print(json.dumps({k.rsplit("/", 1)[-1]: sorted(set(x)) for k, x in v.items()}))
"""
    )
    got = json.loads(out)
    assert got == {"gamma": [1.0], "beta": [0.0], "moving_mean": [0.0], "moving_variance": [1.0]}, got


def test_export_refuses_what_it_cannot_initialize_or_reduce():
    """Loud over wrong: an initializer without a host port raises
    NotImplementedError (never zeros); Keras' default loss reduction raises
    ValueError before tracing (its divisor is a host tuple that exported as
    an unwritten buffer); and a regression loss gets float labels shaped
    like the output, not the classifier's int32 ids."""
    out = _run(
        _EXPORT_PRELUDE
        + """
errors = []
L = keras.layers
kw = dict(batch_size=4, input_shape=(8,), learning_rate=0.01)
m = keras.Sequential([L.Input((8,)), L.Dense(3, kernel_initializer="he_normal"), L.Dense(1)])
try:
    export_train_step(m, sgd(), keras.losses.MeanSquaredError(reduction=None), **kw)
except NotImplementedError as e:
    errors.append("HeNormal" in str(e))
m = keras.Sequential([L.Input((8,)), L.Dense(1)])
try:
    export_train_step(m, sgd(), keras.losses.MeanSquaredError(), **kw)
except ValueError as e:
    errors.append("reduction=None" in str(e))
o = export_train_step(m, sgd(), keras.losses.MeanSquaredError(reduction=None), **kw)
meta = o["meta"]
labels = [meta["labelShape"], meta["labelDtype"]]
print(json.dumps({"errors": errors, "labels": labels, "kernels": meta["kernels"] > 0}))
"""
    )
    got = json.loads(out)
    assert got == {"errors": [True, True], "labels": [[1], "float32"], "kernels": True}, got


def test_linalg_gradients_are_not_silently_zero():
    """tinygrad 0.13 returns ZERO gradients through a realized tensor, and a
    host read (`.item()`) realizes every `.contiguous()` node upstream of it.
    linalg's loops called `.realize()` per step and its validity checks read
    the returned factors: inv / det / solve / cholesky / eigh / svd / lstsq
    all differentiated to exact zeros (found 2026-09-21). Each op's gradient
    is checked against finite differences of the numpy reference."""
    out = _run("""
import json, warnings
import keras_tinygrad, keras
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters
from keras_tinygrad.src.ops import core, linalg
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
B = rng.normal(size=(3, 3)); A0 = (B @ B.T + 3 * np.eye(3)).astype("float32")
R0 = rng.normal(size=(4, 3)).astype("float32"); b0 = rng.normal(size=(3, 2)).astype("float32")
W = rng.normal(size=(8, 8))
def w(a): return W[: a.shape[0], : a.shape[1]] if a.ndim == 2 else W[0, : a.shape[0]]
def sym(X): return (X + X.T) / 2
def fd(f, X, h=1e-3):
    X = X.astype("float64"); g = np.zeros_like(X)
    for i in np.ndindex(*X.shape):
        P = X.copy(); P[i] += h; M = X.copy(); M[i] -= h
        g[i] = (f(P) - f(M)) / (2 * h)
    return g
b64 = b0.astype("float64")
cases = {
    "inv": (A0, linalg.inv, np.linalg.inv),
    "det": (A0, lambda A: linalg.det(A).reshape(1), lambda X: np.array([np.linalg.det(X)])),
    "solve": (A0, lambda A: linalg.solve(A, Tensor(b0)), lambda X: np.linalg.solve(X, b64)),
    "cholesky": (A0, lambda A: linalg.cholesky((A + A.transpose()) / 2), lambda X: np.linalg.cholesky(sym(X))),
    "eigh": (A0, lambda A: linalg.eigh(A)[0], lambda X: np.linalg.eigh(sym(X))[0]),
    "svd": (R0, lambda A: linalg.svd(A)[1], lambda X: np.linalg.svd(X)[1]),
    "qr": (R0, lambda A: linalg.qr(A)[1].abs(), lambda X: np.abs(np.linalg.qr(X)[1])),
    "lstsq": (R0, lambda A: linalg.lstsq(A, Tensor(np.ones((4, 2), "float32"))),
              lambda X: np.linalg.lstsq(X, np.ones((4, 2)), rcond=None)[0]),
    "solve_triangular": (A0, lambda A: linalg.solve_triangular(A, Tensor(b0), lower=True),
                         lambda X: np.linalg.solve(np.tril(X), b64)),
}
results = {}
for name, (X0, f, ref) in cases.items():
    A = Tensor(X0); o = f(A)
    (g,) = (o * Tensor(w(o).astype("float32"))).sum().gradient(A)
    e = fd(lambda X: float((ref(X) * w(ref(X))).sum()), X0)
    results[name] = bool(np.abs(e).max() > 1e-2 and np.allclose(g.numpy(), e, rtol=2e-2, atol=2e-3))
# lu_factor has no one-line numpy reference: nonzero is the receipt.
A = Tensor(A0); (g,) = linalg.lu_factor(A)[0].sum().gradient(A)
results["lu_factor"] = bool(g.numpy().any())
# Inside the trainer's tape: no host read at all (nothing is scheduled while
# the ops build their graphs), and `eig`, whose symmetry gate needs one, is loud.
with core.custom_gradient_tape():
    k0 = GlobalCounters.kernel_count
    linalg.cholesky(Tensor(A0)); linalg.eigh(Tensor(A0)); linalg.svd(Tensor(R0))
    results["no_host_read_under_tape"] = GlobalCounters.kernel_count == k0
    try:
        linalg.eig(Tensor(A0)); results["eig_loud_under_tape"] = False
    except NotImplementedError:
        results["eig_loud_under_tape"] = True
    # A Jacobi iteration too long to hold lazily (keras' own test_svd size:
    # uncapped, that graph stalled an 8-core box) has no gradient: loud here,
    # and refused BEFORE anything is built or scheduled.
    k0 = GlobalCounters.kernel_count
    try:
        linalg.svd(Tensor(np.ones((4, 30, 20), "float32"))); results["big_svd_loud_under_tape"] = False
    except NotImplementedError:
        results["big_svd_loud_under_tape"] = GlobalCounters.kernel_count == k0
# ... and the eager checks still fire.
try:
    linalg.cholesky(Tensor(-A0)); results["cholesky_still_raises"] = False
except ValueError:
    results["cholesky_still_raises"] = True
print(json.dumps(results))
""")
    got = json.loads(out)
    assert got and all(got.values()), got


def test_elastic_transform_runs():
    """The 2026-09-21 series dropped image.py's `draw_seed` import while
    reordering the block: NameError on the first call, invisible to lint
    (backend sources are excluded; `make lint-check` now runs F821 on them)
    and to every other local test."""
    out = _run("""
import keras_tinygrad, keras
import numpy as np
y = keras.ops.image.elastic_transform(np.ones((1, 8, 8, 3), "float32"), seed=0)
print(tuple(y.shape))
""")
    assert out == "(1, 8, 8, 3)", out
