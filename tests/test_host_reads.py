"""Receipts for the host-read audit (2026-09-28): every op that reads a
data-dependent value onto the host must not cut the gradient of anything
sharing nodes with it.

Mechanism (tinygrad 0.13 and 0.14, probed): a host read (`.item()` /
`.numpy()`) realizes its graph, and tinygrad then swaps, on EVERY live
tensor, each node that got materialized (`.contiguous()` nodes and whatever
else the scheduler turns into a buffer) for that buffer — `Tensor.gradient`
returns silent zeros upstream of it. Detaching the read tensor changes
nothing: the nodes are the same objects. linalg hit this first (its loops
realized per step, 2026-09-21); the audit found the same exposure at every
site that reads a tensor-valued argument, predicate or count on the EAGER
train-step path (the JIT refuses host reads loudly at capture, then falls
back to eager). Fix: `core.host_read` — read, then restore every other
tensor's graph (docstring there).

Sites, with their verdicts (each is exercised below through a graph that
shares a `.contiguous()` node with the loss — `numpy.log1p`'s):

- affected before the fix, routed through `host_read`: `Tensor.__bool__`,
  `__float__` / `__int__`, `__array__` (numpy.py's dunder patches);
  `bincount` (min/max reads), `nonzero` (count), `split` / `array_split`
  (indices), `repeat` (counts), `roll` (shift, axis), `pad` (widths),
  `full` (0-d fill), `arange` (bounds), `eye` (dims), `quantile` (q),
  `_host_scalar` (`kaiser` beta); core.py's `cond` (predicate),
  `while_loop` (predicate), `_index_int` (`slice`, `slice_update`,
  `switch` indices). Before the fix every one of these returned a WRONG
  gradient in the probe below (the same table with the plain read).
- NOT affected, verified and left alone: `clip`'s `.item()` calls (numpy
  scalars only — a Tensor bound takes the lazy `maximum`/`minimum` path),
  `trapezoid` (a Tensor `dx` stays lazy; `_host_scalar` sees scalars
  only there), `arange` / `_host_scalar`'s numpy-scalar branches,
  `fori_loop` (python ints, no read at all), `convert_to_numpy` (the
  terminal read by design: predict / evaluate / metrics after the
  gradient).
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent


def _run(code):
    env = dict(os.environ, KERAS_BACKEND="tinygrad", KERAS_TINYGRAD_NO_VERSION_WARNING="1")
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=900,
        preexec_fn=child_limits,
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout.strip().splitlines()[-1]


def test_host_reads_keep_the_gradient_of_a_shared_graph():
    """For every site: build `h = log1p(x)` (a `.contiguous()` node shared
    by the loss and by the value the site reads), call the site with an
    argument derived from `h`, THEN differentiate the loss — the gradient
    must match the analytic 1/(1+x) * w, not zeros. The `control` entry
    is the same check with no read at all (the loss graph itself is sane),
    and `mechanism` shows the raw tinygrad read still zeroes the gradient
    (if that ever stops failing, `host_read` can go)."""
    out = _run("""
import json, warnings
import keras_tinygrad, keras
import numpy as np
from tinygrad import Tensor
from keras_tinygrad.src.ops import core, numpy as knp
warnings.simplefilter("ignore")
x0 = np.array([0.5, 1.0, 2.0, 3.0], dtype="float32")
w0 = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
expect = w0 / (1.0 + x0)
i32 = lambda t: t.cast("int32")
sites = {
    "control": lambda x, h: None,
    "bool": lambda x, h: bool(h.sum() > 0),
    "float": lambda x, h: float(h.sum()),
    "int": lambda x, h: int(h.sum()),
    "array": lambda x, h: np.asarray(h),
    "bincount": lambda x, h: knp.bincount(i32(h)),
    "nonzero": lambda x, h: knp.nonzero(h),
    "split": lambda x, h: knp.split(x, i32(h[0] * 0 + 2)),
    "array_split": lambda x, h: knp.array_split(x, i32(h[0] * 0 + 2)),
    "repeat": lambda x, h: knp.repeat(x, i32(h[:1] * 0 + 2)),
    "roll": lambda x, h: knp.roll(x, i32(h[0] * 0 + 1), i32(h[0] * 0)),
    "pad": lambda x, h: knp.pad(x, [[i32(h[0] * 0 + 1), 0]]),
    "full": lambda x, h: knp.full((2,), h.sum()),
    "arange": lambda x, h: knp.arange(i32(h[0] * 0), i32(h[0] * 0 + 3)),
    "eye": lambda x, h: knp.eye(i32(h[0] * 0 + 2)),
    "quantile": lambda x, h: knp.quantile(x, h[0] * 0 + 0.5),
    "trapezoid_dx": lambda x, h: knp.trapezoid(x, dx=h[0] * 0 + 0.5),
    "kaiser_beta": lambda x, h: knp.kaiser(4, h[0] * 0 + 2.0),
    "cond": lambda x, h: core.cond(h.sum() > 0, lambda: 1, lambda: 0),
    "while_loop": lambda x, h: core.while_loop(lambda i: i < h.sum(), lambda i: i + 1, 0),
    "slice": lambda x, h: core.slice(x, [i32(h[0] * 0 + 1)], [2]),
    "slice_update": lambda x, h: core.slice_update(x, [i32(h[0] * 0 + 1)], Tensor(np.zeros(2, "float32"))),
    "switch": lambda x, h: core.switch(i32(h[0] * 0 + 1), [lambda: 0, lambda: 1]),
    # the raw mechanism, on purpose NOT through host_read
    "mechanism": lambda x, h: h.max().item(),
}
results = {}
for name, site in sites.items():
    x = Tensor(x0); h = knp.log1p(x); loss = (h * Tensor(w0)).sum()
    site(x, h)
    (g,) = loss.gradient(x)
    results[name] = bool(np.allclose(g.numpy(), expect, rtol=1e-3, atol=1e-4))
print(json.dumps(results))
""")
    results = json.loads(out)
    assert results.pop("control"), "the control (no read) failed — the probe graph itself is wrong"
    assert not results.pop("mechanism"), "a raw .item() no longer zeroes the gradient — host_read may be obsolete"
    bad = sorted(k for k, ok in results.items() if not ok)
    assert not bad, f"host reads zeroed the gradient at: {bad}"


def test_host_read_values_and_graph_are_unchanged():
    """`host_read` returns the value the plain read would (dtypes incl.
    bool / int / bfloat16, 0-d and n-d), restores the read tensor's own
    graph as well (a second read recomputes the same value), and leaves
    already-realized tensors' buffers alone (no gratuitous recompute)."""
    out = _run("""
import json
import keras_tinygrad, keras
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters
from keras_tinygrad.src.ops import core
r = {}
x = Tensor(np.array([[1.5, -2.0], [0.0, 4.0]], dtype="float32"))
r["float_nd"] = core.host_read(x * 2).tolist() == [[3.0, -4.0], [0.0, 8.0]]
r["bool_0d"] = core.host_read((x.sum() > 0)).item() is True
r["int_1d"] = core.host_read(x.cast("int32").sum(axis=1)).tolist() == [-1, 4]
r["bf16"] = abs(float(core.host_read(x.cast("bfloat16").sum())) - 3.5) < 1e-2
h = (x + 1).contiguous() * 3
u = h.uop
core.host_read(h)
r["graph_restored"] = h.uop is u
r["reread_same"] = core.host_read(h).tolist() == core.host_read(h).tolist() == [[7.5, -3.0], [3.0, 15.0]]
buf = (x + 1).realize()
k0 = GlobalCounters.kernel_count
core.host_read(buf)
r["realized_stays_realized"] = buf.uop.has_buffer_identity()
core.host_read(buf)
r["no_recompute_of_realized"] = GlobalCounters.kernel_count == k0
print(json.dumps(r))
""")
    results = json.loads(out)
    bad = sorted(k for k, ok in results.items() if not ok)
    assert not bad, f"host_read misbehaved: {bad}"


def test_eager_train_step_with_a_host_read_matches_the_read_free_step():
    """The user-facing shape of the bug: a layer whose `call` reads a value
    host-side (here `keras.ops.cond` on a predicate derived from its own
    activations) trained the SAME as the read-free layer only in the JIT
    (which refuses the read at capture) — the eager fallback trained on
    zero gradients for everything upstream of the read. Both layers must
    now land on identical weights after a fit, eager and JIT alike."""
    out = _run("""
import json, os
import keras_tinygrad, keras
import numpy as np
from keras import ops

class Plain(keras.layers.Layer):
    def call(self, x):
        return ops.log1p(ops.relu(x)) * 2.0

class Reads(keras.layers.Layer):
    def call(self, x):
        h = ops.log1p(ops.relu(x))
        # a data-dependent python branch on the activations (numpy-backend style)
        return ops.cond(ops.sum(h) > -1.0, lambda: h * 2.0, lambda: h * 3.0)

def fit(layer_cls, jit):
    os.environ["KERAS_TINYGRAD_TRAINER_JIT"] = "1" if jit else "0"
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(6), layer_cls(), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.SGD(0.05), loss="mse")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 8)).astype("float32"); y = rng.normal(size=(64, 1)).astype("float32")
    init = [w.tolist() for w in m.get_weights()]
    m.fit(x, y, epochs=2, batch_size=16, verbose=0)
    return init, [w.tolist() for w in m.get_weights()]

init, ref = fit(Plain, jit=False)
r = {"weights_moved": init != ref}
r["plain_jit_matches_eager"] = fit(Plain, jit=True)[1] == ref
for name, jit in (("reads_eager", False), ("reads_jit", True)):
    r[name] = fit(Reads, jit)[1] == ref
print(json.dumps(r))
""")
    results = json.loads(out)
    bad = sorted(k for k, ok in results.items() if not ok)
    assert not bad, f"host-read layer diverged from the read-free layer: {bad}"
