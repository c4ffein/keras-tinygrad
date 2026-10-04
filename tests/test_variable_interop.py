"""`tinygrad_tensor <op> keras.Variable` — the Variable on the right of a
python operator (a keras-core idiom; keras.io's "customizing saving and
serialization" guide does `dense_out + self.my_variable`). Before the patch
in ops/core.py tinygrad raised RuntimeError("Could not infer dtype of
<Variable ...>") before python could try the Variable's reflected op; other
backends never hit this because their Variables implement their framework's
array protocol. Found 2026-09-28 by running keras.io's guides under the
backend.

Subprocess per test (a Keras process is locked to one backend at import),
under tests/_limits.py caps like the other receipts.
"""

import os
import pathlib
import subprocess
import sys

from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent


def run(code):
    env = dict(os.environ, KERAS_BACKEND="tinygrad", PYTHONPATH=str(REPO / "tests"))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900, preexec_fn=child_limits
    )
    assert proc.returncode == 0, f"child failed\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr[-4000:]}"
    return proc.stdout


PRELUDE = """
import keras_tinygrad  # noqa: F401
import keras, numpy as np
from keras import ops
rng = np.random.default_rng(0)
v = keras.Variable(rng.normal(size=(3,)).astype("float32") + 2.0, name="v")   # +2: safe divisor
t = ops.convert_to_tensor(rng.normal(size=(2, 3)).astype("float32") + 2.0)
tn, vn = ops.convert_to_numpy(t), ops.convert_to_numpy(v.value)
"""


def test_every_binary_operator_with_the_variable_on_the_right():
    """Each `t <op> v` equals the same op spelled through keras.ops on the
    variable's value — python fell through to the Variable's reflected op."""
    out = run(
        PRELUDE
        + """
cases = {
    "add": (lambda a, b: a + b, ops.add), "sub": (lambda a, b: a - b, ops.subtract),
    "mul": (lambda a, b: a * b, ops.multiply), "truediv": (lambda a, b: a / b, ops.divide),
    "floordiv": (lambda a, b: a // b, ops.floor_divide), "mod": (lambda a, b: a % b, ops.mod),
    "pow": (lambda a, b: a ** b, ops.power),
    "lt": (lambda a, b: a < b, ops.less), "le": (lambda a, b: a <= b, ops.less_equal),
    "gt": (lambda a, b: a > b, ops.greater), "ge": (lambda a, b: a >= b, ops.greater_equal),
    "eq": (lambda a, b: a == b, ops.equal), "ne": (lambda a, b: a != b, ops.not_equal),
}
for name, (py, ref) in cases.items():
    got = ops.convert_to_numpy(py(t, v)); want = ops.convert_to_numpy(ref(t, v.value))
    assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=name)
m = ops.convert_to_tensor(rng.normal(size=(4, 3)).astype("float32"))
w = keras.Variable(rng.normal(size=(3, 2)).astype("float32"), name="w")
np.testing.assert_allclose(ops.convert_to_numpy(m @ w), ops.convert_to_numpy(ops.matmul(m, w.value)), rtol=1e-5)
bt = ops.convert_to_tensor(np.array([True, False, True])); bv = keras.Variable(np.array([True, True, False]), name="b")
for name, (py, ref) in {"and": (lambda a, b: a & b, ops.logical_and), "or": (lambda a, b: a | b, ops.logical_or),
                        "xor": (lambda a, b: a ^ b, ops.logical_xor)}.items():
    got, want = ops.convert_to_numpy(py(bt, bv)), ops.convert_to_numpy(ref(bt, bv.value))
    np.testing.assert_array_equal(got, want, err_msg=name)
np.testing.assert_array_equal(ops.convert_to_numpy(abs(v)), np.abs(vn))       # Variable.__abs__ -> Tensor.__abs__
np.testing.assert_array_equal(ops.convert_to_numpy(abs(t - 3.0)), np.abs(tn - 3.0))
print("OPS OK", len(cases) + 4)
"""
    )
    assert "OPS OK 17" in out, out


def test_gradient_reaches_a_variable_used_on_the_right():
    """A layer written `inputs * self.w + self.b` (variables on the right)
    trains bit-for-bit like the same layer written through keras.ops — the
    graph is built on the variables' tensors, not on a converted copy."""
    out = run(
        PRELUDE
        + """
class Right(keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],), initializer="ones", name="w")
        self.b = self.add_weight(shape=(input_shape[-1],), initializer="zeros", name="b")
    def call(self, inputs):
        return inputs * self.w + self.b

class Ref(Right):
    def call(self, inputs):
        return ops.add(ops.multiply(inputs, self.w), self.b)

x = rng.normal(size=(64, 3)).astype("float32"); y = (x * 3.0 - 1.0).astype("float32")
def train(layer_cls):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.layers.Input((3,)), layer_cls()])
    m.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
    m.fit(x, y, epochs=2, batch_size=16, verbose=0)
    return [ops.convert_to_numpy(w) for w in m.trainable_weights]
right, ref = train(Right), train(Ref)
for a, b in zip(right, ref):
    np.testing.assert_array_equal(a, b)
assert not np.allclose(right[0], 1.0) and not np.allclose(right[1], 0.0), right
print("GRAD OK", [r.round(3).tolist() for r in right])
"""
    )
    assert "GRAD OK" in out, out


def test_other_operand_types_keep_tinygrads_own_error():
    """The guard is for KerasVariable only: junk on the right still raises
    tinygrad's RuntimeError, exactly as before (additive patch, no behavior
    change for non-Keras users)."""
    out = run(
        PRELUDE
        + """
from tinygrad import Tensor
for junk in (object(), "x", {"a": 1}):
    try:
        t + junk
    except RuntimeError as e:
        assert "Could not infer dtype" in str(e), e
    else:
        raise AssertionError(f"no error for {junk!r}")
assert getattr(Tensor.__add__, "_keras_variable_guard", False), "guard not installed on Tensor.__add__"
print("GUARD OK")
"""
    )
    assert "GUARD OK" in out, out
